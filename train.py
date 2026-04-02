"""
Autoresearch: Time Series Forecasting & Anomaly Detection
Training script. Single-GPU/CPU, single-file.

This is the ONLY file the agent modifies.
Everything is fair game: model architecture, optimizer, hyperparameters,
loss functions, feature engineering, sequence processing, etc.

Usage: python train.py
"""

import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, CACHE_DIR,
    load_prepared_data, make_infinite_dataloader, evaluate
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_SIZE = 64           # LSTM hidden dimension
NUM_LAYERS = 2             # number of LSTM layers
DROPOUT = 0.2              # dropout rate
BIDIRECTIONAL = False       # use bidirectional LSTM
USE_ATTENTION = False       # add attention mechanism on top of LSTM
ANOMALY_HEAD = True         # include anomaly detection head

# Optimization
LEARNING_RATE = 1e-3        # initial learning rate
WEIGHT_DECAY = 1e-5         # L2 regularization
BATCH_SIZE = 64             # batch size
LR_SCHEDULER = "cosine"    # "cosine", "step", "plateau", or "none"
WARMUP_STEPS = 50           # LR warmup steps

# Loss weights
FORECAST_LOSS_WEIGHT = 1.0  # weight for forecasting MSE loss
ANOMALY_LOSS_WEIGHT = 0.3   # weight for anomaly BCE loss

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Simple temporal attention over LSTM outputs."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        scores = self.attn(lstm_out)          # (B, T, 1)
        weights = F.softmax(scores, dim=1)    # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)  # (B, H)
        return context


class ForecastAnomalyModel(nn.Module):
    """
    LSTM-based model for joint forecasting and anomaly detection.
    The agent can modify everything here: architecture, layers, activations.
    """
    def __init__(self, n_features, hidden_size, num_layers, dropout,
                 bidirectional, use_attention, anomaly_head):
        super().__init__()
        self.use_attention = use_attention
        self.anomaly_head = anomaly_head

        # Encoder
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        enc_dim = hidden_size * (2 if bidirectional else 1)

        # Attention (optional)
        if use_attention:
            self.attention = Attention(enc_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.ReLU(),
            nn.Linear(enc_dim // 2, 1),
        )

        # Anomaly head (optional)
        if anomaly_head:
            self.anomaly_detection_head = nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 2),
                nn.ReLU(),
                nn.Linear(enc_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        """
        Args:
            x: (B, T, F) input sequences
        Returns:
            dict with "forecast" (B,) and "anomaly" (B,)
        """
        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(x)  # lstm_out: (B, T, H)

        # Get representation
        if self.use_attention:
            rep = self.attention(lstm_out)    # (B, H)
        else:
            rep = lstm_out[:, -1, :]         # (B, H) last timestep

        rep = self.dropout(rep)

        # Forecast
        forecast = self.forecast_head(rep).squeeze(-1)  # (B,)

        # Anomaly
        if self.anomaly_head:
            anomaly = self.anomaly_detection_head(rep).squeeze(-1)  # (B,)
        else:
            # If no anomaly head, use forecast residual as anomaly signal
            anomaly = torch.zeros_like(forecast)

        return {"forecast": forecast, "anomaly": anomaly}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Load data
splits, metadata, target_scaler = load_prepared_data(device="cpu")
n_features = metadata["n_features"]
seq_length = metadata["seq_length"]
print(f"Features: {n_features}, Seq length: {seq_length}")
print(f"Train: {metadata['n_train']}, Val: {metadata['n_val']}, Test: {metadata['n_test']}")

# Build model
model = ForecastAnomalyModel(
    n_features=n_features,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    use_attention=USE_ATTENTION,
    anomaly_head=ANOMALY_HEAD,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# LR Scheduler
if LR_SCHEDULER == "cosine":
    # Estimate total steps from time budget (rough)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5000, eta_min=LEARNING_RATE * 0.01)
elif LR_SCHEDULER == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
elif LR_SCHEDULER == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200)
else:
    scheduler = None

# Data loader
train_loader = make_infinite_dataloader(splits["train"], BATCH_SIZE, shuffle=True)

# Loss functions
forecast_criterion = nn.MSELoss()
anomaly_criterion = nn.BCELoss()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"Time budget: {TIME_BUDGET}s")
print("Training...")

t_start_training = time.time()
total_training_time = 0
step = 0
smooth_loss = 0
best_val_mae = float("inf")

# Warmup tracking
warmup_done = False

while True:
    t0 = time.time()

    (X_batch, y_f_batch, y_a_batch), epoch = next(train_loader)
    X_batch = X_batch.to(device)
    y_f_batch = y_f_batch.to(device)
    y_a_batch = y_a_batch.to(device)

    # Forward
    output = model(X_batch)

    # Losses
    loss_forecast = forecast_criterion(output["forecast"], y_f_batch)
    loss = FORECAST_LOSS_WEIGHT * loss_forecast

    if ANOMALY_HEAD and output["anomaly"].requires_grad:
        loss_anomaly = anomaly_criterion(output["anomaly"], y_a_batch)
        loss = loss + ANOMALY_LOSS_WEIGHT * loss_anomaly

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # LR schedule
    if scheduler is not None:
        if LR_SCHEDULER == "plateau":
            pass  # updated after validation
        else:
            scheduler.step()

    # Warmup
    if step < WARMUP_STEPS:
        warmup_frac = (step + 1) / WARMUP_STEPS
        for pg in optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * warmup_frac

    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Smoothed loss
    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss.item()
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    # Fast fail
    if math.isnan(loss.item()) or loss.item() > 1000:
        print("FAIL: loss exploded")
        exit(1)

    # Logging
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    remaining = max(0, TIME_BUDGET - total_training_time)
    current_lr = optimizer.param_groups[0]["lr"]

    if step % 50 == 0:
        print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased_loss:.6f} "
              f"| lr: {current_lr:.2e} | dt: {dt*1000:.0f}ms | epoch: {epoch} "
              f"| remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    step += 1

    # Time's up
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
metrics = evaluate(model, splits["val"], target_scaler, BATCH_SIZE, device)

# Also evaluate on test set (for reference, not for agent decisions)
test_metrics = evaluate(model, splits["test"], target_scaler, BATCH_SIZE, device)

# Memory stats
if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

t_end = time.time()

# ---------------------------------------------------------------------------
# Summary (agent parses this)
# ---------------------------------------------------------------------------

print("---")
print(f"val_mae:            {metrics['val_mae']:.6f}")
print(f"val_scaled_mae:     {metrics['val_scaled_mae']:.6f}")
print(f"val_rmse:           {metrics['val_rmse']:.6f}")
print(f"val_r2:             {metrics['val_r2']:.4f}")
print(f"anomaly_f1:         {metrics['anomaly_f1']:.4f}")
print(f"anomaly_precision:  {metrics['anomaly_precision']:.4f}")
print(f"anomaly_recall:     {metrics['anomaly_recall']:.4f}")
print(f"combined_score:     {metrics['combined_score']:.6f}")
print(f"test_mae:           {test_metrics['val_mae']:.6f}")
print(f"test_r2:            {test_metrics['val_r2']:.4f}")
print(f"training_seconds:   {total_training_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
print(f"num_steps:          {step}")
print(f"num_params:         {num_params}")
print(f"hidden_size:        {HIDDEN_SIZE}")
print(f"num_layers:         {NUM_LAYERS}")
print(f"batch_size:         {BATCH_SIZE}")
print(f"learning_rate:      {LEARNING_RATE}")
