"""
Image Denoising — Model & Training Loop
THIS IS THE FILE YOU MODIFY.

Baseline: plain convolutional stack (no skip connections, no residual learning).
Plenty of room for improvement — see program.md for ideas.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    NOISE_SIGMA,
    TIME_BUDGET,
    get_train_loader,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────
# Hyperparameters (feel free to change)
# ──────────────────────────────────────────────────────────────
CHANNELS    = 64        # base channel width
NUM_LAYERS  = 6         # number of conv layers
LR          = 1e-3      # learning rate
BATCH_SIZE  = 32        # training batch size

# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class SimpleDenoiser(nn.Module):
    """
    Plain convolutional denoiser — intentionally basic.

    No skip connections, no downsampling, no residual learning.
    All convolutions are 3×3 with padding=1 (preserves spatial size).
    """

    def __init__(self, channels=CHANNELS, num_layers=NUM_LAYERS):
        super().__init__()
        layers = []
        # input conv
        layers.append(nn.Conv2d(3, channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # hidden convs
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        # output conv
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)  # residual learning: predict noise, subtract


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train():
    # ── Device ────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Model & Optimizer ─────────────────────────────────────
    model = SimpleDenoiser().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = get_train_loader(batch_size=BATCH_SIZE)

    # ── Training loop (time-budgeted) ─────────────────────────
    model.train()
    t_start = time.time()
    step = 0
    epoch = 0
    best_loss = float("inf")

    while True:
        epoch += 1
        for noisy, clean in loader:
            if time.time() - t_start >= TIME_BUDGET:
                break

            noisy = noisy.to(device)
            clean = clean.to(device)

            pred = model(noisy)
            loss = F.l1_loss(pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  step {step:5d}  loss {loss.item():.6f}  "
                      f"[{elapsed:.0f}s / {TIME_BUDGET}s]")

        if time.time() - t_start >= TIME_BUDGET:
            break

    training_time = time.time() - t_start
    print(f"\nTraining done: {step} steps, {epoch} epochs, {training_time:.1f}s")

    # ── Evaluation ────────────────────────────────────────────
    t_eval = time.time()
    val_psnr = evaluate_model(model, device)
    eval_time = time.time() - t_eval
    total_time = time.time() - t_start

    # ── Print results (autoresearch format) ───────────────────
    print(f"\n---")
    print(f"val_psnr:         {val_psnr:.4f}")
    print(f"training_seconds: {training_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_epochs:       {epoch}")
    print(f"num_params:       {n_params}")


if __name__ == "__main__":
    train()
