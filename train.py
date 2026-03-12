"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    FORWARD_HOURS,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
    load_val_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [1, 4, 12, 24, 72, 168]
VOLATILITY_WINDOWS = [24, 168]
MAX_LOOKBACK = 168


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute features from OHLCV data."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts = df["timestamp"].values

    hourly_returns = np.zeros(len(close))
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0

    feature_cols = []

    # 1. Returns over lookback windows
    for lb in RETURN_LOOKBACKS:
        ret = np.full(len(close), np.nan)
        ret[lb:] = close[lb:] / close[:-lb] - 1.0
        feature_cols.append(ret)

    # 2. Volatility (rolling std of hourly returns)
    hr_series = pd.Series(hourly_returns)
    for w in VOLATILITY_WINDOWS:
        vol = hr_series.rolling(w, min_periods=w).std().values
        feature_cols.append(vol)

    # 3. Volume ratio: 24h avg / 168h avg
    vol_series = pd.Series(volume)
    vol_24 = vol_series.rolling(24, min_periods=24).mean().values
    vol_168 = vol_series.rolling(168, min_periods=168).mean().values
    vol_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Hour of day (cyclical)
    hours = pd.to_datetime(ts).hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    # 5. RSI-like momentum (14-period and 48-period)
    for period in [14, 48]:
        gains = np.where(hourly_returns > 0, hourly_returns, 0.0)
        losses = np.where(hourly_returns < 0, -hourly_returns, 0.0)
        avg_gain = pd.Series(gains).rolling(period, min_periods=period).mean().values
        avg_loss = pd.Series(losses).rolling(period, min_periods=period).mean().values
        rsi = np.where(avg_loss > 0, avg_gain / (avg_gain + avg_loss), 0.5)
        feature_cols.append(rsi)

    # 6. High-low range ratio (volatility proxy)
    hl_range = (high - low) / np.where(close > 0, close, 1.0)
    hl_24 = pd.Series(hl_range).rolling(24, min_periods=24).mean().values
    feature_cols.append(hl_24)

    features = np.column_stack(feature_cols)

    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]

    return features, timestamps


def compute_targets(df: pd.DataFrame) -> np.ndarray:
    """Compute 24-hour forward returns."""
    close = df["close"].values.astype(np.float64)
    n = len(close)
    targets = np.full(n, np.nan)
    targets[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0
    return targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

N_FEATURES = len(RETURN_LOOKBACKS) + len(VOLATILITY_WINDOWS) + 1 + 2 + 2 + 1  # 14


class ForwardReturnModel(nn.Module):
    """Linear model with output clipping."""

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.linear(x).squeeze(-1)
        # Hard clip predictions to [-0.03, 0.03] to prevent extreme positions
        return torch.clamp(raw, -0.03, 0.03)


def count_model_params(model: nn.Module | None = None) -> int:
    if model is None:
        model = ForwardReturnModel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Normalization — Winsorized z-score to handle outliers
# ---------------------------------------------------------------------------

_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None


def _normalize(features: np.ndarray, fit: bool = False) -> np.ndarray:
    global _feat_mean, _feat_std
    if fit:
        _feat_mean = np.nanmean(features, axis=0)
        _feat_std = np.nanstd(features, axis=0)
        _feat_std[_feat_std < 1e-8] = 1.0
    result = (features - _feat_mean) / _feat_std
    # Winsorize to [-3, 3] to prevent outlier features from dominating
    result = np.clip(result, -3.0, 3.0)
    return result


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py --evaluate-holdout)
# ---------------------------------------------------------------------------

_trained_model: nn.Module | None = None


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features, timestamps = compute_features(df)
    features = _normalize(features, fit=False)
    features = np.nan_to_num(features, nan=0.0)

    model = _trained_model
    if model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32, device=device)
        preds = model(X).cpu().numpy()

    return preds, timestamps


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    global _trained_model

    total_start = time.time()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading training data...")
    train_df = load_train_data()
    print(f"  {len(train_df)} rows")

    # --- Compute features and targets ---
    features, timestamps = compute_features(train_df)
    targets = compute_targets(train_df)

    targets = targets[MAX_LOOKBACK:]

    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_timestamps = timestamps[valid]

    features = _normalize(features, fit=True)
    features = np.nan_to_num(features, nan=0.0)

    # Clip targets too — we don't care about predicting extreme moves accurately
    targets = np.clip(targets, -0.03, 0.03)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Setup model ---
    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)
    print(f"  Model parameters: {n_params}")

    # SGD with strong momentum and moderate L2 for smooth convergence
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=5e-2, momentum=0.9)
    loss_fn = nn.MSELoss()

    # --- Create DataLoader ---
    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=len(features), shuffle=False, drop_last=False)

    # --- Train ---
    print(f"Training for up to {TIME_BUDGET}s...")
    train_start = time.time()
    epoch = 0

    while time.time() - train_start < TIME_BUDGET:
        epoch += 1
        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() - train_start >= TIME_BUDGET:
                break

        if epoch % 200 == 0 or epoch == 1:
            with torch.no_grad():
                preds_check = model(X_tensor.to(device)).cpu().numpy()
            n_long = np.sum(preds_check > 0.005)
            n_short = np.sum(preds_check < -0.005)
            elapsed = time.time() - train_start
            print(f"  Epoch {epoch:4d} | loss={loss.item():.6f} | long={n_long} short={n_short} | {elapsed:.1f}s")

    training_seconds = time.time() - train_start
    print(f"Training complete: {epoch} epochs in {training_seconds:.1f}s")

    _trained_model = model

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor.to(device)).cpu().numpy()

    print(f"  Pred stats: mean={np.mean(all_preds):.6f}, std={np.std(all_preds):.6f}")
    print(f"  Preds > 0.005: {np.sum(all_preds > 0.005)}, Preds < -0.005: {np.sum(all_preds < -0.005)}")

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = _normalize(val_features, fit=False)
    val_features = np.nan_to_num(val_features, nan=0.0)

    with torch.no_grad():
        X_val = torch.tensor(val_features, dtype=torch.float32, device=device)
        val_preds = model(X_val).cpu().numpy()

    val_result = evaluate_model(val_preds, val_timestamps, n_params, split="val")

    total_seconds = time.time() - total_start

    # --- Print summary ---
    print()
    print("---")
    print(f"score:            {train_result['score']:.4f}")
    print(f"sharpe:           {train_result['sharpe']:.4f}")
    print(f"max_drawdown:     {train_result['max_drawdown']:.1%}")
    print(f"n_trades:         {train_result['n_trades']}")
    print(f"total_return:     {train_result['total_return']:.1%}")
    print(f"n_params:         {n_params}")
    print(f"val_pass:         {str(val_result['val_pass']).lower()}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    main()
