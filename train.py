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
# Model — Mean-reversion with learned scale
# ---------------------------------------------------------------------------

N_FEATURES = len(RETURN_LOOKBACKS) + len(VOLATILITY_WINDOWS) + 1 + 2 + 2 + 1  # 14

# Feature indices (after normalization)
IDX_24H_RETURN = 3  # 24h lookback return


IDX_24H_VOL = 6  # 24h volatility (index in feature array)
IDX_168H_VOL = 7  # 168h volatility


class ForwardReturnModel(nn.Module):
    """Mean-reversion with volatility gating.

    prediction = -24h_return * scale * vol_gate
    vol_gate = 1 when vol is low, 0 when vol is high
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.005))
        self.vol_threshold = nn.Parameter(torch.tensor(0.0))  # z-scored threshold
        self._n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean reversion signal
        signal = -x[:, IDX_24H_RETURN] * self.scale
        # Volatility gate: reduce signal when vol is high
        # Use sigmoid for smooth gating: gate → 0 when vol >> threshold
        vol = x[:, IDX_24H_VOL]
        gate = torch.sigmoid(-(vol - self.vol_threshold) * 2.0)
        return signal * gate


def count_model_params(model: nn.Module | None = None) -> int:
    if model is None:
        model = ForwardReturnModel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Normalization — Winsorized z-score
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

    # Clip targets
    targets = np.clip(targets, -0.05, 0.05)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Setup model ---
    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)
    print(f"  Model parameters: {n_params}")

    # Grid search over scale values to find optimal (only 1 param)
    X_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    print("Grid searching optimal scale and vol_threshold...")
    best_score = -999
    best_scale = 0.005
    best_vol_thresh = 0.0
    signal = -features[:, IDX_24H_RETURN]
    vol = features[:, IDX_24H_VOL]
    close = train_df["close"].values[MAX_LOOKBACK:][valid]
    price_returns = np.zeros(len(close))
    price_returns[1:] = close[1:] / close[:-1] - 1.0

    for scale in np.arange(0.001, 0.030, 0.001):
        for vol_thresh in np.arange(-1.0, 2.0, 0.25):
            gate = 1.0 / (1.0 + np.exp((vol - vol_thresh) * 2.0))
            preds = signal * scale * gate

            positions = np.zeros_like(preds)
            positions[preds > 0.005] = 1.0
            positions[preds < -0.005] = -1.0

            port_returns_arr = positions[:-1] * price_returns[1:]

            # Also compute max drawdown approximation
            equity = np.cumprod(1.0 + port_returns_arr)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dd = np.min(dd)

            if np.std(port_returns_arr) > 0:
                sharpe = np.mean(port_returns_arr) / np.std(port_returns_arr) * np.sqrt(8760)
            else:
                sharpe = 0

            n_trades_approx = np.sum(np.diff(positions) != 0)

            # Score: Sharpe, but penalize if drawdown > 25% or < 30 trades
            score = sharpe
            if abs(max_dd) > 0.25:
                score *= 0.25 / abs(max_dd)  # soft drawdown penalty
            if n_trades_approx < 30:
                score = -999

            if score > best_score:
                best_score = score
                best_scale = scale
                best_vol_thresh = vol_thresh

    print(f"  Best scale: {best_scale:.3f}, vol_thresh: {best_vol_thresh:.2f}, score: {best_score:.4f}")

    # Set the parameters
    with torch.no_grad():
        model.scale.fill_(best_scale)
        model.vol_threshold.fill_(best_vol_thresh)

    # Also do gradient-based fine-tuning of the scale
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss(delta=0.01)

    print(f"Fine-tuning scale for up to {TIME_BUDGET}s...")
    train_start = time.time()
    epoch = 0

    while time.time() - train_start < TIME_BUDGET:
        epoch += 1
        model.train()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            scale_val = model.scale.item()
            elapsed = time.time() - train_start
            print(f"  Epoch {epoch:4d} | loss={loss.item():.6f} | scale={scale_val:.6f} | {elapsed:.1f}s")

        if time.time() - train_start >= TIME_BUDGET:
            break

    training_seconds = time.time() - train_start
    print(f"Training complete: {epoch} epochs in {training_seconds:.1f}s")
    print(f"  Final scale: {model.scale.item():.6f}")

    _trained_model = model

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor).cpu().numpy()

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
