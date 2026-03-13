"""
Autotrader baseline model. Predicts BTC/USD 24-hour forward returns.

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
ZSCORE_WINDOWS = [24, 72, 168]
MAX_LOOKBACK = 168  # maximum lookback window (1 week)


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute features from OHLCV data.

    Returns:
        features: (N, n_features) array where N = len(df) - MAX_LOOKBACK.
        timestamps: (N,) array of pd.Timestamp aligned with features.
    """
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts = df["timestamp"].values

    # Hourly returns (for lookback and volatility calculations)
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
    # Avoid division by zero
    vol_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Rolling z-scores of returns (mean-reversion signals)
    close_series = pd.Series(close)
    for w in ZSCORE_WINDOWS:
        rolling_mean = close_series.rolling(w, min_periods=w).mean().values
        rolling_std = close_series.rolling(w, min_periods=w).std().values
        zscore = np.where(rolling_std > 0, (close - rolling_mean) / rolling_std, 0.0)
        feature_cols.append(zscore)

    # 5. Hour of day (cyclical)
    hours = pd.to_datetime(ts).hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    features = np.column_stack(feature_cols)

    # Trim to valid rows (after max lookback)
    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]

    return features, timestamps


def compute_targets(df: pd.DataFrame) -> np.ndarray:
    """Compute 24-hour forward returns: close[t+24]/close[t] - 1.

    Returns array of same length as df. Last FORWARD_HOURS entries are NaN.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    targets = np.full(n, np.nan)
    targets[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0
    return targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

N_FEATURES = len(RETURN_LOOKBACKS) + len(VOLATILITY_WINDOWS) + 1 + len(ZSCORE_WINDOWS) + 2  # 14 features


class ForwardReturnModel(nn.Module):
    """Feedforward network with dropout: n_features -> 32 -> 16 -> 1"""

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_model_params(model: nn.Module | None = None) -> int:
    """Return total trainable parameter count."""
    if model is None:
        model = ForwardReturnModel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None


def _normalize(features: np.ndarray, fit: bool = False) -> np.ndarray:
    """Z-score normalize features. If fit=True, compute and store stats."""
    global _feat_mean, _feat_std
    if fit:
        _feat_mean = np.nanmean(features, axis=0)
        _feat_std = np.nanstd(features, axis=0)
        _feat_std[_feat_std < 1e-8] = 1.0  # avoid division by zero
    return (features - _feat_mean) / _feat_std


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py --evaluate-holdout)
# ---------------------------------------------------------------------------

_trained_model: nn.Module | None = None


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on arbitrary OHLCV data.

    Used by prepare.py for holdout evaluation. Requires that the model
    has already been trained (i.e., train.py has been run).
    """
    features, timestamps = compute_features(df)
    features = _normalize(features, fit=False)

    # Replace any remaining NaN with 0
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

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
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

    # Align: targets need same trimming as features (MAX_LOOKBACK from start)
    targets = targets[MAX_LOOKBACK:]

    # Drop rows where targets are NaN (last FORWARD_HOURS rows)
    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_timestamps = timestamps[valid]

    # Winsorize targets to reduce extreme outlier influence
    p5, p95 = np.percentile(targets, [5, 95])
    targets = np.clip(targets, p5, p95)

    # Normalize features (fit on training data)
    features = _normalize(features, fit=True)

    # Replace any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Setup model ---
    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)
    print(f"  Model parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()

    # --- Create DataLoader ---
    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=False)

    # --- Train ---
    print(f"Training for up to {TIME_BUDGET}s...")
    train_start = time.time()
    epoch = 0

    while time.time() - train_start < TIME_BUDGET:
        epoch += 1
        epoch_loss = 0.0
        n_batches = 0
        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            mse_loss = mse_fn(pred, y_batch)
            # Directional penalty: penalize predictions that have wrong sign
            sign_match = (pred * y_batch > 0).float()  # 1 if same sign, 0 if not
            dir_penalty = (1.0 - sign_match).mean()
            loss = mse_loss + 0.001 * dir_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if time.time() - train_start >= TIME_BUDGET:
                break

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - train_start
            print(f"  Epoch {epoch:4d} | loss={avg_loss:.6f} | {elapsed:.1f}s")

    training_seconds = time.time() - train_start
    print(f"Training complete: {epoch} epochs in {training_seconds:.1f}s")

    _trained_model = model

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    model.eval()
    with torch.no_grad():
        all_preds = model(X_tensor.to(device)).cpu().numpy()

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
