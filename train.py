"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
TREND_MA_WINDOWS = [24, 72, 168]
ZSCORE_WINDOWS = [72, 168]
MAX_LOOKBACK = 168  # maximum lookback window (1 week)


def compute_vol_168(df: pd.DataFrame) -> np.ndarray:
    """Compute 168h rolling volatility (trimmed to valid start)."""
    close = df["close"].values.astype(np.float64)
    hourly_returns = np.zeros(len(close))
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0
    vol = pd.Series(hourly_returns).rolling(168, min_periods=168).std().values
    return vol[MAX_LOOKBACK:]


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
    vol_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Price relative to moving averages (trend signals)
    close_series = pd.Series(close)
    for w in TREND_MA_WINDOWS:
        ma = close_series.rolling(w, min_periods=w).mean().values
        price_rel_ma = np.where(ma > 0, close / ma - 1.0, 0.0)
        feature_cols.append(price_rel_ma)

    # 5. Rolling z-scores (mean-reversion signals)
    for w in ZSCORE_WINDOWS:
        rolling_mean = close_series.rolling(w, min_periods=w).mean().values
        rolling_std = close_series.rolling(w, min_periods=w).std().values
        zscore = np.where(rolling_std > 0, (close - rolling_mean) / rolling_std, 0.0)
        feature_cols.append(zscore)

    # 6. Rolling max drawdown (crash detector) over 168h
    rolling_max = close_series.rolling(168, min_periods=168).max().values
    rolling_dd = np.where(rolling_max > 0, close / rolling_max - 1.0, 0.0)
    feature_cols.append(rolling_dd)

    # 6. High-low range volatility (24h average)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    hl_range = np.where(close > 0, (high - low) / close, 0.0)
    hl_range_24 = pd.Series(hl_range).rolling(24, min_periods=24).mean().values
    feature_cols.append(hl_range_24)

    # 7. Hour of day (cyclical)
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

_trained_model = None


def count_model_params(model=None) -> int:
    """Return parameter count for the Ridge model."""
    if model is None:
        model = _trained_model
    if model is None:
        return 0
    return model.coef_.size + 1  # coefficients + intercept


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None
_vol_median: float = 0.01  # median 168h volatility from training


def _normalize(features: np.ndarray, fit: bool = False) -> np.ndarray:
    """Z-score normalize features. If fit=True, compute and store stats."""
    global _feat_mean, _feat_std
    if fit:
        _feat_mean = np.nanmean(features, axis=0)
        _feat_std = np.nanstd(features, axis=0)
        _feat_std[_feat_std < 1e-8] = 1.0
    return (features - _feat_mean) / _feat_std


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py --evaluate-holdout)
# ---------------------------------------------------------------------------

def _apply_regime_filter(preds: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Regime filter: no longs during crash, no shorts during strong uptrend."""
    close = df["close"].values.astype(np.float64)

    # Compute 168h returns for regime detection
    ret_168 = np.full(len(close), 0.0)
    ret_168[168:] = close[168:] / close[:-168] - 1.0
    ret_168 = ret_168[MAX_LOOKBACK:][:len(preds)]  # align with predictions

    # Crash filter: no longs when price dropped > 15% in a week
    crash_mask = ret_168 < -0.15
    preds[crash_mask] = np.minimum(preds[crash_mask], 0.0)

    # Bull filter: no shorts when price rose > 15% in a week
    bull_mask = ret_168 > 0.15
    preds[bull_mask] = np.maximum(preds[bull_mask], 0.0)

    return preds


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on arbitrary OHLCV data."""
    features, timestamps = compute_features(df)
    vol = compute_vol_168(df)
    features = _normalize(features, fit=False)
    features = np.nan_to_num(features, nan=0.0)

    model = _trained_model
    if model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    preds = model.predict(features)
    preds = _apply_regime_filter(preds, df)
    return preds, timestamps


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    global _trained_model

    total_start = time.time()

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

    # Compute median volatility for regime filter dampening
    train_vol = compute_vol_168(train_df)
    global _vol_median
    _vol_median = np.nanmedian(train_vol[~np.isnan(train_vol)])
    print(f"  Median 168h volatility: {_vol_median:.6f}")

    # Normalize features (fit on training data)
    features = _normalize(features, fit=True)
    features = np.nan_to_num(features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Train Ridge ---
    print("Training Ridge regression...")
    train_start = time.time()

    model = Ridge(alpha=1.0)
    model.fit(features, targets)

    training_seconds = time.time() - train_start
    print(f"Training complete in {training_seconds:.1f}s")

    _trained_model = model
    n_params = count_model_params(model)
    print(f"  Model parameters (node count): {n_params}")

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    all_preds = model.predict(features)
    all_preds = _apply_regime_filter(all_preds, train_df)

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = _normalize(val_features, fit=False)
    val_features = np.nan_to_num(val_features, nan=0.0)

    val_preds = model.predict(val_features)
    val_preds = _apply_regime_filter(val_preds, val_df)

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
