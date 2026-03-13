"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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
PRED_SCALE = 0.8  # very selective — highest confidence trades only


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
    dt = pd.to_datetime(ts)
    hours = dt.hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    # 8. Day of week (cyclical) — crypto has weekly patterns
    dow = dt.dayofweek
    feature_cols.append(np.sin(2 * np.pi * dow / 7))
    feature_cols.append(np.cos(2 * np.pi * dow / 7))

    # 9. Momentum acceleration: 24h return - 72h return (trend strengthening?)
    ret_24 = np.full(len(close), np.nan)
    ret_24[24:] = close[24:] / close[:-24] - 1.0
    ret_72 = np.full(len(close), np.nan)
    ret_72[72:] = close[72:] / close[:-72] - 1.0
    accel = ret_24 - ret_72 / 3  # normalize 72h to per-24h scale
    feature_cols.append(accel)

    features = np.column_stack(feature_cols)

    # Trim to valid rows (after max lookback)
    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]

    return features, timestamps


def compute_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute vol-adjusted 24-hour forward returns.

    Returns:
        targets: vol-adjusted forward returns (same length as df, NaN-padded)
        vol_168: rolling 168h volatility for rescaling predictions
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    raw_returns = np.full(n, np.nan)
    raw_returns[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0

    # Compute rolling volatility for normalization
    hourly_returns = np.zeros(n)
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0
    vol = pd.Series(hourly_returns).rolling(168, min_periods=168).std().values
    vol = np.where(vol > 1e-8, vol, 1e-8)

    # Vol-adjusted targets: raw_return / (vol * sqrt(24))
    # sqrt(24) scales hourly vol to ~24h vol
    vol_24 = vol * np.sqrt(24)
    targets = raw_returns / vol_24

    return targets, vol


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_trained_model = None


def count_model_params(model=None) -> int:
    """Return approximate parameter count for the GBR model."""
    if model is None:
        model = _trained_model
    if model is None:
        return 0
    n_params = 0
    for estimators in model.estimators_:
        for tree in estimators:
            n_params += tree.tree_.node_count
    return n_params


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
    """Long-only filter with crash protection and vol dampening."""
    close = df["close"].values.astype(np.float64)

    # Long-only: never go short
    preds = np.maximum(preds, 0.0)

    # Crash filter: go flat during crashes (168h return < -15%)
    ret_168 = np.full(len(close), 0.0)
    ret_168[168:] = close[168:] / close[:-168] - 1.0
    ret_168 = ret_168[MAX_LOOKBACK:][:len(preds)]
    crash_mask = ret_168 < -0.15
    preds[crash_mask] = 0.0  # flat during crash

    return preds


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on arbitrary OHLCV data."""
    features, timestamps = compute_features(df)
    vol = compute_vol_168(df)
    features = np.nan_to_num(features, nan=0.0)

    model = _trained_model
    if model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    vol_24 = vol * np.sqrt(24)
    preds = model.predict(features) * vol_24 * PRED_SCALE
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
    targets, train_vol = compute_targets(train_df)

    # Align: targets and vol need same trimming as features (MAX_LOOKBACK from start)
    targets = targets[MAX_LOOKBACK:]
    train_vol = train_vol[MAX_LOOKBACK:]

    # Drop rows where targets are NaN (last FORWARD_HOURS rows)
    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_vol = train_vol[valid]
    train_timestamps = timestamps[valid]

    # Winsorize vol-adjusted targets at ±3 (3 sigma)
    targets = np.clip(targets, -3.0, 3.0)

    # GBR is invariant to feature scaling — skip normalization to avoid
    # distribution mismatch between train/val periods.
    features = np.nan_to_num(features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Train GBR ---
    print(f"Training GBR...")
    train_start = time.time()

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        min_samples_leaf=100,
        max_features=0.8,
        loss="squared_error",
        random_state=42,
    )
    model.fit(features, targets)

    training_seconds = time.time() - train_start
    print(f"Training complete in {training_seconds:.1f}s")

    _trained_model = model
    n_params = count_model_params(model)
    print(f"  Model parameters (node count): {n_params}")

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    # Model predicts vol-adjusted returns, scale back to raw returns
    vol_24_train = train_vol * np.sqrt(24)
    all_preds = model.predict(features) * vol_24_train * PRED_SCALE
    all_preds = _apply_regime_filter(all_preds, train_df)

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = np.nan_to_num(val_features, nan=0.0)
    val_vol = compute_vol_168(val_df)

    vol_24_val = val_vol * np.sqrt(24)
    val_preds = model.predict(val_features) * vol_24_val * PRED_SCALE

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
