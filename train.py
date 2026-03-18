"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from prepare import (
    FORWARD_HOURS,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [1, 4, 12, 24, 48, 72, 168]
VOLATILITY_WINDOWS = [24, 168]
MAX_LOOKBACK = 168  # maximum lookback window (1 week)


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute features from OHLCV data.

    Returns:
        features: (N, n_features) array where N = len(df) - MAX_LOOKBACK.
        timestamps: (N,) array of pd.Timestamp aligned with features.
        vol_safe: (N,) array of 168h rolling vol for target normalization.
    """
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts = df["timestamp"].values

    # Hourly returns (for lookback and volatility calculations)
    hourly_returns = np.zeros(len(close))
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0

    # 168h rolling vol for normalizing return-based features
    hr_series = pd.Series(hourly_returns)
    vol_168h = hr_series.rolling(168, min_periods=168).std().values
    vol_safe = np.where((vol_168h > 0) & ~np.isnan(vol_168h), vol_168h, 1.0)

    feature_cols = []

    # 1. Vol-normalized returns over lookback windows
    for lb in RETURN_LOOKBACKS:
        ret = np.full(len(close), np.nan)
        ret[lb:] = close[lb:] / close[:-lb] - 1.0
        feature_cols.append(ret / vol_safe)

    # 2. Volatility (rolling std of hourly returns) — raw, not normalized
    for w in VOLATILITY_WINDOWS:
        vol = hr_series.rolling(w, min_periods=w).std().values
        feature_cols.append(vol)

    # 3. Vol ratio: 24h vol / 168h vol (vol regime indicator)
    vol_24h = hr_series.rolling(24, min_periods=24).std().values
    vol_ratio = np.where(vol_safe > 0, vol_24h / vol_safe, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Volume ratio: 24h avg / 168h avg
    vol_series = pd.Series(volume)
    vol_24 = vol_series.rolling(24, min_periods=24).mean().values
    vol_168 = vol_series.rolling(168, min_periods=168).mean().values
    volume_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(volume_ratio)

    # 5. High-low range volatility (24h average, normalized by price)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    hl_range = np.where(close > 0, (high - low) / close, 0.0)
    hl_range_24 = pd.Series(hl_range).rolling(24, min_periods=24).mean().values
    feature_cols.append(hl_range_24)

    # 6. RSI-like indicator (proportion of up-hours in window)
    up = (hourly_returns > 0).astype(np.float64)
    for w in [24, 72, 168]:
        rsi = pd.Series(up).rolling(w, min_periods=w).mean().values
        feature_cols.append(rsi - 0.5)  # center around zero

    # 7. Bollinger band position: (close - SMA) / (2 * std)
    close_series = pd.Series(close)
    for w in [24, 72]:
        sma = close_series.rolling(w, min_periods=w).mean().values
        std = close_series.rolling(w, min_periods=w).std().values
        std_safe = np.where(std > 0, std, 1.0)
        bb_pos = (close - sma) / (2 * std_safe)
        feature_cols.append(bb_pos)

    # 8. VWAP deviation (volume-weighted average price vs close)
    vwap_24 = (pd.Series(close * volume).rolling(24, min_periods=24).sum().values /
               pd.Series(volume).rolling(24, min_periods=24).sum().values)
    vwap_dev = np.where(vwap_24 > 0, (close - vwap_24) / vwap_24, 0.0)
    feature_cols.append(vwap_dev)

    # 9. Return autocorrelation at lag 24h (rolling over 168h window)
    # Captures momentum persistence: high autocorrelation = trending market
    ret_24h = np.full(len(close), np.nan)
    ret_24h[24:] = close[24:] / close[:-24] - 1.0
    ret_24h_series = pd.Series(ret_24h)
    ret_24h_lagged = ret_24h_series.shift(24)
    autocorr = ret_24h_series.rolling(168, min_periods=168).corr(ret_24h_lagged).values
    feature_cols.append(np.nan_to_num(autocorr, nan=0.0))

    # 10. Hour of day (cyclical)
    dt = pd.to_datetime(ts)
    hours = dt.hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    # 7. Day of week (cyclical)
    dow = dt.dayofweek
    feature_cols.append(np.sin(2 * np.pi * dow / 7))
    feature_cols.append(np.cos(2 * np.pi * dow / 7))

    features = np.column_stack(feature_cols)

    # Trim to valid rows (after max lookback)
    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]
    vol_trimmed = vol_safe[valid_start:]

    return features, timestamps, vol_trimmed


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
    """Return approximate parameter count for the HistGBR model."""
    if model is None:
        model = _trained_model
    if model is None:
        return 0
    n_params = 0
    for predictor_list in model._predictors:
        for predictor in predictor_list:
            n_params += predictor.get_n_leaf_nodes()
    return n_params


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py evaluate_model)
# ---------------------------------------------------------------------------

def _smooth_predictions(raw_preds: np.ndarray) -> np.ndarray:
    """Apply EMA smoothing — same effective width as 48h SMA but more responsive."""
    return pd.Series(raw_preds).ewm(span=48, min_periods=1).mean().values


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on arbitrary OHLCV data.

    Model predicts in sigma-space; denormalize to raw return space.
    """
    features, timestamps, vol_safe = compute_features(df)
    features = np.nan_to_num(features, nan=0.0)

    model = _trained_model
    if model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    sigma_preds = model.predict(features)
    vol_cap = np.percentile(vol_safe, 90)
    vol_capped = np.minimum(vol_safe, vol_cap)
    raw_preds = sigma_preds * vol_capped
    compressed = 0.010 * np.tanh(raw_preds / 0.010)
    preds = _smooth_predictions(compressed)
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
    features, timestamps, vol_safe = compute_features(train_df)
    targets = compute_targets(train_df)

    # Align: targets need same trimming as features (MAX_LOOKBACK from start)
    targets = targets[MAX_LOOKBACK:]

    # Drop rows where targets are NaN (last FORWARD_HOURS rows)
    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    vol_train = vol_safe[valid]
    train_timestamps = timestamps[valid]

    # Vol-normalize targets first, then winsorize in sigma-space
    targets = targets / vol_train
    targets = np.clip(targets, -5.0, 5.0)

    features = np.nan_to_num(features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Train GBR ---
    print("Training GBR...")
    train_start = time.time()

    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=4,
        learning_rate=0.02,
        min_samples_leaf=100,
        max_bins=255,
        l2_regularization=0.1,
        loss="squared_error",
        random_state=42,
    )
    # Time-decay weighting: recent data is more relevant than old data.
    # Exponential decay so 2022 data is ~5x more weighted than 2018 data.
    # Combined with 1.2x asymmetric weighting on positive returns.
    ts_float = train_timestamps.astype("datetime64[h]").astype(np.float64)
    ts_norm = (ts_float - ts_float.min()) / (ts_float.max() - ts_float.min())
    time_weights = np.exp(0.8 * ts_norm)
    asym_weights = np.where(targets > 0, 1.2, 1.0)
    sample_weights = time_weights * asym_weights
    model.fit(features, targets, sample_weight=sample_weights)

    training_seconds = time.time() - train_start
    print(f"Training complete in {training_seconds:.1f}s")

    _trained_model = model
    n_params = count_model_params(model)
    print(f"  Model parameters (node count): {n_params}")

    # --- Evaluate (black box) ---
    print("Evaluating...")
    result = evaluate_model(predict_on_data)

    total_seconds = time.time() - total_start

    # --- Print summary ---
    print()
    print("---")
    print(f"score:            {result['score']:.4f}")
    print(f"sharpe_min:       {result['sharpe_min']:.4f}")
    print(f"max_drawdown:     {result['max_drawdown']:.1%}")
    print(f"total_trades:     {result['total_trades']}")
    print(f"consistency:      {result['consistency']}")
    print(f"n_params:         {n_params}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    main()