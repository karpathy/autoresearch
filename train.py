"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

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

    # 1a. Volume-weighted cumulative return (emphasizes moves on high volume)
    vol_avg = pd.Series(volume).rolling(168, min_periods=168).mean().values
    vol_avg_safe = np.where(vol_avg > 0, vol_avg, 1.0)
    vol_weight = volume / vol_avg_safe  # relative volume (1.0 = average)
    vw_returns = hourly_returns * vol_weight
    vw_series = pd.Series(vw_returns)
    for lb in [24, 72, 168]:
        vw_cum = vw_series.rolling(lb, min_periods=lb).sum().values
        feature_cols.append(np.nan_to_num(vw_cum / vol_safe, nan=0.0))

    # 1b. Momentum divergence: short-term vs long-term return agreement
    #     Positive = both agree on direction, negative = divergence
    ret_24 = np.full(len(close), np.nan)
    ret_24[24:] = close[24:] / close[:-24] - 1.0
    ret_168 = np.full(len(close), np.nan)
    ret_168[168:] = close[168:] / close[:-168] - 1.0
    # Product of vol-normalized returns: positive when aligned, negative when divergent
    mom_div = (ret_24 / vol_safe) * (ret_168 / vol_safe)
    feature_cols.append(np.nan_to_num(mom_div, nan=0.0))
    feature_cols.append(np.nan_to_num(np.abs(mom_div), nan=0.0))  # magnitude of alignment

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

    # 4b. Volume momentum: rate of change of volume (24h avg vs 72h avg lagged by 24h)
    vol_72 = vol_series.rolling(72, min_periods=72).mean()
    vol_72_lagged = vol_72.shift(24).values
    vol_momentum = np.where(vol_72_lagged > 0, vol_24 / vol_72_lagged - 1.0, 0.0)
    feature_cols.append(np.nan_to_num(vol_momentum, nan=0.0))

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

    # 7b. Bollinger band width: 2*std/sma (narrow = consolidation, wide = trending)
    for w in [24, 72, 168]:
        sma = close_series.rolling(w, min_periods=w).mean().values
        std = close_series.rolling(w, min_periods=w).std().values
        sma_safe = np.where(sma > 0, sma, 1.0)
        bb_width = 2 * std / sma_safe
        feature_cols.append(bb_width)

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

    # 10. Chop / consolidation detection features
    # a) Rolling price range as fraction of price (high in chop, low in trend)
    for w in [168, 720]:  # 1 week, 30 days
        roll_max = pd.Series(close).rolling(w, min_periods=w).max().values
        roll_min = pd.Series(close).rolling(w, min_periods=w).min().values
        price_range = np.where(close > 0, (roll_max - roll_min) / close, 0.0)
        feature_cols.append(price_range)

    # b) Directional efficiency: net move / total path length
    #    Near ±1 in trends, near 0 in chop
    for w in [72, 168]:
        net_move = np.full(len(close), np.nan)
        net_move[w:] = close[w:] - close[:-w]
        path_length = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(
            w, min_periods=w).sum().values
        path_safe = np.where(path_length > 0, path_length, 1.0)
        efficiency = net_move / path_safe
        feature_cols.append(np.nan_to_num(efficiency, nan=0.0))

    # c) Absolute return magnitude (low = chop, high = trending)
    for w in [72, 168]:
        abs_ret = np.full(len(close), np.nan)
        abs_ret[w:] = np.abs(close[w:] / close[:-w] - 1.0)
        abs_ret_norm = abs_ret / vol_safe  # normalize by vol
        feature_cols.append(np.nan_to_num(abs_ret_norm, nan=0.0))

    # 11. Hour of day (cyclical)
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
# Model Helpers
# ---------------------------------------------------------------------------

def count_model_params(models) -> int:
    """Return approximate parameter count for an ensemble."""
    if not models:
        return 0
    n_params = 0
    for model in models:
        for tree in model.estimators_:
            n_params += tree.tree_.node_count
    return n_params


def _smooth_predictions(raw_preds: np.ndarray) -> np.ndarray:
    """Apply EMA smoothing — same effective width as 48h SMA but more responsive."""
    return pd.Series(raw_preds).ewm(span=30, min_periods=1).mean().values


def _confidence_scaled_predict(model, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Produce confidence-scaled predictions and confidence weights from a single model."""
    all_tree_preds = np.array([tree.predict(features) for tree in model.estimators_])
    preds = all_tree_preds.mean(axis=0)
    pred_std = all_tree_preds.std(axis=0)
    confidence = 1.0 / (1.0 + 2.0 * pred_std)
    return preds * confidence, confidence


# ---------------------------------------------------------------------------
# Build Model (recipe)
# ---------------------------------------------------------------------------

def build_model(train_df: pd.DataFrame) -> callable:
    """Train a model on the provided data and return a prediction function.

    This is the recipe: features, architecture, hyperparameters, and prediction
    pipeline. The evaluation system calls this independently for each walk-forward
    window. Each call must be self-contained — no global state.

    Args:
        train_df: OHLCV DataFrame for training. The date range varies —
                  the recipe must work regardless of which years are included.

    Returns:
        predict_fn: Callable that takes an OHLCV DataFrame and returns
                    (sigma_predictions, timestamps, vol) — predictions in
                    sigma-space.
    """
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

    # Vol-normalize targets first, then winsorize in sigma-space
    targets = targets / vol_train
    targets = np.clip(targets, -5.0, 5.0)

    features = np.nan_to_num(features, nan=0.0)

    # --- Train pass 1: feature selection ---
    selector = ExtraTreesRegressor(
        n_estimators=3000,
        max_depth=7,
        min_samples_leaf=600,
        random_state=42,
        n_jobs=-1,
    )
    selector.fit(features, targets)

    # Feature importance pruning
    importances = selector.feature_importances_
    threshold = np.mean(importances)
    selected = importances >= threshold
    features_pruned = features[:, selected]

    # --- Train pass 2: diverse ensemble on pruned features ---
    ensemble_configs = [
        {"n_estimators": 3000, "max_depth": 7, "min_samples_leaf": 600},   # original (strong regularization)
        {"n_estimators": 2000, "max_depth": 5, "min_samples_leaf": 800},   # very shallow + heavy regularization
        {"n_estimators": 2000, "max_depth": 7, "min_samples_leaf": 500},   # moderate regularization variant
        {"n_estimators": 2000, "max_depth": 6, "min_samples_leaf": 700},   # intermediate
        {"n_estimators": 2000, "max_depth": 7, "min_samples_leaf": 600, "max_features": 0.8},  # feature subsampling
    ]

    models = []
    for cfg in ensemble_configs:
        m = ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            **cfg,
        )
        m.fit(features_pruned, targets)
        models.append(m)

    n_params = count_model_params(models)

    # --- Return prediction closure ---
    def predict_fn(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sigma-space predictions on arbitrary OHLCV data."""
        feats, ts, vol = compute_features(df)
        feats = np.nan_to_num(feats, nan=0.0)
        feats = feats[:, selected]

        # Confidence-weighted ensemble: trust each model proportional to its confidence
        weighted_sum = np.zeros(len(feats))
        weight_sum = np.zeros(len(feats))
        for model in models:
            scaled_pred, confidence = _confidence_scaled_predict(model, feats)
            weighted_sum += scaled_pred * confidence
            weight_sum += confidence
        weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        sigma_preds = weighted_sum / weight_sum

        sigma_preds = np.clip(sigma_preds, -2.0, 2.0)
        sigma_preds = sigma_preds * 1.5  # base scale
        sigma_smoothed = _smooth_predictions(sigma_preds)
        return sigma_smoothed, ts, vol

    predict_fn.n_params = n_params
    predict_fn.n_selected = int(selected.sum())
    predict_fn.n_total_features = len(selected)
    return predict_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()

    # --- Load data ---
    print("Loading training data...")
    train_df = load_train_data()
    print(f"  {len(train_df)} rows")

    # --- Build model (local development) ---
    print("Training model...")
    train_start = time.time()
    predict_fn = build_model(train_df)
    training_seconds = time.time() - train_start

    n_params = predict_fn.n_params
    print(f"  Training complete in {training_seconds:.1f}s")
    print(f"  Features: {predict_fn.n_selected}/{predict_fn.n_total_features} selected")
    print(f"  Parameters (node count): {n_params}")

    # --- Evaluate (black box — retrains on all walk-forward windows) ---
    result = evaluate_model(build_model)

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
