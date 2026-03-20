"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import signal
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from prepare import (
    FORWARD_HOURS,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [4, 12, 24, 48, 72, 168]  # 1h removed — noisiest with power transform
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

    # 1b. [REMOVED] Momentum divergence — redundant with constrained returns + power transform

    # 2. Volatility (rolling std of hourly returns) — raw, not normalized
    for w in VOLATILITY_WINDOWS:
        vol = hr_series.rolling(w, min_periods=w).std().values
        feature_cols.append(vol)

    # 3. Vol ratio: 24h vol / 168h vol (vol regime indicator)
    vol_24h = hr_series.rolling(24, min_periods=24).std().values
    vol_ratio = np.where(vol_safe > 0, vol_24h / vol_safe, 1.0)
    feature_cols.append(vol_ratio)

    # 3b. Vol percentile rank: where does current 24h vol sit in its own 720h history?
    # Regime-relative: 0.0 = historically low vol, 1.0 = historically high vol
    vol_24h_series = pd.Series(vol_24h)
    vol_pctile = vol_24h_series.rolling(720, min_periods=168).rank(pct=True).values
    feature_cols.append(np.nan_to_num(vol_pctile - 0.5, nan=0.0))  # center around 0

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

    # 7b. Bollinger band width: 2*std/sma (24h removed — noisy with power transform)
    for w in [72, 168]:
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

    # a2) Trend strength: rolling R-squared of log(price) vs time
    log_close = np.log(np.maximum(close, 1.0))
    log_series = pd.Series(log_close)
    time_idx = pd.Series(np.arange(len(close), dtype=np.float64))
    for w in [168]:
        corr = log_series.rolling(w, min_periods=w).corr(time_idx).values
        r_sq = corr ** 2
        feature_cols.append(np.nan_to_num(r_sq, nan=0.0))

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
    return pd.Series(raw_preds).ewm(span=40, min_periods=1).mean().values


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

    # --- Monotonic constraints: longer-horizon returns must be increasing ---
    # Prevents "strong momentum → predict reversal" pathology across multiple horizons
    mono_cst = np.zeros(features.shape[1], dtype=int)
    mono_cst[2] = 1  # 24h vol-normalized return → monotonically increasing
    mono_cst[3] = 1  # 48h vol-normalized return → monotonically increasing
    mono_cst[4] = 1  # 72h vol-normalized return → monotonically increasing
    mono_cst[5] = 1  # 168h vol-normalized return → monotonically increasing
    mono_cst[6] = 1  # 24h VW cumulative return → monotonically increasing
    mono_cst[28] = 1  # 72h directional efficiency → monotonically increasing
    mono_cst[29] = 1  # 168h directional efficiency → monotonically increasing

    # --- Train: two-model ensemble for diversity ---
    model_conservative = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=4,
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=20,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model_conservative.fit(features, targets)

    model_aggressive = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=4,
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=20,
        max_features=0.8,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model_aggressive.fit(features, targets)

    selected = np.ones(features.shape[1], dtype=bool)
    models = [model_conservative, model_aggressive]
    blend_weights = [0.5, 0.5]

    # Compute and store training prediction bias for demeaning
    train_preds = sum(w * m.predict(features) for w, m in zip(blend_weights, models))
    pred_bias = float(np.mean(train_preds))

    # Approximate param count
    n_params = 0
    for m in models:
        n_params += sum(
            m._predictors[j][0].get_n_leaf_nodes()
            for j in range(len(m._predictors))
        )

    # --- Return prediction closure ---
    def predict_fn(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sigma-space predictions on arbitrary OHLCV data."""
        feats, ts, vol = compute_features(df)
        feats = np.nan_to_num(feats, nan=0.0)
        feats = feats[:, selected]

        # Weighted blend prediction
        preds = [m.predict(feats) for m in models]
        sigma_preds = sum(w * p for w, p in zip(blend_weights, preds))

        sigma_preds = sigma_preds - pred_bias  # remove training-context directional bias
        sigma_preds = np.clip(sigma_preds, -1.5, 1.5)
        # Power transform: amplify predictions away from zero to increase trade count
        # 0.1→0.20, 0.3→0.41, 0.5→0.62, 1.0→1.0 (preserves sign and large signals)
        sigma_preds = np.sign(sigma_preds) * np.abs(sigma_preds) ** 0.7
        sigma_preds = sigma_preds * 0.3  # further dampen to reduce position sizes
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
    def _timeout_handler(signum, frame):
        raise TimeoutError("evaluation exceeded 240s budget")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(240)
    try:
        result = evaluate_model(build_model)
    except TimeoutError:
        print("TIMEOUT: evaluation exceeded 240s budget")
        result = {
            "score": -9999,
            "sharpe_min": 0,
            "max_drawdown": 0,
            "total_trades": 0,
            "consistency": "0/8",
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    total_seconds = time.time() - total_start

    # --- Print summary ---
    print()
    print("---")
    print(f"score:            {result['score']:.4f}")
    print(f"sharpe_min:       {result['sharpe_min']:.4f}")
    print(f"max_drawdown:     {result['max_drawdown']:.1%}")
    print(f"total_trades:     {result['total_trades']}")
    print(f"consistency:      {result['consistency']}")
    print(f"holdout_health:   {result.get('holdout_health', 'N/A')}")
    print(f"epoch:            {result.get('epoch', 'N/A')}")
    print(f"n_params:         {n_params}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    main()
