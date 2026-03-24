"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import signal
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from prepare import (
    FORWARD_HOURS,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [4, 12, 24, 48, 72, 168]  # 1h removed — noisiest
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
    for lb in [12, 48, 168]:
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

    # 7b. Bollinger band width: 2*std/sma (24h removed — noisy)
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

    # 11b. Day of week (cyclical)
    dow = dt.dayofweek
    feature_cols.append(np.sin(2 * np.pi * dow / 7))
    feature_cols.append(np.cos(2 * np.pi * dow / 7))

    # 12. Funding rate features (market positioning signal)
    if "funding_rate" in df.columns:
        fr = df["funding_rate"].values.astype(np.float64)
        fr_series = pd.Series(fr)
        feature_cols.append(fr * 1000)
        fr_cum_24 = fr_series.rolling(24, min_periods=1).sum().values
        feature_cols.append(fr_cum_24 * 100)
        fr_cum_168 = fr_series.rolling(168, min_periods=1).sum().values
        feature_cols.append(fr_cum_168 * 10)

    features = np.column_stack(feature_cols)

    # Trim to valid rows (after max lookback)
    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]
    vol_trimmed = vol_safe[valid_start:]

    return features, timestamps, vol_trimmed


def compute_targets(df: pd.DataFrame, horizon: int = FORWARD_HOURS) -> np.ndarray:
    """Compute forward returns: close[t+horizon]/close[t] - 1.

    Returns array of same length as df. Last `horizon` entries are NaN.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    targets = np.full(n, np.nan)
    targets[:n - horizon] = close[horizon:] / close[:n - horizon] - 1.0
    return targets


def compute_vol_targets(df: pd.DataFrame, horizon: int = FORWARD_HOURS) -> np.ndarray:
    """Compute forward realized vol ratio: std(next `horizon` returns) / trailing 168h vol.

    A ratio of 1.0 = vol stays the same. >1 = vol increasing. <1 = calming.
    Returns array of same length as df. Last `horizon` entries are NaN.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    # Hourly returns
    hourly_ret = np.zeros(n)
    hourly_ret[1:] = close[1:] / close[:-1] - 1.0

    # Forward realized vol: std of hourly returns over next horizon
    hr_series = pd.Series(hourly_ret)
    forward_vol = hr_series.rolling(horizon, min_periods=horizon).std().shift(-horizon).values

    # Trailing 168h vol (same as used in compute_features for vol_safe)
    trailing_vol = hr_series.rolling(168, min_periods=168).std().values
    trailing_vol_safe = np.where((trailing_vol > 0) & ~np.isnan(trailing_vol), trailing_vol, 1.0)

    vol_targets = np.full(n, np.nan)
    valid_mask = ~np.isnan(forward_vol)
    vol_targets[valid_mask] = forward_vol[valid_mask] / trailing_vol_safe[valid_mask]

    return vol_targets


def compute_confidence_targets(df: pd.DataFrame) -> np.ndarray:
    """Compute favorable regime target: 1 = normal conditions, 0 = danger.

    Danger conditions (reduce positions):
      - Forward vol shock: forward 168h realized vol > 1.5x trailing vol
      - Extreme chop: forward 168h directional efficiency < 5th percentile

    ~70-80% of hours are "favorable" → model defaults to full positions.

    Returns array of same length as df. Edges are NaN.
    No sklearn, no trained models, no learned parameters.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    # --- Forward vol shock ---
    hourly_ret = np.zeros(n)
    hourly_ret[1:] = close[1:] / close[:-1] - 1.0
    hr_series = pd.Series(hourly_ret)

    forward_vol = hr_series.rolling(168, min_periods=168).std().shift(-168).values
    trailing_vol = hr_series.rolling(168, min_periods=168).std().values
    trailing_vol_safe = np.where((trailing_vol > 0) & ~np.isnan(trailing_vol), trailing_vol, 1.0)

    vol_shock = np.full(n, False)
    both_valid = ~np.isnan(forward_vol) & ~np.isnan(trailing_vol)
    vol_shock[both_valid] = (forward_vol[both_valid] / trailing_vol_safe[both_valid]) > 1.5

    # --- Extreme chop: low directional efficiency ---
    net_move = np.full(n, np.nan)
    net_move[:n - 168] = np.abs(close[168:] - close[:n - 168])
    abs_hourly = np.abs(np.diff(close, prepend=close[0]))
    path_length = pd.Series(abs_hourly).rolling(168, min_periods=168).sum().shift(-168).values
    path_safe = np.where((path_length > 0) & ~np.isnan(path_length), path_length, 1.0)
    fwd_efficiency = net_move / path_safe

    eff_5th = pd.Series(fwd_efficiency).expanding(min_periods=720).quantile(0.05).values
    extreme_chop = np.full(n, False)
    eff_valid = ~np.isnan(fwd_efficiency) & ~np.isnan(eff_5th)
    extreme_chop[eff_valid] = fwd_efficiency[eff_valid] < eff_5th[eff_valid]

    # --- Combine: favorable = not vol_shock AND not extreme_chop ---
    targets = np.full(n, np.nan)
    any_valid = both_valid | eff_valid
    targets[any_valid] = 1.0
    targets[vol_shock | extreme_chop] = 0.0
    targets[~any_valid] = np.nan

    return targets


# ---------------------------------------------------------------------------
# Model Helpers
# ---------------------------------------------------------------------------

def _smooth_predictions(raw_preds: np.ndarray) -> np.ndarray:
    """Apply light EMA smoothing — reduce micro-noise while preserving signal."""
    return pd.Series(raw_preds).ewm(span=20, min_periods=1).mean().values


# ---------------------------------------------------------------------------
# Build Model (recipe)
# ---------------------------------------------------------------------------

def build_model(train_df: pd.DataFrame, sample_weight=None) -> callable:
    """Train a model on the provided data and return a prediction function.

    Architecture:
      - Return model: single HistGradientBoostingRegressor predicting
        vol-normalized 24h forward returns.
      - Confidence scaler: HGB classifier on favorable conditions target
        (vol shock + chop detection). Asymmetric [0.70, 1.20] adjustment —
        dampens bottom 5-15% of predictions, boosts top 10%.
      - Position scaler: HGB regressor on binary vol target (>0.9 threshold).
        Modulates position size based on predicted vol conditions.

    Pipeline: raw prediction → confidence adj [0.70, 1.20] → position
      scale → scale(0.35) → EMA(20)

    Args:
        train_df: OHLCV DataFrame for training. The date range varies —
                  the recipe must work regardless of which years are included.
        sample_weight: Optional array of per-sample weights (same length as
                  train_df). Passed by the evaluation infrastructure for
                  exponential decay weighting of older samples.

    Returns:
        predict_fn: Callable that takes an OHLCV DataFrame and returns
                    (sigma_predictions, timestamps, vol) — predictions in
                    sigma-space.
    """
    # --- Compute features and targets ---
    features_all, timestamps, vol_safe = compute_features(train_df)
    features_all = np.nan_to_num(features_all, nan=0.0)
    sw_trimmed = sample_weight[MAX_LOOKBACK:] if sample_weight is not None else None

    targets = compute_targets(train_df)[MAX_LOOKBACK:]
    valid = ~np.isnan(targets)
    features = features_all[valid]
    targets = targets[valid]
    vol_train = vol_safe[valid]
    sample_weight = sw_trimmed[valid] if sw_trimmed is not None else None

    # Vol-normalize targets first, then winsorize in sigma-space
    targets = targets / vol_train
    targets = np.clip(targets, -5.0, 5.0)

    # --- Monotonic constraints ---
    # Short-term returns unconstrained to allow mean reversion after extreme moves.
    # Only medium/long-term features keep positive monotonic (momentum).
    mono_cst = np.zeros(features.shape[1], dtype=int)
    # indices 0-2: 4h, 12h, 24h returns — unconstrained (mean reversion)
    # index 6: 24h VW cumulative return — unconstrained
    mono_cst[3] = 1  # 48h vol-normalized return (momentum)
    mono_cst[4] = 1  # 72h vol-normalized return (momentum)
    mono_cst[5] = 1  # 168h vol-normalized return (momentum)
    # indices 28-29: directional efficiency — unconstrained (high efficiency in downtrend = bearish)

    # --- Train: single return model ---
    model = HistGradientBoostingRegressor(
        max_iter=1000,
        max_depth=4,
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=15,
        l2_regularization=1.5,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model.fit(features, targets, sample_weight=sample_weight)

    # --- Train 72h auxiliary model (higher regularization for higher-variance target) ---
    targets_72 = compute_targets(train_df, horizon=72)[MAX_LOOKBACK:]
    valid_72 = ~np.isnan(targets_72)
    tgt_72 = targets_72[valid_72] / vol_safe[valid_72]
    tgt_72 = np.clip(tgt_72, -5.0, 5.0)
    sw_72 = sw_trimmed[valid_72] if sw_trimmed is not None else None

    model_72 = HistGradientBoostingRegressor(
        max_iter=800,
        max_depth=3,
        min_samples_leaf=1000,
        learning_rate=0.01,
        max_leaf_nodes=10,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model_72.fit(features_all[valid_72], tgt_72, sample_weight=sw_72)
    print(f"  72h auxiliary model trained (regularized)")

    # --- Position scaler: regressor on binary vol target ---
    scaler_targets_raw = compute_vol_targets(train_df)
    scaler_targets = scaler_targets_raw[MAX_LOOKBACK:]
    scaler_targets = scaler_targets[valid]  # same valid mask as return targets
    scaler_binary = (scaler_targets > 0.9).astype(np.float64)  # very sensitive: dampen unless vol clearly declining

    # Exclude low-importance features (guided by permutation importance)
    scaler_exclude = {6, 7, 8, 16, 17, 18, 19, 20, 28, 29}
    scaler_feat_mask = [i for i in range(features.shape[1]) if i not in scaler_exclude]
    scaler_features = features[:, scaler_feat_mask]

    pos_scaler = HistGradientBoostingRegressor(
        max_iter=1000,
        max_depth=4,
        min_samples_leaf=400,
        learning_rate=0.02,
        max_leaf_nodes=20,
        l2_regularization=1.5,
        random_state=42,
    )
    # Weight positive examples 3x to focus on high-vol tail
    scaler_sample_weight = np.where(scaler_binary == 1, 3.0, 1.0)
    if sample_weight is not None:
        scaler_sample_weight = scaler_sample_weight * sample_weight
    pos_scaler.fit(scaler_features, scaler_binary, sample_weight=scaler_sample_weight)

    print(f"  Pos scaler target: {scaler_binary.mean()*100:.1f}% positive ({len(scaler_feat_mask)} features)")
    vtp = np.clip(pos_scaler.predict(scaler_features), 0.0, 1.0)
    print(f"  Pos scaler train: mean={vtp.mean():.3f} std={vtp.std():.3f} "
          f"min={vtp.min():.3f} max={vtp.max():.3f} >0.5={100*(vtp>0.5).mean():.1f}%")
    if hasattr(pos_scaler, 'feature_importances_'):
        imp = pos_scaler.feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        print(f"  Pos scaler top features: {', '.join(f'{i}:{imp[i]:.3f}' for i in top_idx)}")

    # --- Confidence scaler: predict favorable conditions (binary classifier) ---
    conf_targets_raw = compute_confidence_targets(train_df)
    conf_targets = conf_targets_raw[MAX_LOOKBACK:]
    conf_targets = conf_targets[valid]

    # Vol/structure features only — no return-directional features
    conf_feat_indices = [9, 10, 11, 12, 13, 14, 15, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
    conf_features = features[:, conf_feat_indices]

    conf_valid = ~np.isnan(conf_targets)
    conf_binary = conf_targets[conf_valid].astype(int)

    conf_model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=3,
        min_samples_leaf=800,
        learning_rate=0.02,
        max_leaf_nodes=10,
        l2_regularization=2.0,
        random_state=42,
    )
    conf_model.fit(
        conf_features[conf_valid],
        conf_binary,
        sample_weight=sample_weight[conf_valid] if sample_weight is not None else None,
    )

    rtp = conf_model.predict_proba(conf_features[conf_valid])[:, 1]
    conf_train_p5 = float(np.percentile(rtp, 5))
    conf_train_p95 = float(np.percentile(rtp, 95))
    print(f"  Confidence target: {conf_binary.mean()*100:.1f}% favorable ({len(conf_feat_indices)} features)")
    print(f"  Confidence train: mean={rtp.mean():.3f} std={rtp.std():.3f} "
          f"p5={conf_train_p5:.3f} p95={conf_train_p95:.3f}")

    # Approximate param count (24h + 48h return models + position scaler + confidence scaler)
    n_params = sum(
        model._predictors[j][0].get_n_leaf_nodes()
        for j in range(len(model._predictors))
    )
    n_params += sum(
        model_72._predictors[j][0].get_n_leaf_nodes()
        for j in range(len(model_72._predictors))
    )
    n_params += sum(
        pos_scaler._predictors[j][0].get_n_leaf_nodes()
        for j in range(len(pos_scaler._predictors))
    )
    n_params += sum(
        conf_model._predictors[j][0].get_n_leaf_nodes()
        for j in range(len(conf_model._predictors))
    )

    # --- Return prediction closure ---
    def predict_fn(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sigma-space predictions on arbitrary OHLCV data."""
        feats, ts, vol = compute_features(df)
        feats = np.nan_to_num(feats, nan=0.0)

        # 24h prediction + 72h asymmetric modulation (+2% agree, -8% disagree)
        pred_24 = model.predict(feats)
        pred_72 = model_72.predict(feats)
        sign_match = np.sign(pred_24) * np.sign(pred_72)
        # Asymmetric: dampen more on disagreement than boost on agreement
        # agree (sign_match=+1): 1.0 + 0.02 = 1.02
        # disagree (sign_match=-1): 1.0 - 0.08 = 0.92
        # neutral (sign_match=0): 1.0 - 0.03 = 0.97
        sigma_preds = pred_24 * (1.0 - 0.03 + 0.05 * sign_match)

        # Confidence scaler — asymmetric threshold, EMA-24
        conf_pred = conf_model.predict_proba(feats[:, conf_feat_indices])[:, 1]
        conf_smooth = pd.Series(conf_pred).ewm(span=24, min_periods=1).mean().values

        # Normalize to [0, 1] using train range
        conf_range = max(conf_train_p95 - conf_train_p5, 1e-6)
        conf_norm = np.clip((conf_smooth - conf_train_p5) / conf_range, 0.0, 1.0)

        # Asymmetric threshold: dampen danger tail, boost favorable tail
        dampen = np.clip((conf_norm - 0.05) / 0.10, 0.0, 1.0)
        boost = np.clip((conf_norm - 0.85) / 0.10, 0.0, 1.0)
        conf_adj = 0.70 + 0.30 * dampen + 0.20 * boost
        sigma_preds = sigma_preds * conf_adj

        # Position scaling — regressor on binary target, clip to [0,1] as pseudo-probability
        scaler_signal = np.clip(pos_scaler.predict(feats[:, scaler_feat_mask]), 0.0, 1.0)
        predict_fn.last_vol_ratio = scaler_signal
        print(f"  Vol eval: mean={scaler_signal.mean():.3f} std={scaler_signal.std():.3f} "
              f"min={scaler_signal.min():.3f} max={scaler_signal.max():.3f} >0.5={100*(scaler_signal>0.5).mean():.1f}%")
        pos_scale = 1.08 - 0.72 * scaler_signal
        sigma_preds = sigma_preds * pos_scale

        sigma_preds = sigma_preds * 0.35
        sigma_smoothed = _smooth_predictions(sigma_preds)
        return sigma_smoothed, ts, vol

    predict_fn.n_params = n_params
    predict_fn.n_selected = features.shape[1]
    predict_fn.n_total_features = features.shape[1]
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
