"""Frozen copy of autotrader feature computation.

This file is a static snapshot of compute_features() from
assets/btc_hourly/train.py. It must exactly match the feature
computation that produced the model artifacts.

DO NOT MODIFY unless retraining with updated features.
"""

import numpy as np
import pandas as pd

# Source commit when this copy was made
SOURCE_COMMIT = "8382079"

RETURN_LOOKBACKS = [4, 12, 24, 48, 72, 168]  # 1h removed — noisiest
VOLATILITY_WINDOWS = [24, 168]
MAX_LOOKBACK = 168  # maximum lookback window (1 week)

FEATURE_COUNT_WITHOUT_FUNDING = 36
FEATURE_COUNT_WITH_FUNDING = 39


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
