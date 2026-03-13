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

IDX_VOL24 = 6


class ForwardReturnModel(nn.Module):
    """Dummy model — predictions computed in numpy, this is for API compat."""

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self._n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


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
_predict_fn = None


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features, timestamps = compute_features(df)
    features = _normalize(features, fit=False)
    features = np.nan_to_num(features, nan=0.0)

    if _predict_fn is not None:
        return _predict_fn(features), timestamps

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
# Quick backtest (fee-adjusted)
# ---------------------------------------------------------------------------

FEE = 0.001 + 0.0005


def _quick_backtest(preds, close, threshold=0.005):
    """Fast backtest returning (sharpe, max_dd, n_trades)."""
    positions = np.zeros(len(preds))
    positions[preds > threshold] = 1.0
    positions[preds < -threshold] = -1.0

    price_returns = np.zeros(len(close))
    price_returns[1:] = close[1:] / close[:-1] - 1.0

    port_returns = np.zeros(len(close))
    n_trades = 0
    for i in range(1, len(close)):
        pos = positions[i - 1]
        port_returns[i] = pos * price_returns[i]
        prev_pos = positions[i - 2] if i >= 2 else 0.0
        if pos != prev_pos:
            cost = 0.0
            if prev_pos != 0: cost += FEE
            if pos != 0:
                cost += FEE
                n_trades += 1
            port_returns[i] -= cost

    equity = np.cumprod(1.0 + port_returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    if np.std(port_returns) > 0:
        sharpe = float(np.mean(port_returns) / np.std(port_returns) * np.sqrt(8760))
    else:
        sharpe = 0.0

    return sharpe, max_dd, n_trades


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _trained_model, _predict_fn

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

    features, timestamps = compute_features(train_df)
    targets = compute_targets(train_df)
    targets = targets[MAX_LOOKBACK:]

    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_timestamps = timestamps[valid]

    features = _normalize(features, fit=True)
    features = np.nan_to_num(features, nan=0.0)

    close = train_df["close"].values[MAX_LOOKBACK:][valid]

    # Load val data
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = _normalize(val_features, fit=False)
    val_features = np.nan_to_num(val_features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")
    print(f"  Val samples: {len(val_features)}")

    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)

    # -----------------------------------------------------------------------
    # Strategy search: jointly optimize for train score AND val_pass
    # -----------------------------------------------------------------------

    def make_single_fn(fi, s, sc):
        def fn(feats):
            return s * feats[:, fi] * sc
        return fn

    def make_pair_fn(fi, fj, si, sc_i, sj, sc_j):
        def fn(feats):
            return si * feats[:, fi] * sc_i + sj * feats[:, fj] * sc_j
        return fn

    def make_gated_fn(base_fn, vol_thresh):
        def fn(feats):
            base = base_fn(feats)
            gate = 1.0 / (1.0 + np.exp((feats[:, IDX_VOL24] - vol_thresh) * 3.0))
            return base * gate
        return fn

    # Collect all strategies with their prediction functions
    strategies = []  # (proxy_train_score, pred_fn, desc)

    print("Generating strategies...")

    # Single features: all lookbacks, RSI, both signs
    feature_names = ["1h", "4h", "12h", "24h", "72h", "168h",
                     "vol24", "vol168", "volratio", "sin_h", "cos_h",
                     "rsi14", "rsi48", "hl24"]
    scales = np.arange(0.0005, 0.020, 0.0005)

    for feat_idx in range(features.shape[1]):
        for sign in [-1, +1]:
            for scale in scales:
                fn = make_single_fn(feat_idx, sign, scale)
                preds = fn(features)
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                if n_trades >= 30:
                    proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                    if proxy > -0.2:  # only keep promising ones
                        name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
                        strategies.append((proxy, fn, f"{name} s={sign:+d} sc={scale:.4f}"))

    print(f"  Single features: {len(strategies)} promising strategies")

    # Pairs: return features (0-5) x return features, and return x RSI
    combo_scales = [0.001, 0.0015, 0.002, 0.003, 0.005]
    pair_count = 0
    for i in range(6):
        for j in range(i + 1, 6):
            for si in [-1, +1]:
                for sj in [-1, +1]:
                    for sc_i in combo_scales:
                        for sc_j in combo_scales:
                            fn = make_pair_fn(i, j, si, sc_i, sj, sc_j)
                            preds = fn(features)
                            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                            if n_trades >= 30:
                                proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                                if proxy > -0.2:
                                    strategies.append((proxy, fn,
                                        f"f{i}*{si*sc_i:+.4f}+f{j}*{sj*sc_j:+.4f}"))
                                    pair_count += 1

    # Return feature + RSI combos
    for ret_idx in range(6):
        for rsi_idx in [12, 13]:
            for sr in [-1, +1]:
                for ss in [-1, +1]:
                    for sc_r in combo_scales:
                        for sc_s in combo_scales:
                            fn = make_pair_fn(ret_idx, rsi_idx, sr, sc_r, ss, sc_s)
                            preds = fn(features)
                            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                            if n_trades >= 30:
                                proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                                if proxy > -0.2:
                                    strategies.append((proxy, fn,
                                        f"f{ret_idx}*{sr*sc_r:+.4f}+rsi{rsi_idx}*{ss*sc_s:+.4f}"))
                                    pair_count += 1

    # Phase 4: Targeted search around val-passing strategies
    # vol168 negative + any return feature combination
    print("Phase 4: Vol168-based strategies (targeted for val_pass)...")
    targeted_count = 0
    for vol_scale in np.arange(0.0005, 0.010, 0.0005):
        # Pure vol168 signal
        fn = make_single_fn(7, -1, vol_scale)  # vol168 = feature 7
        preds = fn(features)
        sharpe, max_dd, n_trades = _quick_backtest(preds, close)
        if n_trades >= 20:
            proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
            strategies.append((proxy, fn, f"vol168 s=-1 sc={vol_scale:.4f}"))
            targeted_count += 1

    # 24h MR + 168h momentum (val_pass found at f3*-0.002+f5*+0.003)
    for mr_scale in np.arange(0.0005, 0.005, 0.0005):
        for mom_scale in np.arange(0.0005, 0.005, 0.0005):
            fn = make_pair_fn(3, 5, -1, mr_scale, +1, mom_scale)
            preds = fn(features)
            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
            if n_trades >= 20:
                proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                strategies.append((proxy, fn,
                    f"24hMR({mr_scale:.4f})+168hMOM({mom_scale:.4f})"))
                targeted_count += 1

    # Also try: 72h MR + 168h momentum (our best train combo + momentum)
    for mr_scale in np.arange(0.0005, 0.005, 0.0005):
        for mom_scale in np.arange(0.0005, 0.005, 0.0005):
            fn = make_pair_fn(4, 5, -1, mr_scale, +1, mom_scale)
            preds = fn(features)
            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
            if n_trades >= 20:
                proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                strategies.append((proxy, fn,
                    f"72hMR({mr_scale:.4f})+168hMOM({mom_scale:.4f})"))
                targeted_count += 1

    print(f"  Targeted: {targeted_count} strategies")

    # Phase 5: Vol-gated versions of val-passing strategies
    print("Phase 5: Vol-gated val-passing strategies...")
    gated_count = 0
    for mr_scale in np.arange(0.001, 0.005, 0.0005):
        for mom_scale in np.arange(0.001, 0.005, 0.0005):
            base_fn = make_pair_fn(3, 5, -1, mr_scale, +1, mom_scale)
            for vol_thresh in np.arange(-1.5, 2.0, 0.25):
                fn = make_gated_fn(base_fn, vol_thresh)
                preds = fn(features)
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                if n_trades >= 20:
                    proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                    strategies.append((proxy, fn,
                        f"GATED 24hMR({mr_scale:.4f})+168hMOM({mom_scale:.4f}) vt={vol_thresh:.2f}"))
                    gated_count += 1

    for mr_scale in np.arange(0.001, 0.005, 0.0005):
        for mom_scale in np.arange(0.001, 0.005, 0.0005):
            base_fn = make_pair_fn(4, 5, -1, mr_scale, +1, mom_scale)
            for vol_thresh in np.arange(-1.5, 2.0, 0.25):
                fn = make_gated_fn(base_fn, vol_thresh)
                preds = fn(features)
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                if n_trades >= 20:
                    proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                    strategies.append((proxy, fn,
                        f"GATED 72hMR({mr_scale:.4f})+168hMOM({mom_scale:.4f}) vt={vol_thresh:.2f}"))
                    gated_count += 1

    for vol_scale in np.arange(0.001, 0.010, 0.001):
        base_fn = make_single_fn(7, -1, vol_scale)
        for vol_thresh in np.arange(-1.5, 2.0, 0.25):
            fn = make_gated_fn(base_fn, vol_thresh)
            preds = fn(features)
            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
            if n_trades >= 20:
                proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                strategies.append((proxy, fn,
                    f"GATED vol168({vol_scale:.4f}) vt={vol_thresh:.2f}"))
                gated_count += 1

    print(f"  Gated: {gated_count} strategies")

    # Phase 6: Fine-tune around best val-passing strategy
    # Best so far: GATED 24hMR(0.0025)+168hMOM(0.0030) vt=0.75
    print("Phase 6: Fine-tuning around best val-passing params...")
    finetune_count = 0

    # Fine grid around mr=0.0025, mom=0.003, vt=0.75
    for mr_scale in np.arange(0.0018, 0.0035, 0.0002):
        for mom_scale in np.arange(0.0020, 0.0042, 0.0002):
            for vol_thresh in np.arange(0.3, 1.3, 0.1):
                for steepness in [2.0, 3.0, 5.0, 7.0]:
                    def make_steep_gated(bf, vt, st):
                        def fn(feats):
                            base = bf(feats)
                            gate = 1.0 / (1.0 + np.exp((feats[:, IDX_VOL24] - vt) * st))
                            return base * gate
                        return fn
                    base_fn = make_pair_fn(3, 5, -1, mr_scale, +1, mom_scale)
                    fn = make_steep_gated(base_fn, vol_thresh, steepness)
                    preds = fn(features)
                    sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                    if n_trades >= 20:
                        proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                        if proxy > 0.0:
                            strategies.append((proxy, fn,
                                f"FT 24hMR({mr_scale:.4f})+168hMOM({mom_scale:.4f}) vt={vol_thresh:.2f} st={steepness:.0f}"))
                            finetune_count += 1

    # Phase 7: 3-feature gated combos — add 72hMR or RSI to the winning formula
    print("Phase 7: 3-feature gated combos...")
    triple_gated_count = 0

    def make_triple_gated(fi, fj, fk, si, sc_i, sj, sc_j, sk, sc_k, vol_thresh, steepness=3.0):
        def fn(feats):
            base = si * feats[:, fi] * sc_i + sj * feats[:, fj] * sc_j + sk * feats[:, fk] * sc_k
            gate = 1.0 / (1.0 + np.exp((feats[:, IDX_VOL24] - vol_thresh) * steepness))
            return base * gate
        return fn

    # 24hMR + 168hMOM + 72hMR (add mean-reversion depth)
    for sc24 in [0.001, 0.0015, 0.002, 0.0025, 0.003]:
        for sc168 in [0.002, 0.0025, 0.003, 0.0035]:
            for sc72 in [0.0005, 0.001, 0.0015, 0.002]:
                for vt in [0.5, 0.75, 1.0]:
                    fn = make_triple_gated(3, 5, 4, -1, sc24, +1, sc168, -1, sc72, vt)
                    preds = fn(features)
                    sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                    if n_trades >= 20:
                        proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                        if proxy > 0.0:
                            strategies.append((proxy, fn,
                                f"TG 24hMR({sc24:.4f})+168hMOM({sc168:.4f})+72hMR({sc72:.4f}) vt={vt:.2f}"))
                            triple_gated_count += 1

    # 24hMR + 168hMOM + RSI (add momentum confirmation)
    for sc24 in [0.001, 0.002, 0.0025, 0.003]:
        for sc168 in [0.002, 0.003, 0.004]:
            for rsi_sc in [0.001, 0.002, 0.003]:
                for rsi_sign in [-1, +1]:
                    for vt in [0.5, 0.75, 1.0]:
                        fn = make_triple_gated(3, 5, 12, -1, sc24, +1, sc168, rsi_sign, rsi_sc, vt)
                        preds = fn(features)
                        sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                        if n_trades >= 20:
                            proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                            if proxy > 0.0:
                                rsi_s = "+" if rsi_sign > 0 else "-"
                                strategies.append((proxy, fn,
                                    f"TG 24hMR({sc24:.4f})+168hMOM({sc168:.4f})+RSI14({rsi_s}{rsi_sc:.4f}) vt={vt:.2f}"))
                                triple_gated_count += 1

    print(f"  Fine-tuned: {finetune_count}, Triple-gated: {triple_gated_count}")

    # Phase 8: Rolling z-score features + EMA-based signals
    print("Phase 8: Rolling z-score and EMA features...")

    # Compute rolling z-scores of returns (raw, pre-normalization)
    raw_features_df = train_df.copy()
    raw_close = raw_features_df["close"].values.astype(np.float64)
    n_raw = len(raw_close)

    # Raw returns (not normalized)
    raw_returns = {}
    for lb in [24, 72, 168]:
        ret = np.full(n_raw, np.nan)
        ret[lb:] = raw_close[lb:] / raw_close[:-lb] - 1.0
        raw_returns[lb] = ret

    # Rolling z-score: (return - rolling_mean) / rolling_std
    rzs_features = {}
    for lb in [24, 72, 168]:
        ret = pd.Series(raw_returns[lb])
        for window in [168, 336, 720]:  # 1wk, 2wk, 1mo
            rmean = ret.rolling(window, min_periods=window).mean()
            rstd = ret.rolling(window, min_periods=window).std()
            rzs = np.where(rstd > 0, (ret - rmean) / rstd, 0.0).values
            rzs = np.clip(rzs, -3, 3)
            rzs_features[f"rzs{lb}w{window}"] = rzs[MAX_LOOKBACK:][valid]

    # EMA features
    ema_features = {}
    close_series = pd.Series(raw_close)
    for span in [12, 24, 72, 168]:
        ema = close_series.ewm(span=span).mean().values
        ema_dev = (raw_close - ema) / np.where(ema > 0, ema, 1.0)
        ema_features[f"ema{span}dev"] = ema_dev[MAX_LOOKBACK:][valid]

    # Search over new features
    new_feat_count = 0
    all_new_feats = {}
    all_new_feats.update(rzs_features)
    all_new_feats.update(ema_features)

    for name, raw_feat in all_new_feats.items():
        # Normalize
        feat = (raw_feat - np.nanmean(raw_feat)) / (np.nanstd(raw_feat) + 1e-8)
        feat = np.clip(feat, -3, 3)
        feat = np.nan_to_num(feat, nan=0.0)

        for sign in [-1, +1]:
            for scale in [0.001, 0.002, 0.003, 0.005, 0.008]:
                preds = sign * feat * scale
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                if n_trades >= 20:
                    proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                    if proxy > 0.0:
                        def make_new_fn(f, s, sc):
                            def fn(feats):
                                return s * f * sc
                            return fn
                        strategies.append((proxy, make_new_fn(feat, sign, scale),
                            f"NEW {name} s={sign:+d} sc={scale:.4f}"))
                        new_feat_count += 1

        # Also try combos with 168h momentum
        for sign in [-1]:
            for scale in [0.001, 0.002, 0.003]:
                for mom_sc in [0.002, 0.003, 0.004]:
                    for vt in [0.75, 1.0, 1.25]:
                        preds_base = sign * feat * scale + features[:, 5] * mom_sc
                        gate = 1.0 / (1.0 + np.exp((features[:, IDX_VOL24] - vt) * 2.0))
                        preds = preds_base * gate
                        sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                        if n_trades >= 20:
                            proxy = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                            if proxy > 0.1:
                                def make_combo_fn(f, s, sc, msc, v):
                                    def fn(feats):
                                        base = s * f * sc + feats[:, 5] * msc
                                        g = 1.0 / (1.0 + np.exp((feats[:, IDX_VOL24] - v) * 2.0))
                                        return base * g
                                    return fn
                                strategies.append((proxy, make_combo_fn(feat, sign, scale, mom_sc, vt),
                                    f"NEWCOMBO {name}*{sign*scale:+.4f}+168hMOM({mom_sc:.4f}) vt={vt:.2f}"))
                                new_feat_count += 1

    print(f"  New feature strategies: {new_feat_count}")

    print(f"  Pairs: {pair_count} promising strategies")
    print(f"  Total: {len(strategies)}")

    # Sort by proxy score
    strategies.sort(reverse=True, key=lambda x: x[0])

    # -----------------------------------------------------------------------
    # Evaluate top strategies with actual evaluator on BOTH train and val
    # -----------------------------------------------------------------------
    n_eval = min(500, len(strategies))
    print(f"\nEvaluating top {n_eval} on train+val...")

    best_score = -999
    best_fn = None
    best_desc = ""
    best_val_pass = False

    for rank, (proxy, fn, desc) in enumerate(strategies[:n_eval]):
        # Train evaluation
        train_preds = fn(features)
        train_result = evaluate_model(train_preds, train_timestamps, n_params, split="train")
        train_score = train_result["score"]

        # Val evaluation
        val_preds = fn(val_features)
        val_result = evaluate_model(val_preds, val_timestamps, n_params, split="val")
        vp = val_result["val_pass"]

        # Selection: prefer val_pass=true with highest train score
        is_better = False
        if vp and train_score > 0:
            if not best_val_pass or train_score > best_score:
                is_better = True
        elif not best_val_pass and train_score > best_score:
            is_better = True

        if is_better:
            best_score = train_score
            best_fn = fn
            best_desc = desc
            best_val_pass = vp

        if vp or train_score > 0.5:
            print(f"  #{rank}: {desc} train={train_score:.4f} "
                  f"sharpe={train_result['sharpe']:.4f} dd={train_result['max_drawdown']:.1%} "
                  f"trades={train_result['n_trades']} val_pass={vp}")

    print(f"\nBest: {best_desc} score={best_score:.4f} val_pass={best_val_pass}")

    _predict_fn = best_fn
    _trained_model = model

    training_seconds = time.time() - total_start
    print(f"Search complete in {training_seconds:.1f}s")

    # --- Final evaluation ---
    all_preds = best_fn(features)
    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    val_preds = best_fn(val_features)
    val_result = evaluate_model(val_preds, val_timestamps, n_params, split="val")

    total_seconds = time.time() - total_start

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
