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
# Model — Multiple mean-reversion signals with vol gating
# ---------------------------------------------------------------------------

N_FEATURES = len(RETURN_LOOKBACKS) + len(VOLATILITY_WINDOWS) + 1 + 2 + 2 + 1  # 14

# Feature indices
IDX_1H = 0; IDX_4H = 1; IDX_12H = 2; IDX_24H = 3; IDX_72H = 4; IDX_168H = 5
IDX_VOL24 = 6; IDX_VOL168 = 7


class ForwardReturnModel(nn.Module):
    """Multi-signal mean reversion with volatility gating.

    Combines mean-reversion signals from multiple timeframes.
    No gradient training — parameters set via grid search.
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        # Weights for each return lookback (negative = mean reversion)
        self.weights = nn.Parameter(torch.zeros(6))  # one per return lookback
        self.vol_thresh = nn.Parameter(torch.tensor(0.0))
        self._n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weighted combination of return features (indices 0-5)
        signal = (x[:, :6] * self.weights.unsqueeze(0)).sum(dim=1)
        # Vol gate
        vol = x[:, IDX_VOL24]
        gate = torch.sigmoid(-(vol - self.vol_thresh) * 3.0)
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
_predict_fn = None  # numpy prediction function


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
# Backtest proxy (fast, fee-adjusted)
# ---------------------------------------------------------------------------

FEE = 0.001 + 0.0005  # fee + slippage per side


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

    close = train_df["close"].values[MAX_LOOKBACK:][valid]

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Setup model ---
    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)
    print(f"  Model parameters: {n_params}")

    # --- Comprehensive grid search ---
    # Use actual evaluate_model to score top candidates (slower but accurate)
    print("Phase 1: Single feature scan...")
    candidates = []  # list of (proxy_score, feat_idx, sign, scale, vol_thresh)
    scales = np.arange(0.001, 0.020, 0.0005)

    for feat_idx in range(6):
        for sign in [-1, +1]:
            for scale in scales:
                preds = sign * features[:, feat_idx] * scale
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                if n_trades >= 30:
                    score = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                    candidates.append((score, feat_idx, sign, scale, 3.0))  # 3.0 = no gating

    # Phase 1b: Vol gating on top single features
    print("Phase 2: Vol gating on top candidates...")
    top_singles = sorted(candidates, reverse=True)[:10]  # top 10 singles
    for _, feat_idx, sign, scale, _ in top_singles:
        for vol_thresh in np.arange(-1.0, 2.5, 0.25):
            gate = 1.0 / (1.0 + np.exp((features[:, IDX_VOL24] - vol_thresh) * 3.0))
            preds = sign * features[:, feat_idx] * scale * gate
            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
            if n_trades >= 30:
                score = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                candidates.append((score, feat_idx, sign, scale, vol_thresh))

    # Phase 2: Two-feature combinations
    print("Phase 3: Two-feature combinations...")
    combo_scales = [0.001, 0.002, 0.003, 0.005]
    for i in range(6):
        for j in range(i + 1, 6):
            for si in [-1, +1]:
                for sj in [-1, +1]:
                    for sc_i in combo_scales:
                        for sc_j in combo_scales:
                            preds = si * features[:, i] * sc_i + sj * features[:, j] * sc_j
                            sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                            if n_trades >= 30:
                                score = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                                # Store as special format with negative feat_idx
                                candidates.append((score, -(i * 10 + j), si, sc_i, sj * sc_j))

    # Sort and pick the best
    candidates.sort(reverse=True)
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Top 5 proxy scores: {[f'{c[0]:.4f}' for c in candidates[:5]]}")

    # Now use evaluate_model on top candidates to find the actual best
    print("Phase 4: Verifying top candidates with actual evaluator...")
    best_actual_score = -999
    best_predict_fn = None
    best_actual_desc = ""

    for rank, cand in enumerate(candidates[:20]):
        proxy_score, feat_id, sign, scale, extra = cand

        if feat_id >= 0:
            # Single feature
            vol_thresh = extra

            def make_fn(fi, s, sc, vt):
                def fn(feats):
                    sig = s * feats[:, fi] * sc
                    if vt < 2.5:
                        gate = 1.0 / (1.0 + np.exp((feats[:, IDX_VOL24] - vt) * 3.0))
                        return sig * gate
                    return sig
                return fn

            pred_fn = make_fn(feat_id, sign, scale, vol_thresh)
            desc = f"feat{feat_id} s={sign:+d} sc={scale:.4f} vt={vol_thresh:.2f}"
        else:
            # Two features
            i = (-feat_id) // 10
            j = (-feat_id) % 10
            sc_j = extra  # sign * scale for j

            def make_fn2(fi, fj, si, sc_i, sc_j_val):
                def fn(feats):
                    return si * feats[:, fi] * sc_i + feats[:, fj] * sc_j_val
                return fn

            pred_fn = make_fn2(i, j, sign, scale, sc_j)
            desc = f"feat{i}+feat{j}"

        train_preds = pred_fn(features)
        result = evaluate_model(train_preds, train_timestamps, n_params, split="train")
        actual_score = result["score"]

        if actual_score > best_actual_score:
            best_actual_score = actual_score
            best_predict_fn = pred_fn
            best_actual_desc = desc
            print(f"  #{rank}: {desc} proxy={proxy_score:.4f} actual={actual_score:.4f} "
                  f"sharpe={result['sharpe']:.4f} dd={result['max_drawdown']:.1%} trades={result['n_trades']}")

    training_seconds = time.time() - total_start
    print(f"\n  Best actual: {best_actual_desc} score={best_actual_score:.4f}")

    global _predict_fn
    _predict_fn = best_predict_fn
    _trained_model = model

    training_seconds = time.time() - total_start
    print(f"Search complete in {training_seconds:.1f}s")

    # --- Evaluate on train split using actual best ---
    print("Evaluating on training data...")
    all_preds = best_predict_fn(features)

    print(f"  Pred stats: mean={np.mean(all_preds):.6f}, std={np.std(all_preds):.6f}")
    print(f"  Preds > 0.005: {np.sum(all_preds > 0.005)}, Preds < -0.005: {np.sum(all_preds < -0.005)}")

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = _normalize(val_features, fit=False)
    val_features = np.nan_to_num(val_features, nan=0.0)

    val_preds = best_predict_fn(val_features)

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
