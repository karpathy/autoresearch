"""
Autotrader data preparation, backtesting engine, and evaluation.

DO NOT MODIFY — the autonomous agent modifies only train.py.

This module provides:
  - Data download and caching (BTC/USD hourly OHLCV from Binance)
  - Strict temporal train/val/holdout splits
  - A backtesting engine that converts predictions into trading signals
  - Black-box composite metric (higher is better)

Public API for train.py:
  load_train_data() -> pd.DataFrame
  evaluate_model(predict_fn, n_params) -> dict
  TIME_BUDGET       -> int (seconds)
  FORWARD_HOURS     -> int
"""

import argparse
import io
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autotrader"
PARQUET_PATH = CACHE_DIR / "btcusdt_1h.parquet"

TIME_BUDGET = 240  # seconds for training

FORWARD_HOURS = 24      # predict 24-hour forward returns
THRESHOLD = 0.005       # 0.5% threshold for long/short signals
POSITION_SCALE = 0.02   # prediction level for full position (±1.0)
FEE_RATE = 0.001        # 0.1% per trade (one side)
SLIPPAGE_RATE = 0.0005  # 0.05% per trade (one side)

# Fixed start of all data
TRAIN_START = pd.Timestamp("2018-01-01")


# ---------------------------------------------------------------------------
# Walk-Forward Splits (computed from data, not hardcoded)
# ---------------------------------------------------------------------------

def _compute_splits(data_end):
    """Compute walk-forward splits: train to T-12mo, val T-12 to T-6mo, holdout T-6mo to T.

    Splits snap to month boundaries. Train gets 3 subperiods (long window
    deserves more granularity), val and holdout get 2 each. Total: 7 subperiods,
    matching the original consistency denominator.
    """
    ref_month = data_end.to_period('M') + 1  # first of next month

    train_start = TRAIN_START
    holdout_end = data_end
    holdout_start = (ref_month - 6).to_timestamp()
    val_end = holdout_start - pd.Timedelta(hours=1)
    val_start = (ref_month - 12).to_timestamp()
    train_end = val_start - pd.Timedelta(hours=1)

    def split_n(start, end, n):
        """Split a time range into n roughly equal subperiods."""
        total = end - start
        chunk = total / n
        subs = []
        for i in range(n):
            s = start + chunk * i
            e = start + chunk * (i + 1) if i < n - 1 else end
            subs.append((s.floor('h'), e.floor('h')))
        return subs

    return [
        (train_start, train_end, split_n(train_start, train_end, 3)),
        (val_start, val_end, split_n(val_start, val_end, 2)),
        (holdout_start, holdout_end, split_n(holdout_start, holdout_end, 2)),
    ]


_cached_splits = None


def _get_splits():
    """Return walk-forward splits, computing them on first call."""
    global _cached_splits
    if _cached_splits is None:
        all_data = _load_all_data()
        data_end = all_data["timestamp"].max()
        _cached_splits = _compute_splits(data_end)
        for name, (s, e, subs) in zip(["Train", "Val", "Holdout"], _cached_splits):
            print(f"  {name}: {s.date()} to {e.date()} ({len(subs)} subperiods)")
    return _cached_splits


# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

_BINANCE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines"
    "/BTCUSDT/1h/BTCUSDT-1h-{year}-{month:02d}.zip"
)

# Binance CSV columns (no header in file)
_BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _download_month(year: int, month: int, max_retries: int = 5) -> pd.DataFrame | None:
    """Download a single month of hourly klines from Binance public data."""
    url = _BINANCE_URL.format(year=year, month=month)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, names=_BINANCE_COLS)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} for {year}-{month:02d} "
                      f"(waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED to download {year}-{month:02d} after "
                      f"{max_retries} attempts: {e}")
                return None


def download_data() -> pd.DataFrame:
    """Download BTC/USD hourly OHLCV data through the current month and cache."""
    # Cache invalidation: if cached data is stale (>60 days old), re-download
    if PARQUET_PATH.exists():
        cached = pd.read_parquet(PARQUET_PATH)
        max_ts = cached["timestamp"].max()
        if (pd.Timestamp.now() - max_ts).days <= 60:
            print(f"Loading cached data from {PARQUET_PATH}")
            return cached
        else:
            print(f"Cached data ends at {max_ts.date()}, re-downloading...")
            PARQUET_PATH.unlink()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading BTC/USD hourly data from Binance...")

    now = pd.Timestamp.now()
    frames = []
    for year in range(2018, now.year + 1):
        for month in range(1, 13):
            if year == now.year and month > now.month:
                break
            print(f"  Downloading {year}-{month:02d}...", end=" ", flush=True)
            df = _download_month(year, month)
            if df is not None:
                print(f"{len(df)} rows")
                frames.append(df)
            else:
                print("skipped (not available)")

    if not frames:
        raise RuntimeError("No data downloaded. Check network connectivity.")

    raw = pd.concat(frames, ignore_index=True)

    # Convert open_time to datetime.
    # Binance changed from milliseconds to microseconds in Jan 2025.
    timestamps = raw["open_time"].values.astype(np.float64)
    is_microseconds = timestamps > 1e15
    ts_seconds = np.where(is_microseconds, timestamps / 1e6, timestamps / 1e3)
    raw["timestamp"] = pd.to_datetime(ts_seconds, unit="s", utc=True)
    raw["timestamp"] = raw["timestamp"].dt.tz_localize(None)

    df = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "open": raw["open"].astype(float),
        "high": raw["high"].astype(float),
        "low": raw["low"].astype(float),
        "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })

    # Sort and deduplicate
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Forward-fill any gaps (create complete hourly index)
    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_idx).ffill().reset_index()
    df = df.rename(columns={"index": "timestamp"})

    # Filter from TRAIN_START — no upper bound, include all available data
    df = df[df["timestamp"] >= TRAIN_START].reset_index(drop=True)

    print(f"Total: {len(df)} hourly candles "
          f"({df['timestamp'].min()} to {df['timestamp'].max()})")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Cached to {PARQUET_PATH}")

    return df


def _load_all_data() -> pd.DataFrame:
    """Load the full dataset (all periods). Internal use only."""
    return download_data()


# ---------------------------------------------------------------------------
# Public Data Loader
# ---------------------------------------------------------------------------

def load_train_data() -> pd.DataFrame:
    """Return OHLCV DataFrame for the training period only.

    The training window is computed from walk-forward splits (everything
    before the validation window). The agent should use ONLY this data.
    """
    df = _load_all_data()
    splits = _get_splits()
    train_start, train_end, _ = splits[0]
    mask = (df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtesting Engine (internal)
# ---------------------------------------------------------------------------

def _backtest(predictions: np.ndarray, close_prices: np.ndarray,
              timestamps: np.ndarray, subperiods: list) -> dict:
    """Run backtest on predictions against actual prices.

    Args:
        predictions: Array of forward-return predictions, one per timestamp.
        close_prices: Array of close prices aligned with predictions.
        timestamps: Array of pd.Timestamp aligned with predictions.
        subperiods: List of (start, end) Timestamp tuples for consistency check.

    Returns:
        dict with keys: sharpe, max_drawdown, n_trades, total_return,
                        subperiod_returns (list of floats).
    """
    n = len(predictions)
    assert len(close_prices) == n and len(timestamps) == n

    # Continuous position sizing: linear ramp from 0 at ±THRESHOLD to ±1.0 at ±POSITION_SCALE
    abs_pred = np.abs(predictions)
    scale_range = POSITION_SCALE - THRESHOLD
    raw_size = np.where(abs_pred > THRESHOLD, (abs_pred - THRESHOLD) / scale_range, 0.0)
    positions = np.clip(raw_size, 0.0, 1.0) * np.sign(predictions)

    # Compute hourly price returns
    price_returns = np.zeros(n)
    price_returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    # Portfolio returns: position at time t earns the return from t to t+1.
    # The position decided at t-1 is held during period t.
    portfolio_returns = np.zeros(n)
    n_trades = 0

    for i in range(1, n):
        pos = positions[i - 1]
        portfolio_returns[i] = pos * price_returns[i]

        # Fee model: cost proportional to position change magnitude
        prev_pos = positions[i - 2] if i >= 2 else 0.0
        pos_change = abs(pos - prev_pos)

        if pos_change > 1e-10:
            cost = pos_change * (FEE_RATE + SLIPPAGE_RATE)
            portfolio_returns[i] -= cost

        # Count trades on zero-crossings only (direction changes)
        if (prev_pos > 0 and pos < 0) or (prev_pos < 0 and pos > 0) \
                or (prev_pos == 0 and pos != 0) or (prev_pos != 0 and pos == 0):
            n_trades += 1

    # Equity curve
    equity = np.cumprod(1.0 + portfolio_returns)

    # Sharpe ratio (annualized from hourly)
    if np.std(portfolio_returns) > 0:
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * math.sqrt(8760)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown))  # negative number

    # Total return
    total_return = float(equity[-1] / equity[0] - 1.0) if len(equity) > 0 else 0.0

    # Subperiod returns
    ts_series = pd.Series(timestamps)
    subperiod_returns = []
    for sp_start, sp_end in subperiods:
        mask = (ts_series >= sp_start) & (ts_series <= sp_end)
        if mask.sum() > 0:
            sp_equity = np.cumprod(1.0 + portfolio_returns[mask.values])
            sp_return = float(sp_equity[-1] - 1.0)
            subperiod_returns.append(sp_return)
        else:
            subperiod_returns.append(0.0)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "total_return": total_return,
        "subperiod_returns": subperiod_returns,
    }


# ---------------------------------------------------------------------------
# Public Evaluation API
# ---------------------------------------------------------------------------

def evaluate_model(predict_fn: callable) -> dict:
    """Black-box model evaluation across all temporal splits.

    Runs the model on training, validation, and holdout data internally.
    Returns a single composite score. The agent never sees per-split
    metrics and cannot determine which split is the bottleneck.

    The score is based on the MINIMUM Sharpe ratio across all splits,
    ensuring the model must generalize to achieve a positive score.

    Args:
        predict_fn: Callable that takes a pd.DataFrame (with columns
            timestamp, open, high, low, close, volume) and returns
            (predictions: np.ndarray, timestamps: np.ndarray).
            Must work on arbitrary date ranges.

    Returns:
        dict with keys: score, sharpe_min, max_drawdown, total_trades,
                        consistency.
        sharpe_min is the minimum Sharpe across all evaluation periods.
        max_drawdown is the worst drawdown across all periods.
        total_trades is the sum across all periods.
        consistency is "N/M" where N profitable subperiods out of M total.
    """
    # Generate predictions on the FULL dataset so features have full
    # lookback context even at the start of val/holdout periods.
    all_data = _load_all_data()
    predictions, timestamps = predict_fn(all_data)
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    timestamps = np.asarray(timestamps).ravel()

    assert len(predictions) == len(timestamps), (
        f"predict_fn returned {len(predictions)} predictions but "
        f"{len(timestamps)} timestamps"
    )

    # Build prediction lookup
    pred_df = pd.DataFrame({"timestamp": timestamps, "prediction": predictions})

    # Backtest each split independently
    split_results = []
    for split_start, split_end, subperiods in _get_splits():
        mask = (all_data["timestamp"] >= split_start) & (all_data["timestamp"] <= split_end)
        split_data = all_data[mask].reset_index(drop=True)

        merged = split_data.merge(pred_df, on="timestamp", how="inner")

        if len(merged) == 0:
            # No predictions for this split — worst possible result
            split_results.append({
                "sharpe": -10.0,
                "max_drawdown": -1.0,
                "n_trades": 0,
                "total_return": -1.0,
                "subperiod_returns": [-1.0] * len(subperiods),
            })
            continue

        bt = _backtest(
            predictions=merged["prediction"].values,
            close_prices=merged["close"].values,
            timestamps=merged["timestamp"].values,
            subperiods=subperiods,
        )
        split_results.append(bt)

    # --- Composite score ---

    # Base: minimum Sharpe across all splits.
    # The model MUST generalize — train-only performance can't compensate
    # for poor out-of-sample performance.
    sharpes = [r["sharpe"] for r in split_results]
    base = min(sharpes)

    # Drawdown: worst across all splits
    worst_dd_raw = min(r["max_drawdown"] for r in split_results)  # most negative
    dd = abs(worst_dd_raw)
    if dd <= 0.10:
        dd_mult = 1.0
    else:
        dd_mult = 1.0 / (1.0 + ((dd - 0.10) / 0.15) ** 2)

    # Trade count: total across all splits, ramp to 100
    # (~33 per split on average for full credit)
    total_trades = sum(r["n_trades"] for r in split_results)
    trade_mult = min(1.0, total_trades / 100.0)

    # Consistency: fraction of ALL subperiods that are profitable.
    # 7 total subperiods: 3 train + 2 val + 2 holdout.
    # A model overfit to train gets at most 3/7 ≈ 0.43x.
    all_sp_returns = []
    for r in split_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0

    score = base * dd_mult * trade_mult * consistency

    return {
        "score": score,
        "sharpe_min": base,
        "max_drawdown": worst_dd_raw,
        "total_trades": total_trades,
        "consistency": f"{n_profitable}/{n_total}",
    }


# ---------------------------------------------------------------------------
# Fresh Holdout Validation (human-only diagnostic)
# ---------------------------------------------------------------------------

def _run_fresh_holdout():
    """Validate the trained model — signal quality on the holdout window."""
    import train as train_module
    from scipy.stats import spearmanr

    all_data = _load_all_data()
    splits = _get_splits()
    holdout_start, holdout_end, _ = splits[2]

    # 1. Train the model (prints normal evaluation output)
    print("\n" + "=" * 50)
    print("Training model (standard pipeline)...")
    print("=" * 50 + "\n")
    train_module.main()

    # Capture the evaluation results
    eval_result = evaluate_model(train_module.predict_on_data)

    # 2. Signal quality on holdout window
    print("\n" + "=" * 50)
    print("=== SIGNAL QUALITY (sigma-space, pre-pipeline) ===")
    print("=" * 50)
    print(f"\nHoldout window: {holdout_start.date()} to {holdout_end.date()}")

    # Raw sigma predictions (no EMA, no vol denorm, no tanh)
    features, feat_timestamps, vol_safe = train_module.compute_features(all_data)
    features = np.nan_to_num(features, nan=0.0)
    sigma_preds = train_module._trained_model.predict(features)

    # Actual 24h forward returns, vol-normalized to sigma-space
    raw_targets = train_module.compute_targets(all_data)
    raw_targets = raw_targets[train_module.MAX_LOOKBACK:]
    actual_sigma = raw_targets / vol_safe

    # Holdout slice (excluding last 24h with no future data)
    holdout_mask = (
        (pd.to_datetime(feat_timestamps) >= holdout_start) &
        (pd.to_datetime(feat_timestamps) <= holdout_end) &
        ~np.isnan(raw_targets)
    )
    n_holdout_hours = holdout_mask.sum()

    if n_holdout_hours == 0:
        print("\nNo holdout hours with known future returns. Cannot assess signal.")
    else:
        sig_preds = sigma_preds[holdout_mask]
        sig_actuals = actual_sigma[holdout_mask]

        # Directional accuracy (excluding low-conviction predictions)
        conviction_mask = np.abs(sig_preds) >= 0.1
        n_low_conviction = (~conviction_mask).sum()
        if conviction_mask.sum() > 0:
            signs_match = np.sign(sig_preds[conviction_mask]) == np.sign(sig_actuals[conviction_mask])
            directional_acc = signs_match.mean() * 100
        else:
            directional_acc = float("nan")

        # Rank correlation (all hours)
        rho, pvalue = spearmanr(sig_preds, sig_actuals)

        print(f"Holdout hours:         {n_holdout_hours}  (excluding last 24h with no future data)")
        print()
        print(f"Directional accuracy:  {directional_acc:.1f}%  (sign match, excluding low-conviction)")
        print(f"  Random baseline:     50.0%")
        print(f"  Low-conviction excluded: {n_low_conviction} hours (|sigma_pred| < 0.1)")
        print()
        print(f"Rank correlation:      {rho:.4f}  (Spearman rho)")
        print(f"  p-value:             {pvalue:.4f}")
        print()
        print(f"Sigma prediction stats:")
        print(f"  mean:  {sig_preds.mean():.4f}    actual mean:  {sig_actuals.mean():.4f}")
        print(f"  std:   {sig_preds.std():.4f}    actual std:   {sig_actuals.std():.4f}")
        print()

        print(f"For comparison, the model's evaluation results:")
        print(f"  score:        {eval_result['score']:.4f}")
        print(f"  sharpe_min:   {eval_result['sharpe_min']:.4f}")
        print(f"  max_drawdown: {eval_result['max_drawdown']:.1%}")
        print(f"  total_trades: {eval_result['total_trades']}")
        print(f"  consistency:  {eval_result['consistency']}")
        print()

        if directional_acc > 53 and rho > 0.03:
            sig_verdict = "SIGNAL PRESENT"
        elif directional_acc > 51 or rho > 0.01:
            sig_verdict = "INCONCLUSIVE"
        else:
            sig_verdict = "NO SIGNAL"

        print(f"Signal verdict: {sig_verdict}")
        print(f"  (SIGNAL PRESENT: accuracy > 53% AND rho > 0.03)")


# ---------------------------------------------------------------------------
# Per-Split Diagnostic (human-only)
# ---------------------------------------------------------------------------

def _run_diagnostic():
    """Per-split and per-subperiod diagnostic breakdown.

    Human-only tool for diagnosing which subperiods are losing money.
    Trains the model, runs the same evaluation as evaluate_model(),
    but prints intermediate per-split results instead of aggregating.
    """
    import train as train_module

    # 1. Train the model
    print("=" * 60)
    print("Training model...")
    print("=" * 60 + "\n")
    train_module.main()

    # 2. Generate predictions on full dataset (same as evaluate_model)
    all_data = _load_all_data()
    predictions, timestamps = train_module.predict_on_data(all_data)
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    timestamps = np.asarray(timestamps).ravel()
    pred_df = pd.DataFrame({"timestamp": timestamps, "prediction": predictions})

    # 3. Backtest each split and collect results
    splits = _get_splits()
    split_names = ["Train", "Val", "Holdout"]
    split_results = []
    split_subperiods = []

    for split_start, split_end, subperiods in splits:
        mask = (all_data["timestamp"] >= split_start) & (all_data["timestamp"] <= split_end)
        split_data = all_data[mask].reset_index(drop=True)
        merged = split_data.merge(pred_df, on="timestamp", how="inner")

        if len(merged) == 0:
            split_results.append({
                "sharpe": -10.0, "max_drawdown": -1.0,
                "n_trades": 0, "total_return": -1.0,
                "subperiod_returns": [-1.0] * len(subperiods),
            })
        else:
            bt = _backtest(
                predictions=merged["prediction"].values,
                close_prices=merged["close"].values,
                timestamps=merged["timestamp"].values,
                subperiods=subperiods,
            )
            split_results.append(bt)
        split_subperiods.append(subperiods)

    # 4. Compute composite score (same formula as evaluate_model)
    sharpes = [r["sharpe"] for r in split_results]
    base = min(sharpes)
    worst_dd_raw = min(r["max_drawdown"] for r in split_results)
    dd = abs(worst_dd_raw)
    dd_mult = 1.0 if dd <= 0.10 else 1.0 / (1.0 + ((dd - 0.10) / 0.15) ** 2)
    total_trades = sum(r["n_trades"] for r in split_results)
    trade_mult = min(1.0, total_trades / 100.0)
    all_sp_returns = []
    for r in split_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0
    score = base * dd_mult * trade_mult * consistency

    # 5. Print diagnostic breakdown
    print("\n" + "=" * 60)
    print("=== DIAGNOSTIC BREAKDOWN ===")
    print("(Human-only — this information is hidden from the agent)")
    print("=" * 60)
    print(f"\nOverall: score={score:.4f}, sharpe_min={base:.4f}, "
          f"consistency={n_profitable}/{n_total}")
    print(f"  dd_mult={dd_mult:.4f}, trade_mult={trade_mult:.4f}")

    losing_subperiods = []
    sp_idx = 0

    for i, (name, (s, e, subperiods), result) in enumerate(
            zip(split_names, splits, split_results)):
        print(f"\n--- Split {i+1}: {name} ({s.date()} to {e.date()}) ---")
        print(f"  Sharpe: {result['sharpe']:.4f}"
              f"{'  <-- MIN' if result['sharpe'] == base else ''}")
        print(f"  Max drawdown: {result['max_drawdown']:.1%}")
        print(f"  Trades: {result['n_trades']}")
        print(f"  Total return: {result['total_return']:+.1%}")

        for j, ((sp_start, sp_end), sp_ret) in enumerate(
                zip(subperiods, result["subperiod_returns"])):
            mark = "+" if sp_ret > 0 else "-"
            symbol = "OK" if sp_ret > 0 else "LOSS"
            print(f"  Subperiod {j+1} ({sp_start.date()} to {sp_end.date()}): "
                  f"return {sp_ret:+.2%}  {symbol}")
            if sp_ret <= 0:
                losing_subperiods.append(f"{name}-{j+1}")
            sp_idx += 1

    print(f"\nLosing subperiods: {', '.join(losing_subperiods) if losing_subperiods else 'None'}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotrader data preparation and validation")
    parser.add_argument("--validate", action="store_true",
                        help="Run signal quality validation on holdout window")
    parser.add_argument("--diagnose", action="store_true",
                        help="Per-split diagnostic breakdown (human-only)")
    args = parser.parse_args()

    if args.validate:
        _run_fresh_holdout()
    elif args.diagnose:
        _run_diagnostic()
    else:
        df = download_data()
        print(f"\nData summary:")
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        splits = _get_splits()
        for name, (s, e, subs) in zip(["Train", "Val", "Holdout"], splits):
            n = ((df["timestamp"] >= s) & (df["timestamp"] <= e)).sum()
            print(f"  {name:8s} {s.date()} to {e.date()}  ({n} rows, {len(subs)} subperiods)")