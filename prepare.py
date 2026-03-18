"""
Autotrader data preparation, backtesting engine, and evaluation.

DO NOT MODIFY — the autonomous agent modifies only train.py.

This module provides:
  - Data download and caching (BTC/USD hourly OHLCV from Binance)
  - Walk-forward evaluation across multiple independent windows
  - A backtesting engine that converts predictions into trading signals
  - Black-box composite metric (higher is better)

Public API for train.py:
  load_train_data() -> pd.DataFrame
  evaluate_model(build_model_fn) -> dict
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
SIGMA_THRESHOLD = 0.20   # sigma threshold for trading (regime-invariant)
SIGMA_FULL_POSITION = 0.50  # sigma for full position size (±1.0)
FEE_RATE = 0.001        # 0.1% per trade (one side)
SLIPPAGE_RATE = 0.0005  # 0.05% per trade (one side)

# Fixed start of all data
TRAIN_START = pd.Timestamp("2018-01-01")


# ---------------------------------------------------------------------------
# Walk-Forward Windows
# ---------------------------------------------------------------------------

def _get_walk_forward_windows():
    """4 walk-forward windows: 3yr train, 1yr eval, non-overlapping eval periods.

    Each window trains on a different 3-year period and evaluates on the
    following calendar year. The recipe (build_model) is retrained independently
    for each window.

    | Window | Training        | Eval   |
    |--------|-----------------|--------|
    | 1      | 2019-01 – 2021-12 | 2022 |
    | 2      | 2020-01 – 2022-12 | 2023 |
    | 3      | 2021-01 – 2023-12 | 2024 |
    | 4      | 2022-01 – 2024-12 | 2025 |
    """
    T = pd.Timestamp
    return [
        {
            "train_start": T("2019-01-01"),
            "train_end": T("2021-12-31 23:00:00"),
            "eval_start": T("2022-01-01"),
            "eval_end": T("2022-12-31 23:00:00"),
            "subperiods": [
                (T("2022-01-01"), T("2022-06-30 23:00:00"), "2022 H1"),
                (T("2022-07-01"), T("2022-12-31 23:00:00"), "2022 H2"),
            ],
        },
        {
            "train_start": T("2020-01-01"),
            "train_end": T("2022-12-31 23:00:00"),
            "eval_start": T("2023-01-01"),
            "eval_end": T("2023-12-31 23:00:00"),
            "subperiods": [
                (T("2023-01-01"), T("2023-06-30 23:00:00"), "2023 H1"),
                (T("2023-07-01"), T("2023-12-31 23:00:00"), "2023 H2"),
            ],
        },
        {
            "train_start": T("2021-01-01"),
            "train_end": T("2023-12-31 23:00:00"),
            "eval_start": T("2024-01-01"),
            "eval_end": T("2024-12-31 23:00:00"),
            "subperiods": [
                (T("2024-01-01"), T("2024-06-30 23:00:00"), "2024 H1"),
                (T("2024-07-01"), T("2024-12-31 23:00:00"), "2024 H2"),
            ],
        },
        {
            "train_start": T("2022-01-01"),
            "train_end": T("2024-12-31 23:00:00"),
            "eval_start": T("2025-01-01"),
            "eval_end": T("2025-12-31 23:00:00"),
            "subperiods": [
                (T("2025-01-01"), T("2025-06-30 23:00:00"), "2025 H1"),
                (T("2025-07-01"), T("2025-12-31 23:00:00"), "2025 H2"),
            ],
        },
    ]


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
    """Return OHLCV DataFrame for the most recent walk-forward window's training data.

    Returns Window 4 training data: 2022-01-01 to 2024-12-31. This is what
    the agent uses for local development. The scored evaluation retrains on
    all windows internally via build_model.
    """
    df = _load_all_data()
    windows = _get_walk_forward_windows()
    w = windows[-1]  # Window 4 (most recent)
    mask = (df["timestamp"] >= w["train_start"]) & (df["timestamp"] <= w["train_end"])
    result = df[mask].reset_index(drop=True)
    print(f"  Training data: {w['train_start'].date()} to {w['train_end'].date()} "
          f"({len(result)} rows)")
    return result


# ---------------------------------------------------------------------------
# Backtesting Engine (internal)
# ---------------------------------------------------------------------------

def _backtest(sigma_predictions: np.ndarray, close_prices: np.ndarray,
              timestamps: np.ndarray, subperiods: list) -> dict:
    """Run backtest on sigma-space predictions against actual prices.

    Position sizing is done in sigma-space (regime-invariant).
    Portfolio returns are computed in dollar terms against actual prices.

    Args:
        sigma_predictions: Array of sigma-space predictions, one per timestamp.
        close_prices: Array of close prices aligned with predictions.
        timestamps: Array of pd.Timestamp aligned with predictions.
        subperiods: List of (start, end, label) tuples for consistency check.

    Returns:
        dict with keys: sharpe, max_drawdown, n_trades, total_return,
                        subperiod_returns (list of floats).
    """
    n = len(sigma_predictions)
    assert len(close_prices) == n and len(timestamps) == n

    # Continuous position sizing in sigma-space: linear ramp from 0 at
    # ±SIGMA_THRESHOLD to ±1.0 at ±SIGMA_FULL_POSITION
    abs_sigma = np.abs(sigma_predictions)
    sigma_scale = SIGMA_FULL_POSITION - SIGMA_THRESHOLD
    raw_size = np.where(abs_sigma > SIGMA_THRESHOLD,
                        (abs_sigma - SIGMA_THRESHOLD) / sigma_scale, 0.0)
    positions = np.clip(raw_size, 0.0, 1.0) * np.sign(sigma_predictions)

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
    for sp_start, sp_end, *_ in subperiods:
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
# Internal: run build_model on a window and backtest
# ---------------------------------------------------------------------------

def _eval_window(build_model_fn, all_data, window, window_idx):
    """Train a model on one window's training data and backtest on its eval period.

    Returns (backtest_result, train_seconds, eval_preds) where eval_preds is
    the array of sigma predictions on the eval period (for diagnostics).
    """
    train_mask = ((all_data["timestamp"] >= window["train_start"]) &
                  (all_data["timestamp"] <= window["train_end"]))
    train_data = all_data[train_mask].reset_index(drop=True)

    t0 = time.time()
    predict_fn = build_model_fn(train_data)
    train_seconds = time.time() - t0

    print(f"  Window {window_idx}: trained in {train_seconds:.1f}s, ", end="", flush=True)

    # Generate predictions on full dataset (for feature lookback context)
    sigma_preds, timestamps, vol = predict_fn(all_data)
    sigma_preds = np.asarray(sigma_preds, dtype=np.float64).ravel()
    timestamps = np.asarray(timestamps).ravel()

    assert len(sigma_preds) == len(timestamps), (
        f"predict_fn returned {len(sigma_preds)} predictions but "
        f"{len(timestamps)} timestamps"
    )

    pred_df = pd.DataFrame({"timestamp": timestamps, "sigma_pred": sigma_preds})

    # Extract eval period and backtest
    eval_mask = ((all_data["timestamp"] >= window["eval_start"]) &
                 (all_data["timestamp"] <= window["eval_end"]))
    eval_data = all_data[eval_mask].reset_index(drop=True)
    merged = eval_data.merge(pred_df, on="timestamp", how="inner")

    if len(merged) == 0:
        print("no predictions!")
        return {
            "sharpe": -10.0, "max_drawdown": -1.0,
            "n_trades": 0, "total_return": -1.0,
            "subperiod_returns": [-1.0] * len(window["subperiods"]),
        }, train_seconds, np.array([])

    eval_preds = merged["sigma_pred"].values

    bt = _backtest(
        sigma_predictions=eval_preds,
        close_prices=merged["close"].values,
        timestamps=merged["timestamp"].values,
        subperiods=window["subperiods"],
    )
    print("evaluated")
    return bt, train_seconds, eval_preds


# ---------------------------------------------------------------------------
# Public Evaluation API
# ---------------------------------------------------------------------------

def evaluate_model(build_model_fn: callable) -> dict:
    """Black-box walk-forward evaluation of a model recipe.

    The recipe (build_model_fn) is retrained independently on each of 4
    walk-forward windows. Each window has 3 years of training data and
    1 year of non-overlapping evaluation. The composite score is the
    worst-case performance across all windows.

    The agent sees only the composite score. Per-window results are hidden.

    Args:
        build_model_fn: Callable that takes a pd.DataFrame (training data)
            and returns a predict_fn. The predict_fn takes a pd.DataFrame
            and returns (sigma_predictions, timestamps, vol).

    Returns:
        dict with keys: score, sharpe_min, max_drawdown, total_trades,
                        consistency.
    """
    all_data = _load_all_data()
    windows = _get_walk_forward_windows()

    print(f"Evaluating ({len(windows)} walk-forward windows)...")
    window_results = []
    for i, w in enumerate(windows):
        bt, _, _ = _eval_window(build_model_fn, all_data, w, i + 1)
        window_results.append(bt)

    # --- Composite score ---

    # Base: minimum Sharpe across all windows.
    sharpes = [r["sharpe"] for r in window_results]
    base = min(sharpes)

    # Drawdown: worst across all windows
    worst_dd_raw = min(r["max_drawdown"] for r in window_results)  # most negative
    dd = abs(worst_dd_raw)
    if dd <= 0.10:
        dd_mult = 1.0
    else:
        dd_mult = 1.0 / (1.0 + ((dd - 0.10) / 0.15) ** 2)

    # Trade count: per-window exponential. Min across windows penalizes
    # going silent on any window.
    window_trade_mults = []
    for w, result in zip(windows, window_results):
        eval_hours = (w["eval_end"] - w["eval_start"]).total_seconds() / 3600
        scale = eval_hours / (FORWARD_HOURS * 7)
        window_trade_mults.append(1 - math.exp(-result["n_trades"] / scale))
    trade_mult = min(window_trade_mults)
    total_trades = sum(r["n_trades"] for r in window_results)

    # Consistency: fraction of ALL subperiods (4 windows × 2 = 8) that are profitable.
    all_sp_returns = []
    for r in window_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0

    if base >= 0:
        score = base * dd_mult * trade_mult * consistency
    else:
        score = base / max(dd_mult, 0.01) / max(trade_mult, 0.01) / max(consistency, 0.01)

    return {
        "score": score,
        "sharpe_min": base,
        "max_drawdown": worst_dd_raw,
        "total_trades": total_trades,
        "consistency": f"{n_profitable}/{n_total}",
    }


# ---------------------------------------------------------------------------
# Per-Window Diagnostic (human-only)
# ---------------------------------------------------------------------------

def _pred_stats_line(preds):
    """One-line prediction distribution summary."""
    abs_p = np.abs(preds)
    e10 = (abs_p > 0.10).sum() / len(preds) * 100
    e20 = (abs_p > 0.20).sum() / len(preds) * 100
    e30 = (abs_p > 0.30).sum() / len(preds) * 100
    return (f"  Predictions: mean={preds.mean():+.4f}, std={preds.std():.4f}, "
            f"|pred|>0.10: {e10:.0f}%, >0.20: {e20:.0f}%, >0.30: {e30:.0f}%")


def _run_diagnostic():
    """Per-window diagnostic breakdown with prediction stats.

    Human-only tool. Retrains the recipe on each walk-forward window and
    prints per-window backtest results and prediction distributions. Also
    trains on 2023-2025 and evaluates on 2026+ as the true holdout.
    """
    import train as train_module

    all_data = _load_all_data()
    windows = _get_walk_forward_windows()

    print("=" * 60)
    print("Training and evaluating across walk-forward windows...")
    print("=" * 60 + "\n")

    window_results = []
    window_preds = []
    window_train_times = []
    for i, w in enumerate(windows):
        bt, train_time, eval_preds = _eval_window(
            train_module.build_model, all_data, w, i + 1)
        window_results.append(bt)
        window_preds.append(eval_preds)
        window_train_times.append(train_time)

    # Compute composite score (same formula as evaluate_model)
    sharpes = [r["sharpe"] for r in window_results]
    base = min(sharpes)
    worst_dd_raw = min(r["max_drawdown"] for r in window_results)
    dd = abs(worst_dd_raw)
    dd_mult = 1.0 if dd <= 0.10 else 1.0 / (1.0 + ((dd - 0.10) / 0.15) ** 2)

    window_trade_mults = []
    for w, result in zip(windows, window_results):
        eval_hours = (w["eval_end"] - w["eval_start"]).total_seconds() / 3600
        scale = eval_hours / (FORWARD_HOURS * 7)
        window_trade_mults.append(1 - math.exp(-result["n_trades"] / scale))
    trade_mult = min(window_trade_mults)
    total_trades = sum(r["n_trades"] for r in window_results)

    all_sp_returns = []
    for r in window_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0

    if base >= 0:
        score = base * dd_mult * trade_mult * consistency
    else:
        score = base / max(dd_mult, 0.01) / max(trade_mult, 0.01) / max(consistency, 0.01)

    # Print diagnostic breakdown
    print("\n" + "=" * 60)
    print("=== DIAGNOSTIC BREAKDOWN ===")
    print("(Human-only — this information is hidden from the agent)")
    print("=" * 60)
    print(f"\nOverall: score={score:.4f}, sharpe_min={base:.4f}, "
          f"consistency={n_profitable}/{n_total}")
    print(f"  dd_mult={dd_mult:.4f}, trade_mult={trade_mult:.4f}")

    losing_subperiods = []

    for i, (w, result, preds) in enumerate(
            zip(windows, window_results, window_preds)):
        train_range = f"{w['train_start'].date()}-{w['train_end'].date()}"
        eval_range = f"{w['eval_start'].date()}-{w['eval_end'].date()}"
        print(f"\n--- Window {i+1}: Train {train_range}, Eval {eval_range} ---")
        print(f"  Sharpe: {result['sharpe']:.4f}"
              f"{'  <-- MIN' if result['sharpe'] == base else ''}")
        print(f"  Max drawdown: {result['max_drawdown']:.1%}")
        print(f"  Trades: {result['n_trades']}  (trade_mult={window_trade_mults[i]:.4f})"
              f"{'  <-- MIN' if window_trade_mults[i] == trade_mult else ''}")
        print(f"  Total return: {result['total_return']:+.1%}")

        for (sp_start, sp_end, label), sp_ret in zip(
                w["subperiods"], result["subperiod_returns"]):
            symbol = "OK" if sp_ret > 0 else "LOSS"
            print(f"  {label} ({sp_start.date()} to {sp_end.date()}): "
                  f"return {sp_ret:+.2%}  {symbol}")
            if sp_ret <= 0:
                losing_subperiods.append(label)

        if len(preds) > 0:
            print(_pred_stats_line(preds))

    print(f"\nLosing subperiods: {', '.join(losing_subperiods) if losing_subperiods else 'None'}")

    # True holdout: train on 2023-2025, evaluate on 2026+
    T = pd.Timestamp
    th_train_start = T("2023-01-01")
    th_train_end = T("2025-12-31 23:00:00")
    th_eval_start = T("2026-01-01")

    th_eval_data = all_data[all_data["timestamp"] >= th_eval_start]
    if len(th_eval_data) > 0:
        th_eval_end = th_eval_data["timestamp"].max()
        th_window = {
            "train_start": th_train_start,
            "train_end": th_train_end,
            "eval_start": th_eval_start,
            "eval_end": th_eval_end,
            "subperiods": [(th_eval_start, th_eval_end, "2026 YTD")],
        }
        bt_th, _, th_preds = _eval_window(
            train_module.build_model, all_data, th_window, "holdout")
        print(f"\n--- True Holdout (UNSCORED) ---")
        print(f"  (Trained on {th_train_start.date()}-{th_train_end.date()}, "
              f"evaluated on {th_eval_start.date()}-{th_eval_end.date()})")
        print(f"  Sharpe: {bt_th['sharpe']:.4f}")
        print(f"  Max drawdown: {bt_th['max_drawdown']:.1%}")
        print(f"  Trades: {bt_th['n_trades']}")
        print(f"  Total return: {bt_th['total_return']:+.1%}")
        print(f"  2026 YTD: return {bt_th['subperiod_returns'][0]:+.2%}")
        if len(th_preds) > 0:
            print(_pred_stats_line(th_preds))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotrader data preparation and diagnostics")
    parser.add_argument("--diagnose", action="store_true",
                        help="Per-window diagnostic breakdown + prediction stats + true holdout (human-only)")
    args = parser.parse_args()

    if args.diagnose:
        _run_diagnostic()
    else:
        df = download_data()
        print(f"\nData summary:")
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        windows = _get_walk_forward_windows()
        for i, w in enumerate(windows):
            n_train = ((df["timestamp"] >= w["train_start"]) &
                       (df["timestamp"] <= w["train_end"])).sum()
            n_eval = ((df["timestamp"] >= w["eval_start"]) &
                      (df["timestamp"] <= w["eval_end"])).sum()
            print(f"  Window {i+1}: "
                  f"Train {w['train_start'].date()}-{w['train_end'].date()} ({n_train} rows), "
                  f"Eval {w['eval_start'].date()}-{w['eval_end'].date()} ({n_eval} rows, "
                  f"{len(w['subperiods'])} subperiods)")
