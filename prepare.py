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

TIME_BUDGET = 120  # seconds for training

FORWARD_HOURS = 24      # predict 24-hour forward returns
THRESHOLD = 0.005       # 0.5% threshold for long/short signals
FEE_RATE = 0.001        # 0.1% per trade (one side)
SLIPPAGE_RATE = 0.0005  # 0.05% per trade (one side)

# Temporal split boundaries (inclusive)
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END   = pd.Timestamp("2022-12-31 23:00:00")
VAL_START   = pd.Timestamp("2023-01-01")
VAL_END     = pd.Timestamp("2024-06-30 23:00:00")
HOLDOUT_START = pd.Timestamp("2024-07-01")
HOLDOUT_END   = pd.Timestamp("2025-12-31 23:00:00")

# Subperiods for consistency check — 7 total across all splits
TRAIN_SUBPERIODS = [
    (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31 23:00:00")),
    (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31 23:00:00")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31 23:00:00")),
]

VAL_SUBPERIODS = [
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:00:00")),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30 23:00:00")),
]

HOLDOUT_SUBPERIODS = [
    (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31 23:00:00")),
    (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:00:00")),
]

# All splits for evaluation
_SPLITS = [
    (TRAIN_START, TRAIN_END, TRAIN_SUBPERIODS),
    (VAL_START, VAL_END, VAL_SUBPERIODS),
    (HOLDOUT_START, HOLDOUT_END, HOLDOUT_SUBPERIODS),
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
    """Download BTC/USD hourly OHLCV data and cache as parquet."""
    if PARQUET_PATH.exists():
        print(f"Loading cached data from {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading BTC/USD hourly data from Binance...")

    frames = []
    for year in range(2018, 2026):
        for month in range(1, 13):
            ts = pd.Timestamp(f"{year}-{month:02d}-01")
            if ts > HOLDOUT_END + pd.DateOffset(months=1):
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

    # Filter to our data range
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= HOLDOUT_END)].reset_index(drop=True)

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
    """Return OHLCV DataFrame for the training period only (2018-2022).

    This is the ONLY data the agent should use for training. The evaluation
    function handles all other splits internally.
    """
    df = _load_all_data()
    mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
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

    # Determine positions: +1 = long, -1 = short, 0 = flat
    positions = np.zeros(n, dtype=np.float64)
    positions[predictions > THRESHOLD] = 1.0
    positions[predictions < -THRESHOLD] = -1.0

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

        # Count trades: position change incurs fees + slippage
        prev_pos = positions[i - 2] if i >= 2 else 0.0

        if pos != prev_pos:
            cost = 0.0
            if prev_pos != 0.0:
                cost += FEE_RATE + SLIPPAGE_RATE  # close old
            if pos != 0.0:
                cost += FEE_RATE + SLIPPAGE_RATE  # open new
                n_trades += 1
            portfolio_returns[i] -= cost

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

def evaluate_model(predict_fn: callable, n_params: int) -> dict:
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
        n_params: Number of trainable parameters in the model.

    Returns:
        dict with keys: score, sharpe_min, max_drawdown, total_trades,
                        consistency, n_params.
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
    for split_start, split_end, subperiods in _SPLITS:
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

    # Trade count: total across all splits, ramp to 150
    # (~50 per split on average for full credit)
    total_trades = sum(r["n_trades"] for r in split_results)
    trade_mult = min(1.0, total_trades / 150.0)

    # Consistency: fraction of ALL subperiods that are profitable.
    # 7 total subperiods across train (3) + val (2) + holdout (2).
    # A model overfit to train gets at most 3/7 ≈ 0.43x.
    all_sp_returns = []
    for r in split_results:
        all_sp_returns.extend(r["subperiod_returns"])
    n_profitable = sum(1 for ret in all_sp_returns if ret > 0)
    n_total = len(all_sp_returns)
    consistency = n_profitable / n_total if n_total > 0 else 0.0

    # Parameter count: gentle complexity penalty
    n_params_k = n_params / 1000.0
    param_mult = 1.0 / (1.0 + 0.01 * n_params_k)

    score = base * dd_mult * trade_mult * consistency * param_mult

    return {
        "score": score,
        "sharpe_min": base,
        "max_drawdown": worst_dd_raw,
        "total_trades": total_trades,
        "consistency": f"{n_profitable}/{n_total}",
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = download_data()
    print(f"\nData summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    train = load_train_data()
    print(f"  Train: {len(train)} rows "
          f"({train['timestamp'].iloc[0]} to {train['timestamp'].iloc[-1]})")
    n_val = ((df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)).sum()
    n_holdout = ((df["timestamp"] >= HOLDOUT_START) & (df["timestamp"] <= HOLDOUT_END)).sum()
    print(f"  Val:     {n_val} rows")
    print(f"  Holdout: {n_holdout} rows")