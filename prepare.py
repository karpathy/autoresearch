"""
Autotrader data preparation, backtesting engine, and evaluation.

DO NOT MODIFY — the autonomous agent modifies only train.py.

This module provides:
  - Data download and caching (BTC/USD hourly OHLCV from Binance)
  - Strict temporal train/val/holdout splits
  - A backtesting engine that converts predictions into trading signals
  - Composite metric computation (higher is better)
  - Validation pass/fail without leaking the actual metric

Public API for train.py:
  load_train_data() -> pd.DataFrame
  load_val_data()   -> pd.DataFrame
  evaluate_model(predictions, timestamps, n_params, split) -> dict
  TIME_BUDGET       -> int (seconds)
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

# Subperiods for consistency check (train window)
TRAIN_SUBPERIODS = [
    (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31 23:00:00")),
    (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31 23:00:00")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31 23:00:00")),
]

# Subperiods for consistency check (val window)
VAL_SUBPERIODS = [
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:00:00")),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30 23:00:00")),
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
# Public Data Loaders
# ---------------------------------------------------------------------------

def load_train_data() -> pd.DataFrame:
    """Return OHLCV DataFrame for the training period only (2018-2022)."""
    df = _load_all_data()
    mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    return df[mask].reset_index(drop=True)


def load_val_data() -> pd.DataFrame:
    """Return OHLCV DataFrame for the validation period only (2023-01 to 2024-06).

    No targets — prevents lookahead bias. The agent uses this to generate
    predictions, which are then evaluated by evaluate_model().
    """
    df = _load_all_data()
    mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
    return df[mask].reset_index(drop=True)


def _load_holdout_data() -> pd.DataFrame:
    """Return OHLCV DataFrame for the holdout period. Internal use only."""
    df = _load_all_data()
    mask = (df["timestamp"] >= HOLDOUT_START) & (df["timestamp"] <= HOLDOUT_END)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtesting Engine (internal)
# ---------------------------------------------------------------------------

def _compute_forward_returns(close: np.ndarray) -> np.ndarray:
    """Compute FORWARD_HOURS-ahead percentage returns.

    Returns array of same length as input; last FORWARD_HOURS entries are NaN.
    """
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0
    return fwd


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


def _compute_score(sharpe: float, max_drawdown: float, n_trades: int,
                   subperiod_returns: list, n_params: int,
                   min_subperiods_profitable: int) -> float:
    """Compute composite score with penalty multiplier.

    score = sharpe_ratio * penalty_multiplier

    penalty_multiplier is 0.0 if:
      - n_trades < 30
      - |max_drawdown| > 25%
      - fewer than min_subperiods_profitable subperiods are profitable

    Soft penalty for model complexity:
      penalty_multiplier *= 1 / (1 + 0.02 * n_params_thousands)
    """
    penalty = 1.0

    if n_trades < 30:
        penalty = 0.0
    if abs(max_drawdown) > 0.25:
        penalty = 0.0

    profitable_periods = sum(1 for r in subperiod_returns if r > 0)
    if profitable_periods < min_subperiods_profitable:
        penalty = 0.0

    n_params_k = n_params / 1000.0
    penalty *= 1.0 / (1.0 + 0.02 * n_params_k)

    return sharpe * penalty


# ---------------------------------------------------------------------------
# Public Evaluation API
# ---------------------------------------------------------------------------

def evaluate_model(predictions: np.ndarray, timestamps: np.ndarray,
                   n_params: int, split: str = "train") -> dict:
    """Evaluate model predictions via backtesting.

    Args:
        predictions: 1-D array of forward-return predictions.
        timestamps: 1-D array of pd.Timestamp (same length as predictions).
            Each timestamp is the time at which the prediction was made.
        n_params: Number of trainable parameters in the model.
        split: "train" or "val".

    Returns:
        dict with keys: score, sharpe, max_drawdown, n_trades, total_return, val_pass.
        When split="val", only val_pass is meaningful; other fields are None.
    """
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    timestamps = np.asarray(timestamps).ravel()
    assert len(predictions) == len(timestamps), (
        f"predictions ({len(predictions)}) and timestamps ({len(timestamps)}) "
        f"must have the same length"
    )

    all_data = _load_all_data()

    if split == "train":
        mask = (all_data["timestamp"] >= TRAIN_START) & (all_data["timestamp"] <= TRAIN_END)
        subperiods = TRAIN_SUBPERIODS
        min_profitable = 2
    elif split == "val":
        mask = (all_data["timestamp"] >= VAL_START) & (all_data["timestamp"] <= VAL_END)
        subperiods = VAL_SUBPERIODS
        min_profitable = len(VAL_SUBPERIODS)
    elif split == "holdout":
        mask = (all_data["timestamp"] >= HOLDOUT_START) & (all_data["timestamp"] <= HOLDOUT_END)
        subperiods = [
            (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31 23:00:00")),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:00:00")),
        ]
        min_profitable = len(subperiods)
    else:
        raise ValueError(f"split must be 'train', 'val', or 'holdout', got '{split}'")

    split_data = all_data[mask].reset_index(drop=True)

    # Align predictions with price data by timestamp
    pred_df = pd.DataFrame({"timestamp": timestamps, "prediction": predictions})
    merged = split_data.merge(pred_df, on="timestamp", how="inner")

    if len(merged) == 0:
        raise ValueError(
            f"No matching timestamps between predictions and {split} data. "
            f"Predictions span {timestamps[0]} to {timestamps[-1]}, "
            f"split data spans {split_data['timestamp'].iloc[0]} to "
            f"{split_data['timestamp'].iloc[-1]}"
        )

    bt = _backtest(
        predictions=merged["prediction"].values,
        close_prices=merged["close"].values,
        timestamps=merged["timestamp"].values,
        subperiods=subperiods,
    )

    score = _compute_score(
        sharpe=bt["sharpe"],
        max_drawdown=bt["max_drawdown"],
        n_trades=bt["n_trades"],
        subperiod_returns=bt["subperiod_returns"],
        n_params=n_params,
        min_subperiods_profitable=min_profitable,
    )

    if split == "train":
        return {
            "score": score,
            "sharpe": bt["sharpe"],
            "max_drawdown": bt["max_drawdown"],
            "n_trades": bt["n_trades"],
            "total_return": bt["total_return"],
            "val_pass": None,
        }
    elif split == "val":
        return {
            "score": None,
            "sharpe": None,
            "max_drawdown": None,
            "n_trades": None,
            "total_return": None,
            "val_pass": score > 0,
        }
    else:
        return {
            "score": score,
            "sharpe": bt["sharpe"],
            "max_drawdown": bt["max_drawdown"],
            "n_trades": bt["n_trades"],
            "total_return": bt["total_return"],
            "val_pass": None,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_holdout_evaluation():
    """Run holdout evaluation. For human use only — never called by agent."""
    print("=" * 60)
    print("HOLDOUT EVALUATION")
    print("=" * 60)
    print()
    print("This evaluates the current model on held-out data")
    print("(2024-07-01 to 2025-12-31) that the agent never sees.")
    print()

    try:
        import train as train_module
    except ImportError:
        print("ERROR: Could not import train.py. Make sure it exists.")
        sys.exit(1)

    holdout_data = _load_holdout_data()
    print(f"Holdout data: {len(holdout_data)} rows "
          f"({holdout_data['timestamp'].iloc[0]} to "
          f"{holdout_data['timestamp'].iloc[-1]})")

    if not hasattr(train_module, "predict_on_data"):
        print("ERROR: train.py must define predict_on_data(df) -> (predictions, timestamps)")
        sys.exit(1)

    predictions, timestamps = train_module.predict_on_data(holdout_data)
    n_params = train_module.count_model_params()

    result = evaluate_model(predictions, timestamps, n_params, split="holdout")

    print()
    print("--- HOLDOUT RESULTS ---")
    print(f"score:        {result['score']:.4f}")
    print(f"sharpe:       {result['sharpe']:.4f}")
    print(f"max_drawdown: {result['max_drawdown']:.1%}")
    print(f"n_trades:     {result['n_trades']}")
    print(f"total_return: {result['total_return']:.1%}")
    print(f"n_params:     {n_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotrader data preparation")
    parser.add_argument("--evaluate-holdout", action="store_true",
                        help="Run holdout evaluation (human use only)")
    args = parser.parse_args()

    if args.evaluate_holdout:
        _run_holdout_evaluation()
    else:
        df = download_data()
        print(f"\nData summary:")
        print(f"  Rows: {len(df)}")
        print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        train = load_train_data()
        val = load_val_data()
        holdout = _load_holdout_data()
        print(f"  Train:   {len(train)} rows ({train['timestamp'].iloc[0]} to {train['timestamp'].iloc[-1]})")
        print(f"  Val:     {len(val)} rows ({val['timestamp'].iloc[0]} to {val['timestamp'].iloc[-1]})")
        print(f"  Holdout: {len(holdout)} rows ({holdout['timestamp'].iloc[0]} to {holdout['timestamp'].iloc[-1]})")
