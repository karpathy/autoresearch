"""
BTC/USD hourly data preparation, evaluation wrapper, and diagnostics.

DO NOT MODIFY — the autonomous agent modifies only train.py.

This module provides:
  - Data download and caching (BTC/USD hourly OHLCV from Binance)
  - Walk-forward evaluation via core infrastructure
  - Black-box composite metric (higher is better)

Public API for train.py:
  load_train_data() -> pd.DataFrame
  evaluate_model(build_model_fn) -> dict
  TIME_BUDGET       -> int (seconds)
  FORWARD_HOURS     -> int
"""

import argparse
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core.config import AssetConfig, BacktestConfig, WalkForwardConfig
from core.evaluation import evaluate_model as _core_evaluate_model
from core.evaluation import run_diagnostic as _core_run_diagnostic


# ---------------------------------------------------------------------------
# BTC-Specific Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autotrader"
PARQUET_PATH = CACHE_DIR / "btcusdt_1h.parquet"
TRAIN_START = pd.Timestamp("2018-01-01")

_BTC_CONFIG = AssetConfig(
    backtest=BacktestConfig(
        sigma_threshold=0.20,
        sigma_full_position=0.50,
        fee_rate=0.001,
        slippage_rate=0.0005,
    ),
    walk_forward=WalkForwardConfig(
        forward_hours=24,
        time_budget=240,
        epoch_length=30,
        holdout_ok_threshold=0.0,
        holdout_warn_threshold=-1.0,
    ),
    eval_count_path=CACHE_DIR / "eval_count",
    salt_env_var="AUTOTRADER_HOLDOUT_SALT",
)

# Re-export constants for train.py compatibility
FORWARD_HOURS = _BTC_CONFIG.walk_forward.forward_hours   # 24
TIME_BUDGET = _BTC_CONFIG.walk_forward.time_budget        # 240


# ---------------------------------------------------------------------------
# Walk-Forward Windows (BTC-specific)
# ---------------------------------------------------------------------------

def _get_walk_forward_windows():
    """5 expanding walk-forward windows: train from data start, 1yr eval.

    Each window trains on ALL available data from 2018-01-01 up to the year
    before its eval period. Sample weight decay (configured in core) ensures
    recent data dominates while older data provides regime diversity.

    | Window | Training          | Eval   |
    |--------|-------------------|--------|
    | 0      | 2018-01 – 2020-12 | 2021   |
    | 1      | 2018-01 – 2021-12 | 2022   |
    | 2      | 2018-01 – 2022-12 | 2023   |
    | 3      | 2018-01 – 2023-12 | 2024   |
    | 4      | 2018-01 – 2024-12 | 2025   |
    """
    T = pd.Timestamp
    return [
        {
            "train_start": T("2018-01-01"),
            "train_end": T("2020-12-31 23:00:00"),
            "eval_start": T("2021-01-01"),
            "eval_end": T("2021-12-31 23:00:00"),
            "subperiods": [
                (T("2021-01-01"), T("2021-06-30 23:00:00"), "2021 H1"),
                (T("2021-07-01"), T("2021-12-31 23:00:00"), "2021 H2"),
            ],
        },
        {
            "train_start": T("2018-01-01"),
            "train_end": T("2021-12-31 23:00:00"),
            "eval_start": T("2022-01-01"),
            "eval_end": T("2022-12-31 23:00:00"),
            "subperiods": [
                (T("2022-01-01"), T("2022-06-30 23:00:00"), "2022 H1"),
                (T("2022-07-01"), T("2022-12-31 23:00:00"), "2022 H2"),
            ],
        },
        {
            "train_start": T("2018-01-01"),
            "train_end": T("2022-12-31 23:00:00"),
            "eval_start": T("2023-01-01"),
            "eval_end": T("2023-12-31 23:00:00"),
            "subperiods": [
                (T("2023-01-01"), T("2023-06-30 23:00:00"), "2023 H1"),
                (T("2023-07-01"), T("2023-12-31 23:00:00"), "2023 H2"),
            ],
        },
        {
            "train_start": T("2018-01-01"),
            "train_end": T("2023-12-31 23:00:00"),
            "eval_start": T("2024-01-01"),
            "eval_end": T("2024-12-31 23:00:00"),
            "subperiods": [
                (T("2024-01-01"), T("2024-06-30 23:00:00"), "2024 H1"),
                (T("2024-07-01"), T("2024-12-31 23:00:00"), "2024 H2"),
            ],
        },
        {
            "train_start": T("2018-01-01"),
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
# Data Download (BTC-specific)
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

    Returns Window 4 (index -1) training data: 2022-01-01 to 2024-12-31.
    This is what the agent uses for local development. The evaluation
    retrains on all 5 windows internally via build_model.
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
# Public Evaluation API (wraps core with BTC config)
# ---------------------------------------------------------------------------

def evaluate_model(build_model_fn: callable) -> dict:
    """Black-box walk-forward evaluation with rotating holdout.

    Wraps core.evaluation.evaluate_model with BTC-specific configuration.
    """
    return _core_evaluate_model(
        build_model_fn=build_model_fn,
        load_data_fn=_load_all_data,
        windows=_get_walk_forward_windows(),
        config=_BTC_CONFIG,
    )


# ---------------------------------------------------------------------------
# Per-Window Diagnostic (human-only)
# ---------------------------------------------------------------------------

def _run_diagnostic():
    """Per-window diagnostic breakdown — BTC wrapper.

    Does NOT increment the eval counter.
    """
    import train as train_module

    all_data = _load_all_data()

    # Construct the true holdout window (2026+) if data exists
    T = pd.Timestamp
    th_eval_data = all_data[all_data["timestamp"] >= T("2026-01-01")]
    extra_holdout_window = None
    if len(th_eval_data) > 0:
        extra_holdout_window = {
            "train_start": T("2018-01-01"),
            "train_end": T("2025-12-31 23:00:00"),
            "eval_start": T("2026-01-01"),
            "eval_end": th_eval_data["timestamp"].max(),
            "subperiods": [(T("2026-01-01"), th_eval_data["timestamp"].max(), "2026 YTD")],
        }

    _core_run_diagnostic(
        build_model_fn=train_module.build_model,
        load_data_fn=_load_all_data,
        windows=_get_walk_forward_windows(),
        config=_BTC_CONFIG,
        extra_holdout_window=extra_holdout_window,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC/USD hourly data preparation and diagnostics")
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
