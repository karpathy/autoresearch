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
from core.evaluation import compute_decay_weights
from core.evaluation import evaluate_model as _core_evaluate_model
from core.evaluation import run_diagnostic as _core_run_diagnostic


# ---------------------------------------------------------------------------
# BTC-Specific Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autotrader"
PARQUET_PATH = CACHE_DIR / "btcusdt_1h.parquet"
_FUNDING_PARQUET = CACHE_DIR / "binance_funding_btcusdt.parquet"
_FEAR_GREED_PARQUET = CACHE_DIR / "fear_greed_index.parquet"
_BLOCKCHAIN_PARQUET = CACHE_DIR / "blockchain_metrics.parquet"
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


# ---------------------------------------------------------------------------
# Supplementary Data Sources
# ---------------------------------------------------------------------------

_FUNDING_URL = (
    "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    "/BTCUSDT/BTCUSDT-fundingRate-{year}-{month:02d}.zip"
)


def _download_funding_month(year: int, month: int, max_retries: int = 5) -> pd.DataFrame | None:
    """Download a single month of funding rate data from Binance public data."""
    url = _FUNDING_URL.format(year=year, month=month)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} for funding {year}-{month:02d} "
                      f"(waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED to download funding {year}-{month:02d} after "
                      f"{max_retries} attempts: {e}")
                return None


def _download_funding_rate(max_retries: int = 5) -> pd.DataFrame | None:
    """Download Binance BTCUSDT perpetual funding rate history and cache.

    Uses Binance public data (data.binance.vision) to avoid API geo-blocking.
    Returns hourly DataFrame with columns [timestamp, funding_rate].
    The 8-hour settlement rate is forward-filled to hourly resolution.
    Returns None on failure (caller fills with defaults).
    """
    if _FUNDING_PARQUET.exists():
        age_hours = (time.time() - _FUNDING_PARQUET.stat().st_mtime) / 3600
        if age_hours < 24:
            return pd.read_parquet(_FUNDING_PARQUET)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Binance BTCUSDT funding rate history...")

    # BTCUSDT perpetual launched ~Sep 2019
    now = pd.Timestamp.now()
    frames = []
    for year in range(2019, now.year + 1):
        for month in range(1, 13):
            if year == 2019 and month < 9:
                continue
            if year == now.year and month > now.month:
                break
            print(f"  Downloading funding {year}-{month:02d}...", end=" ", flush=True)
            df = _download_funding_month(year, month, max_retries)
            if df is not None:
                print(f"{len(df)} records")
                frames.append(df)
            else:
                print("skipped")

    if not frames:
        print("  WARNING: No funding rate data downloaded")
        return None

    raw = pd.concat(frames, ignore_index=True)

    # Parse: the CSV has columns like 'calc_time' or 'fundingTime' and 'last_funding_rate' or 'fundingRate'
    # Detect column names (Binance changed format over time)
    if "calc_time" in raw.columns:
        ts_col = "calc_time"
    elif "fundingTime" in raw.columns:
        ts_col = "fundingTime"
    else:
        # Try first column that looks like a timestamp
        ts_col = raw.columns[0]

    if "last_funding_rate" in raw.columns:
        rate_col = "last_funding_rate"
    elif "fundingRate" in raw.columns:
        rate_col = "fundingRate"
    else:
        rate_col = raw.columns[1]

    # Convert timestamps (may be ms or us, same logic as OHLCV)
    timestamps = pd.to_numeric(raw[ts_col], errors="coerce").values.astype(np.float64)
    is_microseconds = timestamps > 1e15
    ts_seconds = np.where(is_microseconds, timestamps / 1e6, timestamps / 1e3)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts_seconds, unit="s", utc=True).tz_localize(None),
        "funding_rate": pd.to_numeric(raw[rate_col], errors="coerce").astype(np.float64),
    })
    df = df.dropna(subset=["timestamp", "funding_rate"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Resample 8h intervals to hourly via forward-fill
    df = df.set_index("timestamp").resample("1h").ffill().reset_index()

    print(f"  Funding rate: {len(df)} hourly records "
          f"({df['timestamp'].min()} to {df['timestamp'].max()})")
    df.to_parquet(_FUNDING_PARQUET, index=False)
    return df


def _download_fear_greed(max_retries: int = 5) -> pd.DataFrame | None:
    """Download Alternative.me Crypto Fear & Greed Index and cache.

    Returns hourly DataFrame with columns [timestamp, fear_greed].
    Daily values are forward-filled to hourly resolution.
    Returns None on failure (caller fills with defaults).
    """
    if _FEAR_GREED_PARQUET.exists():
        age_hours = (time.time() - _FEAR_GREED_PARQUET.stat().st_mtime) / 3600
        if age_hours < 24:
            return pd.read_parquet(_FEAR_GREED_PARQUET)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Fear & Greed Index...")

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()["data"]
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} for Fear & Greed "
                      f"(waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  WARNING: Failed to download Fear & Greed after "
                      f"{max_retries} attempts: {e}")
                return None

    if not data:
        print("  WARNING: No Fear & Greed data received")
        return None

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            [int(r["timestamp"]) for r in data], unit="s", utc=True
        ).tz_localize(None),
        "fear_greed": [int(r["value"]) for r in data],
    })
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Resample daily to hourly via forward-fill
    df = df.set_index("timestamp").resample("1h").ffill().reset_index()

    print(f"  Fear & Greed: {len(df)} hourly records "
          f"({df['timestamp'].min()} to {df['timestamp'].max()})")
    df.to_parquet(_FEAR_GREED_PARQUET, index=False)
    return df


def _download_blockchain_metrics(max_retries: int = 5) -> pd.DataFrame | None:
    """Download Bitcoin network metrics from Blockchain.com and cache.

    Returns hourly DataFrame with columns [timestamp, hash_rate, tx_count, tx_volume_usd].
    Daily values are forward-filled to hourly resolution.
    Returns None on failure (caller fills with defaults).
    """
    if _BLOCKCHAIN_PARQUET.exists():
        age_days = (time.time() - _BLOCKCHAIN_PARQUET.stat().st_mtime) / 86400
        if age_days < 7:
            return pd.read_parquet(_BLOCKCHAIN_PARQUET)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading blockchain network metrics...")

    metrics = {
        "hash-rate": "hash_rate",
        "n-transactions": "tx_count",
        "estimated-transaction-volume-usd": "tx_volume_usd",
    }

    metric_dfs = []
    for api_name, col_name in metrics.items():
        url = f"https://api.blockchain.info/charts/{api_name}?timespan=all&format=json&cors=true"
        success = False
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                values = resp.json()["values"]
                success = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Retry {attempt + 1}/{max_retries} for {api_name} "
                          f"(waiting {wait}s): {e}")
                    time.sleep(wait)
                else:
                    print(f"  WARNING: Failed to download {api_name} after "
                          f"{max_retries} attempts: {e}")

        if not success:
            continue

        mdf = pd.DataFrame({
            "timestamp": pd.to_datetime(
                [v["x"] for v in values], unit="s", utc=True
            ).tz_localize(None),
            col_name: [float(v["y"]) for v in values],
        })
        mdf = mdf.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        metric_dfs.append(mdf)
        print(f"  {api_name}: {len(mdf)} daily records")

    if not metric_dfs:
        print("  WARNING: No blockchain metrics downloaded")
        return None

    # Merge all metrics on timestamp
    merged = metric_dfs[0]
    for mdf in metric_dfs[1:]:
        merged = merged.merge(mdf, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Resample daily to hourly via forward-fill
    merged = merged.set_index("timestamp").resample("1h").ffill().reset_index()

    print(f"  Blockchain metrics: {len(merged)} hourly records "
          f"({merged['timestamp'].min()} to {merged['timestamp'].max()})")
    merged.to_parquet(_BLOCKCHAIN_PARQUET, index=False)
    return merged


def _merge_supplementary(
    ohlcv_df: pd.DataFrame,
    supplementary: list[tuple[pd.DataFrame | None, dict[str, float]]],
) -> pd.DataFrame:
    """Merge supplementary data sources onto the OHLCV hourly grid.

    Args:
        ohlcv_df: Authoritative OHLCV DataFrame with hourly timestamps.
        supplementary: List of (dataframe_or_none, defaults_dict) tuples.
            If dataframe is None, columns are filled with default values.

    Returns:
        Enriched DataFrame with all supplementary columns added.
    """
    result = ohlcv_df.copy()
    supp_cols = []

    for supp_df, defaults in supplementary:
        if supp_df is None:
            for col, default in defaults.items():
                result[col] = default
                supp_cols.append(col)
        else:
            result = result.merge(supp_df, on="timestamp", how="left")
            supp_cols.extend(defaults.keys())

    # Forward-fill gaps, then fill remaining NaN (pre-availability) with defaults
    for supp_df, defaults in supplementary:
        for col, default in defaults.items():
            if col in result.columns:
                result[col] = result[col].ffill().fillna(default)

    # Verify no NaN in supplementary columns
    for col in supp_cols:
        n_null = result[col].isna().sum()
        if n_null > 0:
            print(f"  WARNING: {n_null} NaN values remain in {col}, filling with default")
            default = next(d[col] for _, d in supplementary if col in d)
            result[col] = result[col].fillna(default)

    return result


def _load_all_data() -> pd.DataFrame:
    """Load the full dataset with supplementary features. Internal use only."""
    ohlcv_df = download_data()

    # Download supplementary data sources (non-critical — failures fill with defaults)
    funding_df = _download_funding_rate()
    fear_greed_df = _download_fear_greed()
    blockchain_df = _download_blockchain_metrics()

    supplementary = [
        (funding_df, {"funding_rate": 0.0001}),
        (fear_greed_df, {"fear_greed": 50}),
        (blockchain_df, {"hash_rate": 0.0, "tx_count": 0.0, "tx_volume_usd": 0.0}),
    ]

    return _merge_supplementary(ohlcv_df, supplementary)


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
    """Per-window diagnostic breakdown — BTC wrapper with vol model stats.

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

    # --- Vol model diagnostic pass ---
    # Retrains each window to extract vol model predictions on eval periods.
    # Adds ~50s for human-only diagnostic use.
    windows = _get_walk_forward_windows()
    wf = _BTC_CONFIG.walk_forward

    # Get aligned timestamps from compute_features (same for all calls on all_data)
    _, ts_all, _ = train_module.compute_features(all_data)

    print("=" * 60)
    print("=== VOL MODEL DIAGNOSTIC ===")
    print("=" * 60)

    for i, w in enumerate(windows):
        train_mask = ((all_data["timestamp"] >= w["train_start"]) &
                      (all_data["timestamp"] <= w["train_end"]))
        train_data = all_data[train_mask].reset_index(drop=True)

        sample_weight = None
        if wf.decay_half_life_years > 0:
            sample_weight = compute_decay_weights(
                train_data["timestamp"].values,
                w["eval_start"],
                wf.decay_half_life_years,
            )

        predict_fn = train_module.build_model(train_data, sample_weight=sample_weight)
        predict_fn(all_data)  # populates predict_fn.last_vol_ratio

        vol_ratio = predict_fn.last_vol_ratio

        # Align vol_ratio with eval period timestamps
        eval_mask = ((pd.to_datetime(ts_all) >= w["eval_start"]) &
                     (pd.to_datetime(ts_all) <= w["eval_end"]))
        eval_vr = vol_ratio[eval_mask]

        eval_range = f"{w['eval_start'].date()}\u2014{w['eval_end'].date()}"
        print(f"\n  Window {i} ({eval_range}):")
        if len(eval_vr) > 0:
            print(f"    Vol ratio pred: mean={eval_vr.mean():.3f}, std={eval_vr.std():.3f}, "
                  f"min={eval_vr.min():.3f}, max={eval_vr.max():.3f}")
            print(f"    Low vol  (<0.8): {(eval_vr < 0.8).mean()*100:4.0f}% of hours "
                  f"\u2192 positions amplified up to 2\u00d7")
            print(f"    High vol (>1.5): {(eval_vr > 1.5).mean()*100:4.0f}% of hours "
                  f"\u2192 positions reduced to \u226467%")
            print(f"    Extreme  (>2.0): {(eval_vr > 2.0).mean()*100:4.0f}% of hours "
                  f"\u2192 positions reduced to \u226450%")
        else:
            print("    (no eval data)")

    print()


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

        # Refresh supplementary data caches
        print("\nDownloading supplementary data sources...")
        _download_funding_rate()
        _download_fear_greed()
        _download_blockchain_metrics()

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
