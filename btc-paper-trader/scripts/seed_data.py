"""Seed the historical parquet file for the paper trader.

Copies from the autotrader cache (preferred) or backfills from
the Binance REST API.

Usage:
    cd btc-paper-trader
    python scripts/seed_data.py                    # Auto-detect source
    python scripts/seed_data.py --from-cache       # Force cache copy
    python scripts/seed_data.py --from-api          # Force API backfill
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUTOTRADER_CACHE = Path.home() / ".cache" / "autotrader"
OHLCV_CACHE = AUTOTRADER_CACHE / "btcusdt_1h.parquet"
FUNDING_CACHE = AUTOTRADER_CACHE / "binance_funding_btcusdt.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "btcusdt_1h.parquet"


def seed_from_cache() -> pd.DataFrame:
    """Copy and merge OHLCV + funding rate from autotrader cache."""
    if not OHLCV_CACHE.exists():
        print(f"Cache not found: {OHLCV_CACHE}")
        sys.exit(1)

    print(f"Loading OHLCV from {OHLCV_CACHE}")
    df = pd.read_parquet(OHLCV_CACHE)
    print(f"  {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Merge funding rate if available
    if FUNDING_CACHE.exists():
        print(f"Loading funding rate from {FUNDING_CACHE}")
        funding = pd.read_parquet(FUNDING_CACHE)
        print(f"  {len(funding)} hourly funding records")

        df = df.merge(funding[["timestamp", "funding_rate"]], on="timestamp", how="left")
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0001)
        print(f"  Merged: {df['funding_rate'].notna().sum()} rows with funding data")
    else:
        print("  No funding rate cache found, filling with default 0.0001")
        df["funding_rate"] = 0.0001

    return df


def seed_from_api() -> pd.DataFrame:
    """Backfill from Binance REST API."""
    from src.data import backfill_candles

    print("Backfilling from Binance API (this may take several minutes)...")
    df = backfill_candles(start_date="2018-01-01")

    # No funding rate from REST API backfill — use default
    df["funding_rate"] = 0.0001
    print("  No funding rate available from REST API backfill, using default")

    return df


def validate(df: pd.DataFrame) -> bool:
    """Validate the seeded DataFrame."""
    print("\nValidation:")
    ok = True

    # Check required columns
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"  FAIL: Missing columns: {missing}")
        ok = False
    else:
        print("  OK: All required columns present")

    # Check row count
    if len(df) < 1000:
        print(f"  FAIL: Only {len(df)} rows (expected >1000)")
        ok = False
    else:
        print(f"  OK: {len(df)} rows")

    # Check for NaN in OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            print(f"  WARN: {n_nan} NaN in {col}")

    # Check timestamp continuity (allow small gaps)
    ts_diff = df["timestamp"].diff().dropna()
    expected_gap = pd.Timedelta(hours=1)
    gaps = ts_diff[ts_diff > expected_gap * 2]
    if len(gaps) > 0:
        print(f"  WARN: {len(gaps)} timestamp gaps > 2 hours")
        for idx in gaps.index[:5]:
            print(f"    Gap at {df['timestamp'].iloc[idx-1]} → {df['timestamp'].iloc[idx]}")
    else:
        print("  OK: Timestamp continuity")

    # Check data range
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: ${df['close'].min():.0f} to ${df['close'].max():.0f}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Seed historical data for paper trader")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-cache", action="store_true", help="Copy from autotrader cache")
    group.add_argument("--from-api", action="store_true", help="Backfill from Binance API")
    args = parser.parse_args()

    # Auto-detect source
    if args.from_cache:
        df = seed_from_cache()
    elif args.from_api:
        df = seed_from_api()
    elif OHLCV_CACHE.exists():
        print("Auto-detected autotrader cache, using it")
        df = seed_from_cache()
    else:
        print("No autotrader cache found, falling back to API backfill")
        df = seed_from_api()

    # Sort and dedup
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    # Validate
    if not validate(df):
        print("\nValidation failed. Fix issues before proceeding.")
        sys.exit(1)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
