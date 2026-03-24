"""Data fetching and parquet management.

OHLCV candles from Binance US (geo-friendly).
Funding rate from Kraken Futures (Binance Futures blocked in US).
Manages the historical parquet file with atomic writes and deduplication.
"""

import logging
import os
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Kraken Futures funding rate is annualized absolute.
# Convert to Binance-equivalent per-8h rate: rate / (365.25 * 3)
_KRAKEN_ANNUAL_TO_8H = 1.0 / (365.25 * 3)


def validate_candle(candle: dict) -> list[str]:
    """Validate candle data. Returns list of issues (empty if OK)."""
    issues = []
    for field in ("open", "high", "low", "close"):
        v = candle.get(field)
        if v is None or not np.isfinite(v) or v <= 0:
            issues.append(f"{field}={v} (must be positive and finite)")
    if candle.get("high", 0) < candle.get("low", 0):
        issues.append(f"high={candle['high']} < low={candle['low']}")
    v = candle.get("volume")
    if v is not None and (not np.isfinite(v) or v < 0):
        issues.append(f"volume={v} (must be non-negative)")
    return issues


def backfill_recent_gap(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    base_url: str = "https://api.binance.us",
) -> pd.DataFrame:
    """Fill gap between last parquet timestamp and now.

    Fetches missing candles from Binance US (up to 1000 per request).
    Returns updated DataFrame with gap filled.
    """
    latest_ts = df["timestamp"].max()
    start_ms = int((latest_ts + pd.Timedelta(hours=1)).timestamp() * 1000)
    end_ms = int(pd.Timestamp.now("UTC").tz_localize(None).timestamp() * 1000)

    url = f"{base_url}/api/v3/klines"
    new_rows = []
    current_start = start_ms

    while current_start < end_ms:
        try:
            resp = requests.get(url, params={
                "symbol": symbol, "interval": "1h",
                "startTime": current_start, "limit": 1000,
            }, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            logger.warning(f"Backfill fetch failed: {e}")
            break

        if not klines:
            break

        for k in klines:
            if len(k) < 6:
                continue
            new_rows.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").tz_localize(None),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        current_start = klines[-1][0] + 1
        time.sleep(0.3)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Carry forward funding_rate if column exists
        if "funding_rate" in df.columns:
            new_df["funding_rate"] = df["funding_rate"].iloc[-1]
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        if "funding_rate" in df.columns:
            df["funding_rate"] = df["funding_rate"].ffill()
        logger.info(f"Backfilled {len(new_rows)} candles")

    return df


def fetch_latest_candle(
    symbol: str = "BTCUSDT",
    base_url: str = "https://api.binance.us",
    retry_attempts: int = 3,
    retry_delay: int = 60,
) -> dict | None:
    """Fetch the most recently completed 1h candle from Binance US.

    Returns dict with keys: timestamp, open, high, low, close, volume.
    Returns None if all retries fail.
    """
    url = f"{base_url}/api/v3/klines"
    params = {"symbol": symbol, "interval": "1h", "limit": 2}

    for attempt in range(retry_attempts):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()

            if len(klines) < 2:
                logger.warning("Received fewer than 2 candles")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                continue

            # The second-to-last candle is the most recently completed one
            candle = klines[-2]

            # Validate response structure
            if not isinstance(candle, list) or len(candle) < 6:
                logger.warning(f"Malformed candle response: {candle!r:.200}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                continue

            # Binance kline format: [open_time, open, high, low, close, volume, ...]
            ts_ms = candle[0]
            ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None)

            # Validate: timestamp should be within the last 2 hours
            now = pd.Timestamp.now("UTC").tz_localize(None)
            age_hours = (now - ts).total_seconds() / 3600
            if age_hours > 2:
                logger.warning(f"Candle timestamp {ts} is {age_hours:.1f}h old")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue

            return {
                "timestamp": ts,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            }

        except Exception as e:
            logger.warning(f"Candle fetch attempt {attempt + 1}/{retry_attempts} failed: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)

    logger.error("All candle fetch attempts failed")
    return None


def fetch_latest_funding(
    kraken_futures_url: str = "https://futures.kraken.com",
    kraken_symbol: str = "PF_XBTUSD",
) -> tuple[float, pd.Timestamp] | None:
    """Fetch the most recent funding rate from Kraken Futures.

    Kraken's fundingRate is annualized absolute — we convert to
    Binance-equivalent per-8h rate for model compatibility.

    Returns (rate, timestamp) tuple, or None on failure.
    """
    url = f"{kraken_futures_url}/derivatives/api/v3/tickers/{kraken_symbol}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ticker = data.get("ticker", {})

        if not ticker:
            logger.warning("Empty Kraken ticker response")
            return None

        # Convert annualized rate to per-8h (Binance convention)
        if "fundingRate" not in ticker:
            logger.warning("Kraken ticker missing fundingRate field")
            return None
        try:
            annual_rate = float(ticker["fundingRate"])
        except (ValueError, TypeError):
            logger.warning(f"Kraken fundingRate not parseable: {ticker.get('fundingRate')!r}")
            return None
        if not np.isfinite(annual_rate):
            logger.warning(f"Kraken fundingRate not finite: {annual_rate}")
            return None
        per_8h_rate = annual_rate * _KRAKEN_ANNUAL_TO_8H

        # Parse timestamp from lastTime
        last_time = ticker.get("lastTime", "")
        if last_time:
            ts = pd.Timestamp(last_time).tz_localize(None)
        else:
            ts = pd.Timestamp.now(tz=None).floor("h")

        logger.info(
            f"Kraken funding: annual={annual_rate:.4f}, "
            f"per_8h={per_8h_rate:.6f}"
        )
        return (per_8h_rate, ts)

    except Exception as e:
        logger.warning(f"Kraken funding rate fetch failed: {e}")
        return None


def load_parquet(path: str) -> pd.DataFrame:
    """Load parquet file. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    return df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet with atomic write (temp + rename)."""
    tmp_path = path + ".tmp"
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to save parquet {path}: {e}")
        # Clean up orphaned tmp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def append_candle(
    df: pd.DataFrame,
    candle: dict,
    funding_rate: float | None = None,
) -> pd.DataFrame:
    """Append a candle to the DataFrame, with deduplication and funding forward-fill.

    Returns updated DataFrame (original is not modified).
    """
    ts = candle["timestamp"]

    # Deduplication check
    if ts in df["timestamp"].values:
        logger.info(f"Candle {ts} already exists, skipping")
        return df

    new_row = {
        "timestamp": ts,
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
    }

    # Handle funding rate
    if "funding_rate" in df.columns:
        if funding_rate is not None:
            new_row["funding_rate"] = funding_rate
        else:
            # Forward-fill from most recent known value
            new_row["funding_rate"] = df["funding_rate"].iloc[-1]

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df = new_df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill any NaN in funding_rate
    if "funding_rate" in new_df.columns:
        new_df["funding_rate"] = new_df["funding_rate"].ffill()

    return new_df


def backfill_candles(
    start_date: str = "2018-01-01",
    symbol: str = "BTCUSDT",
    base_url: str = "https://api.binance.us",
) -> pd.DataFrame:
    """Backfill historical candles from Binance US REST API.

    Paginates forward using limit=1000 requests.
    Used by seed_data.py for initial setup only.
    """
    url = f"{base_url}/api/v3/klines"
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    all_candles = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": current_start,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            logger.error(f"Backfill failed at {current_start}: {e}")
            time.sleep(5)
            continue

        if not klines:
            break

        for k in klines:
            all_candles.append({
                "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").tz_localize(None),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        # Move start to after the last candle
        current_start = klines[-1][0] + 1
        print(f"  Fetched {len(all_candles)} candles through {all_candles[-1]['timestamp']}")
        time.sleep(0.5)  # Rate limiting

    df = pd.DataFrame(all_candles)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df
