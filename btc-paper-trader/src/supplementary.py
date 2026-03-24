"""Supplementary data collection: order book snapshots and open interest.

Order book: Binance US spot (same data as training, US-accessible).
Open interest: Kraken Futures (Binance Futures blocked in US).

These data sources are NOT used by the current model but are accumulated
for future model iterations. Stored in separate parquet files.
"""

import json
import logging
import os
import zlib

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_orderbook_snapshot(
    symbol: str = "BTCUSDT",
    base_url: str = "https://api.binance.us",
    depth_limit: int = 1000,
) -> dict | None:
    """Fetch order book depth snapshot from Binance US.

    Returns dict with summary metrics and compressed raw levels,
    or None on failure.
    """
    url = f"{base_url}/api/v3/depth"
    params = {"symbol": symbol, "limit": depth_limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Order book fetch failed: {e}")
        return None

    bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

    if not bids or not asks:
        logger.warning("Empty order book")
        return None

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid_price * 10000

    # Compute volume within percentage bands from mid
    def volume_within_pct(levels, mid, pct, side):
        total = 0.0
        for price, qty in levels:
            if side == "bid" and price >= mid * (1 - pct / 100):
                total += qty
            elif side == "ask" and price <= mid * (1 + pct / 100):
                total += qty
        return total

    bid_vol_05 = volume_within_pct(bids, mid_price, 0.5, "bid")
    bid_vol_1 = volume_within_pct(bids, mid_price, 1.0, "bid")
    bid_vol_2 = volume_within_pct(bids, mid_price, 2.0, "bid")
    ask_vol_05 = volume_within_pct(asks, mid_price, 0.5, "ask")
    ask_vol_1 = volume_within_pct(asks, mid_price, 1.0, "ask")
    ask_vol_2 = volume_within_pct(asks, mid_price, 2.0, "ask")

    def imbalance(bid_v, ask_v):
        total = bid_v + ask_v
        return (bid_v - ask_v) / total if total > 0 else 0.0

    # Compress raw top-100 levels as JSON blob
    raw_top100 = {
        "bids": bids[:100],
        "asks": asks[:100],
    }
    raw_json = json.dumps(raw_top100, separators=(",", ":"))
    raw_compressed = zlib.compress(raw_json.encode(), level=6)

    return {
        "timestamp": pd.Timestamp.now(tz=None).floor("h"),
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "bid_volume_0_5pct": bid_vol_05,
        "bid_volume_1pct": bid_vol_1,
        "bid_volume_2pct": bid_vol_2,
        "ask_volume_0_5pct": ask_vol_05,
        "ask_volume_1pct": ask_vol_1,
        "ask_volume_2pct": ask_vol_2,
        "imbalance_0_5pct": imbalance(bid_vol_05, ask_vol_05),
        "imbalance_1pct": imbalance(bid_vol_1, ask_vol_1),
        "raw_levels": raw_compressed,
    }


def fetch_open_interest(
    kraken_futures_url: str = "https://futures.kraken.com",
    kraken_symbol: str = "PF_XBTUSD",
    btc_price: float | None = None,
) -> dict | None:
    """Fetch current open interest from Kraken Futures.

    Returns dict with OI in BTC and USD, or None on failure.
    """
    url = f"{kraken_futures_url}/derivatives/api/v3/tickers/{kraken_symbol}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ticker = data.get("ticker", {})
    except Exception as e:
        logger.warning(f"Open interest fetch failed: {e}")
        return None

    oi = float(ticker.get("openInterest", 0))
    # Use Kraken mark price if btc_price not provided
    price = btc_price or float(ticker.get("markPrice", 0))
    oi_usd = oi * price if price else 0.0

    return {
        "timestamp": pd.Timestamp.now(tz=None).floor("h"),
        "open_interest": oi,
        "open_interest_usd": oi_usd,
    }


def append_supplementary_row(path: str, row: dict) -> None:
    """Append a row to a supplementary parquet file.

    If the file doesn't exist, creates it. Uses atomic write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    new_row_df = pd.DataFrame([row])

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        # Dedup by timestamp
        if row["timestamp"] in existing["timestamp"].values:
            logger.info(f"Supplementary row {row['timestamp']} already exists in {path}")
            return
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    tmp_path = path + ".tmp"
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
