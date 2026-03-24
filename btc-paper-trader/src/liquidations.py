"""Persistent websocket aggregator for Kraken Futures liquidation feed.

Runs as a separate systemd service (not part of the hourly cron).
Connects to Kraken Futures websocket, aggregates liquidation events
into hourly buckets, and writes one row per hour to parquet.

Note: Binance Futures websocket is geo-blocked in the US, so we use
Kraken Futures instead. The data format differs but the aggregated
metrics (counts, volumes, imbalance) are comparable.

Usage:
    python -m src.liquidations --config config.yaml
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Kraken Futures websocket for trade/liquidation events
WS_URL = "wss://futures.kraken.com/ws/v1"
PARQUET_PATH = "data/liquidations_1h.parquet"


class HourlyBucket:
    """Accumulator for liquidation events within one hour."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.long_liq_count = 0
        self.long_liq_volume = 0.0
        self.short_liq_count = 0
        self.short_liq_volume = 0.0
        self.max_single_liq = 0.0
        self.hour_start = pd.Timestamp.now(tz=None).floor("h")

    def add_event(self, side: str, quantity: float, price: float):
        """Add a liquidation event to the current bucket."""
        notional = quantity * price
        self.max_single_liq = max(self.max_single_liq, notional)

        if side == "BUY":
            # BUY = short position being liquidated
            self.short_liq_count += 1
            self.short_liq_volume += notional
        else:
            # SELL = long position being liquidated
            self.long_liq_count += 1
            self.long_liq_volume += notional

    def to_row(self) -> dict:
        """Convert bucket to a row dict for parquet."""
        total_vol = self.long_liq_volume + self.short_liq_volume
        net_imbalance = (
            (self.long_liq_volume - self.short_liq_volume) / total_vol
            if total_vol > 0
            else 0.0
        )

        return {
            "timestamp": self.hour_start,
            "long_liq_count": self.long_liq_count,
            "long_liq_volume": self.long_liq_volume,
            "short_liq_count": self.short_liq_count,
            "short_liq_volume": self.short_liq_volume,
            "max_single_liq": self.max_single_liq,
            "net_liq_imbalance": net_imbalance,
        }


def _append_row(row: dict, path: str) -> None:
    """Append a row to the liquidations parquet file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame([row])

    if os.path.exists(path):
        existing = pd.read_parquet(path)
        if row["timestamp"] in existing["timestamp"].values:
            logger.info(f"Liquidation row {row['timestamp']} already exists")
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    tmp_path = path + ".tmp"
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


async def _run_websocket(parquet_path: str) -> None:
    """Main websocket loop with auto-reconnect."""
    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed. Run: pip install websockets")
        sys.exit(1)

    bucket = HourlyBucket()
    reconnect_delay = 1  # exponential backoff starts at 1s

    while True:
        try:
            logger.info(f"Connecting to {WS_URL}")
            async with websockets.connect(WS_URL, ping_interval=30, ping_timeout=10) as ws:
                # Subscribe to PF_XBTUSD trades (includes liquidations)
                subscribe_msg = json.dumps({
                    "event": "subscribe",
                    "feed": "trade",
                    "product_ids": ["PF_XBTUSD"],
                })
                await ws.send(subscribe_msg)
                logger.info("Connected to Kraken liquidation feed")
                reconnect_delay = 1  # Reset backoff on successful connect

                async for message in ws:
                    try:
                        data = json.loads(message)

                        # Skip subscription confirmations and heartbeats
                        feed = data.get("feed", "")
                        if feed != "trade":
                            continue

                        # Kraken trade format: side, qty, price, type
                        # Liquidation trades have type "liquidation"
                        trade_type = data.get("type", "")
                        if trade_type != "liquidation":
                            continue

                        side = data.get("side", "").upper()  # "buy" or "sell"
                        quantity = float(data.get("qty", 0))
                        price = float(data.get("price", 0))

                        # Kraken: buy liquidation = short squeezed, sell = long liquidated
                        # Map to Binance convention: BUY = short liq, SELL = long liq
                        mapped_side = "BUY" if side == "BUY" else "SELL"

                        # Check if we've crossed an hour boundary
                        current_hour = pd.Timestamp.now(tz=None).floor("h")
                        if current_hour > bucket.hour_start:
                            # Flush the completed bucket
                            row = bucket.to_row()
                            _append_row(row, parquet_path)
                            logger.info(
                                f"Flushed liquidation bucket: {row['long_liq_count']}L "
                                f"${row['long_liq_volume']:.0f} / "
                                f"{row['short_liq_count']}S "
                                f"${row['short_liq_volume']:.0f}"
                            )
                            bucket.reset()

                        bucket.add_event(mapped_side, quantity, price)

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse liquidation event: {e}")

        except Exception as e:
            logger.warning(
                f"Websocket disconnected: {e}. "
                f"Reconnecting in {reconnect_delay}s"
            )

            # Log the gap
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.warning(f"Connection gap started at {now}")

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 300)  # Cap at 5 minutes


def main():
    """Entry point for the liquidation aggregator service."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Liquidation websocket aggregator")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/liquidations.log"),
        ],
    )

    # Load config
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    parquet_path = config.get("supplementary", {}).get(
        "liquidations_path", PARQUET_PATH
    )

    # Handle SIGTERM gracefully (systemd stop)
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Starting liquidation aggregator")
    asyncio.run(_run_websocket(parquet_path))


if __name__ == "__main__":
    main()
