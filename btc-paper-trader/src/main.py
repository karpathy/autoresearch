"""Hourly cron entry point — orchestrates the full paper trading pipeline.

Usage:
    python -m src.main                  # Hourly inference run
    python -m src.main --report         # Generate and deliver daily report

Exit codes:
    0 = success
    1 = data fetch failed (non-critical)
    2 = inference failed (critical)
"""

import argparse
import fcntl
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from .alerts import run_health_checks, write_alerts
from .data import (
    append_candle,
    backfill_recent_gap,
    fetch_latest_candle,
    fetch_latest_funding,
    load_parquet,
    save_parquet,
    validate_candle,
)
from .inference import InferenceResult, compute_position, load_artifacts, run_inference, validate_artifacts
from .logging_config import (
    append_daily_summary,
    append_prediction_row,
    append_trade_row,
    setup_system_log,
)
from .portfolio import (
    PortfolioState,
    compute_daily_summary,
    load_portfolio_state,
    save_portfolio_state,
    update_portfolio,
)
from .report import deliver_report, generate_report
from .supplementary import append_supplementary_row, fetch_open_interest, fetch_orderbook_snapshot

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _acquire_lock(lock_path: str):
    """Acquire an exclusive file lock. Returns file handle or None if locked."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        fh.close()
        return None


def run_hourly(config: dict) -> int:
    """Execute the hourly inference pipeline.

    Returns exit code: 0=success, 1=data fetch failed, 2=inference failed.
    """
    # Prevent concurrent runs
    lock_path = os.path.join(os.path.dirname(config["data"]["parquet_path"]), ".lockfile")
    lock_fh = _acquire_lock(lock_path)
    if lock_fh is None:
        logger.warning("Another run is in progress (lock held), exiting")
        return 0

    try:
        return _run_hourly_inner(config)
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def _run_hourly_inner(config: dict) -> int:
    """Inner hourly pipeline (called with lock held)."""
    start_time = time.time()
    data_cfg = config["data"]
    model_cfg = config["model"]
    trading_cfg = config["trading"]
    log_cfg = config["logging"]

    def elapsed():
        return f"[{time.time() - start_time:.1f}s]"

    logger.info(f"{elapsed()} === Hourly run starting ===")

    # --- Load model artifacts ---
    artifact_path = model_cfg["artifact_path"]
    try:
        artifacts = load_artifacts(artifact_path)
        if not validate_artifacts(artifacts):
            logger.error(f"{elapsed()} Artifact validation failed")
            return 2
        logger.info(f"{elapsed()} Artifacts loaded: commit={artifacts['commit']}")
    except Exception as e:
        logger.error(f"{elapsed()} Failed to load artifacts: {e}")
        return 2

    # --- Load historical data ---
    parquet_path = data_cfg["parquet_path"]
    try:
        df = load_parquet(parquet_path)
        logger.info(f"{elapsed()} Parquet loaded: {len(df)} rows")
    except Exception as e:
        logger.error(f"{elapsed()} Failed to load parquet: {e}")
        return 2

    # --- Backfill gap if data is stale ---
    latest_ts = df["timestamp"].max()
    import pandas as pd
    gap_hours = (pd.Timestamp.now("UTC").tz_localize(None) - latest_ts).total_seconds() / 3600
    if gap_hours > 2:
        logger.info(f"{elapsed()} Data gap: {gap_hours:.0f}h since {latest_ts}. Backfilling...")
        df = backfill_recent_gap(
            df, symbol=data_cfg["symbol"], base_url=data_cfg["binance_base_url"],
        )
        save_parquet(df, parquet_path)
        logger.info(f"{elapsed()} Backfill complete: {len(df)} rows")

    # --- Fetch latest candle ---
    candle = fetch_latest_candle(
        symbol=data_cfg["symbol"],
        base_url=data_cfg["binance_base_url"],
        retry_attempts=data_cfg.get("fetch_retry_attempts", 3),
        retry_delay=data_cfg.get("fetch_retry_delay_seconds", 60),
    )
    if candle is None:
        logger.warning(f"{elapsed()} Failed to fetch latest candle")
        return 1

    # Validate candle data
    issues = validate_candle(candle)
    if issues:
        logger.warning(f"{elapsed()} Bad candle rejected: {'; '.join(issues)}")
        return 1
    logger.info(f"{elapsed()} Candle: {candle['timestamp']} close=${candle['close']:.2f}")

    # --- Fetch funding rate (from Kraken Futures) ---
    funding_result = fetch_latest_funding(
        kraken_futures_url=data_cfg.get("kraken_futures_url", "https://futures.kraken.com"),
        kraken_symbol=data_cfg.get("kraken_symbol", "PF_XBTUSD"),
    )
    funding_rate = funding_result[0] if funding_result else None
    logger.info(f"{elapsed()} Funding rate: {funding_rate:.6f}" if funding_rate else f"{elapsed()} Funding rate: forward-fill")

    # --- Append to DataFrame and save ---
    prev_len = len(df)
    df = append_candle(df, candle, funding_rate)
    save_parquet(df, parquet_path)
    new_rows = len(df) - prev_len
    logger.info(f"{elapsed()} Parquet saved: {len(df)} rows ({'+' + str(new_rows) if new_rows else 'dedup'})")

    # --- Fetch supplementary data (non-critical) ---
    _fetch_supplementary(config, candle["close"])

    # --- Run inference ---
    logger.info(f"{elapsed()} Running inference on {len(df)} rows...")
    try:
        result = run_inference(df, artifacts)
        logger.info(
            f"{elapsed()} Inference done: pred_final={result.pred_final:.4f}, "
            f"position={result.position:.2f}"
        )
    except Exception as e:
        logger.error(f"{elapsed()} Inference failed: {e}", exc_info=True)
        return 2

    # --- Update portfolio ---
    state_path = os.path.join(os.path.dirname(parquet_path), "portfolio_state.json")
    state = load_portfolio_state(state_path)

    new_position = compute_position(
        result.pred_final,
        sigma_threshold=trading_cfg["sigma_threshold"],
        sigma_full=trading_cfg["sigma_full_position"],
    )

    new_state, metrics = update_portfolio(
        state,
        new_position,
        candle["close"],
        fee_rate=trading_cfg["fee_rate"],
        slippage_rate=trading_cfg["slippage_rate"],
    )
    save_portfolio_state(new_state, state_path)

    logger.info(
        f"Portfolio: value={metrics['portfolio_value']:.4f}, "
        f"drawdown={metrics['drawdown']:.2%}"
    )

    # --- Log prediction row ---
    pred_row = {
        "timestamp": str(candle["timestamp"]),
        "pred_24_raw": result.pred_24_raw,
        "pred_72_raw": result.pred_72_raw,
        "pred_72_smoothed": result.pred_72_smoothed,
        "sign_agree": result.sign_agree,
        "pred_after_72h": result.pred_after_72h,
        "conf_prob": result.conf_prob,
        "conf_smoothed": result.conf_smoothed,
        "conf_norm": result.conf_norm,
        "conf_adj": result.conf_adj,
        "pred_after_conf": result.pred_after_conf,
        "pos_scaler_signal": result.pos_scaler_signal,
        "pos_scale": result.pos_scale,
        "pred_after_pos": result.pred_after_pos,
        "pred_after_scale": result.pred_after_scale,
        "pred_final": result.pred_final,
        "position": metrics["position"],
        "position_prev": metrics["position_prev"],
        "position_delta": metrics["position_delta"],
        "fee_cost": metrics["fee_cost"],
        "btc_price": candle["close"],
        "btc_return_1h": metrics["btc_return_1h"],
    }
    append_prediction_row(log_cfg["prediction_log"], pred_row)

    # --- Log trade (if position changed) ---
    if metrics["position_changed"]:
        direction = "flat"
        if new_position > 1e-6:
            direction = "long"
        elif new_position < -1e-6:
            direction = "short"

        trade_row = {
            "timestamp": str(candle["timestamp"]),
            "direction": direction,
            "size": abs(new_position),
            "entry_price": candle["close"],
            "pred_sigma": result.pred_final,
            "conf_adj": result.conf_adj,
            "pos_scale": result.pos_scale,
        }
        append_trade_row(log_cfg["trade_log"], trade_row)
        logger.info(f"Trade logged: {direction} {abs(new_position):.2f}")

    # --- Daily summary (first run of new day) ---
    today = str(candle["timestamp"].date()) if hasattr(candle["timestamp"], "date") else str(candle["timestamp"])[:10]
    _maybe_log_daily_summary(log_cfg, today, new_state)

    # --- Health checks ---
    alerts = run_health_checks(
        config=config,
        df=df,
        pred_final=result.pred_final,
        portfolio_value=new_state.portfolio_value,
        peak_value=new_state.peak_value,
        artifact_trained_at=artifacts["trained_at"],
    )
    if alerts:
        write_alerts(alerts, config["alerts"]["alert_file"])
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")

    logger.info(f"{elapsed()} === Hourly run complete ===")
    return 0


def _fetch_supplementary(config: dict, btc_price: float) -> None:
    """Fetch order book snapshot and open interest (non-critical)."""
    data_cfg = config["data"]
    supp_cfg = config.get("supplementary", {})

    try:
        ob = fetch_orderbook_snapshot(
            symbol=data_cfg["symbol"],
            base_url=data_cfg["binance_base_url"],
        )
        if ob:
            ob_path = supp_cfg.get("orderbook_path", "data/orderbook_1h.parquet")
            append_supplementary_row(ob_path, ob)
            logger.info(f"Order book snapshot: spread={ob['spread_bps']:.1f}bps")
    except Exception as e:
        logger.warning(f"Order book fetch failed (non-critical): {e}")

    try:
        oi = fetch_open_interest(
            kraken_futures_url=data_cfg.get("kraken_futures_url", "https://futures.kraken.com"),
            kraken_symbol=data_cfg.get("kraken_symbol", "PF_XBTUSD"),
            btc_price=btc_price,
        )
        if oi:
            oi_path = supp_cfg.get("open_interest_path", "data/open_interest_1h.parquet")
            append_supplementary_row(oi_path, oi)
            logger.info(f"Open interest: {oi['open_interest']:.2f} BTC")
    except Exception as e:
        logger.warning(f"Open interest fetch failed (non-critical): {e}")


def _maybe_log_daily_summary(log_cfg: dict, today: str, state: PortfolioState) -> None:
    """Log daily summary if it hasn't been logged for today yet."""
    summary_path = log_cfg["daily_summary_log"]

    # Check if we already have a summary for today
    if os.path.exists(summary_path):
        import csv

        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("date") == today:
                    return  # Already logged

    summary = compute_daily_summary(log_cfg["prediction_log"], today, state)
    if summary:
        append_daily_summary(summary_path, summary)
        logger.info(f"Daily summary logged for {today}")


def run_daily_report(config: dict) -> int:
    """Generate and deliver the daily report."""
    log_cfg = config["logging"]
    model_cfg = config["model"]

    # Load artifacts for metadata
    try:
        artifacts = load_artifacts(model_cfg["artifact_path"])
    except Exception as e:
        logger.error(f"Failed to load artifacts for report: {e}")
        artifacts = {"commit": "unknown", "trained_at": "unknown"}

    # Load portfolio state
    state_path = os.path.join(
        os.path.dirname(config["data"]["parquet_path"]),
        "portfolio_state.json",
    )
    state = load_portfolio_state(state_path)

    # Check for active alerts
    alert_file = config["alerts"]["alert_file"]
    alerts = []
    if os.path.exists(alert_file):
        with open(alert_file) as f:
            # Get last 5 alerts
            all_alerts = f.readlines()
            alerts = [a.strip() for a in all_alerts[-5:] if a.strip()]

    report = generate_report(
        config=config,
        prediction_log_path=log_cfg["prediction_log"],
        portfolio_state=state,
        artifact_metadata=artifacts,
        alerts=alerts,
    )

    deliver_report(report, config)
    return 0


def main():
    parser = argparse.ArgumentParser(description="BTC Paper Trader")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--report", action="store_true", help="Generate daily report instead of hourly run")
    args = parser.parse_args()

    # Change to script directory so relative paths in config work
    script_dir = Path(__file__).resolve().parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    # Setup logging
    log_cfg = config["logging"]
    setup_system_log(
        log_cfg["system_log"],
        max_bytes=log_cfg.get("system_log_max_bytes", 10 * 1024 * 1024),
        backup_count=log_cfg.get("system_log_backup_count", 3),
    )

    if args.report:
        sys.exit(run_daily_report(config))
    else:
        sys.exit(run_hourly(config))


if __name__ == "__main__":
    main()
