"""Replay historical data through the paper trading pipeline.

Infrastructure stress test: exercises every code path across thousands
of hourly cycles. Results must NOT be used for decision gates.

Usage:
    cd btc-paper-trader
    python scripts/replay.py --start 2026-01-01 --end 2026-03-24
    python scripts/replay.py --start 2026-01-01  # defaults to yesterday
"""

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_parquet
from src.inference import (
    FullInferenceResult,
    compute_position,
    load_artifacts,
    run_inference_full,
    validate_artifacts,
)
from src.logging_config import (
    DAILY_SUMMARY_FIELDS,
    PREDICTION_FIELDS,
    TRADE_FIELDS,
    append_daily_summary,
    append_prediction_row,
    append_trade_row,
)
from src.portfolio import (
    PortfolioState,
    compute_bip_fees,
    compute_rolling_sharpe,
    update_portfolio,
)


def replay(
    start: str,
    end: str,
    config_path: str = "config.yaml",
    output_dir: str = "logs/replay",
):
    """Run historical replay."""
    import yaml

    start_time = time.time()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    trading_cfg = config["trading"]
    bip_cfg = config.get("bip_tracking", {})

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    pred_log = os.path.join(output_dir, "predictions.csv")
    trade_log = os.path.join(output_dir, "trades.csv")
    summary_log = os.path.join(output_dir, "daily_summary.csv")
    report_path = os.path.join(output_dir, "summary_report.txt")

    # Clean previous replay output
    for f in [pred_log, trade_log, summary_log, report_path]:
        if os.path.exists(f):
            os.unlink(f)

    # Load data and artifacts
    print("Loading data and artifacts...")
    df = load_parquet(config["data"]["parquet_path"])
    artifacts = load_artifacts(config["model"]["artifact_path"])
    if not validate_artifacts(artifacts):
        print("ERROR: Artifact validation failed")
        sys.exit(1)

    print(f"  Data: {len(df)} rows ({df['timestamp'].min()} to {df['timestamp'].max()})")
    print(f"  Model: commit={artifacts['commit']}")

    # Parse date range
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Validate range is within data
    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()
    if start_ts < data_start:
        print(f"ERROR: --start {start_ts} is before data start {data_start}")
        sys.exit(1)
    if end_ts > data_end:
        print(f"WARNING: --end {end_ts} is after data end {data_end}, clamping")
        end_ts = data_end

    # Run inference once on the full dataset
    print("Running inference on full dataset (one-time)...")
    inference_start = time.time()
    result = run_inference_full(df, artifacts)
    print(f"  Inference complete in {time.time() - inference_start:.1f}s")
    print(f"  {len(result.pred_final)} prediction rows")

    # Build a timestamp index for fast lookup
    ts_index = pd.DatetimeIndex(result.timestamps)
    close_prices = df.set_index("timestamp").reindex(ts_index)["close"].values
    funding_rates = np.zeros(len(ts_index))
    if "funding_rate" in df.columns:
        funding_rates = df.set_index("timestamp").reindex(ts_index)["funding_rate"].fillna(0.0).values

    # Find replay range indices
    replay_mask = (ts_index >= start_ts) & (ts_index <= end_ts)
    replay_indices = np.where(replay_mask)[0]

    if len(replay_indices) == 0:
        print(f"ERROR: No data in range {start_ts} to {end_ts}")
        sys.exit(1)

    n_hours = len(replay_indices)
    print(f"\nReplaying {n_hours} hours: {ts_index[replay_indices[0]]} to {ts_index[replay_indices[-1]]}")

    # Initialize portfolio state (fresh for replay)
    state = PortfolioState()
    prev_day = None

    # Tracking stats
    n_trades = 0
    n_position_changes = 0
    nan_predictions = 0
    inf_values = 0
    total_fee_cost = 0.0
    total_funding_cost = 0.0
    total_bip_cost = 0.0
    max_consecutive_flat = 0
    current_flat_streak = 0
    hours_positioned = 0

    # Step through hour by hour
    for step, idx in enumerate(replay_indices):
        ts = ts_index[idx]
        price = float(close_prices[idx])
        fr = float(funding_rates[idx])

        # Get predictions for this hour
        pf = float(result.pred_final[idx])

        # Check for NaN/Inf
        if np.isnan(pf):
            nan_predictions += 1
            pf = 0.0
        if np.isinf(pf):
            inf_values += 1
            pf = 0.0

        # Compute position
        position = compute_position(
            pf,
            sigma_threshold=trading_cfg["sigma_threshold"],
            sigma_full=trading_cfg["sigma_full_position"],
        )

        # Update portfolio
        new_state, metrics = update_portfolio(
            state, position, price,
            fee_rate=trading_cfg["fee_rate"],
            slippage_rate=trading_cfg["slippage_rate"],
            funding_rate=fr,
        )

        # BIP fees
        bip = compute_bip_fees(
            position_delta=metrics["position_delta"],
            btc_price=price,
            contract_size=bip_cfg.get("contract_size", 0.01),
            fee_per_contract=bip_cfg.get("fee_per_contract", 0.46),
            slippage_bps=bip_cfg.get("slippage_bps", 5.0),
        )

        # Track stats
        total_fee_cost += metrics["fee_cost"]
        total_funding_cost += metrics["funding_cost"]
        total_bip_cost += bip["total_bip_cost"]

        if metrics["position_changed"]:
            n_position_changes += 1
        if abs(position) > 1e-6:
            hours_positioned += 1
            current_flat_streak = 0
        else:
            current_flat_streak += 1
            max_consecutive_flat = max(max_consecutive_flat, current_flat_streak)

        # Compute BTC return for logging
        btc_return = metrics["btc_return_1h"]

        # Log prediction row
        pred_row = {
            "timestamp": str(ts),
            "pred_24_raw": result.pred_24_raw[idx],
            "pred_72_raw": result.pred_72_raw[idx],
            "pred_72_smoothed": result.pred_72_smoothed[idx],
            "sign_agree": result.sign_agree[idx],
            "pred_after_72h": result.pred_after_72h[idx],
            "conf_prob": result.conf_prob[idx],
            "conf_smoothed": result.conf_smoothed[idx],
            "conf_norm": result.conf_norm[idx],
            "conf_adj": result.conf_adj[idx],
            "pred_after_conf": result.pred_after_72h[idx] * result.conf_adj[idx],
            "pos_scaler_signal": result.pos_scaler_signal[idx],
            "pos_scale": result.pos_scale[idx],
            "pred_after_pos": result.pred_after_72h[idx] * result.conf_adj[idx] * result.pos_scale[idx],
            "pred_after_scale": result.pred_after_scale[idx],
            "pred_final": pf,
            "position": metrics["position"],
            "position_prev": metrics["position_prev"],
            "position_delta": metrics["position_delta"],
            "fee_cost": metrics["fee_cost"],
            "funding_rate": fr,
            "funding_cost": metrics["funding_cost"],
            "btc_price": price,
            "btc_return_1h": btc_return,
            "bip_n_contracts": bip["n_contracts"],
            "bip_fee_cost": bip["total_bip_cost"],
        }
        append_prediction_row(pred_log, pred_row)

        # Log trade
        if metrics["position_changed"]:
            direction = "flat"
            if position > 1e-6:
                direction = "long"
            elif position < -1e-6:
                direction = "short"
            n_trades += 1

            trade_row = {
                "timestamp": str(ts),
                "direction": direction,
                "size": abs(position),
                "entry_price": price,
                "pred_sigma": pf,
                "conf_adj": result.conf_adj[idx],
                "pos_scale": result.pos_scale[idx],
            }
            append_trade_row(trade_log, trade_row)

        # Daily summary at day boundary
        current_day = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
        if prev_day is not None and current_day != prev_day:
            _log_replay_daily_summary(summary_log, pred_log, prev_day, new_state)
        prev_day = current_day

        state = new_state

        # Progress
        if (step + 1) % 500 == 0 or step == n_hours - 1:
            print(f"  [{step + 1}/{n_hours}] {ts} portfolio={state.portfolio_value:.4f} "
                  f"pos={position:+.2f} trades={n_trades}")

    # Final daily summary
    if prev_day:
        _log_replay_daily_summary(summary_log, pred_log, prev_day, state)

    elapsed = time.time() - start_time

    # Count output rows
    pred_rows = _count_csv_rows(pred_log)
    trade_rows = _count_csv_rows(trade_log)
    summary_rows = _count_csv_rows(summary_log)
    n_days = (end_ts - start_ts).days + 1

    # Generate summary report
    report = f"""Replay Summary — {start} to {end}
==========================================

Infrastructure checks:
  Total hours replayed:     {n_hours:,}
  Prediction log rows:      {pred_rows:,}  (expect == hours)
  Trade log rows:           {trade_rows}
  Daily summaries:          {summary_rows}     (expect ~{n_days} days)
  NaN predictions:          {nan_predictions}      (expect 0)
  Infinite values:          {inf_values}      (expect 0)
  Replay time:              {elapsed:.1f}s

Portfolio checks:
  Final portfolio value:    {state.portfolio_value:.4f}
  Max drawdown:             {(state.portfolio_value - state.peak_value) / state.peak_value * 100 if state.peak_value > 0 else 0:.2f}%
  Total trades:             {n_trades}
  Position changes:         {n_position_changes}
  Hours positioned:         {hours_positioned} / {n_hours} ({hours_positioned / n_hours * 100:.1f}%)
  Max position size:        {_max_abs_position(pred_log):.2f}
  Max consecutive flat:     {max_consecutive_flat} hours

Cost checks:
  Total fee costs:          {total_fee_cost * 100:.3f}% of portfolio
  Total funding costs:      {total_funding_cost * 100:.3f}% of portfolio
  Cumulative funding:       {state.cumulative_funding_cost * 100:.3f}%
  Total BIP fees:           ${total_bip_cost:.2f}

Consistency checks:
  Prediction range:         [{result.pred_final[replay_indices].min():.4f}, {result.pred_final[replay_indices].max():.4f}] sigma
  |pred| > 0.20 frequency:  {(np.abs(result.pred_final[replay_indices]) > 0.20).mean() * 100:.1f}% of hours
  Conf scaler range:        [{result.conf_adj[replay_indices].min():.3f}, {result.conf_adj[replay_indices].max():.3f}]
  Position scaler range:    [{result.pos_scale[replay_indices].min():.3f}, {result.pos_scale[replay_indices].max():.3f}]
  Funding rate range:       [{funding_rates[replay_indices].min():.6f}, {funding_rates[replay_indices].max():.6f}]

Daily summary checks:
  Days with positive return: (see {summary_log})
  Running Sharpe (final):    {compute_rolling_sharpe(pred_log, days=9999):.2f}
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'=' * 50}")
    print(report)
    print(f"Replay output: {output_dir}/")


def _log_replay_daily_summary(summary_log, pred_log, date, state):
    """Compute and append daily summary for replay."""
    if not os.path.exists(pred_log):
        return
    df = pd.read_csv(pred_log)
    if len(df) == 0:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    day_data = df[df["date"] == date]
    if len(day_data) == 0:
        return

    funding = day_data["funding_cost"] if "funding_cost" in day_data.columns else 0.0
    hourly_rets = day_data["position_prev"] * day_data["btc_return_1h"] - day_data["fee_cost"] - funding
    daily_return = float((1 + hourly_rets).prod() - 1)
    positions = day_data["position"].values

    row = {
        "date": date,
        "portfolio_value": state.portfolio_value,
        "daily_return": daily_return,
        "drawdown": (state.portfolio_value - state.peak_value) / state.peak_value if state.peak_value > 0 else 0.0,
        "n_trades_today": int((day_data["position_delta"].abs() > 1e-6).sum()),
        "avg_position_size": float(np.mean(np.abs(positions))),
        "max_position_size": float(np.max(np.abs(positions))),
        "hours_flat": int((np.abs(positions) < 1e-6).sum()),
        "sharpe_running": compute_rolling_sharpe(pred_log, days=30),
        "total_funding_cost": float(day_data["funding_cost"].sum()) if "funding_cost" in day_data.columns else 0.0,
    }
    append_daily_summary(summary_log, row)


def _count_csv_rows(path):
    """Count data rows in a CSV file (excluding header)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def _max_abs_position(pred_log):
    """Get max absolute position from prediction log."""
    if not os.path.exists(pred_log):
        return 0.0
    df = pd.read_csv(pred_log)
    if len(df) == 0 or "position" not in df.columns:
        return 0.0
    return float(df["position"].abs().max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay historical data through paper trading pipeline")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default="logs/replay", help="Output directory")
    args = parser.parse_args()

    # Default end to yesterday
    if args.end is None:
        args.end = str((pd.Timestamp.now("UTC").tz_localize(None) - timedelta(days=1)).date())

    # Change to project root
    os.chdir(Path(__file__).resolve().parent.parent)

    replay(
        start=args.start,
        end=args.end,
        config_path=args.config,
        output_dir=args.output_dir,
    )
