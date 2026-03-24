"""Health checks and alert notifications."""

import logging
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def run_health_checks(
    config: dict,
    df: pd.DataFrame,
    pred_final: float,
    portfolio_value: float,
    peak_value: float,
    artifact_trained_at: str,
) -> list[str]:
    """Run all health checks, return list of alert strings (empty if all OK)."""
    alerts = []
    alert_cfg = config.get("alerts", {})

    # 1. Data freshness: latest candle < 2 hours old
    latest_ts = df["timestamp"].max()
    now = pd.Timestamp.now(tz=None)
    age_hours = (now - latest_ts).total_seconds() / 3600
    if age_hours > 2:
        alerts.append(f"WARN: Data stale — latest candle is {age_hours:.1f}h old ({latest_ts})")

    # 2. Prediction sanity
    sanity_threshold = alert_cfg.get("prediction_sanity_threshold", 2.0)
    if abs(pred_final) > sanity_threshold:
        alerts.append(
            f"ALERT: Prediction {pred_final:.4f} exceeds sanity threshold {sanity_threshold}"
        )

    # 3. Portfolio drawdown
    dd_threshold = alert_cfg.get("drawdown_threshold", -0.10)
    drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
    if drawdown < dd_threshold:
        alerts.append(
            f"ALERT: Drawdown {drawdown:.1%} exceeds threshold {dd_threshold:.1%}"
        )

    # 4. Model staleness
    staleness_days = alert_cfg.get("model_staleness_days", 30)
    try:
        trained = datetime.fromisoformat(artifact_trained_at)
        age = (datetime.now(timezone.utc) - trained).days
        if age > staleness_days:
            alerts.append(
                f"WARN: Model artifact is {age} days old (threshold: {staleness_days})"
            )
    except (ValueError, TypeError):
        alerts.append("WARN: Could not parse artifact trained_at timestamp")

    # 5. Disk space
    try:
        usage = shutil.disk_usage("/")
        pct_used = usage.used / usage.total
        if pct_used > 0.90:
            alerts.append(f"WARN: Disk {pct_used:.0%} full")
    except Exception:
        pass  # Skip on platforms where this fails

    return alerts


def write_alerts(alerts: list[str], path: str) -> None:
    """Append timestamped alerts to the alert log file."""
    if not alerts:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(path, "a") as f:
        for alert in alerts:
            line = f"[{now}] {alert}\n"
            f.write(line)
            logger.warning(alert)
