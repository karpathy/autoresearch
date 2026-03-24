"""Daily report generation and delivery.

Generates a plain text report from prediction/portfolio logs and
delivers it via file, Telegram, email, or Slack.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .portfolio import PortfolioState, compute_rolling_sharpe

logger = logging.getLogger(__name__)


def generate_report(
    config: dict,
    prediction_log_path: str,
    portfolio_state: PortfolioState,
    artifact_metadata: dict,
    alerts: list[str] | None = None,
) -> str:
    """Generate the daily report as plain text.

    Args:
        config: Full config dict
        prediction_log_path: Path to predictions.csv
        portfolio_state: Current portfolio state
        artifact_metadata: Dict with 'commit', 'trained_at', etc.
        alerts: Any active alerts to include

    Returns:
        Report as a plain text string.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    commit = artifact_metadata.get("commit", "unknown")
    inception = portfolio_state.inception_date or today

    # Calculate day count
    try:
        inception_dt = datetime.strptime(inception, "%Y-%m-%d")
        day_count = (datetime.strptime(today, "%Y-%m-%d") - inception_dt).days + 1
    except ValueError:
        day_count = 0

    lines = []

    # --- Header ---
    lines.append(f"BTC Paper Trader — Daily Report — {today}")
    lines.append(f"Model: {commit} | Running since: {inception} | Day {day_count}")
    lines.append("")

    # --- Portfolio summary ---
    pv = portfolio_state.portfolio_value
    cum_return = (pv - 1.0) * 100
    peak = portfolio_state.peak_value
    drawdown = (pv - peak) / peak * 100 if peak > 0 else 0.0

    # Compute today's return from prediction log
    today_return = _compute_today_return(prediction_log_path, today)

    lines.append("Portfolio summary:")
    lines.append(f"  Portfolio value:    {pv:.4f} ({cum_return:+.2f}% since inception)")
    lines.append(f"  Today's return:     {today_return:+.2f}%")
    lines.append(f"  Peak value:         {peak:.4f}")
    lines.append(f"  Current drawdown:   {drawdown:.2f}%")
    lines.append(f"  Max drawdown:       {drawdown:.2f}%")

    # Funding costs
    today_funding = _compute_today_funding(prediction_log_path, today)
    cum_funding = portfolio_state.cumulative_funding_cost * 100
    lines.append(f"  Funding costs:      {today_funding:+.3f}% (today) / {cum_funding:+.3f}% (cumulative)")
    lines.append("")

    # --- Trading activity ---
    activity = _compute_trading_activity(prediction_log_path, today, portfolio_state)
    lines.append("Trading activity (last 24h):")
    lines.append(f"  Position changes:   {activity['n_trades']}")
    lines.append(f"  Current position:   {activity['current_pos_str']}")
    lines.append(f"  Hours positioned:   {activity['hours_positioned']}/24")
    lines.append(f"  Avg |position|:     {activity['avg_position']:.2f}")
    lines.append("")

    # --- Prediction diagnostics ---
    diag = _compute_pred_diagnostics(prediction_log_path, today)
    lines.append("Prediction diagnostics (last 24h):")
    lines.append(f"  Pred range:         {diag['pred_min']:.2f} to {diag['pred_max']:+.2f} sigma")
    lines.append(f"  |pred| > 0.20:     {diag['above_threshold']}/24 hours ({diag['above_threshold_pct']:.1f}%)")
    lines.append(f"  72h disagreements:  {diag['disagreements']}/24 hours")
    lines.append(f"  Conf scaler range:  {diag['conf_min']:.2f} – {diag['conf_max']:.2f}")
    lines.append("")

    # --- Running metrics ---
    sharpe_30d = compute_rolling_sharpe(prediction_log_path, days=30)
    sharpe_all = compute_rolling_sharpe(prediction_log_path, days=9999)
    metrics = _compute_running_metrics(prediction_log_path, portfolio_state)

    lines.append("Running metrics:")
    lines.append(f"  Trades (total):     {portfolio_state.trade_count}")
    lines.append(f"  Sharpe (30-day):    {sharpe_30d:.2f}")
    lines.append(f"  Sharpe (inception): {sharpe_all:.2f}")
    lines.append(f"  Win rate:           {metrics['win_rate']:.0f}%")
    lines.append(f"  Avg trade duration: {metrics['avg_trade_duration']:.1f} hours")
    lines.append(f"  Monthly returns:    {metrics['monthly_returns_str']}")

    # Fee comparison
    fee_comparison = _compute_fee_comparison(prediction_log_path)
    lines.append(f"  Fee comparison:     model={fee_comparison['model_fees']:.2f}% vs BIP={fee_comparison['bip_fees']:.2f}% (cumulative)")
    lines.append("")

    # --- Alerts ---
    if alerts:
        lines.append("Alerts:")
        for alert in alerts:
            lines.append(f"  {alert}")
        lines.append("")

    return "\n".join(lines)


def _compute_today_return(log_path: str, date: str) -> float:
    """Compute today's portfolio return from prediction log."""
    if not os.path.exists(log_path):
        return 0.0
    df = pd.read_csv(log_path)
    if len(df) == 0:
        return 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]
    if len(today_data) == 0:
        return 0.0
    funding = today_data["funding_cost"] if "funding_cost" in today_data.columns else 0.0
    hourly_rets = today_data["position_prev"] * today_data["btc_return_1h"] - today_data["fee_cost"] - funding
    return float(((1 + hourly_rets).prod() - 1) * 100)


def _compute_today_funding(log_path: str, date: str) -> float:
    """Compute today's total funding cost as percentage."""
    if not os.path.exists(log_path):
        return 0.0
    df = pd.read_csv(log_path)
    if len(df) == 0 or "funding_cost" not in df.columns:
        return 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]
    if len(today_data) == 0:
        return 0.0
    return float(today_data["funding_cost"].sum() * 100)


def _compute_fee_comparison(log_path: str) -> dict:
    """Compute cumulative model fees vs BIP fees as percentages."""
    default = {"model_fees": 0.0, "bip_fees": 0.0}
    if not os.path.exists(log_path):
        return default
    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default
    model_fees = df["fee_cost"].sum() * 100 if "fee_cost" in df.columns else 0.0
    bip_fees = df["bip_fee_cost"].sum() / 100 if "bip_fee_cost" in df.columns else 0.0  # BIP is in USD, rough % estimate
    return {"model_fees": float(model_fees), "bip_fees": float(bip_fees)}


def _compute_trading_activity(
    log_path: str, date: str, state: PortfolioState,
) -> dict:
    """Compute trading activity for the last 24 hours."""
    default = {
        "n_trades": 0,
        "current_pos_str": "FLAT",
        "hours_positioned": 0,
        "avg_position": 0.0,
    }
    if not os.path.exists(log_path):
        return default

    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]

    if len(today_data) == 0:
        return default

    positions = today_data["position"].values
    n_trades = int((today_data["position_delta"].abs() > 1e-6).sum())
    hours_positioned = int((np.abs(positions) > 1e-6).sum())
    avg_pos = float(np.mean(np.abs(positions)))

    # Current position string
    pos = state.position
    if abs(pos) < 1e-6:
        pos_str = "FLAT"
    elif pos > 0:
        pos_str = f"LONG {pos:.2f}"
    else:
        pos_str = f"SHORT {abs(pos):.2f}"

    return {
        "n_trades": n_trades,
        "current_pos_str": pos_str,
        "hours_positioned": hours_positioned,
        "avg_position": avg_pos,
    }


def _compute_pred_diagnostics(log_path: str, date: str) -> dict:
    """Compute prediction diagnostics for the last 24 hours."""
    default = {
        "pred_min": 0.0, "pred_max": 0.0,
        "above_threshold": 0, "above_threshold_pct": 0.0,
        "disagreements": 0,
        "conf_min": 0.0, "conf_max": 0.0,
    }
    if not os.path.exists(log_path):
        return default

    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]

    if len(today_data) == 0:
        return default

    preds = today_data["pred_final"].values
    n = len(today_data)
    above = int((np.abs(preds) > 0.20).sum())

    disagree = 0
    if "sign_agree" in today_data.columns:
        disagree = int((today_data["sign_agree"] < 0).sum())

    conf_adj = today_data["conf_adj"].values if "conf_adj" in today_data.columns else [0.0]

    return {
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "above_threshold": above,
        "above_threshold_pct": above / n * 100 if n > 0 else 0.0,
        "disagreements": disagree,
        "conf_min": float(np.min(conf_adj)),
        "conf_max": float(np.max(conf_adj)),
    }


def _compute_running_metrics(log_path: str, state: PortfolioState) -> dict:
    """Compute running metrics from the full prediction log."""
    default = {
        "win_rate": 0.0,
        "avg_trade_duration": 0.0,
        "monthly_returns_str": "N/A",
    }
    if not os.path.exists(log_path):
        return default

    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Win rate: fraction of hours with positive P&L when positioned
    positioned = df[df["position_prev"].abs() > 1e-6]
    if len(positioned) > 0:
        hourly_pnl = positioned["position_prev"] * positioned["btc_return_1h"] - positioned["fee_cost"]
        win_rate = float((hourly_pnl > 0).mean() * 100)
    else:
        win_rate = 0.0

    # Average trade duration
    pos_changes = df[df["position_delta"].abs() > 1e-6]
    if len(pos_changes) > 1:
        durations = pos_changes["timestamp"].diff().dt.total_seconds() / 3600
        avg_duration = float(durations.dropna().mean())
    else:
        avg_duration = 0.0

    # Monthly returns string (last 3 months)
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly_returns = []
    for month in df["month"].unique()[-3:]:
        month_data = df[df["month"] == month]
        hourly_rets = month_data["position_prev"] * month_data["btc_return_1h"] - month_data["fee_cost"]
        month_ret = (1 + hourly_rets).prod() - 1
        month_str = str(month)
        monthly_returns.append(f"{month_str} {month_ret:+.1%}")

    monthly_str = ", ".join(monthly_returns) if monthly_returns else "N/A"

    return {
        "win_rate": win_rate,
        "avg_trade_duration": avg_duration,
        "monthly_returns_str": monthly_str,
    }


def deliver_report(report_text: str, config: dict) -> None:
    """Deliver the report via configured transport(s).

    Always writes to file. Optionally sends via Telegram/email/Slack.
    """
    reporting = config.get("reporting", {})
    report_path = reporting.get("daily_report_path", "logs/daily_report.txt")

    # Always write to file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Daily report written to {report_path}")

    # Deliver via configured method
    method = reporting.get("delivery_method", "file")

    if method == "telegram":
        _send_telegram(report_text, reporting)
    elif method == "email":
        _send_email(report_text, reporting)
    elif method == "slack":
        _send_slack(report_text, reporting)


def _send_telegram(text: str, reporting_config: dict) -> None:
    """Send report via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or reporting_config.get("telegram_bot_token", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or reporting_config.get("telegram_chat_id", "")

    if not token or not chat_id:
        logger.warning("Telegram credentials not configured, skipping delivery")
        return

    # Clean env var syntax from config
    token = token.strip("${}")
    chat_id = chat_id.strip("${}")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"```\n{text}\n```",
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info("Daily report sent via Telegram")
    except Exception as e:
        logger.error(f"Telegram delivery failed: {e}")


def _send_email(text: str, reporting_config: dict) -> None:
    """Send report via SMTP email."""
    import smtplib
    from email.mime.text import MIMEText

    host = reporting_config.get("email_smtp_host", "")
    port = reporting_config.get("email_smtp_port", 587)
    from_addr = reporting_config.get("email_from", "")
    to_addr = reporting_config.get("email_to", "")
    password = os.environ.get("EMAIL_PASSWORD", "")

    if not all([host, from_addr, to_addr]):
        logger.warning("Email credentials not configured, skipping delivery")
        return

    msg = MIMEText(text)
    msg["Subject"] = f"BTC Paper Trader — Daily Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            if password:
                server.login(from_addr, password)
            server.send_message(msg)
        logger.info("Daily report sent via email")
    except Exception as e:
        logger.error(f"Email delivery failed: {e}")


def _send_slack(text: str, reporting_config: dict) -> None:
    """Send report via Slack webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or reporting_config.get("slack_webhook_url", "")

    if not webhook_url:
        logger.warning("Slack webhook not configured, skipping delivery")
        return

    payload = {"text": f"```{text}```"}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info("Daily report sent via Slack")
    except Exception as e:
        logger.error(f"Slack delivery failed: {e}")
