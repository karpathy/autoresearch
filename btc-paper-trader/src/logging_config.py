"""Logging configuration: CSV loggers and rotating system log."""

import csv
import logging
import os
from logging.handlers import RotatingFileHandler

PREDICTION_FIELDS = [
    "timestamp",
    "pred_24_raw",
    "pred_72_raw",
    "pred_72_smoothed",
    "sign_agree",
    "pred_after_72h",
    "conf_prob",
    "conf_smoothed",
    "conf_norm",
    "conf_adj",
    "pred_after_conf",
    "pos_scaler_signal",
    "pos_scale",
    "pred_after_pos",
    "pred_after_scale",
    "pred_final",
    "position",
    "position_prev",
    "position_delta",
    "fee_cost",
    "btc_price",
    "btc_return_1h",
]

TRADE_FIELDS = [
    "timestamp",
    "direction",
    "size",
    "entry_price",
    "pred_sigma",
    "conf_adj",
    "pos_scale",
]

DAILY_SUMMARY_FIELDS = [
    "date",
    "portfolio_value",
    "daily_return",
    "drawdown",
    "n_trades_today",
    "avg_position_size",
    "max_position_size",
    "hours_flat",
    "sharpe_running",
]


def setup_system_log(
    path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure rotating file handler for system log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("paper_trader")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count,
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to stderr for cron job visibility
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


def _append_csv_row(path: str, row: dict, fields: list[str]) -> None:
    """Append a row to a CSV file. Write header if file doesn't exist."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def append_prediction_row(path: str, row: dict) -> None:
    """Append one row to the prediction log CSV."""
    _append_csv_row(path, row, PREDICTION_FIELDS)


def append_trade_row(path: str, row: dict) -> None:
    """Append one row to the trade log CSV."""
    _append_csv_row(path, row, TRADE_FIELDS)


def append_daily_summary(path: str, row: dict) -> None:
    """Append one row to the daily summary CSV."""
    _append_csv_row(path, row, DAILY_SUMMARY_FIELDS)
