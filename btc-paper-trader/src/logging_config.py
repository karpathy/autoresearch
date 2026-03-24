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
    "funding_rate",
    "funding_cost",
    "btc_price",
    "btc_return_1h",
    "bip_n_contracts",
    "bip_fee_cost",
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
    "total_funding_cost",
]


def setup_system_log(
    path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure rotating file handler for system log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Use root logger so all modules (src.data, src.inference, etc.) are captured
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeat calls
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count,
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to stderr for cron job visibility
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


_MAX_CSV_BYTES = 50 * 1024 * 1024  # 50MB


def _rotate_if_large(path: str, max_bytes: int = _MAX_CSV_BYTES) -> None:
    """Rotate file to .1 backup if it exceeds max_bytes."""
    if os.path.exists(path) and os.path.getsize(path) > max_bytes:
        backup = path + ".1"
        if os.path.exists(backup):
            os.unlink(backup)
        os.rename(path, backup)
        logging.getLogger(__name__).info(f"Rotated {path} ({max_bytes // 1024 // 1024}MB limit)")


def _append_csv_row(path: str, row: dict, fields: list[str]) -> None:
    """Append a row to a CSV file. Write header if file doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _rotate_if_large(path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to write CSV row to {path}: {e}")


def append_prediction_row(path: str, row: dict) -> None:
    """Append one row to the prediction log CSV."""
    _append_csv_row(path, row, PREDICTION_FIELDS)


def append_trade_row(path: str, row: dict) -> None:
    """Append one row to the trade log CSV."""
    _append_csv_row(path, row, TRADE_FIELDS)


def append_daily_summary(path: str, row: dict) -> None:
    """Append one row to the daily summary CSV."""
    _append_csv_row(path, row, DAILY_SUMMARY_FIELDS)
