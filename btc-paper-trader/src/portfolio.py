"""Portfolio state tracking, P&L calculation, and daily summaries."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Simulated portfolio state persisted between hourly runs."""

    position: float = 0.0  # -1.0 to +1.0
    portfolio_value: float = 1.0  # unit portfolio
    peak_value: float = 1.0
    trade_count: int = 0
    prev_btc_price: float = 0.0  # last hour's close
    last_trade_timestamp: str = ""
    inception_date: str = ""
    daily_returns: list = field(default_factory=list)  # for rolling Sharpe


def load_portfolio_state(path: str) -> PortfolioState:
    """Load portfolio state from JSON file. Returns default if missing."""
    if not os.path.exists(path):
        return PortfolioState()

    with open(path) as f:
        data = json.load(f)

    return PortfolioState(
        position=data.get("position", 0.0),
        portfolio_value=data.get("portfolio_value", 1.0),
        peak_value=data.get("peak_value", 1.0),
        trade_count=data.get("trade_count", 0),
        prev_btc_price=data.get("prev_btc_price", 0.0),
        last_trade_timestamp=data.get("last_trade_timestamp", ""),
        inception_date=data.get("inception_date", ""),
        daily_returns=data.get("daily_returns", []),
    )


def save_portfolio_state(state: PortfolioState, path: str) -> None:
    """Save portfolio state to JSON with atomic write."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(asdict(state), f, indent=2, default=str)
    os.replace(tmp_path, path)


def update_portfolio(
    state: PortfolioState,
    new_position: float,
    btc_price: float,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> tuple[PortfolioState, dict]:
    """Update portfolio with new position and price.

    Returns (new_state, metrics_dict) where metrics contains
    all values needed for prediction/trade logging.
    """
    position_prev = state.position
    prev_price = state.prev_btc_price

    # Compute BTC return (0 on first run)
    if prev_price > 0:
        btc_return = (btc_price - prev_price) / prev_price
    else:
        btc_return = 0.0

    # Position change and fees
    position_delta = abs(new_position - position_prev)
    fee_cost = position_delta * (fee_rate + slippage_rate)

    # Portfolio return: position P&L minus transaction costs
    portfolio_return = position_prev * btc_return - fee_cost
    new_value = state.portfolio_value * (1 + portfolio_return)
    new_peak = max(state.peak_value, new_value)
    drawdown = (new_value - new_peak) / new_peak if new_peak > 0 else 0.0

    # Track if position changed
    position_changed = abs(position_delta) > 1e-6
    new_trade_count = state.trade_count + (1 if position_changed else 0)
    new_trade_ts = (
        str(datetime.now(tz=None)) if position_changed else state.last_trade_timestamp
    )

    # Set inception date on first run
    inception = state.inception_date or str(datetime.now(tz=None).date())

    new_state = PortfolioState(
        position=new_position,
        portfolio_value=new_value,
        peak_value=new_peak,
        trade_count=new_trade_count,
        prev_btc_price=btc_price,
        last_trade_timestamp=new_trade_ts,
        inception_date=inception,
        daily_returns=state.daily_returns,
    )

    metrics = {
        "position": new_position,
        "position_prev": position_prev,
        "position_delta": position_delta,
        "fee_cost": fee_cost,
        "btc_return_1h": btc_return,
        "portfolio_return": portfolio_return,
        "portfolio_value": new_value,
        "peak_value": new_peak,
        "drawdown": drawdown,
        "position_changed": position_changed,
    }

    return new_state, metrics


def compute_daily_summary(
    prediction_log_path: str,
    date: str,
    portfolio_state: PortfolioState,
) -> dict | None:
    """Compute daily summary from prediction log CSV.

    Args:
        prediction_log_path: Path to predictions.csv
        date: Date string (YYYY-MM-DD) to summarize
        portfolio_state: Current portfolio state (for cumulative metrics)

    Returns dict of summary fields, or None if no data for the date.
    """
    if not os.path.exists(prediction_log_path):
        return None

    df = pd.read_csv(prediction_log_path)
    if len(df) == 0:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)

    day_data = df[df["date"] == date]
    if len(day_data) == 0:
        return None

    # Daily return from portfolio returns
    if "fee_cost" in day_data.columns and "btc_return_1h" in day_data.columns:
        hourly_returns = (
            day_data["position_prev"] * day_data["btc_return_1h"] - day_data["fee_cost"]
        )
        daily_return = (1 + hourly_returns).prod() - 1
    else:
        daily_return = 0.0

    # Position statistics
    positions = day_data["position"].values
    n_trades = int((day_data["position_delta"].abs() > 1e-6).sum())
    avg_pos = float(np.mean(np.abs(positions)))
    max_pos = float(np.max(np.abs(positions)))
    hours_flat = int((np.abs(positions) < 1e-6).sum())

    # Rolling 30-day Sharpe
    sharpe = compute_rolling_sharpe(prediction_log_path, days=30)

    return {
        "date": date,
        "portfolio_value": portfolio_state.portfolio_value,
        "daily_return": daily_return,
        "drawdown": (portfolio_state.portfolio_value - portfolio_state.peak_value)
        / portfolio_state.peak_value
        if portfolio_state.peak_value > 0
        else 0.0,
        "n_trades_today": n_trades,
        "avg_position_size": avg_pos,
        "max_position_size": max_pos,
        "hours_flat": hours_flat,
        "sharpe_running": sharpe,
    }


def compute_rolling_sharpe(
    prediction_log_path: str,
    days: int = 30,
) -> float:
    """Compute annualized Sharpe ratio from last N days of hourly returns."""
    if not os.path.exists(prediction_log_path):
        return 0.0

    df = pd.read_csv(prediction_log_path)
    if len(df) == 0:
        return 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    recent = df[df["timestamp"] >= cutoff]

    if len(recent) < 48:  # Need at least 2 days of data
        return 0.0

    # Hourly portfolio returns
    hourly_returns = recent["position_prev"] * recent["btc_return_1h"] - recent["fee_cost"]
    mean_return = hourly_returns.mean()
    std_return = hourly_returns.std()

    if std_return < 1e-10:
        return 0.0

    # Annualize: 8760 hours per year
    sharpe = (mean_return / std_return) * np.sqrt(8760)
    return float(sharpe)
