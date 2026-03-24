"""Backtesting engine: convert sigma-space predictions to positions and compute metrics.

Position sizing is done in sigma-space (regime-invariant).
Portfolio returns are computed in dollar terms against actual prices.
"""

import math

import numpy as np
import pandas as pd

from core.config import BacktestConfig


def backtest(sigma_predictions: np.ndarray, close_prices: np.ndarray,
             timestamps: np.ndarray, subperiods: list,
             config: BacktestConfig,
             funding_rates: np.ndarray | None = None) -> dict:
    """Run backtest on sigma-space predictions against actual prices.

    Args:
        sigma_predictions: Array of sigma-space predictions, one per timestamp.
        close_prices: Array of close prices aligned with predictions.
        timestamps: Array of pd.Timestamp aligned with predictions.
        subperiods: List of (start, end, label) tuples for consistency check.
        config: Backtesting parameters (thresholds, fees, slippage).
        funding_rates: Optional array of per-8h funding rates aligned with
            predictions. When provided and config.model_funding_cost is True,
            hourly funding cost = position * (funding_rate / 8) is deducted.

    Returns:
        dict with keys: sharpe, max_drawdown, n_trades, total_return,
                        subperiod_returns (list of floats).
    """
    n = len(sigma_predictions)
    assert len(close_prices) == n and len(timestamps) == n

    # Continuous position sizing in sigma-space: linear ramp from 0 at
    # ±sigma_threshold to ±1.0 at ±sigma_full_position
    abs_sigma = np.abs(sigma_predictions)
    sigma_scale = config.sigma_full_position - config.sigma_threshold
    raw_size = np.where(abs_sigma > config.sigma_threshold,
                        (abs_sigma - config.sigma_threshold) / sigma_scale, 0.0)
    positions = np.clip(raw_size, 0.0, 1.0) * np.sign(sigma_predictions)

    # Compute hourly price returns
    price_returns = np.zeros(n)
    price_returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    # Portfolio returns: position at time t earns the return from t to t+1.
    # The position decided at t-1 is held during period t.
    portfolio_returns = np.zeros(n)
    n_trades = 0

    # Prepare funding rate array (per-8h → per-hour)
    has_funding = (config.model_funding_cost
                   and funding_rates is not None
                   and len(funding_rates) == n)
    if has_funding:
        hourly_funding = np.nan_to_num(funding_rates, nan=0.0) / 8.0

    for i in range(1, n):
        pos = positions[i - 1]
        portfolio_returns[i] = pos * price_returns[i]

        # Fee model: cost proportional to position change magnitude
        prev_pos = positions[i - 2] if i >= 2 else 0.0
        pos_change = abs(pos - prev_pos)

        if pos_change > 1e-10:
            cost = pos_change * (config.fee_rate + config.slippage_rate)
            portfolio_returns[i] -= cost

        # Funding cost: position * hourly_funding_rate (every hour when positioned)
        if has_funding:
            portfolio_returns[i] -= pos * hourly_funding[i]

        # Count trades on zero-crossings only (direction changes)
        if (prev_pos > 0 and pos < 0) or (prev_pos < 0 and pos > 0) \
                or (prev_pos == 0 and pos != 0) or (prev_pos != 0 and pos == 0):
            n_trades += 1

    # Equity curve
    equity = np.cumprod(1.0 + portfolio_returns)

    # Sharpe ratio (annualized from hourly)
    if np.std(portfolio_returns) > 0:
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * math.sqrt(8760)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown))  # negative number

    # Total return
    total_return = float(equity[-1] / equity[0] - 1.0) if len(equity) > 0 else 0.0

    # Subperiod returns
    ts_series = pd.Series(timestamps)
    subperiod_returns = []
    for sp_start, sp_end, *_ in subperiods:
        mask = (ts_series >= sp_start) & (ts_series <= sp_end)
        if mask.sum() > 0:
            sp_equity = np.cumprod(1.0 + portfolio_returns[mask.values])
            sp_return = float(sp_equity[-1] - 1.0)
            subperiod_returns.append(sp_return)
        else:
            subperiod_returns.append(0.0)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "total_return": total_return,
        "subperiod_returns": subperiod_returns,
    }
