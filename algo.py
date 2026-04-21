"""
Strategy file. This is the ONLY file the research agent edits.

The contract is the `strategy(data)` function below: `test.py` will import
it, hand it a `load.Data` scoped to the active split, and score the
returned weights. Everything else in this file can be reshaped as needed.

The current implementation is a placeholder: 100% long SPY, rebalanced
every daily bar. Its Sharpe is whatever SPY's Sharpe happens to be on the
evaluated window (minus a one-time entry cost). Delete or replace it
with your own ideas.
"""

from __future__ import annotations

import pandas as pd

from load import Data


def strategy(data: Data) -> pd.DataFrame:
    """Return target portfolio weights.

    Parameters
    ----------
    data : load.Data
        Pre-scoped to the split the harness is evaluating on. Use
        ``data.universe(...)`` and ``data.bars(...)`` to pull whatever you
        need; both are automatically bounded to the active window.

    Returns
    -------
    pandas.DataFrame
        - Index   : tz-aware UTC ``DatetimeIndex``, strictly increasing, no
                    duplicates. Every timestamp MUST be a real bar timestamp
                    from ``data.bars(..., timeframe=...)`` at either ``"day"``
                    or ``"30min"`` resolution (mix of the two in one frame
                    is not supported).
        - Columns : symbols, all drawn from ``data.universe()``.
        - Values  : target weight of the symbol in portfolio NAV at that
                    row's timestamp. +0.5 = 50% long, -0.25 = 25% short,
                    0 or NaN = flat. No cap on gross exposure -- leverage
                    is on you.

    Semantics
    ---------
    The row at time ``t`` is executed at that bar's close and held until
    the row at ``t+1``. Returns earned from ``t`` to ``t+1`` are
    ``W[t] . (P[t+1] / P[t] - 1)``. Transaction cost is charged at every
    rebalance (including the initial entry from zero).
    """
    # Placeholder: buy SPY on day 1, hold to the end. Zero turnover after
    # the initial entry.
    bars = data.bars("SPY", timeframe="day")
    timestamps = bars.index.get_level_values("timestamp")
    return pd.DataFrame({"SPY": 1.0}, index=timestamps)
