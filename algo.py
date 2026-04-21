"""
Cointegration pair trade: TLT vs EDV.

Both are long-duration Treasury ETFs (TLT ~17yr, EDV ~25yr strips).
The spread is structurally mean-reverting but wider than pure trackers.
OLS hedge ratio computed on a rolling window. Trade z-score of spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import Data

LOOKBACK  = 120     # rolling window for hedge ratio + z-score
ENTRY_Z   = 1.5    # wider threshold → lower turnover, survives costs
EXIT_Z    = 0.25   # exit early enough to not over-trade
HALF_LEG  = 0.5    # each leg 50% NAV, market-neutral


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope: y = a + h*x."""
    X = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def strategy(data: Data) -> pd.DataFrame:
    bars = data.bars(["TLT", "EDV"], timeframe="day")
    close = bars["close"].unstack("symbol").ffill().dropna()

    log_tlt = np.log(close["TLT"])
    log_edv = np.log(close["EDV"])

    # Rolling OLS hedge ratio + z-score of residual spread
    spread = pd.Series(np.nan, index=close.index)
    for i in range(LOOKBACK, len(close)):
        win_y = log_tlt.iloc[i - LOOKBACK: i].values
        win_x = log_edv.iloc[i - LOOKBACK: i].values
        h = _ols_hedge(win_y, win_x)
        spread.iloc[i] = log_tlt.iloc[i] - h * log_edv.iloc[i]

    roll_mean = spread.rolling(LOOKBACK).mean()
    roll_std  = spread.rolling(LOOKBACK).std()
    z = (spread - roll_mean) / roll_std

    tlt_w = pd.Series(0.0, index=close.index)
    edv_w = pd.Series(0.0, index=close.index)

    position = 0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pass
        elif position == 0:
            if zt > ENTRY_Z:
                position = -1   # TLT expensive vs EDV: short TLT, long EDV
            elif zt < -ENTRY_Z:
                position = 1    # TLT cheap: long TLT, short EDV
        elif position == 1 and zt >= EXIT_Z:
            position = 0
        elif position == -1 and zt <= -EXIT_Z:
            position = 0

        tlt_w.iloc[t] =  HALF_LEG * position
        edv_w.iloc[t] = -HALF_LEG * position

    weights = pd.DataFrame({"TLT": tlt_w, "EDV": edv_w}, index=close.index)
    return weights
