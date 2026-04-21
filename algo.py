"""
Cointegration pair trade: SPY vs IVV.

Both track the S&P 500. The log-price spread is structurally mean-reverting.
We trade the z-score of that spread with a simple threshold entry/exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import Data

LOOKBACK = 60       # rolling window for z-score
ENTRY_Z  = 1.0     # enter when |z| crosses this
EXIT_Z   = 0.0     # exit when spread mean-reverts to zero
HALF_LEG = 0.5     # each leg is 50% of NAV (market-neutral, 1x gross)


def strategy(data: Data) -> pd.DataFrame:
    bars = data.bars(["SPY", "IVV"], timeframe="day")
    close = bars["close"].unstack("symbol").ffill()

    spread = np.log(close["SPY"]) - np.log(close["IVV"])
    roll_mean = spread.rolling(LOOKBACK).mean()
    roll_std  = spread.rolling(LOOKBACK).std()
    z = (spread - roll_mean) / roll_std

    spy_w = pd.Series(0.0, index=close.index)
    ivv_w = pd.Series(0.0, index=close.index)

    position = 0  # +1 = long SPY / short IVV, -1 = opposite
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pass
        elif position == 0:
            if zt > ENTRY_Z:
                position = -1   # SPY expensive: short SPY, long IVV
            elif zt < -ENTRY_Z:
                position = 1    # SPY cheap: long SPY, short IVV
        elif position == 1 and zt >= EXIT_Z:
            position = 0
        elif position == -1 and zt <= EXIT_Z:
            position = 0

        spy_w.iloc[t] =  HALF_LEG * position
        ivv_w.iloc[t] = -HALF_LEG * position

    weights = pd.DataFrame({"SPY": spy_w, "IVV": ivv_w}, index=close.index)
    weights.index = close.index
    return weights
