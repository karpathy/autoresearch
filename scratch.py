"""Pair screening: find ETF pairs with mean-reverting spreads."""
import numpy as np
import pandas as pd
from load import Data

data = Data(split="train")

# Candidate pairs - ETFs that are economically related but not identical trackers
candidates = [
    ("SPY", "IVV"),   # S&P 500 trackers (baseline - we know this is too tight)
    ("HYG", "JNK"),   # High yield bond ETFs
    ("GLD", "IAU"),   # Gold ETFs
    ("TLT", "IEF"),   # Long vs medium Treasury
    ("XLF", "KBE"),   # Financials vs Banks
    ("XLK", "QQQ"),   # Tech sector vs Nasdaq-100
    ("IWM", "MDY"),   # Small cap vs Mid cap
    ("XLE", "OIH"),   # Energy vs Oil services
    ("XLU", "XLP"),   # Utilities vs Consumer Staples (defensive pair)
    ("EEM", "VWO"),   # Emerging markets ETFs
    ("LQD", "HYG"),   # Investment grade vs High yield bonds
    ("TLT", "EDV"),   # Long Treasury vs Treasury strips
    ("SHY", "BIL"),   # Short-term Treasuries
]

def half_life(spread):
    """Estimate mean-reversion half-life via AR(1) regression."""
    s = spread.dropna()
    delta = s.diff().dropna()
    lagged = s.shift(1).dropna()
    # Align
    lagged, delta = lagged.align(delta, join="inner")
    # OLS: delta = a + b * lagged
    X = np.column_stack([np.ones(len(lagged)), lagged.values])
    b = np.linalg.lstsq(X, delta.values, rcond=None)[0]
    lam = b[1]
    if lam >= 0 or lam <= -2:
        return np.inf
    return -np.log(2) / lam

def hedge_ratio(p1, p2):
    """OLS hedge ratio: log(p1) = a + h * log(p2)."""
    lp1 = np.log(p1.dropna())
    lp2 = np.log(p2.dropna())
    lp1, lp2 = lp1.align(lp2, join="inner")
    X = np.column_stack([np.ones(len(lp2)), lp2.values])
    b = np.linalg.lstsq(X, lp1.values, rcond=None)[0]
    return b[1]

print(f"{'Pair':<16} {'Corr':>6} {'HalfLife':>10} {'SpreadStd':>10} {'Obs':>6}")
print("-" * 55)

results = []
for sym1, sym2 in candidates:
    try:
        bars = data.bars([sym1, sym2], timeframe="day")
        close = bars["close"].unstack("symbol").ffill().dropna()
        if len(close) < 200:
            print(f"{sym1}/{sym2:<12} too few bars ({len(close)})")
            continue

        p1, p2 = close[sym1], close[sym2]
        corr = p1.corr(p2)

        h = hedge_ratio(p1, p2)
        spread = np.log(p1) - h * np.log(p2)
        hl = half_life(spread)
        spread_std = spread.std()

        print(f"{sym1+'/'+sym2:<16} {corr:6.4f} {hl:10.1f} {spread_std:10.6f} {len(close):6d}")
        results.append((sym1, sym2, corr, hl, spread_std, len(close)))
    except Exception as e:
        print(f"{sym1}/{sym2}: ERROR - {e}")

print()
print("Best candidates (hl < 60, corr > 0.95):")
for r in sorted(results, key=lambda x: x[3]):
    sym1, sym2, corr, hl, spread_std, n = r
    if hl < 60 and corr > 0.95:
        print(f"  {sym1}/{sym2}: corr={corr:.4f}, half_life={hl:.1f}d, spread_std={spread_std:.6f}")
