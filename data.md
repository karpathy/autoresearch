# Data

All market data access goes through `load.Data`. That's the only API you need.

```python
from load import Data

data = Data(split="train")

data.universe()              # -> list[str], every available symbol
data.universe("sp500")       # -> current S&P 500 (503 symbols)
data.universe("etf")         # -> curated ETFs (324 symbols)

bars = data.bars("SPY")                              # one symbol, daily
bars = data.bars(["AAPL", "MSFT", "NVDA"])           # multi-symbol, daily
bars = data.bars(["KO", "PEP"], timeframe="30min")   # 30-min, session only
```

## What you see

- **`Data(split="train")`** — 2017-01-01 through 2022-12-31, inclusive. This
  is the only split you should ever construct. Use it for both idea
  generation and for fitting parameters.
- The evaluation harness (`test.py`) handles the test split itself. You do
  not construct it, and attempting to will raise.

## Universe (fixed)

| Category | Count | Access                     |
|----------|-------|----------------------------|
| S&P 500  | 503   | `data.universe("sp500")`   |
| ETFs     | 324   | `data.universe("etf")`     |
| **all**  | **827** | `data.universe()`        |

ETFs span broad equity (SPY/VOO/QQQ), style + factor, sector SPDRs, industry
(semis, biotech, banks, REITs, energy, miners, cyber, cloud, ARK, BITO),
international (developed + EM, single-country), full Treasury curve + credit
+ TIPS + muni + EM debt, commodities, currencies, volatility (VXX/UVXY/SVXY),
leveraged + inverse (2x/3x broad, sector, rates), preferreds, MLPs, mREITs.

Requesting a symbol not in the universe raises `ValueError`.

## Timeframes

| Timeframe | Bars / full trading day                      | Tradable?                        |
|-----------|----------------------------------------------|----------------------------------|
| `"day"`   | 1                                            | Always.                          |
| `"30min"` | 13, labelled 9:30, 10:00, …, 15:30 ET        | Every returned bar is in-session.|

`"30min"` is filtered to the US regular session. The 9:30 ET bar is a clean
session-open bar (its `open` is the real 9:30 opening print), and the 15:30
ET bar closes at the bell. Extended hours and the closing-auction print are
excluded — if you got a bar, trading at that timestamp is valid.

No other timeframes are supported.

## Bars shape

All `bars()` calls return a pandas `DataFrame`:

- Index: `MultiIndex` of `(symbol, timestamp)`, timestamps in UTC.
- Columns: `open, high, low, close, volume, trade_count, vwap`.

Feed is SIP (consolidated tape), adjustment is `all` (splits + dividends +
spin-offs applied). Volumes are real consolidated values.

```python
close = data.bars(["AAPL","MSFT"]).loc[:, "close"].unstack("symbol")
# DatetimeIndex x symbol wide-form for downstream work.
```

## Things to know

- **Partial history is normal.** Symbols that IPO'd after 2017 or were added
  to the curated ETF set later (e.g. BITO starts Oct 2021) will have bars
  starting on their listing date, not on 2017-01-01. Don't assume every
  symbol has a bar on every day.
- **First call for a (symbol, timeframe) pair hits the network** and can take
  a few seconds per symbol at `"30min"`. Subsequent calls are served from
  an on-disk cache and are ~instant. Design your experiments so you don't
  re-fetch the same data on every iteration.
- **Deduping, alignment, and NaN handling are your responsibility.** `bars()`
  returns what Alpaca has; it does not forward-fill, align indexes across
  symbols, or drop partial rows.
- **Survivorship bias**: the S&P 500 list is current membership, not
  point-in-time. Names that were dropped from the index between 2017 and now
  are missing. Treat absolute Sharpe numbers on equities-only strategies as
  mildly optimistic.
