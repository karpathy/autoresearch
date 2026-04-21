# Data

All data access goes through `load.Data`. Don't read files in this directory
directly — use the API.

```python
from load import Data

data = Data(split="train")
data.universe()                 # list[str] — every symbol available
data.universe("sp500")          # just current S&P 500
data.universe("etf")            # just curated ETFs

    data.bars("SPY")                                   # daily OHLCV
    data.bars(["AAPL", "MSFT"])                        # multi-symbol daily
    data.bars(["KO", "PEP"], timeframe="30min")        # 30-min (regular session)
```

## Split boundary

| Split | Window (UTC, inclusive)                | Access                                  |
|-------|----------------------------------------|-----------------------------------------|
| train | 2017-01-01 .. 2022-12-31               | `Data(split="train")` — always allowed  |
| test  | 2023-01-01 .. 2025-12-31               | `Data(split="test")` — gated (see note) |

Constructing `Data(split="test")` raises `PermissionError` unless the
environment variable `AUTORESEARCH_ALLOW_TEST=1` is set. Only the evaluation
harness sets this. The research loop never sees test data.

## Universe

**503 current S&P 500 constituents** (from the datahub CSV mirror, cached in
`sp500.csv` the first time `load.py` runs) plus **324 curated ETFs** covering:

- Broad US equity (SPY/VOO/IVV/VTI/…, size segments, RSP)
- Style and factor (value/growth/quality/momentum/low-vol/dividend)
- Sector SPDRs + Vanguard mirrors (XLK/VGT, XLF/VFH, …)
- Industries (semis, biotech, banks, real estate, defense, retail, homebuilders, miners, gold/silver, energy, renewables, fintech, cybersecurity, cloud, gaming/AI, ARK)
- International developed (Europe, Japan, UK, single-country)
- International emerging (China, India, Brazil, …)
- Full Treasury curve (BIL/SGOV → TLT/EDV/ZROZ) and credit (IG/HY/bank loans/TIPS/muni/EM debt)
- Commodities (metals, energy, agriculture, broad baskets)
- Currencies (USD index, EUR/JPY/GBP/CAD/AUD/CHF, CNY, EM basket)
- Volatility (VXX/UVXY/SVXY/…)
- Leveraged + inverse (2x/3x broad, sector, rates)
- Preferreds, MLPs, mortgage REITs, specialty

Survivorship bias: the S&P 500 list reflects *current* membership, not
point-in-time. Strategies are optimistically evaluated relative to reality.

## Timeframes

| Timeframe | Bars / full trading day         | Notes                                                                 |
|-----------|----------------------------------|-----------------------------------------------------------------------|
| `"day"`   | 1                                | Standard daily bar.                                                   |
| `"30min"` | 13 (9:30, 10:00, …, 15:30 ET)   | Regular US session only. Extended-hours and closing-auction bars dropped. |

30-minute bars are labelled on their left edge. The 9:30 ET bar (open-auction
print through 10:00) starts the session cleanly — its `open` is the real
9:30 regular-session open, not a pre-market print. The 15:30 ET bar covers
15:30–16:00 and closes at the bell.

Any timestamp returned by `data.bars(..., "30min")` is a valid trade time;
trading is implicitly disabled outside market hours.

## Bars shape

All calls return a pandas `DataFrame` with:

- Index: `MultiIndex` of `(symbol, timestamp)`, timestamps in UTC.
- Columns: `open, high, low, close, volume, trade_count, vwap`.

Feed is SIP (consolidated tape), adjustment is `all` (splits + dividends +
spin-offs). Volume and VWAP are real consolidated values, not IEX-only.

Symbols that IPO'd or were added to ETF coverage after 2017 will have
truncated history on the left edge (e.g. BITO starts Oct 2021). Don't assume
every symbol has a bar on every day.

## Cache layout

`load.Data` caches on-demand at:

- `data/cache/day/<symbol>.pkl`
- `data/cache/30min/<symbol>.pkl`

Each file holds the full 2017–2025 range for that symbol at that timeframe
(sliced to the requested split at read time). First access fetches from
Alpaca; subsequent accesses are milliseconds. The cache directory is
gitignored and populated entirely on demand.

Empty placeholders are written for symbols that Alpaca returns no data for
(delisted names, bad tickers, too-recent IPOs outside the fetched window)
to avoid re-requesting them. Transient network/SSL errors do *not* write
placeholders — the symbol is left uncached and retried on the next call.
