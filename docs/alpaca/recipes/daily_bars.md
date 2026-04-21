# Recipe: Daily Bars → Clean DataFrame

Task pattern: "Get me [N years] of daily bars for [stocks | ETFs | universe]."

This is the default path for most research. Ask the API for daily bars
directly — don't pull minutes and downsample.

## Minimal version

```python
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
)

req = StockBarsRequest(
    symbol_or_symbols=["SPY", "QQQ", "IWM", "TLT"],
    timeframe=TimeFrame.Day,
    start="2020-01-01",
    end  ="2024-12-31",
    adjustment="all",
    feed="iex",
)
df = client.get_stock_bars(req).df
```

Output: DataFrame with MultiIndex `(symbol, timestamp)`, columns
`open, high, low, close, volume, trade_count, vwap`.

## Ready-to-use function

```python
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = StockHistoricalDataClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
        )
    return _client

def fetch_daily_bars(symbols, start, end=None, adjustment="all"):
    """Fetch adjusted daily bars for a list of symbols (stocks or ETFs).

    Returns a DataFrame with MultiIndex (symbol, timestamp).
    Omit `end` to let the API default to 15 min ago (safe on free tier).
    """
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=adjustment,
        feed="iex",
    )
    return _get_client().get_stock_bars(req).df

df = fetch_daily_bars(["SPY", "QQQ"], "2020-01-01", "2024-12-31")
```

## Common reshapes

```python
# Wide form, one column per symbol
close = df["close"].unstack("symbol")

# Simple returns
rets = close.pct_change().dropna()

# Log returns
import numpy as np
log_rets = np.log(close).diff().dropna()

# Align to NY trading dates (drop the UTC time-of-day component)
close.index = close.index.tz_convert("America/New_York").normalize().tz_localize(None)
```

## Single-symbol slice

```python
spy = df.xs("SPY", level="symbol")
spy["ret"] = spy["close"].pct_change()
```

## Budget check

Daily bars are small: ~252 rows per symbol per year. 10 symbols × 5 years =
12.6k rows, one API call, one page, done in <1 second. You can request
hundreds of symbols across decades without issue.

Persist to disk if you're iterating on analyses and don't want to re-fetch:

```python
df.to_parquet("bars.parquet")
# later
import pandas as pd
df = pd.read_parquet("bars.parquet")
```

## Handling symbols with partial history

If a symbol didn't exist for the full range, its bars start from its listing
date. The wide form handles this cleanly — missing days are `NaN`, not zero:

```python
close = df["close"].unstack("symbol")
# NVDA has bars back to 2017-09; a symbol IPO'd in 2022 will have
# NaNs before 2022 in this frame. Don't fillna(0) — use dropna() per
# operation, or compute returns before aligning.
```
