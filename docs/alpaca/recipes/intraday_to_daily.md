# Recipe: Resample Intraday Bars to Daily

**First, a check:** do you actually need to do this? If you want daily bars
for research, ask the API for daily bars (`TimeFrame.Day`) — it's one call,
one page, and the aggregation is done server-side from the full trade tape.

You should only resample locally when:

1. You already have intraday bars in memory (pulled for some other reason)
   and want daily stats without a second API call.
2. You need a **non-standard daily boundary** — e.g., regular session only
   (9:30 AM – 4:00 PM ET), excluding pre/post-market; or a custom "day"
   that spans 6 AM – 6 PM.
3. You need **aggregations the API doesn't provide** — median bar price,
   first/last 30-minute-only stats, time-weighted averages, etc.

Otherwise, use `recipes/daily_bars.md`.

## The aggregations that matter

Daily OHLCV from intraday bars requires a **different aggregation per column**:

| Column        | How to aggregate across a day                            |
|---------------|----------------------------------------------------------|
| `open`        | first value of the day                                   |
| `high`        | max of intraday highs                                    |
| `low`         | min of intraday lows                                     |
| `close`       | last value of the day                                    |
| `volume`      | sum                                                      |
| `trade_count` | sum                                                      |
| `vwap`        | **volume-weighted mean**, not arithmetic mean (see below) |

A plain `df.resample("1D").mean()` will quietly ruin your OHLC. Always
specify the aggregation per column.

## Pull intraday and resample to daily

```python
import os, pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
)

req = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start="2024-10-01",
    end  ="2024-10-31",
    adjustment="all",
    feed="iex",
)
minutes = client.get_stock_bars(req).df
```

### Standard resampling (all session hours included)

```python
def resample_to_daily(df):
    """Aggregate intraday bars to daily. Expects MultiIndex (symbol, timestamp)
    or a single DatetimeIndex."""
    agg = {
        "open":        "first",
        "high":        "max",
        "low":         "min",
        "close":       "last",
        "volume":      "sum",
        "trade_count": "sum",
    }
    if isinstance(df.index, pd.MultiIndex):
        # Group by symbol, then resample time within each group
        daily = (
            df.groupby(level="symbol")
              .resample("1D", level="timestamp")
              .agg(agg)
        )
    else:
        daily = df.resample("1D").agg(agg)

    # VWAP: volume-weighted mean of intraday VWAPs
    # ∑(vwap_i × volume_i) / ∑(volume_i)
    if "vwap" in df.columns:
        weighted = df["vwap"] * df["volume"]
        if isinstance(df.index, pd.MultiIndex):
            num = weighted.groupby(level="symbol").resample("1D", level="timestamp").sum()
            den = df["volume"].groupby(level="symbol").resample("1D", level="timestamp").sum()
        else:
            num = weighted.resample("1D").sum()
            den = df["volume"].resample("1D").sum()
        daily["vwap"] = num / den.replace(0, pd.NA)

    # Drop days that had no bars (weekends, holidays)
    daily = daily.dropna(subset=["open"])
    return daily

daily = resample_to_daily(minutes)
```

### Regular session only (9:30 AM – 4:00 PM ET)

Sometimes you want daily bars that reflect **only** regular trading hours,
with pre- and post-market stripped. Filter before resampling:

```python
def resample_regular_session(df):
    """Same as resample_to_daily but restricts to 9:30–16:00 NY time."""
    # Get NY-local timestamps for filtering
    if isinstance(df.index, pd.MultiIndex):
        ts = df.index.get_level_values("timestamp").tz_convert("America/New_York")
    else:
        ts = df.index.tz_convert("America/New_York")

    # 9:30 <= time < 16:00
    minutes_of_day = ts.hour * 60 + ts.minute
    mask = (minutes_of_day >= 570) & (minutes_of_day < 960)   # 9:30 … 16:00

    return resample_to_daily(df[mask])

daily_rth = resample_regular_session(minutes)
```

### Custom session window (e.g., first 30 minutes, or last hour)

Filter to the window you care about, then aggregate to daily the same way:

```python
# First 30 minutes: 9:30–10:00 ET
ts = minutes.index.get_level_values("timestamp").tz_convert("America/New_York")
mod = ts.hour * 60 + ts.minute
opening_30m = minutes[(mod >= 570) & (mod < 600)]

opening_daily = resample_to_daily(opening_30m)
# opening_daily.open = price at 9:30
# opening_daily.close = price at 9:59
# opening_daily.volume = 30-minute total
```

## Sanity-check the result

After resampling, a quick comparison against API-provided daily bars catches
most mistakes — they should agree on `open` and `close` within rounding when
using the same session boundaries:

```python
api_daily = client.get_stock_bars(StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Day,
    start="2024-10-01", end="2024-10-31",
    adjustment="all", feed="iex",
)).df.xs("SPY", level="symbol")

local_daily = resample_to_daily(minutes).xs("SPY", level="symbol") \
    if isinstance(resample_to_daily(minutes).index, pd.MultiIndex) \
    else resample_to_daily(minutes)

# API daily bars use the full trade tape; local resampling uses only the bars
# you pulled. On IEX they should match closely in open/close, and differ in
# volume (resampled sum matches IEX-only; API daily is also IEX-only here).
compare = local_daily[["open", "close"]].join(
    api_daily[["open", "close"]], lsuffix="_local", rsuffix="_api"
)
```

## Notes

- `resample("1D")` groups by **UTC midnight boundaries**, not NY midnight.
  For daily bars aligned to NY trading days, convert the index first:
  ```python
  df = df.tz_convert("America/New_York")
  daily = df.resample("1D").agg(agg)
  ```
- Overnight gaps (yesterday's close → today's open) are a feature, not a
  bug — daily bars should reflect them.
- If you're resampling minute bars to hourly instead of daily, the same
  aggregation dict works; just change `resample("1H")`.
