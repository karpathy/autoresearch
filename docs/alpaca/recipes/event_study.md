# Recipe: Event Study

Task pattern: "How does [ticker | universe] behave around [earnings | FOMC |
M&A announcements]?" Given a list of `(symbol, event_date)` tuples, pull a
window of bars around each event and align them on event-relative time.

## The core pattern

```python
import os, pandas as pd
from datetime import timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
)

events = [
    ("AAPL", "2024-02-01"),
    ("AAPL", "2024-05-02"),
    ("AAPL", "2024-08-01"),
    ("AAPL", "2024-10-31"),
    ("MSFT", "2024-01-30"),
    ("MSFT", "2024-04-25"),
    # ...
]

WINDOW_BEFORE = 10   # trading days before the event
WINDOW_AFTER  = 10   # trading days after

def event_window(symbol, event_date, before=WINDOW_BEFORE, after=WINDOW_AFTER):
    """Daily bars with a `t` column in trading days relative to the event.
    t=0 is the event day (or the next trading day if the event is a weekend)."""
    ev = pd.Timestamp(event_date)
    # Overfetch calendar days to cover weekends/holidays, then trim to N trading days
    start = (ev - timedelta(days=before * 2 + 10)).strftime("%Y-%m-%d")
    end   = (ev + timedelta(days=after  * 2 + 10)).strftime("%Y-%m-%d")

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start, end=end,
        adjustment="all",
        feed="iex",
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        return None
    df = df.xs(symbol, level="symbol")
    df.index = df.index.tz_convert("America/New_York").normalize().tz_localize(None)

    idx = df.index.searchsorted(ev)
    if idx >= len(df):
        return None
    event_idx = idx
    lo = max(0, event_idx - before)
    hi = min(len(df), event_idx + after + 1)
    window = df.iloc[lo:hi].copy()
    window["t"] = range(lo - event_idx, hi - event_idx)
    window["symbol"] = symbol
    window["event_date"] = ev
    return window
```

## Collect and stack

```python
import time

panels = []
for sym, date in events:
    w = event_window(sym, date)
    if w is not None and 0 in w["t"].values:
        panels.append(w)
    time.sleep(0.05)   # stay under 200 req/min on the free tier

panel = pd.concat(panels, ignore_index=False)
```

## Compute event-relative returns

```python
panel = panel.sort_values(["symbol", "event_date", "t"])
panel["ret"] = (
    panel.groupby(["symbol", "event_date"])["close"]
         .pct_change()
)

# Cumulative return from start of window
panel["cum_ret"] = (
    panel.groupby(["symbol", "event_date"])["ret"]
         .cumsum()
         .fillna(0)
)

# Average cumulative return by event-relative day
car = panel.groupby("t")["cum_ret"].mean()
```

`car` is a Series indexed -10…+10 of the average cumulative return across
all events. Plot it, or apply a significance test.

## Scaling up

- For >50 events, one `event_window` call each still stays under the 200
  req/min limit with the small `sleep` above.
- To minimize API calls entirely: pull each symbol's full history over the
  union of event dates ± window once, cache to Parquet, then do all
  windowing in pandas:

  ```python
  per_symbol = {}
  for sym in set(s for s, _ in events):
      dates_for_sym = [d for s, d in events if s == sym]
      req = StockBarsRequest(
          symbol_or_symbols=sym,
          timeframe=TimeFrame.Day,
          start=min(dates_for_sym), end=max(dates_for_sym),
          adjustment="all", feed="iex",
      )
      per_symbol[sym] = client.get_stock_bars(req).df.xs(sym, level="symbol")
  # then slice windows from each DataFrame locally
  ```

## Notes

- The "event date" convention matters: is it the announcement date (usually
  after close) or the first trading day after? Decide once and apply
  consistently. `searchsorted` above picks the first trading day on or
  after the event — if your announcements happen after-hours, that's
  the right choice.
- If two events for the same symbol are closer than `before + after` trading
  days, their windows overlap. That's fine for averaging but will bias
  any test that assumes independence — dedupe if needed.
- IEX volume is unreliable, so **any event study involving volume**
  (abnormal volume, turnover spikes) is problematic on the free tier.
  Price-based studies (CAR, drift, volatility around the event) are fine.
