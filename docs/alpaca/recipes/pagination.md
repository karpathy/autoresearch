# Recipe: Pagination

Alpaca caps every historical endpoint at 10,000 rows per page (default 1,000)
and returns a `next_page_token`. The limit is **across all symbols in one
request**, not per symbol — if you ask for 1,000 bars of `[AAPL, MSFT]` and
AAPL alone has 900 in range, you will get 900 AAPL + 100 MSFT on page 1,
then the rest of MSFT on page 2.

## With alpaca-py (the easy case)

The SDK paginates automatically when you don't pass `limit`:

```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

req = StockBarsRequest(
    symbol_or_symbols=["AAPL", "MSFT"],
    timeframe=TimeFrame.Minute,
    start="2024-01-01",
    end  ="2024-12-31",
    feed="iex",
)
df = client.get_stock_bars(req).df   # all pages concatenated
```

That's the happy path. It works for any single call. Two reasons you'd need
to paginate manually:

1. You want to process each page as it arrives (save to disk, compute
   rolling stats) rather than hold the full result in memory.
2. You're calling raw REST because the SDK doesn't expose a parameter you need.

## Raw REST pagination

```python
import os, time, requests

HEADERS = {
    "APCA-API-KEY-ID":     os.environ["ALPACA_API_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
}

def paginate(url, params, throttle=0.3):
    """Yield successive response bodies, handling rate limits."""
    params = dict(params)
    while True:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 1))
            time.sleep(max(1, reset - int(time.time())))
            continue
        r.raise_for_status()
        body = r.json()
        yield body
        token = body.get("next_page_token")
        if not token:
            return
        params["page_token"] = token
        if throttle:
            time.sleep(throttle)

url = "https://data.alpaca.markets/v2/stocks/bars"
params = {
    "symbols":    "AAPL,MSFT,NVDA",
    "timeframe":  "1Day",
    "start":      "2020-01-01",
    "end":        "2024-12-31",
    "adjustment": "all",
    "feed":       "iex",
    "limit":      10000,
}

all_bars = []
for page in paginate(url, params):
    for symbol, bars in page["bars"].items():
        for b in bars:
            b["symbol"] = symbol
            all_bars.append(b)
```

## Streaming each page to disk (large pulls)

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

writer = None
schema = None
out = Path("bars.parquet")

try:
    for page in paginate(url, params):
        rows = []
        for symbol, bars in page["bars"].items():
            for b in bars:
                rows.append({**b, "symbol": symbol})
        if not rows:
            continue
        table = pa.Table.from_pylist(rows)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out, schema)
        writer.write_table(table.cast(schema))
finally:
    if writer:
        writer.close()
```

Peak memory stays bounded by one page (≤10k rows) instead of the full result.

## Cursor gotchas

- `next_page_token` is opaque. Don't parse or guess at it.
- Tokens are tied to the original query parameters. If you change `start`,
  `end`, `symbols`, etc. between pages, the token becomes invalid.
- If the API returns `next_page_token: null` **and** zero rows, the symbol
  has no data in the range — not an error.
- On the free tier, bulk pulls that span hundreds of pages can exhaust the
  200 req/min limit. With `throttle=0.3` you'll do ~200 pages/min — just
  at the ceiling. Raise to `0.5` for comfortable headroom.
