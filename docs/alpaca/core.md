# Core — Auth, Endpoints, Feed, Limits

Applies to every equity data request. Read this before `stocks.md`.

## Auth

```python
import os
from alpaca.data.historical import StockHistoricalDataClient

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
)
```

The same credentials work whether you have a live account or a paper account
— market data is read-only and doesn't distinguish.

## Base URL (if calling REST directly)

```
https://data.alpaca.markets/v2/stocks/…
```

Prefer the SDK. Drop to raw REST only when you need a parameter the SDK
doesn't expose or you're debugging a response shape.

## Feed: IEX

On the free tier you use the **IEX** feed (Investors Exchange), a single US
exchange accounting for ~2.5% of consolidated trading volume. What this means
in practice:

| Field                     | Reliability on IEX                               |
|---------------------------|--------------------------------------------------|
| `open`, `high`, `low`, `close` | Accurate reference prices — IEX saw these trades, and the print reflects the market-wide level at those moments. Fine for any price-based signal. |
| `volume`                  | Only IEX's share (~2.5% of market). Use for relative comparisons on IEX, not absolute liquidity. |
| `trade_count`             | Only trades executed on IEX. Often zero or very low for mid/small caps. |
| `vwap`                    | IEX-only VWAP. Usable for relative work, not as a benchmark price. |

**Rule of thumb:** if the signal would still work on a less-liquid exchange,
IEX data is fine. If the signal depends on seeing the majority of the tape
(VPIN, volume-profile indicators, VWAP-benchmarked execution analysis),
IEX will give you misleading results.

## Rate limits

- Free tier: **200 requests/minute**.
- On `429`, back off. The response headers `X-RateLimit-Remaining` and
  `X-RateLimit-Reset` (Unix timestamp) tell you where you are.
- A large historical pull that paginates many times can hit the limit
  quickly. Sleep ~0.3s between page requests to stay well under.

## Pagination — always required

Every historical endpoint returns at most `limit` rows per page (max 10,000,
default 1,000) plus a `next_page_token`. The limit is **total rows across
all symbols**, not per-symbol — if you ask for 1,000 rows of bars for
`[AAPL, MSFT]` and AAPL alone has 1,000 in range, you get only AAPL on
page 1 and MSFT on later pages.

The SDK handles this for you automatically when you don't pass `limit`:

```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

req = StockBarsRequest(
    symbol_or_symbols=["AAPL", "MSFT"],
    timeframe=TimeFrame.Day,
    start="2024-01-01",
    end="2024-12-31",
    adjustment="all",
    feed="iex",
)
df = client.get_stock_bars(req).df   # all pages concatenated
```

If you paginate manually (raw REST or processing page-by-page for memory),
see `recipes/pagination.md`.

## Time handling

- Timestamps returned by the API are UTC, RFC-3339.
- Bars are labelled with the **left edge** of their interval.
- Daily bars roll over at midnight NY-local time. A daily bar labelled
  `2024-03-14T04:00:00Z` corresponds to the US trading day of 2024-03-14
  (the offset shifts to 05:00Z outside DST).
- `start` / `end` accept `YYYY-MM-DD` (midnight UTC) or full RFC-3339.
  Both bounds are inclusive.

### The 15-minute rule

Recent trade data on IEX is embargoed by 15 minutes. If you pass an `end`
within the last 15 minutes, you will get a 403 with
`subscription does not permit querying recent SIP data`. Workarounds:

```python
from datetime import datetime, timedelta, timezone

# Explicit: 16 minutes ago, UTC
end = datetime.now(timezone.utc) - timedelta(minutes=16)

# Or simply omit `end` — the API defaults to "15 min ago" for free-tier users.
req = StockBarsRequest(
    symbol_or_symbols="SPY",
    timeframe=TimeFrame.Day,
    start="2024-01-01",
    # end omitted
    adjustment="all",
    feed="iex",
)
```

For daily-bar research this rule is never a problem — you're never pulling
data less than 15 minutes old.

## Failure modes — what the error actually means

| Status | Cause and fix                                                     |
|--------|-------------------------------------------------------------------|
| 400    | Bad parameter. Check `timeframe` syntax, symbol format, date range. |
| 401    | Missing or malformed auth headers. Check env vars.               |
| 403 + `subscription does not permit`  | You queried recent (<15min) data or requested a feed you're not subscribed to. Push `end` back 15 min or omit it. |
| 403 + other | Symbol may require a feed you don't have (OTC, institutional).  |
| 404    | Symbol not recognized. Check capitalization and OTC status (`/v2/assets/<symbol>`). |
| 429    | Rate limit. Back off to under 200 req/min.                       |
| 500    | Retry with exponential backoff, max 3 tries.                     |

## Symbol renames (the `asof` parameter)

When a company renames (FB → META on 2022-06-09), a historical query for
the new symbol `META` returns the old `FB` bars as well, labelled as `META`
— by default. This is usually what you want for backtests.

If you specifically need history unmapped (only bars that traded under the
queried symbol), pass `asof='-'`.

## Price adjustments (the `adjustment` parameter)

| Value       | Adjusts for                          |
|-------------|--------------------------------------|
| `raw`       | Nothing (the API default)            |
| `split`     | Forward and reverse splits           |
| `dividend`  | Cash dividends and ETF distributions |
| `spin-off`  | Spin-offs                            |
| `all`       | All of the above                     |

**Use `all` for any study longer than a few weeks.** Using `raw` on a
dividend-paying stock or ETF inserts spurious gap-downs on every ex-div date,
which will corrupt return series, moving averages, and anything else that
assumes price continuity.
