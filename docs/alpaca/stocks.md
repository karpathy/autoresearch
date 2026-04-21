# Stocks & ETFs — Historical Market Data

Read `core.md` first.

```python
from alpaca.data.historical import StockHistoricalDataClient
client = StockHistoricalDataClient(key, secret)
```

Applies equally to stocks (`AAPL`, `MSFT`) and ETFs (`SPY`, `QQQ`, `XLK`,
`TLT`). Same endpoints, same symbol format, same request classes.

## Endpoint map

| What you need              | SDK method                     | Request class              |
|----------------------------|--------------------------------|----------------------------|
| OHLCV bars (history)       | `get_stock_bars`               | `StockBarsRequest`         |
| Individual trades          | `get_stock_trades`             | `StockTradesRequest`       |
| NBBO quotes                | `get_stock_quotes`             | `StockQuotesRequest`       |
| Latest trade (1 symbol)    | `get_stock_latest_trade`       | `StockLatestTradeRequest`  |
| Latest quote (1 symbol)    | `get_stock_latest_quote`       | `StockLatestQuoteRequest`  |
| Latest minute bar          | `get_stock_latest_bar`         | `StockLatestBarRequest`    |
| Snapshot (quote+trade+bar) | `get_stock_snapshot`           | `StockSnapshotRequest`     |

## Bars — the primary endpoint

Most research starts here. The API aggregates trades into bars at whatever
timeframe you request — **daily bars are a first-class API call, not
something you build from minutes**.

```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

req = StockBarsRequest(
    symbol_or_symbols=["SPY", "QQQ", "AAPL"],
    timeframe=TimeFrame.Day,           # the common case
    start="2020-01-01",
    end="2024-12-31",
    adjustment="all",
    feed="iex",
)
df = client.get_stock_bars(req).df
```

Output: pandas DataFrame with MultiIndex `(symbol, timestamp)`, columns
`open, high, low, close, volume, trade_count, vwap`.

### Timeframes the API supports

| Constant                   | Produces                                         |
|----------------------------|--------------------------------------------------|
| `TimeFrame.Day`            | 1 bar per US trading day                         |
| `TimeFrame.Hour`           | 1-hour bars                                      |
| `TimeFrame.Minute`         | 1-minute bars                                    |
| `TimeFrame.Week`           | 1 bar per calendar week                          |
| `TimeFrame.Month`          | 1 bar per calendar month                         |
| `TimeFrame(5, TimeFrameUnit.Minute)` | 5-minute bars                          |
| `TimeFrame(15, TimeFrameUnit.Minute)` | 15-minute bars                        |

Valid ranges (from the REST spec): `[1–59]Min`, `[1–23]Hour`, `1Day`,
`1Week`, `[1,2,3,4,6,12]Month`.

**Ask for the timeframe you need.** Don't pull minutes and downsample unless
you need a custom session boundary (extended hours, half-days handled a
specific way) or a non-standard aggregation. See
`recipes/intraday_to_daily.md` for those cases.

### Bar columns and what to trust on IEX

| Column         | Trust on IEX                                             |
|----------------|----------------------------------------------------------|
| `open`, `close`  | Reliable reference prices                              |
| `high`, `low`    | Reliable as high/low **of IEX trades** — usually close to market high/low but not guaranteed to match the consolidated tape exactly. Fine for signals, suspect for point-in-time extremes |
| `volume`       | IEX-only (~2.5% of market). Don't treat as market volume |
| `trade_count`  | IEX-only trade count                                     |
| `vwap`         | IEX-only VWAP. Useful for relative work within IEX, not as a market benchmark |

## Reshape and work with the result

```python
# Drop the MultiIndex to wide form (close-only)
close = df["close"].unstack("symbol")   # DatetimeIndex × symbol columns
rets  = close.pct_change().dropna()

# Single symbol slice
spy = df.xs("SPY", level="symbol")

# Align to NY trading days (normalize UTC timestamps to session dates)
close.index = close.index.tz_convert("America/New_York").normalize()
```

## Trades and quotes — tick data

Only reach for these when you specifically need tick-level data
(microstructure studies, trade-imbalance signals). Volume is huge;
narrow the time range aggressively.

```python
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest

# One minute of AAPL trades on IEX
trades = client.get_stock_trades(StockTradesRequest(
    symbol_or_symbols="AAPL",
    start="2024-11-04T14:30:00Z",
    end  ="2024-11-04T14:31:00Z",
    feed="iex",
)).df

quotes = client.get_stock_quotes(StockQuotesRequest(
    symbol_or_symbols="AAPL",
    start="2024-11-04T14:30:00Z",
    end  ="2024-11-04T14:31:00Z",
    feed="iex",
)).df
```

IEX shows only its own trades/quotes, so these datasets are dramatically
thinner than SIP equivalents. For most intraday strategy research, minute
bars are the right granularity and trades/quotes are overkill.

## Latest and snapshot — cheap one-shot lookups

```python
from alpaca.data.requests import StockSnapshotRequest

snap = client.get_stock_snapshot(
    StockSnapshotRequest(symbol_or_symbols=["SPY", "QQQ"])
)
# dict[str, Snapshot] — each Snapshot has:
#   .latest_trade, .latest_quote, .minute_bar, .daily_bar, .previous_daily_bar
spy = snap["SPY"]
print(spy.daily_bar.close, spy.previous_daily_bar.close)
```

Useful for: "where's SPY right now", "what was yesterday's close",
"is this symbol still trading". One call, no pagination.

## Practical notes

- **History depth.** IEX data starts around **September 2017** for most
  symbols; earlier history requires a different vendor. Symbols listed
  after 2017 have data from their IPO/listing date.
- **Halts and inactive symbols** return empty bar sets rather than errors.
  If a response has zero rows, check `/v2/assets/<symbol>` for `status`
  and `tradable`.
- **Extended hours.** By default, bars cover 4:00 AM — 8:00 PM ET
  (pre-market through after-hours). For regular session only (9:30 AM —
  4:00 PM ET), filter after fetching:

  ```python
  ny = df.index.get_level_values("timestamp").tz_convert("America/New_York")
  regular_session = df[(ny.hour >= 9) & (ny.hour < 16) & ~((ny.hour == 9) & (ny.minute < 30))]
  ```

- **`trade_count` and `vwap` are `NaN` on IEX bars before 2019.** Don't
  rely on them for pre-2019 historical studies without spot-checking.
