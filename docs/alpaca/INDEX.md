# Alpaca Market Data — Agent Reference

You are a quant research agent. To pull market data you write and execute Python
against the Alpaca Market Data API. Do not ask the user for data; fetch it.

## How to use these docs

Load files on demand. Do not read everything upfront.

1. **Always read first (small, required):**
   - `core.md` — auth, feeds, rate limits, pagination, time handling, failure modes.
2. **Read when working with equities:**
   - `stocks.md` — bars, trades, quotes, latest, snapshots for stocks and ETFs.
3. **Read when the task matches the pattern:**
   - `recipes/daily_bars.md` — the default path. Pull daily OHLCV for a universe.
   - `recipes/intraday_to_daily.md` — resample intraday bars to daily *locally*
     when you need custom session boundaries or aggregations the API doesn't provide.
   - `recipes/event_study.md` — fetch aligned bars around a list of event dates.
   - `recipes/pagination.md` — correctly iterate `next_page_token` for large pulls.

## Environment

- `alpaca-py` is already installed. Do not `pip install`.
- Credentials live in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (env vars).

## Equities coverage — includes ETFs

The equity endpoints cover both common stocks and ETFs with no distinction.
`SPY`, `QQQ`, `XLK`, `TLT`, `TQQQ` are all valid symbols on the same
`/v2/stocks/…` endpoints as `AAPL` or `MSFT`. Price adjustments (`all`) apply
to ETF distributions the same way they do to stock dividends and splits.

One thing to remember: **leveraged and inverse ETFs** (`TQQQ`, `SQQQ`, `SOXL`,
etc.) reset their leverage daily. Even fully adjusted, their long-horizon
returns diverge from the underlying index — that's a product feature, not a
data problem.

## Defaults to assume unless the task says otherwise

- Library: `alpaca-py`.
- Feed: `iex` (no subscription available).
- Output: pandas DataFrame, UTC `DatetimeIndex`.
- Timeframe: **daily** (`TimeFrame.Day`). Request daily bars directly from the
  API — don't pull minutes and downsample unless there's a specific reason.
- Adjustment: `all` (splits + dividends + spin-offs). Required for any backtest
  or cross-date comparison.

## Non-negotiables

- **Never hardcode credentials.** Read from env vars.
- **Always paginate.** Page size caps at 10,000 rows (default 1,000). Loop on
  `next_page_token` until it is `None`.
- **On IEX, `end` must be ≥15 min in the past** for any query including recent
  data. Passing `datetime.now()` will 403. Use `datetime.utcnow() - timedelta(minutes=16)`
  or omit `end` to let the API default it.
- **IEX volume is not the full market volume.** It's ~2.5% of consolidated
  US volume. Close/open/high/low are accurate reference prices, but any signal
  that uses `volume`, `vwap`, or `trade_count` will be noisy. Avoid
  volume-weighted or liquidity-based signals unless you've sanity-checked them.

## When to stop and ask the user

- Credentials are missing from the environment.
- The requested history exceeds what IEX has for a given symbol (IEX data
  starts 2017-09 for most symbols; earlier history needs a different source).
- The task would return more than ~500k rows. Ask whether to narrow the
  universe, raise the timeframe, or persist to Parquet page-by-page.
- The task depends on accurate intraday volume (VWAP execution studies,
  liquidity signals, tape-reading). IEX can't deliver this honestly.
