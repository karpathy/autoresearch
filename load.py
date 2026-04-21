"""
Canonical data-loading API for the research agent.

Expose a fixed universe (current S&P 500 + a curated set of ~200 liquid ETFs)
at daily or 30-minute resolution (OHLCV + trade_count + vwap). Pulls from
Alpaca on demand and caches per-symbol under ``data/cache/<timeframe>/`` so
subsequent calls are instant.

Agent usage
-----------

    from load import Data

    data = Data(split="train")
    data.universe()                # -> list[str], every available symbol
    data.universe("sp500")         # -> S&P 500 only
    data.universe("etf")           # -> curated ETFs only

    bars = data.bars("SPY")                             # one symbol, daily
    bars = data.bars(["AAPL", "MSFT", "NVDA"])          # many symbols, daily
    bars = data.bars(["KO", "PEP"], timeframe="30min")  # 30-minute bars

    # Shape: MultiIndex (symbol, timestamp), columns:
    #   open, high, low, close, volume, trade_count, vwap

Supported timeframes: ``"day"`` and ``"30min"``.

30-minute bars are restricted to the US regular trading session (9:30am-4:00pm
ET). Bars are labelled on their left edge: 9:30, 10:00, 10:30, 11:00, ..., 15:30
ET -- thirteen bars per full trading day (fewer on early-close days like the
Friday after Thanksgiving). Every returned bar is a clean session window:
the 9:30 bar's `open` is the regular-session opening print, and the 15:30
bar's `close` is the 16:00 regular-session closing print. No extended-hours
contamination.

Because only in-session bars are returned, any timestamp from
``data.bars(..., "30min")`` is a valid trade time. Trading is implicitly
disabled outside market hours.

Split boundary (hard)
---------------------

``Data(split="train")`` returns bars from 2017-01-01 through 2022-12-31.
``Data(split="test")`` returns bars from 2023-01-01 through 2025-12-31, but
refuses to construct unless ``AUTORESEARCH_ALLOW_TEST=1`` is set in the
environment. Only the evaluation harness sets this; the training loop never
sees test data.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_ROOT = DATA_DIR / "cache"

_TIMEFRAMES = {
    "day":   (TimeFrame.Day,                            CACHE_ROOT / "day"),
    "30min": (TimeFrame(30, TimeFrameUnit.Minute),      CACHE_ROOT / "30min"),
}

# Full fetched range. Train/test windows are sliced from this single per-symbol
# cache at read time, so each symbol is pulled from Alpaca at most once.
_FETCH_START = datetime(2017, 1, 1,  tzinfo=timezone.utc)
_FETCH_END   = datetime(2025, 12, 31, tzinfo=timezone.utc)

# Regular US session in ET, expressed as HHMM ints on left-edge bar labels:
# 9:30, 10:00, 10:30, ..., 15:30 (inclusive). The 15:30 bar covers 15:30-16:00
# and closes exactly at the bell; the 16:00 bar (closing-auction print) is
# excluded so every returned timestamp is tradable in regular hours.
_SESSION_OPEN_HHMM = 930
_SESSION_LAST_OPEN_HHMM = 1530

SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp("2017-01-01", tz="UTC"),
              pd.Timestamp("2022-12-31 23:59:59", tz="UTC")),
    "test":  (pd.Timestamp("2023-01-01", tz="UTC"),
              pd.Timestamp("2025-12-31 23:59:59", tz="UTC")),
}

_ALLOW_TEST_ENV = "AUTORESEARCH_ALLOW_TEST"

_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

_BATCH_SIZE = 50
_SLEEP_BETWEEN_BATCHES = 0.3  # free tier rate limit is 200 req/min

_SP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
_SP500_CACHE = DATA_DIR / "sp500.csv"


# --------------------------------------------------------------------------- #
# Curated ETF list
# --------------------------------------------------------------------------- #
#
# Chosen to give the agent broad cross-asset optionality:
# equities (broad / style / sector / industry / international), bonds
# (government / corporate / HY / TIPS / muni / EM), commodities, currencies,
# volatility, leveraged & inverse products. Roughly ordered by category.
# ETFs launched after ~2020 have truncated train history; the agent's code
# should expect missing data on the left edge.

CURATED_ETFS: tuple[str, ...] = (
    # Broad US equity
    "SPY", "VOO", "IVV", "VTI", "ITOT", "SCHB", "SPLG",
    "QQQ", "QQQM", "DIA", "IWB", "IWM", "IWV", "MDY", "IJH", "IJR",
    "VO", "VB", "SCHA", "SCHM", "SCHX", "RSP",

    # Style / factor
    "VTV", "VUG", "IWD", "IWF", "IWN", "IWO", "IWS", "IWP",
    "VBR", "VBK", "VOE", "VOT", "MGK", "MGV",
    "MTUM", "QUAL", "VLUE", "USMV", "SPLV", "SPHQ", "SPHB",
    "DVY", "SCHD", "VIG", "NOBL", "VYM", "HDV", "DGRO",

    # Sector SPDRs
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLU", "XLI", "XLB", "XLRE", "XLC",
    # Vanguard sector mirrors
    "VGT", "VFH", "VHT", "VCR", "VDC", "VDE", "VPU", "VIS", "VAW", "VNQ", "VOX",

    # Industry / thematic equity
    "SMH", "SOXX", "XSD", "PSI",
    "XBI", "IBB", "IHI", "IHF", "XPH",
    "KRE", "KBE", "KIE", "IAT",
    "IYR", "REZ", "REM",
    "ITA", "XAR", "PPA",
    "XRT", "XHB", "ITB", "XHE",
    "XME", "PICK", "COPX", "REMX",
    "GDX", "GDXJ", "SIL", "SILJ", "NUGT", "JNUG",
    "XOP", "OIH", "FRAK",
    "TAN", "ICLN", "QCLN", "PBW", "FAN", "URA", "URNM", "LIT",
    "HACK", "CIBR", "FDN", "IGV", "IPAY", "SKYY", "IPO", "MOAT",
    "JETS", "AWAY", "PEJ",
    "KWEB", "EMQQ",
    "ROBO", "BOTZ", "AIQ", "IRBO", "HERO", "ESPO",
    "BLOK", "FINX",
    "BITO",

    # ARK
    "ARKK", "ARKG", "ARKW", "ARKQ", "ARKF",

    # International — developed
    "EFA", "VEA", "IEFA", "VXUS", "SCHF",
    "VGK", "IEV", "EZU", "HEDJ",
    "EWJ", "DXJ", "HEWJ",
    "EWU", "EWG", "EWQ", "EWI", "EWP", "EWD", "EWN", "EWL",
    "EWC", "EWA", "EWH", "EWS", "EWY", "EWT",

    # International — emerging
    "EEM", "VWO", "IEMG", "SCHE", "EEMV", "EEMA",
    "FXI", "MCHI", "ASHR", "CQQQ", "YINN", "YANG",
    "INDA", "INDY", "EWZ", "EZA", "EWW", "TUR", "ECH", "EPOL",

    # Treasuries / duration
    "SHY", "SHV", "BIL", "SGOV",
    "IEI", "VGSH",
    "IEF", "VGIT",
    "TLH", "TLT", "VGLT", "EDV", "ZROZ",
    "GOVT",

    # Credit — IG / HY / bank loans
    "AGG", "BND", "BIV", "BSV", "SCHZ",
    "LQD", "VCIT", "VCSH", "VCLT", "IGIB", "IGSB",
    "HYG", "JNK", "SHYG", "USHY", "FALN", "ANGL",
    "BKLN", "SRLN",

    # Inflation / muni / global
    "TIP", "VTIP", "SCHP", "STIP",
    "MUB", "VTEB", "SUB", "HYD",
    "EMB", "PCY", "EMLC", "BNDX", "IGOV",

    # Commodities
    "GLD", "IAU", "GLDM", "SGOL",
    "SLV", "SIVR",
    "PPLT", "PALL",
    "USO", "BNO", "UNG", "UGA",
    "DBC", "PDBC", "GSG", "COMT",
    "DBA", "CORN", "WEAT", "SOYB", "CANE",
    "DBE", "DBO", "DBB",

    # Currency
    "UUP", "UDN", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF", "CYB", "CEW",

    # Volatility
    "VXX", "VIXY", "UVXY", "SVXY", "VXZ",

    # Leveraged / inverse — broad equity
    "SSO", "SDS", "UPRO", "SPXU", "SPXL", "SPXS",
    "QLD", "QID", "TQQQ", "SQQQ",
    "DDM", "DXD", "UDOW", "SDOW",
    "UWM", "TWM", "TNA", "TZA",
    "SH", "PSQ", "DOG", "RWM",

    # Leveraged / inverse — sectors / niches
    "SOXL", "SOXS", "LABU", "LABD",
    "FAS", "FAZ", "DRN", "DRV",
    "ERX", "ERY", "DUST", "JDST",
    "EDC", "EDZ",

    # Leveraged / inverse — rates
    "TMF", "TMV", "UBT", "TBT", "TBF",

    # Preferreds / MLPs / specialty
    "PFF", "PGX", "PGF", "PFFD",
    "AMLP", "MLPX",
    "MORT",
    "WOOD", "CUT",
)


# --------------------------------------------------------------------------- #
# Universe resolution
# --------------------------------------------------------------------------- #

_SP500_CACHE_MEM: list[str] | None = None


def _sp500_symbols() -> list[str]:
    """Current S&P 500 constituents. Cached both on disk and in memory."""
    global _SP500_CACHE_MEM
    if _SP500_CACHE_MEM is not None:
        return _SP500_CACHE_MEM

    if _SP500_CACHE.exists():
        df = pd.read_csv(_SP500_CACHE)
    else:
        df = pd.read_csv(_SP500_URL)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_SP500_CACHE, index=False)

    sym_col = "Symbol" if "Symbol" in df.columns else "symbol"
    syms = (
        df[sym_col]
        .astype(str)
        .str.replace("-", ".", regex=False)
        .str.upper()
        .unique()
        .tolist()
    )
    _SP500_CACHE_MEM = sorted(syms)
    return _SP500_CACHE_MEM


def _all_universe_symbols() -> list[str]:
    return sorted(set(_sp500_symbols()) | set(CURATED_ETFS))


# --------------------------------------------------------------------------- #
# Data API
# --------------------------------------------------------------------------- #

def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        columns=_BAR_COLUMNS,
        index=pd.MultiIndex.from_tuples([], names=["symbol", "timestamp"]),
    )


class Data:
    """Daily OHLCV bars for the canonical universe, scoped to a split.

    Constructed with ``split="train"`` or ``split="test"``. The second form
    requires ``AUTORESEARCH_ALLOW_TEST=1`` in the environment and is intended
    only for the evaluation harness.
    """

    def __init__(self, split: str):
        if split not in SPLITS:
            raise ValueError(
                f"split must be one of {list(SPLITS)}; got {split!r}"
            )
        if split == "test" and os.environ.get(_ALLOW_TEST_ENV) != "1":
            raise PermissionError(
                "Data(split='test') is gated. The research loop only sees "
                "'train'; test data is reserved for the evaluation harness. "
                f"If you are the harness, set {_ALLOW_TEST_ENV}=1."
            )
        self.split = split
        self.start, self.end = SPLITS[split]
        self._client: StockHistoricalDataClient | None = None

    # ---------- universe ---------- #

    def universe(self, category: str = "all") -> list[str]:
        """Return the canonical symbol list.

        ``category`` ∈ {"all", "sp500", "etf"}.
        """
        if category == "all":
            return _all_universe_symbols()
        if category == "sp500":
            return list(_sp500_symbols())
        if category == "etf":
            return list(CURATED_ETFS)
        raise ValueError(
            f"category must be one of 'all', 'sp500', 'etf'; got {category!r}"
        )

    # ---------- bars ---------- #

    def bars(self, symbols: str | list[str], timeframe: str = "day") -> pd.DataFrame:
        """Return OHLCV bars for ``symbols`` within this split's window.

        Parameters
        ----------
        symbols : str or list[str]
            One or more symbols from the canonical universe.
        timeframe : str
            ``"day"`` or ``"30min"``. 30-minute bars are restricted to the
            US regular session (left-edge ET labels 9:30, 10:00, ..., 15:30 --
            thirteen bars per full trading day).

        Returns
        -------
        pandas.DataFrame
            MultiIndex (symbol, timestamp) with columns
            open, high, low, close, volume, trade_count, vwap. Timestamps
            are strictly within this split's date range.
        """
        if timeframe not in _TIMEFRAMES:
            raise ValueError(
                f"timeframe must be one of {list(_TIMEFRAMES)}; got {timeframe!r}"
            )
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = list(dict.fromkeys(symbols))  # dedupe, preserve order
        if not symbols:
            return _empty_bars()

        universe = set(_all_universe_symbols())
        bad = [s for s in symbols if s not in universe]
        if bad:
            preview = ", ".join(bad[:10]) + ("..." if len(bad) > 10 else "")
            raise ValueError(
                f"{len(bad)} symbol(s) not in canonical universe: {preview}. "
                f"Call Data.universe() to see the available list."
            )

        cache_dir = _TIMEFRAMES[timeframe][1]
        missing = [s for s in symbols if not (cache_dir / f"{s}.pkl").exists()]
        if missing:
            self._fetch_and_cache(missing, timeframe)

        frames = []
        for s in symbols:
            path = cache_dir / f"{s}.pkl"
            if not path.exists():
                continue
            df = pd.read_pickle(path)
            if df.empty:
                continue
            ts = df.index.get_level_values("timestamp")
            df = df[(ts >= self.start) & (ts <= self.end)]
            if df.empty:
                continue
            if timeframe == "30min":
                ts_et = df.index.get_level_values("timestamp").tz_convert("America/New_York")
                hhmm = ts_et.hour * 100 + ts_et.minute
                df = df[(hhmm >= _SESSION_OPEN_HHMM) & (hhmm <= _SESSION_LAST_OPEN_HHMM)]
                if df.empty:
                    continue
            frames.append(df)

        if not frames:
            return _empty_bars()
        return pd.concat(frames).sort_index()

    # ---------- internals ---------- #

    def _get_client(self) -> StockHistoricalDataClient:
        if self._client is None:
            load_dotenv()
            api_key = os.environ.get("ALPACA_API_KEY")
            secret = os.environ.get("ALPACA_SECRET_KEY")
            if not api_key or not secret:
                raise RuntimeError(
                    "ALPACA_API_KEY / ALPACA_SECRET_KEY missing from env (.env)."
                )
            self._client = StockHistoricalDataClient(api_key, secret)
        return self._client

    def _fetch_and_cache(self, symbols: list[str], timeframe: str) -> None:
        """Fetch bars at ``timeframe`` for ``symbols`` over the full retention
        window and cache each symbol to ``<cache_dir>/<sym>.pkl``. Missing
        symbols get an empty-DataFrame placeholder so we don't retry them on
        the next call.
        """
        tf, cache_dir = _TIMEFRAMES[timeframe]
        cache_dir.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        t0 = time.time()
        n = len(symbols)
        # 30-min pulls ~13x more rows per symbol than daily; shrink batches
        # so a single request doesn't balloon to 100k+ rows and many pages.
        batch_size = _BATCH_SIZE if timeframe == "day" else max(1, _BATCH_SIZE // 5)

        for i in range(0, n, batch_size):
            batch = symbols[i:i + batch_size]
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=tf,
                start=_FETCH_START,
                end=_FETCH_END,
                adjustment="all",
                # SIP (consolidated tape) — accurate volume, full history back
                # to 2017. IEX (what Alpaca's docs default to) is missing most
                # pre-mid-2020 bars on this account.
                feed="sip",
            )
            # Retry transient failures (SSL flakes, 429s, 5xx) with backoff.
            # A request that definitively returns empty is NOT a failure.
            df = None
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    df = client.get_stock_bars(req).df
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(1.5 * (attempt + 1))
            if df is None:
                # All retries failed — leave cache untouched so a later call
                # can retry (rather than poisoning these symbols as "empty").
                print(
                    f"  ! batch {batch[0]}..{batch[-1]} failed after 3 tries, "
                    f"leaving uncached: {last_exc}",
                    file=sys.stderr,
                )
                done = i + len(batch)
                elapsed = time.time() - t0
                eta = (elapsed / done) * (n - done) if done else 0.0
                print(
                    f"[fetch {timeframe}] {done:>5}/{n} (skipped {len(batch)}) | "
                    f"elapsed {elapsed:>5.0f}s | eta {eta:>5.0f}s",
                    flush=True,
                )
                time.sleep(_SLEEP_BETWEEN_BATCHES)
                continue

            present = (
                set(df.index.get_level_values("symbol").unique())
                if not df.empty else set()
            )
            for s in batch:
                path = cache_dir / f"{s}.pkl"
                if s in present:
                    df.xs(s, level="symbol", drop_level=False).to_pickle(path)
                else:
                    # Request succeeded but this symbol returned no data
                    # (delisted, too recent, bad symbol). Cache the empty
                    # placeholder so we don't re-request it.
                    _empty_bars().to_pickle(path)

            done = i + len(batch)
            elapsed = time.time() - t0
            eta = (elapsed / done) * (n - done) if done else 0.0
            print(
                f"[fetch {timeframe}] {done:>5}/{n} | "
                f"elapsed {elapsed:>5.0f}s | eta {eta:>5.0f}s",
                flush=True,
            )
            time.sleep(_SLEEP_BETWEEN_BATCHES)
