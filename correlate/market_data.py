"""
Market data fetcher for target company financial outcomes.

Uses yfinance to pull stock prices, then derives returns and other features
that can be correlated against RiskWise indices.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from .config import AnalysisConfig

logger = logging.getLogger(__name__)


def fetch_stock_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Fetch historical stock data for the target ticker.

    Returns a DataFrame indexed by date with columns:
        Close, Volume, Returns, LogReturns, Volatility_20d
    """
    ticker = yf.Ticker(config.target_ticker)
    hist = ticker.history(period=config.market_data_period)

    if hist.empty:
        raise ValueError(
            f"No data returned for ticker '{config.target_ticker}'. "
            "Check that the ticker symbol is valid."
        )

    df = pd.DataFrame(index=hist.index)
    df.index.name = "date"
    df.index = df.index.tz_localize(None)  # strip timezone for clean merges

    df["Close"] = hist["Close"]
    df["Volume"] = hist["Volume"]

    # Daily returns
    df["Returns"] = df["Close"].pct_change()
    df["LogReturns"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility (20-day rolling std of log returns)
    df["Volatility_20d"] = df["LogReturns"].rolling(window=20).std() * np.sqrt(252)

    # Cumulative returns (useful for level-vs-level correlation)
    df["CumReturns"] = (1 + df["Returns"]).cumprod() - 1

    logger.info(
        "Fetched %d days of %s data (%s to %s)",
        len(df), config.target_ticker,
        df.index.min().date(), df.index.max().date(),
    )
    return df


def resample_to_business_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample index data to business-day frequency via forward-fill."""
    return df.resample("B").ffill()


def align_series(
    index_series: pd.DataFrame,
    market_series: pd.DataFrame,
    min_overlap: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Align two time series on their overlapping date range.
    Returns None if insufficient overlap.

    Both inputs should be DatetimeIndex DataFrames.
    """
    # Forward-fill index data to business days to match market calendar
    index_resampled = resample_to_business_daily(index_series)

    # Find overlapping range
    common_idx = index_resampled.index.intersection(market_series.index)
    if len(common_idx) < min_overlap:
        return None

    start, end = common_idx.min(), common_idx.max()
    idx_aligned = index_resampled.loc[start:end].dropna()
    mkt_aligned = market_series.loc[start:end].dropna()

    # Re-intersect after dropna
    common_idx = idx_aligned.index.intersection(mkt_aligned.index)
    if len(common_idx) < min_overlap:
        return None

    return idx_aligned.loc[common_idx], mkt_aligned.loc[common_idx]
