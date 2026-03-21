#!/usr/bin/env python3
"""
RiskWise Correlation Analysis Pipeline — Main Orchestrator

Usage:
    # With environment variables:
    export RISKWISE_DB_HOST=localhost
    export RISKWISE_DB_NAME=riskwise
    export RISKWISE_DB_USER=riskwise
    export RISKWISE_DB_PASSWORD=secret

    uv run python -m correlate.run --ticker PEP --name "PepsiCo"
    uv run python -m correlate.run --ticker KO --name "Coca-Cola"
    uv run python -m correlate.run --ticker AAPL --name "Apple Inc."
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from .config import AnalysisConfig, DBConfig
from .db import RiskWiseDB
from .market_data import fetch_stock_data, align_series
from .analysis import CorrelationAnalyzer
from .report import generate_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("correlate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RiskWise Correlation Analysis Pipeline"
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., PEP)")
    parser.add_argument("--name", required=True, help="Company name (e.g., PepsiCo)")
    parser.add_argument("--period", default="5y", help="Market data lookback (default: 5y)")
    parser.add_argument("--max-lag", type=int, default=90, help="Max lead/lag days (default: 90)")
    parser.add_argument("--significance", type=float, default=0.05, help="Alpha (default: 0.05)")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--use-levels", action="store_true",
                       help="Correlate with price levels instead of returns")
    parser.add_argument("--db-host", help="Override RISKWISE_DB_HOST")
    parser.add_argument("--db-name", help="Override RISKWISE_DB_NAME")
    parser.add_argument("--db-user", help="Override RISKWISE_DB_USER")
    parser.add_argument("--db-password", help="Override RISKWISE_DB_PASSWORD")
    parser.add_argument("--db-schema", default="public", help="DB schema (default: public)")
    parser.add_argument("--index-table", help="Override RISKWISE_INDEX_TABLE")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    db_config = DBConfig()
    if args.db_host:
        db_config.host = args.db_host
    if args.db_name:
        db_config.database = args.db_name
    if args.db_user:
        db_config.user = args.db_user
    if args.db_password:
        db_config.password = args.db_password
    if args.db_schema:
        db_config.schema = args.db_schema

    config = AnalysisConfig(
        target_ticker=args.ticker,
        target_name=args.name,
        db=db_config,
        market_data_period=args.period,
        max_lag_days=args.max_lag,
        significance_level=args.significance,
        output_dir=args.output_dir,
        use_returns=not args.use_levels,
    )
    if args.index_table:
        config.index_table_pattern = args.index_table

    t0 = time.time()

    # ---- Step 1: Connect to DB and discover indices ----
    logger.info("Connecting to RiskWise database at %s:%s/%s",
                config.db.host, config.db.port, config.db.database)
    db = RiskWiseDB(config)

    logger.info("Discovering available indices...")
    indices = db.discover_indices()
    logger.info("Found %d indices", len(indices))

    if not indices:
        logger.error("No indices found in database. Check schema and table configuration.")
        sys.exit(1)

    for idx in indices[:10]:
        logger.info("  → %s (%d obs, %s to %s)", idx.name, idx.n_observations,
                    idx.start_date, idx.end_date)
    if len(indices) > 10:
        logger.info("  ... and %d more", len(indices) - 10)

    # ---- Step 2: Fetch market data ----
    logger.info("Fetching %s stock data for %s (%s)...",
                config.market_data_period, config.target_name, config.target_ticker)
    market_df = fetch_stock_data(config)

    # Choose target series based on config
    if config.use_returns:
        target_series = market_df["LogReturns"].dropna()
        logger.info("Using log returns as target variable")
    else:
        target_series = market_df["Close"]
        logger.info("Using price levels as target variable")

    # ---- Step 3: Fetch all indices (wide format) ----
    logger.info("Fetching all index time series from database...")
    index_wide = db.get_all_indices_wide()
    logger.info("Index data shape: %s", index_wide.shape)

    # Forward-fill to business days and align with market calendar
    index_wide.index = pd.to_datetime(index_wide.index)
    index_wide = index_wide.resample("B").ffill()

    # ---- Step 4: Run analysis ----
    logger.info("Running correlation analysis across %d indices...", len(index_wide.columns))
    analyzer = CorrelationAnalyzer(config)
    summary = analyzer.run_full_analysis(index_wide, target_series)

    elapsed = time.time() - t0
    logger.info("Analysis complete in %.1f seconds", elapsed)
    logger.info("  Tested: %d indices", summary.n_indices_tested)
    logger.info("  Significant: %d", summary.n_significant)
    logger.info("  Leading indicators: %d", len(summary.top_leading))
    logger.info("  Contemporaneous: %d", len(summary.top_contemporaneous))
    logger.info("  Lagging: %d", len(summary.top_lagging))
    logger.info("  Cointegrated: %d", len(summary.cointegrated))

    # ---- Step 5: Generate reports ----
    logger.info("Generating reports...")
    paths = generate_reports(summary, config)
    logger.info("Technical report: %s", paths["technical_path"])
    logger.info("Executive summary: %s", paths["executive_path"])

    # ---- Print summary to stdout ----
    print("\n" + "=" * 70)
    print(f"RISKWISE CORRELATION ANALYSIS: {summary.target_name} ({summary.target_ticker})")
    print("=" * 70)
    print(f"Indices tested:      {summary.n_indices_tested}")
    print(f"Significant:         {summary.n_significant}")
    print(f"Leading indicators:  {len(summary.top_leading)}")
    print(f"Cointegrated:        {len(summary.cointegrated)}")
    print(f"Granger causal:      {sum(1 for r in summary.results if r.granger_p_value and r.granger_p_value < config.significance_level)}")
    print(f"Analysis time:       {elapsed:.1f}s")
    print(f"Technical report:    {paths['technical_path']}")
    print(f"Executive summary:   {paths['executive_path']}")

    if summary.top_leading:
        print(f"\nTop leading indicator: {summary.top_leading[0].index_name}")
        top = summary.top_leading[0]
        print(f"  Lead time: {top.optimal_lag_days} days")
        print(f"  Correlation at lag: {top.lagged_correlation:+.4f}")
        if top.granger_p_value:
            print(f"  Granger p-value: {top.granger_p_value:.4f}")

    print("=" * 70)

    db.close()
    return summary


if __name__ == "__main__":
    main()
