#!/usr/bin/env python3
"""
RiskWise Correlation Analysis Pipeline — Main Orchestrator

Three-stage pipeline:
  Stage 1: Statistical correlation analysis (Pearson, Granger, cointegration, etc.)
  Stage 2: ML predictive modeling (Ridge, Lasso, RF, XGBoost with walk-forward CV)
  Stage 3: SHAP explainability (interactions, temporal dynamics, feature selection)

Usage:
    # With environment variables:
    export RISKWISE_DB_HOST=localhost
    export RISKWISE_DB_NAME=riskwise
    export RISKWISE_DB_USER=riskwise
    export RISKWISE_DB_PASSWORD=secret

    # Full pipeline (all 3 stages):
    uv run python -m correlate.run --ticker PEP --name "PepsiCo"

    # Stats only (skip ML + SHAP):
    uv run python -m correlate.run --ticker PEP --name "PepsiCo" --stats-only

    # Skip SHAP explainability:
    uv run python -m correlate.run --ticker PEP --name "PepsiCo" --no-explain
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
from .ml import MLPredictor, MLConfig
from .explainability import ExplainabilityAnalyzer
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

    # Stage control
    parser.add_argument("--stats-only", action="store_true",
                       help="Run Stage 1 only (statistical correlations)")
    parser.add_argument("--ml-only", action="store_true",
                       help="Run Stage 2 only (ML models — still computes correlations for feature selection)")
    parser.add_argument("--no-explain", action="store_true",
                       help="Skip Stage 3 (SHAP explainability)")
    parser.add_argument("--explain-only", action="store_true",
                       help="Run Stage 3 only (still runs Stage 1+2 for inputs)")

    # ML-specific
    parser.add_argument("--ml-horizons", type=str, default="1,5,10,20",
                       help="Comma-separated prediction horizons in days (default: 1,5,10,20)")
    parser.add_argument("--ml-top-features", type=int, default=30,
                       help="Max features to use in ML models (default: 30)")
    parser.add_argument("--ml-min-train", type=int, default=252,
                       help="Minimum training window in days (default: 252)")
    parser.add_argument("--ml-use-all-indices", action="store_true",
                       help="Use all indices as features, not just significant ones")

    # Explainability-specific
    parser.add_argument("--explain-horizon", type=int, default=None,
                       help="Which horizon to run SHAP analysis on (default: longest)")
    parser.add_argument("--explain-model", type=str, default=None,
                       help="Which model to explain (default: best from Stage 2)")

    # Database
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

    ml_config = MLConfig(
        horizons=[int(h) for h in args.ml_horizons.split(",")],
        top_n_features=args.ml_top_features,
        min_train_days=args.ml_min_train,
        use_significant_only=not args.ml_use_all_indices,
    )

    # Determine which stages to run
    run_ml = not args.stats_only
    run_explain = run_ml and not args.no_explain and not args.stats_only

    t0 = time.time()

    # ==================================================================
    # Data Loading (shared by all stages)
    # ==================================================================

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

    # Fetch market data
    logger.info("Fetching %s stock data for %s (%s)...",
                config.market_data_period, config.target_name, config.target_ticker)
    market_df = fetch_stock_data(config)

    if config.use_returns:
        target_series = market_df["LogReturns"].dropna()
        logger.info("Using log returns as target variable")
    else:
        target_series = market_df["Close"]
        logger.info("Using price levels as target variable")

    # Fetch all indices (wide format)
    logger.info("Fetching all index time series from database...")
    index_wide = db.get_all_indices_wide()
    logger.info("Index data shape: %s", index_wide.shape)

    index_wide.index = pd.to_datetime(index_wide.index)
    index_wide = index_wide.resample("B").ffill()

    # ==================================================================
    # Stage 1: Statistical Correlation Analysis
    # ==================================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 1: Statistical Correlation Analysis")
    logger.info("=" * 60)

    analyzer = CorrelationAnalyzer(config)
    corr_summary = analyzer.run_full_analysis(index_wide, target_series)

    t_stats = time.time() - t0
    logger.info("Stage 1 complete in %.1f seconds", t_stats)
    logger.info("  Tested: %d indices", corr_summary.n_indices_tested)
    logger.info("  Significant: %d", corr_summary.n_significant)
    logger.info("  Leading indicators: %d", len(corr_summary.top_leading))
    logger.info("  Contemporaneous: %d", len(corr_summary.top_contemporaneous))
    logger.info("  Lagging: %d", len(corr_summary.top_lagging))
    logger.info("  Cointegrated: %d", len(corr_summary.cointegrated))

    # ==================================================================
    # Stage 2: Machine Learning Predictive Analysis
    # ==================================================================

    ml_summary = None
    if run_ml:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 2: Machine Learning Predictive Analysis")
        logger.info("=" * 60)

        t_ml_start = time.time()
        predictor = MLPredictor(config, ml_config)

        try:
            ml_summary = predictor.run(
                index_wide, target_series, correlation_summary=corr_summary
            )
            t_ml = time.time() - t_ml_start
            logger.info("Stage 2 complete in %.1f seconds", t_ml)
            logger.info("  Features used: %d", ml_summary.n_features_used)
            for h, best in ml_summary.best_model_per_horizon.items():
                logger.info(
                    "  %dd: best=%s, dir_acc=%.1f%%, R²=%.4f",
                    h, best.model_name,
                    best.directional_accuracy * 100, best.r2,
                )
        except Exception as e:
            logger.error("Stage 2 failed: %s", e)
            logger.info("Continuing with Stage 1 results only...")

    # ==================================================================
    # Stage 3: SHAP Explainability Analysis
    # ==================================================================

    explain_summary = None
    if run_explain:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 3: SHAP Explainability Analysis")
        logger.info("=" * 60)

        t_explain_start = time.time()
        explainer = ExplainabilityAnalyzer(config, ml_config)

        try:
            explain_summary = explainer.run(
                index_wide, target_series,
                ml_summary=ml_summary,
                horizon=args.explain_horizon,
                model_name=args.explain_model,
            )
            if explain_summary:
                t_explain = time.time() - t_explain_start
                logger.info("Stage 3 complete in %.1f seconds", t_explain)
                logger.info("  Top base index: %s",
                          explain_summary.base_index_rankings[0][0]
                          if explain_summary.base_index_rankings else "N/A")
                logger.info("  Interactions found: %d", len(explain_summary.top_interactions))
                logger.info("  Temporal regimes: %d", len(explain_summary.temporal_regimes))
                if explain_summary.selection_result:
                    sel = explain_summary.selection_result
                    logger.info("  SHAP feature selection: %d → %d features (acc: %.1f%%)",
                              sel.n_original_features, sel.n_selected_features,
                              sel.selected_directional_accuracy * 100)
        except Exception as e:
            logger.error("Stage 3 failed: %s", e)
            logger.info("Continuing without explainability results...")

    # ==================================================================
    # Report Generation
    # ==================================================================

    logger.info("")
    logger.info("Generating reports...")
    paths = generate_reports(
        corr_summary, config,
        ml_summary=ml_summary,
        explain_summary=explain_summary,
    )
    logger.info("Technical report: %s", paths["technical_path"])
    logger.info("Executive summary: %s", paths["executive_path"])

    elapsed = time.time() - t0

    # ==================================================================
    # Summary to stdout
    # ==================================================================

    print("\n" + "=" * 70)
    print(f"RISKWISE ANALYSIS: {corr_summary.target_name} ({corr_summary.target_ticker})")
    print("=" * 70)

    print("\n--- Stage 1: Statistical Correlations ---")
    print(f"Indices tested:      {corr_summary.n_indices_tested}")
    print(f"Significant:         {corr_summary.n_significant}")
    print(f"Leading indicators:  {len(corr_summary.top_leading)}")
    print(f"Cointegrated:        {len(corr_summary.cointegrated)}")
    granger_n = sum(1 for r in corr_summary.results
                    if r.granger_p_value and r.granger_p_value < config.significance_level)
    print(f"Granger causal:      {granger_n}")

    if corr_summary.top_leading:
        top = corr_summary.top_leading[0]
        print(f"\nTop leading indicator: {top.index_name}")
        print(f"  Lead time: {top.optimal_lag_days} days")
        print(f"  Correlation at lag: {top.lagged_correlation:+.4f}")
        if top.granger_p_value:
            print(f"  Granger p-value: {top.granger_p_value:.4f}")

    if ml_summary:
        print("\n--- Stage 2: ML Predictive Models ---")
        print(f"Features used:       {ml_summary.n_features_used}")
        for h in sorted(ml_summary.best_model_per_horizon.keys()):
            best = ml_summary.best_model_per_horizon[h]
            baseline = ml_summary.baseline_rmse.get(h, 0)
            sig = "*" if best.directional_p_value < 0.05 else ""
            beat = "✓" if baseline > 0 and best.rmse < baseline else ""
            print(f"  {h:>2}d: {best.model_name:<15} "
                  f"dir={best.directional_accuracy:.1%}{sig}  "
                  f"R²={best.r2:+.4f}  "
                  f"RMSE={best.rmse:.6f} {beat}")

        if ml_summary.feature_importances:
            print(f"\nTop 5 predictive features:")
            for fi in ml_summary.feature_importances[:5]:
                print(f"  {fi.feature_name}: {fi.importance_mean:.4f}")

    if explain_summary:
        print("\n--- Stage 3: SHAP Explainability ---")
        print(f"Model explained:     {explain_summary.model_name} ({explain_summary.horizon_days}d)")
        if explain_summary.base_index_rankings:
            print("Top 5 base indices by SHAP:")
            for name, imp in explain_summary.base_index_rankings[:5]:
                print(f"  {name}: {imp:.6f}")
        if explain_summary.top_interactions:
            cross = [ix for ix in explain_summary.top_interactions
                    if ix.base_index_a != ix.base_index_b]
            if cross:
                print(f"Top cross-index interaction:")
                ix = cross[0]
                print(f"  {ix.base_index_a} × {ix.base_index_b}: "
                      f"{ix.interaction_direction} ({ix.interaction_strength:.6f})")
        if explain_summary.selection_result:
            sel = explain_summary.selection_result
            print(f"SHAP feature selection: {sel.n_original_features} → "
                  f"{sel.n_selected_features} features")
            print(f"  Accuracy: {sel.selected_directional_accuracy:.1%} "
                  f"(change: {sel.improvement:+.1%})")
            print(f"  Essential indices: {', '.join(sel.selected_base_indices[:5])}")

    print(f"\nTotal time:          {elapsed:.1f}s")
    print(f"Technical report:    {paths['technical_path']}")
    print(f"Executive summary:   {paths['executive_path']}")
    print("=" * 70)

    db.close()
    return corr_summary, ml_summary, explain_summary


if __name__ == "__main__":
    main()
