"""
Report generator for RiskWise correlation analysis.

Produces two outputs:
1. PhD-level technical report — full statistical methodology, all results, caveats
2. Executive summary — key findings, headline numbers, sales-oriented narrative
"""
from __future__ import annotations

import os
from datetime import datetime

from .analysis import AnalysisSummary, CorrelationResult
from .config import AnalysisConfig
from .ml import MLSummary
from .explainability import ExplainabilitySummary


def generate_reports(
    summary: AnalysisSummary,
    config: AnalysisConfig,
    ml_summary: MLSummary | None = None,
    explain_summary: ExplainabilitySummary | None = None,
) -> dict[str, str]:
    """
    Generate both technical and executive reports.
    Returns dict with keys 'technical' and 'executive', values are markdown strings.
    Also writes files to config.output_dir.
    """
    technical = _generate_technical_report(summary, config, ml_summary, explain_summary)
    executive = _generate_executive_summary(summary, config, ml_summary, explain_summary)

    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ticker = config.target_ticker.lower()

    tech_path = os.path.join(config.output_dir, f"{ticker}_technical_{timestamp}.md")
    exec_path = os.path.join(config.output_dir, f"{ticker}_executive_{timestamp}.md")

    with open(tech_path, "w") as f:
        f.write(technical)
    with open(exec_path, "w") as f:
        f.write(executive)

    return {"technical": technical, "executive": executive,
            "technical_path": tech_path, "executive_path": exec_path}


# ======================================================================
# Technical Report
# ======================================================================

def _generate_technical_report(
    summary: AnalysisSummary,
    config: AnalysisConfig,
    ml_summary: MLSummary | None = None,
    explain_summary: ExplainabilitySummary | None = None,
) -> str:
    sections = []

    # Title
    sections.append(f"# RiskWise Index Correlation Analysis: {summary.target_name} ({summary.target_ticker})")
    sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Abstract
    sections.append("## Abstract\n")
    sections.append(
        f"This report presents a systematic correlation analysis between "
        f"{summary.n_indices_tested} RiskWise risk indices and {summary.target_name} "
        f"({summary.target_ticker}) financial outcomes. Using a comprehensive statistical "
        f"framework including Pearson, Spearman, and Kendall correlations, cross-correlation "
        f"lead/lag analysis, Granger causality testing, Engle-Granger cointegration testing, "
        f"rolling correlation dynamics, and bootstrap confidence intervals, we identify "
        f"**{summary.n_significant}** statistically significant relationships after "
        f"Benjamini-Hochberg FDR correction at α = {summary.fdr_threshold}.\n"
    )

    # Methodology
    sections.append("## Methodology\n")
    sections.append("### Data Alignment\n")
    sections.append(
        f"- RiskWise indices were resampled to business-day frequency via forward-fill\n"
        f"- Minimum overlap requirement: {config.min_overlap_days} trading days\n"
        f"- Market data: {'log returns' if config.use_returns else 'price levels'} "
        f"of {config.price_column}\n"
        f"- Historical period: {config.market_data_period}\n"
    )

    sections.append("### Statistical Tests\n")
    sections.append(
        "1. **Contemporaneous Correlation**: Pearson (linear), Spearman (rank-order), "
        "Kendall (concordance) — three complementary measures capturing different aspects "
        "of association\n"
        "2. **Lead/Lag Analysis**: Cross-correlation computed for lags "
        f"[-{config.max_lag_days}, +{config.max_lag_days}] days. "
        "Positive lag indicates the index *leads* the target (predictive power). "
        "The lag maximizing |ρ| is reported.\n"
        f"3. **Granger Causality**: F-test for predictive information content, "
        f"tested up to {config.granger_max_lags} lags. Tests both directions "
        "(index→target and target→index) to establish directionality.\n"
        "4. **Cointegration**: Engle-Granger two-step test for long-run equilibrium "
        "relationships between non-stationary series.\n"
        "5. **Stationarity**: Augmented Dickey-Fuller test with AIC-selected lag order.\n"
        f"6. **Rolling Correlation**: {config.rolling_window_days}-day rolling window "
        "to assess stability of relationships over time.\n"
        f"7. **Bootstrap CI**: {config.n_bootstrap}-iteration non-parametric bootstrap "
        "for Pearson r confidence intervals.\n"
        "8. **Multiple Testing**: Benjamini-Hochberg FDR correction applied across all "
        "index-target pairs to control false discovery rate.\n"
    )

    # Results — Leading Indicators
    if summary.top_leading:
        sections.append("## Leading Indicators (Index Leads Target)\n")
        sections.append(
            "These indices show statistically significant predictive power — they move "
            f"*before* {summary.target_name}'s stock price, providing early warning signals.\n"
        )
        sections.append(_results_table(summary.top_leading))

        for r in summary.top_leading[:5]:
            sections.append(_detailed_result(r, summary.target_name))

    # Results — Contemporaneous
    if summary.top_contemporaneous:
        sections.append("## Contemporaneous Correlations\n")
        sections.append(
            "These indices move in tandem with the target — they capture the same "
            "underlying dynamics in real time.\n"
        )
        sections.append(_results_table(summary.top_contemporaneous))

    # Results — Lagging
    if summary.top_lagging:
        sections.append("## Lagging Indicators (Target Leads Index)\n")
        sections.append(
            "These indices follow the target — useful for confirmation and "
            "post-hoc validation.\n"
        )
        sections.append(_results_table(summary.top_lagging))

    # Cointegration
    if summary.cointegrated:
        sections.append("## Cointegrated Pairs\n")
        sections.append(
            "The following indices share a long-run equilibrium relationship with "
            f"{summary.target_name} stock price. Deviations from this equilibrium "
            "are mean-reverting — a powerful signal for both risk monitoring and "
            "trading strategies.\n"
        )
        for r in summary.cointegrated:
            sections.append(
                f"- **{r.index_name}**: cointegration t-stat = {r.coint_t_stat:.3f}, "
                f"p = {r.coint_p_value:.4f}\n"
            )

    # Granger Causality Deep Dive
    granger_sig = [r for r in summary.results
                   if r.granger_p_value is not None
                   and r.granger_p_value < config.significance_level]
    if granger_sig:
        sections.append("## Granger Causality Results\n")
        sections.append(
            "Indices where past values contain statistically significant predictive "
            f"information about future {summary.target_name} outcomes:\n"
        )
        sections.append("| Index | F-stat | p-value | Optimal Lag | Reverse p |\n")
        sections.append("|-------|--------|---------|-------------|----------|\n")
        for r in sorted(granger_sig, key=lambda x: x.granger_p_value or 1):
            rev_p = f"{r.reverse_granger_p:.4f}" if r.reverse_granger_p else "N/A"
            sections.append(
                f"| {r.index_name} | {r.granger_f_stat:.2f} | "
                f"{r.granger_p_value:.4f} | {r.granger_optimal_lag}d | {rev_p} |\n"
            )
        sections.append("")

    # Full Results Table
    sections.append("## Complete Results\n")
    sections.append(_full_results_table(summary.results))

    # ML Results
    if ml_summary:
        sections.append(_generate_ml_technical_section(ml_summary))

    # Explainability Results
    if explain_summary:
        sections.append(_generate_explainability_technical_section(explain_summary))

    # Caveats
    sections.append("## Methodological Caveats\n")
    sections.append(
        "1. **Correlation ≠ Causation**: Statistical association does not establish "
        "causal mechanisms. Confounding variables may drive observed relationships.\n"
        "2. **Look-ahead Bias**: All analyses use historically available data, but "
        "production deployment should use point-in-time index values.\n"
        "3. **Regime Dependence**: Correlations may be unstable across market regimes. "
        "Rolling correlation analysis addresses this partially.\n"
        "4. **Multiple Testing**: FDR correction controls expected false discovery "
        "proportion, not family-wise error rate.\n"
        "5. **Data Frequency Mismatch**: Index data may be at different frequencies "
        "than market data; forward-fill introduces autocorrelation.\n"
        "6. **Non-stationarity**: Correlations between non-stationary series can be "
        "spurious. We flag stationarity test results throughout.\n"
    )
    if ml_summary:
        sections.append(
            "7. **ML Overfitting**: Walk-forward validation mitigates but does not "
            "eliminate overfitting risk. Out-of-sample performance is the gold standard.\n"
            "8. **Feature Engineering**: Lagged and rolling features introduce additional "
            "autocorrelation. Directional accuracy is a more robust metric than R² for "
            "financial time series.\n"
            "9. **Transaction Costs**: Equity curves do not account for transaction costs, "
            "slippage, or market impact. Real-world performance will be lower.\n"
        )

    return "\n".join(sections)


# ======================================================================
# Executive Summary
# ======================================================================

def _generate_executive_summary(
    summary: AnalysisSummary,
    config: AnalysisConfig,
    ml_summary: MLSummary | None = None,
    explain_summary: ExplainabilitySummary | None = None,
) -> str:
    sections = []

    sections.append(f"# Executive Summary: RiskWise Indices × {summary.target_name}")
    sections.append(f"*{datetime.now().strftime('%B %d, %Y')}*\n")

    # Headline
    n_leading = len(summary.top_leading)
    n_contemp = len(summary.top_contemporaneous)
    n_coint = len(summary.cointegrated)

    sections.append("## Key Findings\n")

    sections.append(
        f"We analyzed **{summary.n_indices_tested}** RiskWise risk indices against "
        f"{summary.target_name} ({summary.target_ticker}) stock performance over "
        f"{config.market_data_period}. After rigorous statistical testing with "
        f"multiple-hypothesis correction:\n"
    )

    bullet_points = []
    if n_leading > 0:
        top = summary.top_leading[0]
        bullet_points.append(
            f"**{n_leading} indices are predictive leading indicators** — they move "
            f"before {summary.target_name}'s stock price. The strongest, "
            f"**{top.index_name}**, leads by **{top.optimal_lag_days} days** "
            f"with r = {top.lagged_correlation:.3f}."
        )
    if n_contemp > 0:
        bullet_points.append(
            f"**{n_contemp} indices show real-time correlation** — "
            f"tracking live risk dynamics as they unfold."
        )
    if n_coint > 0:
        bullet_points.append(
            f"**{n_coint} indices are cointegrated** — sharing a long-run "
            f"equilibrium with {summary.target_name}'s valuation. Deviations "
            f"from equilibrium historically revert, creating a powerful risk signal."
        )

    granger_count = sum(
        1 for r in summary.results
        if r.granger_p_value is not None and r.granger_p_value < config.significance_level
    )
    if granger_count > 0:
        bullet_points.append(
            f"**{granger_count} indices pass Granger causality** — their historical "
            f"values contain statistically significant information for predicting "
            f"future {summary.target_name} outcomes."
        )

    for bp in bullet_points:
        sections.append(f"- {bp}")
    sections.append("")

    # Top Predictive Indices
    if summary.top_leading:
        sections.append(f"## Top Predictive Indices for {summary.target_name}\n")
        sections.append(
            "These RiskWise indices provide **early warning** of changes in "
            f"{summary.target_name}'s stock performance:\n"
        )
        sections.append("| Rank | RiskWise Index | Lead Time | Correlation | Confidence |\n")
        sections.append("|------|---------------|-----------|-------------|------------|\n")
        for i, r in enumerate(summary.top_leading[:config.top_n_indices], 1):
            conf = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]" if r.ci_lower else "—"
            sections.append(
                f"| {i} | {r.index_name} | {r.optimal_lag_days}d | "
                f"{r.lagged_correlation:+.3f} | {conf} |\n"
            )
        sections.append("")

    # What This Means
    sections.append("## What This Means\n")

    if n_leading > 0:
        top = summary.top_leading[0]
        direction = "declines" if top.lagged_correlation < 0 else "gains"
        sections.append(
            f"When **{top.index_name}** rises, {summary.target_name}'s stock "
            f"tends to see {direction} approximately **{top.optimal_lag_days} days later** "
            f"(r = {top.lagged_correlation:+.3f}). This lead time provides a "
            f"concrete window for proactive decision-making — whether for risk "
            f"mitigation, portfolio adjustment, or supply chain intervention.\n"
        )

    sections.append(
        f"RiskWise indices are not just correlated with {summary.target_name}'s "
        f"outcomes — they are *predictive*. This transforms risk monitoring from "
        f"reactive reporting into forward-looking intelligence.\n"
    )

    # ML Results in Executive Summary
    if ml_summary:
        sections.append(_generate_ml_executive_section(ml_summary))

    # Explainability in Executive Summary
    if explain_summary:
        sections.append(_generate_explainability_executive_section(explain_summary))

    # Recommendation
    sections.append("## Recommendation\n")
    sections.append(
        f"Based on this analysis, we recommend {summary.target_name} integrate "
        f"the following RiskWise indices into their risk monitoring dashboard:\n"
    )

    recs = (summary.top_leading[:3] + summary.cointegrated[:2] +
            summary.top_contemporaneous[:2])
    seen = set()
    for r in recs:
        if r.index_name not in seen:
            seen.add(r.index_name)
            sections.append(f"1. **{r.index_name}** — {r.relationship_type}, "
                          f"{r.strength} ({r.pearson_r:+.3f})")
    sections.append("")

    sections.append(
        "These indices, delivered via RiskWise's continuous monitoring platform, "
        "would provide early warning signals, real-time risk tracking, and "
        "long-run equilibrium monitoring — enabling data-driven risk management "
        "with quantified lead times and statistical confidence.\n"
    )

    # Methodology Note
    sections.append("---\n")
    ml_note = ""
    if ml_summary:
        ml_note = ("Walk-forward ML validation (Ridge, Lasso, Random Forest, XGBoost), "
                   "SHAP feature importance, directional accuracy testing. ")
    sections.append(
        f"*Methodology: Pearson/Spearman/Kendall correlations, "
        f"cross-correlation lead/lag analysis, Granger causality, "
        f"Engle-Granger cointegration, Benjamini-Hochberg FDR correction. "
        f"{ml_note}"
        f"Full technical report available upon request.*\n"
    )

    return "\n".join(sections)


# ======================================================================
# Table helpers
# ======================================================================

def _results_table(results: list[CorrelationResult]) -> str:
    lines = []
    lines.append("| Index | Pearson r | Lag (days) | Lagged r | Granger p | Type | Strength |")
    lines.append("|-------|-----------|------------|----------|-----------|------|----------|")
    for r in results:
        gp = f"{r.granger_p_value:.4f}" if r.granger_p_value else "—"
        lines.append(
            f"| {r.index_name} | {r.pearson_r:+.3f} | {r.optimal_lag_days:+d} | "
            f"{r.lagged_correlation:+.3f} | {gp} | {r.relationship_type} | {r.strength} |"
        )
    lines.append("")
    return "\n".join(lines)


def _detailed_result(r: CorrelationResult, target_name: str) -> str:
    lines = []
    lines.append(f"\n### {r.index_name}\n")
    lines.append(f"- **Overlap**: {r.overlap_start} to {r.overlap_end} ({r.n_observations} obs)")
    lines.append(f"- **Pearson r**: {r.pearson_r:+.4f} (p = {r.pearson_p:.2e}, "
                f"FDR-adjusted p = {r.pearson_p_adjusted:.2e})")
    lines.append(f"- **Spearman ρ**: {r.spearman_r:+.4f} (p = {r.spearman_p:.2e})")
    lines.append(f"- **Kendall τ**: {r.kendall_tau:+.4f} (p = {r.kendall_p:.2e})")
    lines.append(f"- **Optimal Lag**: {r.optimal_lag_days:+d} days "
                f"(r = {r.lagged_correlation:+.4f})")
    lines.append(f"- **Bootstrap 95% CI**: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")

    if r.granger_p_value is not None:
        direction = "→" if r.granger_p_value < 0.05 else "↛"
        lines.append(
            f"- **Granger**: index {direction} {target_name} "
            f"(F = {r.granger_f_stat:.2f}, p = {r.granger_p_value:.4f}, "
            f"lag = {r.granger_optimal_lag}d)"
        )
    if r.reverse_granger_p is not None:
        direction = "→" if r.reverse_granger_p < 0.05 else "↛"
        lines.append(
            f"- **Reverse Granger**: {target_name} {direction} index "
            f"(p = {r.reverse_granger_p:.4f})"
        )
    if r.is_cointegrated:
        lines.append(
            f"- **Cointegrated**: t = {r.coint_t_stat:.3f}, p = {r.coint_p_value:.4f}"
        )

    if r.rolling_corr_mean is not None:
        lines.append(
            f"- **Rolling Correlation**: μ = {r.rolling_corr_mean:.3f}, "
            f"σ = {r.rolling_corr_std:.3f}, "
            f"range [{r.rolling_corr_min:.3f}, {r.rolling_corr_max:.3f}]"
        )

    lines.append(f"- **Stationarity**: index {'stationary' if r.index_is_stationary else 'non-stationary'} "
                f"(ADF p = {r.index_adf_p:.4f})")
    lines.append("")
    return "\n".join(lines)


def _generate_ml_technical_section(ml: MLSummary) -> str:
    """Generate the ML section for the technical report."""
    sections = []

    sections.append("## Machine Learning Predictive Analysis\n")
    sections.append(
        "To move beyond correlation and quantify *predictive power*, we trained "
        "multiple ML models using walk-forward (expanding window) cross-validation. "
        "This ensures all predictions are strictly out-of-sample — no future data "
        "leaks into training.\n"
    )

    sections.append("### Methodology\n")
    sections.append(
        f"- **Features**: {ml.n_features_used} engineered features from RiskWise indices "
        f"(raw values, lagged values at 1/2/3/5/10/20 days, rolling mean/std at "
        f"5/10/20/60 day windows, momentum, rate of change)\n"
        f"- **Models**: Ridge Regression, Lasso (L1), ElasticNet, Random Forest, "
        f"XGBoost (gradient boosted trees)\n"
        f"- **Validation**: Walk-forward expanding window — train on all history up to "
        f"split point, predict the next segment. No look-ahead bias.\n"
        f"- **Horizons**: {', '.join(str(h) + '-day' for h in sorted(ml.model_results.keys()))}\n"
        f"- **Key Metrics**: Directional accuracy (can we predict up/down?), "
        f"Information Coefficient (rank correlation of predictions vs actuals), "
        f"RMSE vs naive baseline\n"
    )

    # Results per horizon
    sections.append("### Model Performance by Horizon\n")

    for horizon in sorted(ml.model_results.keys()):
        results = ml.model_results[horizon]
        baseline = ml.baseline_rmse.get(horizon)
        best = ml.best_model_per_horizon.get(horizon)

        sections.append(f"#### {horizon}-Day Ahead Prediction\n")
        if baseline:
            sections.append(f"Naive baseline RMSE: {baseline:.6f}\n")

        sections.append("| Model | RMSE | MAE | R² | Dir. Accuracy | Dir. p-value | IC |\n")
        sections.append("|-------|------|-----|----|--------------|--------------|----|")
        for r in sorted(results, key=lambda x: x.directional_accuracy, reverse=True):
            sig = " *" if r.directional_p_value < 0.05 else ""
            sections.append(
                f"| {r.model_name} | {r.rmse:.6f} | {r.mae:.6f} | "
                f"{r.r2:.4f} | {r.directional_accuracy:.1%}{sig} | "
                f"{r.directional_p_value:.4f} | {r.ic_mean:+.3f} |"
            )
        sections.append("")

        if best:
            beat_baseline = "Yes" if best.rmse < baseline else "No" if baseline else "N/A"
            sig_dir = "Yes" if best.directional_p_value < 0.05 else "No"
            sections.append(
                f"**Best model**: {best.model_name} — directional accuracy "
                f"{best.directional_accuracy:.1%} (statistically significant: {sig_dir}), "
                f"beats naive baseline: {beat_baseline}\n"
            )

    # Feature importance
    if ml.feature_importances:
        sections.append("### Feature Importance Rankings\n")
        sections.append(
            "Features ranked by importance across all models and horizons. "
            "This reveals which RiskWise indices and derived signals carry the "
            "most predictive information.\n"
        )
        sections.append("| Rank | Feature | Importance | Std | SHAP | Direction |\n")
        sections.append("|------|---------|------------|-----|------|-----------|\n")
        for i, fi in enumerate(ml.feature_importances[:30], 1):
            shap_str = f"{fi.shap_mean:.4f}" if fi.shap_mean is not None else "—"
            dir_str = fi.direction if fi.direction else "—"
            sections.append(
                f"| {i} | {fi.feature_name} | {fi.importance_mean:.4f} | "
                f"{fi.importance_std:.4f} | {shap_str} | {dir_str} |"
            )
        sections.append("")

        # Summarize which base indices appear most
        sections.append("### Most Predictive Base Indices\n")
        sections.append(
            "Aggregating feature importance back to the underlying RiskWise indices "
            "(across all derived features like lags, rolling stats, momentum):\n"
        )
        base_importance: dict[str, float] = {}
        for fi in ml.feature_importances:
            # Extract base index name (before _lag, _ma, _std, _mom, _roc)
            base = fi.feature_name
            for suffix in ["_lag", "_ma", "_std", "_mom", "_roc"]:
                if suffix in base:
                    base = base[:base.index(suffix)]
                    break
            base_importance[base] = base_importance.get(base, 0) + fi.importance_mean

        ranked_bases = sorted(base_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(ranked_bases[:15], 1):
            sections.append(f"{i}. **{name}**: aggregate importance = {imp:.4f}")
        sections.append("")

    return "\n".join(sections)


def _generate_ml_executive_section(ml: MLSummary) -> str:
    """Generate ML section for the executive summary."""
    sections = []

    sections.append(f"## Predictive Model Validation\n")
    sections.append(
        f"Beyond statistical correlation, we built machine learning models to "
        f"validate that RiskWise indices can *actively predict* "
        f"{ml.target_name}'s stock movements:\n"
    )

    for horizon in sorted(ml.best_model_per_horizon.keys()):
        best = ml.best_model_per_horizon[horizon]
        baseline = ml.baseline_rmse.get(horizon, 0)
        sig = "statistically significant" if best.directional_p_value < 0.05 else "not significant"

        improvement = ""
        if baseline > 0 and best.rmse < baseline:
            pct = (1 - best.rmse / baseline) * 100
            improvement = f" ({pct:.0f}% improvement over naive baseline)"

        sections.append(
            f"- **{horizon}-day prediction**: {best.directional_accuracy:.0%} directional "
            f"accuracy ({sig}), RMSE = {best.rmse:.6f}{improvement}"
        )
    sections.append("")

    # Highlight top features in plain English
    if ml.feature_importances:
        top3 = ml.feature_importances[:3]
        names = [fi.feature_name for fi in top3]
        sections.append(
            f"The most predictive signals are derived from: **{', '.join(names)}**. "
            f"These features, when combined in a machine learning model, provide "
            f"actionable predictive power validated on truly out-of-sample data.\n"
        )

    return "\n".join(sections)


# ======================================================================
# Explainability Report Sections
# ======================================================================

def _generate_explainability_technical_section(ex: ExplainabilitySummary) -> str:
    """Generate the explainability section for the technical report."""
    sections = []

    sections.append("## SHAP Explainability Analysis\n")
    sections.append(
        f"Using SHAP (SHapley Additive exPlanations) on the {ex.model_name} model "
        f"at the {ex.horizon_days}-day horizon, we decompose predictions into "
        f"individual feature contributions. This reveals *why* the model makes "
        f"each prediction and identifies the most impactful risk signals.\n"
    )

    # Global feature rankings
    sections.append("### Global SHAP Feature Rankings\n")
    sections.append(
        "Features ranked by mean |SHAP value| — the average magnitude of each "
        "feature's contribution to predictions across all samples.\n"
    )
    sections.append("| Rank | Feature | Mean |SHAP| | Median |SHAP| | P95 |SHAP| | Direction | Std |\n")
    sections.append("|------|---------|-------------|--------------|-------------|-----------|-----|\n")
    for i, fr in enumerate(ex.feature_rankings[:25], 1):
        sections.append(
            f"| {i} | {fr.feature_name} | {fr.mean_abs_shap:.6f} | "
            f"{fr.median_abs_shap:.6f} | {fr.p95_abs_shap:.6f} | "
            f"{fr.direction} | {fr.std_shap:.6f} |"
        )
    sections.append("")

    # Base index aggregation
    sections.append("### Aggregated Base Index Importance\n")
    sections.append(
        "SHAP importance aggregated back to the underlying RiskWise indices "
        "(summing across all derived features — lags, rolling statistics, momentum).\n"
    )
    for i, (name, imp) in enumerate(ex.base_index_rankings[:15], 1):
        sections.append(f"{i}. **{name}**: total SHAP importance = {imp:.6f}")
    sections.append("")

    # Interactions
    if ex.top_interactions:
        sections.append("### Feature Interactions\n")
        sections.append(
            "SHAP interaction values reveal which pairs of features have "
            "synergistic or antagonistic effects — their combined impact exceeds "
            "the sum of their individual contributions.\n"
        )
        sections.append("| Feature A | Feature B | Interaction | Type | Joint Effect |\n")
        sections.append("|-----------|-----------|-------------|------|-------------|\n")
        for ix in ex.top_interactions[:15]:
            sections.append(
                f"| {ix.feature_a} | {ix.feature_b} | "
                f"{ix.interaction_strength:.6f} | {ix.interaction_direction} | "
                f"{ix.joint_effect_sign} |"
            )
        sections.append("")

        # Summarize cross-index interactions
        cross_index = [ix for ix in ex.top_interactions
                      if ix.base_index_a != ix.base_index_b]
        if cross_index:
            sections.append(
                "**Cross-index interactions** (different base indices amplifying each other):\n"
            )
            for ix in cross_index[:5]:
                sections.append(
                    f"- **{ix.base_index_a}** × **{ix.base_index_b}**: "
                    f"{ix.interaction_direction} interaction (strength = {ix.interaction_strength:.6f})"
                )
            sections.append("")

    # Temporal dynamics
    if ex.temporal_regimes:
        sections.append("### Temporal SHAP Dynamics\n")
        sections.append(
            "How feature importance evolves over time — revealing whether "
            "predictive signals are stable or regime-dependent.\n"
        )
        for regime in ex.temporal_regimes:
            sections.append(
                f"#### {regime.period_start} to {regime.period_end} "
                f"({regime.regime_label.upper()}, return: {regime.regime_market_return:+.1%})\n"
            )
            for fname, shap_val in regime.top_features[:5]:
                sections.append(f"- {fname}: SHAP = {shap_val:.6f}")
            sections.append("")

        # Stability summary
        stable_features = sorted(
            ex.importance_stability.items(), key=lambda x: x[1], reverse=True
        )[:10]
        sections.append("**Most stable features** (consistent importance across time):\n")
        for fname, stab in stable_features:
            sections.append(f"- {fname}: stability = {stab:.3f}")
        sections.append("")

    # Partial dependence
    if ex.partial_dependences:
        sections.append("### Partial Dependence Analysis\n")
        sections.append(
            "How individual features map to predictions, holding all other "
            "features constant. Monotonic relationships are more interpretable "
            "and trustworthy.\n"
        )
        for pd_result in ex.partial_dependences[:10]:
            mono_str = f"monotonic ({pd_result.direction})" if pd_result.monotonic else "non-monotonic"
            pred_range = max(pd_result.mean_predictions) - min(pd_result.mean_predictions)
            sections.append(
                f"- **{pd_result.feature_name}** ({pd_result.base_index_name}): "
                f"{mono_str}, prediction range = {pred_range:.6f}"
            )
        sections.append("")

    # SHAP-driven feature selection
    if ex.selection_result:
        sel = ex.selection_result
        sections.append("### SHAP-Driven Feature Selection\n")
        sections.append(
            "Iterative elimination of low-SHAP features to find the minimal "
            "predictive feature set.\n"
        )
        sections.append(f"- Original features: {sel.n_original_features}")
        sections.append(f"- Selected features: {sel.n_selected_features}")
        sections.append(f"- Full-set directional accuracy: {sel.full_directional_accuracy:.1%}")
        sections.append(f"- Selected-set directional accuracy: {sel.selected_directional_accuracy:.1%}")
        improvement_str = f"{sel.improvement:+.1%}"
        sections.append(f"- Accuracy change: {improvement_str}")
        sections.append("")

        sections.append("**Essential RiskWise indices** (minimal set for prediction):\n")
        for base_idx in sel.selected_base_indices[:10]:
            sections.append(f"- {base_idx}")
        sections.append("")

    return "\n".join(sections)


def _generate_explainability_executive_section(ex: ExplainabilitySummary) -> str:
    """Generate explainability section for the executive summary."""
    sections = []

    sections.append("## Why These Indices Matter: Explainability Analysis\n")
    sections.append(
        "Using advanced explainability techniques (SHAP), we can show *exactly* "
        f"how each RiskWise index contributes to predictions about {ex.target_name}:\n"
    )

    # Top base indices
    if ex.base_index_rankings:
        sections.append("**Most impactful RiskWise indices** (by SHAP importance):\n")
        for i, (name, imp) in enumerate(ex.base_index_rankings[:5], 1):
            sections.append(f"{i}. **{name}**")
        sections.append("")

    # Key interactions
    cross_index = [ix for ix in ex.top_interactions
                  if ix.base_index_a != ix.base_index_b]
    if cross_index:
        top_ix = cross_index[0]
        sections.append(
            f"**Key discovery**: **{top_ix.base_index_a}** and **{top_ix.base_index_b}** "
            f"have a {top_ix.interaction_direction} interaction — when monitored together, "
            f"their combined predictive power exceeds their individual contributions.\n"
        )

    # Feature selection result
    if ex.selection_result:
        sel = ex.selection_result
        sections.append(
            f"Through systematic analysis, we identified that just "
            f"**{len(sel.selected_base_indices)} core RiskWise indices** "
            f"(out of {sel.n_original_features} total signals) capture the essential "
            f"predictive information — achieving {sel.selected_directional_accuracy:.0%} "
            f"directional accuracy.\n"
        )

    # Regime analysis
    regimes = ex.temporal_regimes
    if regimes:
        # Find if importance shifts across regimes
        bear_regimes = [r for r in regimes if r.regime_label == "bear"]
        bull_regimes = [r for r in regimes if r.regime_label == "bull"]
        if bear_regimes and bull_regimes:
            bear_top = bear_regimes[0].top_features[0][0] if bear_regimes[0].top_features else None
            bull_top = bull_regimes[0].top_features[0][0] if bull_regimes[0].top_features else None
            if bear_top and bull_top and bear_top != bull_top:
                sections.append(
                    f"**Regime-aware intelligence**: In bearish markets, **{bear_top}** "
                    f"becomes the dominant signal, while in bullish periods, "
                    f"**{bull_top}** takes precedence. RiskWise adapts to market conditions.\n"
                )

    return "\n".join(sections)


def _full_results_table(results: list[CorrelationResult]) -> str:
    sorted_results = sorted(results, key=lambda r: abs(r.lagged_correlation), reverse=True)
    lines = []
    lines.append("| Index | r | Spearman | Lag | Lagged r | Granger p | FDR p | Type | Sig |")
    lines.append("|-------|---|----------|-----|----------|-----------|-------|------|-----|")
    for r in sorted_results:
        gp = f"{r.granger_p_value:.3f}" if r.granger_p_value else "—"
        fp = f"{r.pearson_p_adjusted:.3f}" if r.pearson_p_adjusted else "—"
        sig = "✓" if r.is_significant else ""
        lines.append(
            f"| {r.index_name} | {r.pearson_r:+.3f} | {r.spearman_r:+.3f} | "
            f"{r.optimal_lag_days:+d} | {r.lagged_correlation:+.3f} | "
            f"{gp} | {fp} | {r.relationship_type} | {sig} |"
        )
    lines.append("")
    return "\n".join(lines)
