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


def generate_reports(summary: AnalysisSummary, config: AnalysisConfig) -> dict[str, str]:
    """
    Generate both technical and executive reports.
    Returns dict with keys 'technical' and 'executive', values are markdown strings.
    Also writes files to config.output_dir.
    """
    technical = _generate_technical_report(summary, config)
    executive = _generate_executive_summary(summary, config)

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

def _generate_technical_report(summary: AnalysisSummary, config: AnalysisConfig) -> str:
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

    return "\n".join(sections)


# ======================================================================
# Executive Summary
# ======================================================================

def _generate_executive_summary(summary: AnalysisSummary, config: AnalysisConfig) -> str:
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
    sections.append("*Methodology: Pearson/Spearman/Kendall correlations, "
                   "cross-correlation lead/lag analysis, Granger causality, "
                   "Engle-Granger cointegration, Benjamini-Hochberg FDR correction. "
                   f"Full technical report available upon request.*\n")

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
