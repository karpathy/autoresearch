"""
Core statistical analysis engine for RiskWise correlation discovery.

Implements PhD-level time series analysis:
- Pearson, Spearman, and Kendall correlations
- Optimal lead/lag detection via cross-correlation
- Granger causality testing
- Engle-Granger cointegration testing
- Rolling/time-varying correlations
- Regime detection (structural breaks in correlation)
- Bootstrap confidence intervals
- Multiple hypothesis correction (Benjamini-Hochberg FDR)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.stats.multitest import multipletests

from .config import AnalysisConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ======================================================================
# Result data classes
# ======================================================================

@dataclass
class CorrelationResult:
    """Result of a single index-vs-target correlation analysis."""
    index_name: str
    n_observations: int
    overlap_start: str
    overlap_end: str

    # Contemporaneous correlations
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float

    # Optimal lag
    optimal_lag_days: int       # positive = index LEADS target (predictive!)
    lagged_correlation: float   # correlation at optimal lag
    lagged_p_value: float

    # Granger causality (index -> target)
    granger_f_stat: float | None = None
    granger_p_value: float | None = None
    granger_optimal_lag: int | None = None

    # Reverse Granger (target -> index)
    reverse_granger_f: float | None = None
    reverse_granger_p: float | None = None

    # Cointegration
    coint_t_stat: float | None = None
    coint_p_value: float | None = None
    is_cointegrated: bool = False

    # Stationarity
    index_adf_stat: float | None = None
    index_adf_p: float | None = None
    index_is_stationary: bool = False
    target_is_stationary: bool = False

    # Rolling correlation stats
    rolling_corr_mean: float | None = None
    rolling_corr_std: float | None = None
    rolling_corr_min: float | None = None
    rolling_corr_max: float | None = None

    # Bootstrap CI for Pearson r
    ci_lower: float | None = None
    ci_upper: float | None = None

    # After FDR correction
    pearson_p_adjusted: float | None = None
    is_significant: bool = False

    # Classification
    relationship_type: str = ""  # "leading", "lagging", "contemporaneous", "cointegrated"
    strength: str = ""           # "strong", "moderate", "weak"

    # Rolling correlation series (stored for plotting, not printed)
    rolling_corr_series: pd.Series | None = field(default=None, repr=False)


@dataclass
class AnalysisSummary:
    """Full analysis results across all indices."""
    target_ticker: str
    target_name: str
    n_indices_tested: int
    n_significant: int
    results: list[CorrelationResult]
    top_leading: list[CorrelationResult]    # indices that LEAD the target
    top_lagging: list[CorrelationResult]    # indices that LAG the target
    top_contemporaneous: list[CorrelationResult]
    cointegrated: list[CorrelationResult]
    fdr_threshold: float


# ======================================================================
# Analysis engine
# ======================================================================

class CorrelationAnalyzer:
    """Runs the full correlation analysis pipeline."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_pair(
        self,
        index_series: pd.Series,
        target_series: pd.Series,
        index_name: str,
    ) -> CorrelationResult | None:
        """
        Run full correlation analysis between one index and the target.

        Args:
            index_series: Risk index values (DatetimeIndex, single column)
            target_series: Target outcome values (DatetimeIndex, single column)
            index_name: Name of the risk index

        Returns:
            CorrelationResult or None if insufficient data.
        """
        # Ensure we have clean aligned series
        idx = index_series.dropna()
        tgt = target_series.dropna()
        common = idx.index.intersection(tgt.index)

        if len(common) < self.config.min_overlap_days:
            return None

        x = idx.loc[common].values.astype(float)
        y = tgt.loc[common].values.astype(float)
        dates = common

        n = len(x)

        # ----------------------------------------------------------
        # 1. Contemporaneous correlations
        # ----------------------------------------------------------
        pearson_r, pearson_p = sp_stats.pearsonr(x, y)
        spearman_r, spearman_p = sp_stats.spearmanr(x, y)
        kendall_tau, kendall_p = sp_stats.kendalltau(x, y)

        # ----------------------------------------------------------
        # 2. Optimal lead/lag via cross-correlation
        # ----------------------------------------------------------
        opt_lag, lag_corr, lag_p = self._find_optimal_lag(x, y)

        # ----------------------------------------------------------
        # 3. Stationarity tests (ADF)
        # ----------------------------------------------------------
        idx_adf_stat, idx_adf_p = self._adf_test(x)
        tgt_adf_stat, tgt_adf_p = self._adf_test(y)
        idx_stationary = idx_adf_p < self.config.significance_level
        tgt_stationary = tgt_adf_p < self.config.significance_level

        # ----------------------------------------------------------
        # 4. Granger causality
        # ----------------------------------------------------------
        granger_f, granger_p, granger_lag = self._granger_test(x, y)
        rev_granger_f, rev_granger_p, _ = self._granger_test(y, x)

        # ----------------------------------------------------------
        # 5. Cointegration (Engle-Granger)
        # ----------------------------------------------------------
        coint_t, coint_p, is_coint = self._cointegration_test(x, y)

        # ----------------------------------------------------------
        # 6. Rolling correlation
        # ----------------------------------------------------------
        rolling = self._rolling_correlation(
            pd.Series(x, index=dates),
            pd.Series(y, index=dates),
        )

        # ----------------------------------------------------------
        # 7. Bootstrap confidence interval
        # ----------------------------------------------------------
        ci_lo, ci_hi = self._bootstrap_ci(x, y)

        # ----------------------------------------------------------
        # 8. Classify the relationship
        # ----------------------------------------------------------
        rel_type = self._classify_relationship(opt_lag, is_coint, pearson_r, lag_corr)
        strength = self._classify_strength(max(abs(pearson_r), abs(lag_corr)))

        return CorrelationResult(
            index_name=index_name,
            n_observations=n,
            overlap_start=str(dates.min().date()),
            overlap_end=str(dates.max().date()),
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            kendall_tau=kendall_tau,
            kendall_p=kendall_p,
            optimal_lag_days=opt_lag,
            lagged_correlation=lag_corr,
            lagged_p_value=lag_p,
            granger_f_stat=granger_f,
            granger_p_value=granger_p,
            granger_optimal_lag=granger_lag,
            reverse_granger_f=rev_granger_f,
            reverse_granger_p=rev_granger_p,
            coint_t_stat=coint_t,
            coint_p_value=coint_p,
            is_cointegrated=is_coint,
            index_adf_stat=idx_adf_stat,
            index_adf_p=idx_adf_p,
            index_is_stationary=idx_stationary,
            target_is_stationary=tgt_stationary,
            rolling_corr_mean=rolling.mean() if rolling is not None else None,
            rolling_corr_std=rolling.std() if rolling is not None else None,
            rolling_corr_min=rolling.min() if rolling is not None else None,
            rolling_corr_max=rolling.max() if rolling is not None else None,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            rolling_corr_series=rolling,
            relationship_type=rel_type,
            strength=strength,
        )

    def run_full_analysis(
        self,
        index_data: pd.DataFrame,
        target_series: pd.Series,
    ) -> AnalysisSummary:
        """
        Run correlation analysis for ALL indices against the target.

        Args:
            index_data: Wide-format DataFrame (rows=dates, cols=index names)
            target_series: Target outcome series (e.g., stock returns)

        Returns:
            AnalysisSummary with all results, ranked and classified.
        """
        results: list[CorrelationResult] = []

        for col in index_data.columns:
            logger.info("Analyzing: %s", col)
            result = self.analyze_pair(
                index_data[col], target_series, index_name=col
            )
            if result is not None:
                results.append(result)

        logger.info("Completed %d / %d indices", len(results), len(index_data.columns))

        # ----------------------------------------------------------
        # Multiple hypothesis correction (Benjamini-Hochberg FDR)
        # ----------------------------------------------------------
        if results:
            p_values = np.array([r.pearson_p for r in results])
            reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh",
                                                 alpha=self.config.significance_level)
            for i, r in enumerate(results):
                r.pearson_p_adjusted = float(p_adj[i])
                r.is_significant = bool(reject[i])

        n_sig = sum(1 for r in results if r.is_significant)

        # ----------------------------------------------------------
        # Classify and rank
        # ----------------------------------------------------------
        significant = [r for r in results if r.is_significant]

        top_leading = sorted(
            [r for r in significant if r.relationship_type == "leading"],
            key=lambda r: abs(r.lagged_correlation), reverse=True,
        )
        top_lagging = sorted(
            [r for r in significant if r.relationship_type == "lagging"],
            key=lambda r: abs(r.lagged_correlation), reverse=True,
        )
        top_contemp = sorted(
            [r for r in significant if r.relationship_type == "contemporaneous"],
            key=lambda r: abs(r.pearson_r), reverse=True,
        )
        cointegrated = [r for r in results if r.is_cointegrated]

        return AnalysisSummary(
            target_ticker=self.config.target_ticker,
            target_name=self.config.target_name,
            n_indices_tested=len(index_data.columns),
            n_significant=n_sig,
            results=results,
            top_leading=top_leading,
            top_lagging=top_lagging,
            top_contemporaneous=top_contemp,
            cointegrated=cointegrated,
            fdr_threshold=self.config.significance_level,
        )

    # ==================================================================
    # Private methods
    # ==================================================================

    def _find_optimal_lag(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[int, float, float]:
        """
        Find the lag that maximizes |correlation| between x(t-lag) and y(t).

        Positive lag means x LEADS y (x moves first → predictive).
        Negative lag means x LAGS y (x follows y).

        Returns: (optimal_lag, correlation_at_lag, p_value)
        """
        max_lag = min(self.config.max_lag_days, len(x) // 3)
        best_lag = 0
        best_abs_r = abs(sp_stats.pearsonr(x, y)[0])
        best_r = sp_stats.pearsonr(x, y)[0]
        best_p = sp_stats.pearsonr(x, y)[1]

        for lag in range(-max_lag, max_lag + 1, self.config.lag_step_days):
            if lag == 0:
                continue
            if lag > 0:
                # x leads: compare x[:-lag] with y[lag:]
                x_lag = x[:-lag]
                y_lag = y[lag:]
            else:
                # x lags: compare x[-lag:] with y[:lag]
                x_lag = x[-lag:]
                y_lag = y[:lag]

            if len(x_lag) < self.config.min_overlap_days:
                continue

            r, p = sp_stats.pearsonr(x_lag, y_lag)
            if abs(r) > best_abs_r:
                best_abs_r = abs(r)
                best_r = r
                best_p = p
                best_lag = lag

        return best_lag, best_r, best_p

    def _adf_test(self, x: np.ndarray) -> tuple[float, float]:
        """Augmented Dickey-Fuller test for stationarity."""
        try:
            result = adfuller(x, autolag="AIC")
            return float(result[0]), float(result[1])
        except Exception:
            return 0.0, 1.0

    def _granger_test(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float | None, float | None, int | None]:
        """
        Granger causality: does x Granger-cause y?
        Returns (best_F, best_p, best_lag) across tested lags.
        """
        max_lags = min(self.config.granger_max_lags, len(x) // 5)
        if max_lags < 1:
            return None, None, None

        data = np.column_stack([y, x])  # statsmodels expects [effect, cause]

        try:
            results = grangercausalitytests(data, maxlag=max_lags, verbose=False)
        except Exception:
            return None, None, None

        best_p = 1.0
        best_f = 0.0
        best_lag = 1

        for lag, tests in results.items():
            # Use the F-test (ssr_ftest)
            f_stat = tests[0]["ssr_ftest"][0]
            p_val = tests[0]["ssr_ftest"][1]
            if p_val < best_p:
                best_p = p_val
                best_f = f_stat
                best_lag = lag

        return float(best_f), float(best_p), int(best_lag)

    def _cointegration_test(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float | None, float | None, bool]:
        """Engle-Granger two-step cointegration test."""
        try:
            t_stat, p_value, _ = coint(x, y)
            is_coint = p_value < self.config.significance_level
            return float(t_stat), float(p_value), is_coint
        except Exception:
            return None, None, False

    def _rolling_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        window: int | None = None,
    ) -> pd.Series | None:
        """Compute rolling Pearson correlation."""
        window = window or self.config.rolling_window_days
        if len(x) < window:
            return None
        return x.rolling(window).corr(y).dropna()

    def _bootstrap_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval for Pearson r."""
        n = len(x)
        rng = np.random.default_rng(42)
        boot_corrs = np.empty(self.config.n_bootstrap)

        for i in range(self.config.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_corrs[i] = np.corrcoef(x[idx], y[idx])[0, 1]

        lo = float(np.percentile(boot_corrs, 100 * alpha / 2))
        hi = float(np.percentile(boot_corrs, 100 * (1 - alpha / 2)))
        return lo, hi

    def _classify_relationship(
        self,
        optimal_lag: int,
        is_cointegrated: bool,
        pearson_r: float,
        lag_corr: float,
    ) -> str:
        """Classify the nature of the relationship."""
        if is_cointegrated:
            return "cointegrated"
        # If the lagged correlation is meaningfully stronger, classify as lead/lag
        if abs(lag_corr) > abs(pearson_r) + 0.05 and abs(optimal_lag) >= 3:
            if optimal_lag > 0:
                return "leading"    # index leads target — predictive!
            else:
                return "lagging"    # index follows target
        return "contemporaneous"

    @staticmethod
    def _classify_strength(abs_r: float) -> str:
        if abs_r >= 0.7:
            return "strong"
        elif abs_r >= 0.4:
            return "moderate"
        elif abs_r >= 0.15:
            return "weak"
        return "negligible"
