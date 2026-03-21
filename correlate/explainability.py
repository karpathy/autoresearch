"""
Explainability-driven signal discovery for RiskWise index analysis.

Stage 3 of the pipeline: uses SHAP as an active tool for finding signal, not just
explaining models after the fact.

Key analyses:
1. SHAP feature importance — global and per-sample
2. SHAP interaction values — which pairs of indices have synergistic predictive power
3. SHAP temporal dynamics — how feature importance changes over time (regime detection)
4. SHAP-driven feature selection — iteratively prune features using SHAP rankings
5. Partial dependence — how each index's value maps to predicted outcomes
6. Cohort analysis — SHAP values segmented by market regime (bull/bear/volatile)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig
from .ml import MLConfig, MLSummary, build_features, walk_forward_split

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ======================================================================
# Result data classes
# ======================================================================

@dataclass
class SHAPFeatureResult:
    """SHAP-derived importance for a single feature."""
    feature_name: str
    base_index_name: str          # Underlying RiskWise index name
    mean_abs_shap: float          # Mean |SHAP| across samples
    mean_signed_shap: float       # Mean signed SHAP (direction of effect)
    std_shap: float               # Variability of SHAP values
    max_abs_shap: float           # Maximum impact observed
    direction: str                # "positive" or "negative"
    # Percentile-based importance (robust to outliers)
    median_abs_shap: float = 0.0
    p95_abs_shap: float = 0.0    # 95th percentile of |SHAP|


@dataclass
class SHAPInteraction:
    """SHAP interaction between two features."""
    feature_a: str
    feature_b: str
    base_index_a: str
    base_index_b: str
    interaction_strength: float    # Mean absolute interaction SHAP value
    interaction_direction: str     # "synergistic" or "antagonistic"
    # When both are high, does prediction go up or down?
    joint_effect_sign: str         # "positive" or "negative"


@dataclass
class SHAPTemporalRegime:
    """SHAP importance within a time period / regime."""
    period_start: str
    period_end: str
    regime_label: str              # "bull", "bear", "volatile", "calm"
    n_samples: int
    top_features: list[tuple[str, float]]  # (feature_name, mean_abs_shap)
    regime_market_return: float    # Market return during this period


@dataclass
class PartialDependence:
    """Partial dependence of prediction on a single feature."""
    feature_name: str
    base_index_name: str
    grid_values: list[float]       # Feature values
    mean_predictions: list[float]  # Mean predicted value at each grid point
    std_predictions: list[float]   # Std of predictions (confidence band)
    monotonic: bool                # Is the relationship monotonic?
    direction: str                 # "increasing" or "decreasing" (if monotonic)


@dataclass
class SHAPSelectionResult:
    """Result of SHAP-driven iterative feature selection."""
    n_original_features: int
    n_selected_features: int
    selected_features: list[str]
    selected_base_indices: list[str]
    # Performance with selected vs all features
    full_directional_accuracy: float
    selected_directional_accuracy: float
    improvement: float             # Accuracy gain from selection (can be positive!)


@dataclass
class ExplainabilitySummary:
    """Complete explainability analysis results."""
    target_ticker: str
    target_name: str
    model_name: str
    horizon_days: int

    # Core SHAP results
    feature_rankings: list[SHAPFeatureResult]
    base_index_rankings: list[tuple[str, float]]  # Aggregated to base index level

    # Interactions
    top_interactions: list[SHAPInteraction]

    # Temporal dynamics
    temporal_regimes: list[SHAPTemporalRegime]
    importance_stability: dict[str, float]  # feature -> correlation of SHAP across time halves

    # Partial dependence
    partial_dependences: list[PartialDependence]

    # Feature selection
    selection_result: SHAPSelectionResult | None = None

    # Raw SHAP matrix for downstream use (not serialized in reports)
    shap_values_matrix: np.ndarray | None = field(default=None, repr=False)
    feature_names: list[str] = field(default_factory=list)


# ======================================================================
# Main explainability engine
# ======================================================================

class ExplainabilityAnalyzer:
    """
    Uses SHAP as an active signal discovery tool.

    Rather than just reporting feature importances, this module:
    1. Identifies which indices genuinely drive predictions (vs noise)
    2. Discovers synergistic interactions between indices
    3. Detects regime changes in feature importance over time
    4. Maps out exactly how each index value translates to predicted outcomes
    5. Performs SHAP-guided feature selection to find the minimal predictive set
    """

    def __init__(self, config: AnalysisConfig, ml_config: MLConfig | None = None):
        self.config = config
        self.ml_config = ml_config or MLConfig()

    def run(
        self,
        index_data: pd.DataFrame,
        target_series: pd.Series,
        ml_summary: MLSummary | None = None,
        horizon: int | None = None,
        model_name: str | None = None,
    ) -> ExplainabilitySummary | None:
        """
        Run full SHAP explainability analysis.

        Args:
            index_data: Wide-format index DataFrame (resampled to business days)
            target_series: Target variable (returns or prices)
            ml_summary: Optional — used to pick best model/horizon
            horizon: Override which horizon to analyze
            model_name: Override which model to explain
        """
        if not HAS_SHAP:
            logger.error("shap package not installed — cannot run explainability analysis")
            return None

        # Pick horizon and model
        if horizon is None:
            if ml_summary and ml_summary.best_model_per_horizon:
                # Use the longest horizon (most interesting for sales)
                horizon = max(ml_summary.best_model_per_horizon.keys())
            else:
                horizon = 20  # Default

        if model_name is None:
            if ml_summary and horizon in ml_summary.best_model_per_horizon:
                model_name = ml_summary.best_model_per_horizon[horizon].model_name
            elif HAS_XGBOOST:
                model_name = "XGBoost"
            else:
                model_name = "RandomForest"

        logger.info("Explainability analysis: %s model, %d-day horizon", model_name, horizon)

        # Select indices (use ML summary's feature set if available)
        selected = None
        if ml_summary and ml_summary.feature_names:
            # Extract base index names from the ML feature names
            base_names = set()
            for fname in ml_summary.feature_names:
                base = self._extract_base_name(fname)
                base_names.add(base)
            selected = [c for c in index_data.columns
                       if c.replace(" ", "_")[:30] in base_names]

        # Build features and target
        features, targets = build_features(
            index_data, target_series, self.ml_config, selected
        )
        all_data = pd.concat(
            [features, targets[horizon].rename("target")], axis=1
        ).dropna()

        if len(all_data) < self.ml_config.min_train_days + 50:
            logger.error("Insufficient data for explainability: %d rows", len(all_data))
            return None

        feature_cols = [c for c in all_data.columns if c != "target"]
        X = all_data[feature_cols]
        y = all_data["target"]
        feature_names = list(X.columns)

        logger.info("Explainability matrix: %d rows × %d features", len(X), len(feature_names))

        # Train the model on full data for global explanations
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        model = self._build_model(model_name)
        model.fit(X_scaled, y.values)

        # ---- 1. Compute SHAP values ----
        logger.info("Computing SHAP values...")
        shap_values, explainer = self._compute_shap(model, X_scaled, model_name)

        # ---- 2. Feature rankings ----
        logger.info("Ranking features by SHAP importance...")
        feature_rankings = self._rank_features(shap_values, feature_names)

        # Aggregate to base index level
        base_rankings = self._aggregate_to_base_indices(feature_rankings)

        # ---- 3. Interaction values ----
        logger.info("Computing SHAP interactions...")
        interactions = self._compute_interactions(
            model, X_scaled, feature_names, model_name
        )

        # ---- 4. Temporal dynamics ----
        logger.info("Analyzing temporal SHAP dynamics...")
        temporal, stability = self._temporal_analysis(
            shap_values, X, y, feature_names
        )

        # ---- 5. Partial dependence ----
        logger.info("Computing partial dependence curves...")
        pd_results = self._partial_dependence(
            model, X_scaled, feature_names, feature_rankings[:20]
        )

        # ---- 6. SHAP-driven feature selection ----
        logger.info("Running SHAP-driven feature selection...")
        selection = self._shap_feature_selection(
            X_scaled, y.values, feature_names, feature_rankings,
            model_name, horizon
        )

        logger.info("Explainability analysis complete")
        logger.info("  Top feature: %s (SHAP = %.6f)",
                    feature_rankings[0].feature_name if feature_rankings else "N/A",
                    feature_rankings[0].mean_abs_shap if feature_rankings else 0)
        logger.info("  Top base index: %s",
                    base_rankings[0][0] if base_rankings else "N/A")
        logger.info("  Interactions found: %d", len(interactions))
        logger.info("  Temporal regimes: %d", len(temporal))

        return ExplainabilitySummary(
            target_ticker=self.config.target_ticker,
            target_name=self.config.target_name,
            model_name=model_name,
            horizon_days=horizon,
            feature_rankings=feature_rankings,
            base_index_rankings=base_rankings,
            top_interactions=interactions,
            temporal_regimes=temporal,
            importance_stability=stability,
            partial_dependences=pd_results,
            selection_result=selection,
            shap_values_matrix=shap_values,
            feature_names=feature_names,
        )

    # ==================================================================
    # SHAP computation
    # ==================================================================

    def _compute_shap(
        self,
        model,
        X_scaled: np.ndarray,
        model_name: str,
    ) -> tuple[np.ndarray, shap.Explainer]:
        """Compute SHAP values using the appropriate explainer."""
        # Sample for large datasets
        max_samples = 2000
        if len(X_scaled) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_scaled), max_samples, replace=False)
            idx.sort()  # Keep temporal order
            X_sample = X_scaled[idx]
        else:
            X_sample = X_scaled

        if model_name in ("XGBoost", "RandomForest"):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer as fallback for linear models (more features than LinearExplainer)
            background = shap.kmeans(X_scaled, min(50, len(X_scaled)))
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer.shap_values(X_sample)
        return shap_values, explainer

    # ==================================================================
    # Feature ranking
    # ==================================================================

    def _rank_features(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> list[SHAPFeatureResult]:
        """Rank features by SHAP importance with detailed statistics."""
        results = []
        abs_shap = np.abs(shap_values)

        for i, fname in enumerate(feature_names):
            col_shap = shap_values[:, i]
            col_abs = abs_shap[:, i]

            mean_signed = float(np.mean(col_shap))
            results.append(SHAPFeatureResult(
                feature_name=fname,
                base_index_name=self._extract_base_name(fname),
                mean_abs_shap=float(np.mean(col_abs)),
                mean_signed_shap=mean_signed,
                std_shap=float(np.std(col_shap)),
                max_abs_shap=float(np.max(col_abs)),
                direction="positive" if mean_signed > 0 else "negative",
                median_abs_shap=float(np.median(col_abs)),
                p95_abs_shap=float(np.percentile(col_abs, 95)),
            ))

        results.sort(key=lambda r: r.mean_abs_shap, reverse=True)
        return results

    def _aggregate_to_base_indices(
        self,
        feature_rankings: list[SHAPFeatureResult],
    ) -> list[tuple[str, float]]:
        """Aggregate SHAP importance back to underlying RiskWise index names."""
        base_totals: dict[str, float] = {}
        for fr in feature_rankings:
            base = fr.base_index_name
            base_totals[base] = base_totals.get(base, 0) + fr.mean_abs_shap
        return sorted(base_totals.items(), key=lambda x: x[1], reverse=True)

    # ==================================================================
    # Interaction analysis
    # ==================================================================

    def _compute_interactions(
        self,
        model,
        X_scaled: np.ndarray,
        feature_names: list[str],
        model_name: str,
        max_pairs: int = 20,
    ) -> list[SHAPInteraction]:
        """
        Compute SHAP interaction values between features.

        For tree models, uses exact TreeSHAP interactions.
        For others, approximates via correlation of SHAP values.
        """
        interactions = []

        if model_name in ("XGBoost", "RandomForest"):
            try:
                # TreeSHAP interaction values (exact but expensive)
                explainer = shap.TreeExplainer(model)
                # Use a small sample for interaction computation (O(n * features²))
                sample_size = min(300, len(X_scaled))
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_scaled), sample_size, replace=False)
                X_sample = X_scaled[idx]

                logger.info("Computing TreeSHAP interaction values (%d samples)...", sample_size)
                interaction_values = explainer.shap_interaction_values(X_sample)
                # interaction_values shape: (n_samples, n_features, n_features)

                # Find strongest off-diagonal interactions
                n_features = len(feature_names)
                pair_strengths = []
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        strength = float(np.mean(np.abs(interaction_values[:, i, j])))
                        pair_strengths.append((i, j, strength))

                pair_strengths.sort(key=lambda x: x[2], reverse=True)

                for i, j, strength in pair_strengths[:max_pairs]:
                    if strength < 1e-8:
                        continue
                    mean_interaction = float(np.mean(interaction_values[:, i, j]))
                    interactions.append(SHAPInteraction(
                        feature_a=feature_names[i],
                        feature_b=feature_names[j],
                        base_index_a=self._extract_base_name(feature_names[i]),
                        base_index_b=self._extract_base_name(feature_names[j]),
                        interaction_strength=strength,
                        interaction_direction="synergistic" if mean_interaction > 0 else "antagonistic",
                        joint_effect_sign="positive" if mean_interaction > 0 else "negative",
                    ))

            except Exception as e:
                logger.warning("TreeSHAP interactions failed: %s. Using correlation fallback.", e)
                interactions = self._correlation_based_interactions(
                    X_scaled, feature_names, model, max_pairs
                )
        else:
            interactions = self._correlation_based_interactions(
                X_scaled, feature_names, model, max_pairs
            )

        return interactions

    def _correlation_based_interactions(
        self,
        X_scaled: np.ndarray,
        feature_names: list[str],
        model,
        max_pairs: int,
    ) -> list[SHAPInteraction]:
        """
        Approximate interactions by looking at how pairs of features jointly
        affect predictions beyond their individual effects.
        """
        interactions = []

        # Compute SHAP values
        background = shap.kmeans(X_scaled, min(50, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict, background)
        sample_size = min(200, len(X_scaled))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), sample_size, replace=False)
        shap_vals = explainer.shap_values(X_scaled[idx])

        # Look for correlated SHAP values (features that amplify each other)
        n_features = min(30, len(feature_names))  # Top 30 to limit computation
        abs_mean = np.mean(np.abs(shap_vals), axis=0)
        top_idx = np.argsort(abs_mean)[-n_features:]

        pair_strengths = []
        for ii, i in enumerate(top_idx):
            for j in top_idx[ii + 1:]:
                # Interaction = correlation of |SHAP_i| * sign(SHAP_j)
                corr = float(np.abs(np.corrcoef(shap_vals[:, i], shap_vals[:, j])[0, 1]))
                if not np.isnan(corr):
                    pair_strengths.append((i, j, corr))

        pair_strengths.sort(key=lambda x: x[2], reverse=True)
        for i, j, strength in pair_strengths[:max_pairs]:
            if strength < 0.1:
                continue
            interactions.append(SHAPInteraction(
                feature_a=feature_names[i],
                feature_b=feature_names[j],
                base_index_a=self._extract_base_name(feature_names[i]),
                base_index_b=self._extract_base_name(feature_names[j]),
                interaction_strength=strength,
                interaction_direction="synergistic",
                joint_effect_sign="positive",
            ))

        return interactions

    # ==================================================================
    # Temporal SHAP dynamics
    # ==================================================================

    def _temporal_analysis(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        n_windows: int = 6,
    ) -> tuple[list[SHAPTemporalRegime], dict[str, float]]:
        """
        Analyze how SHAP values evolve over time.

        Splits data into time windows and computes SHAP importance per window.
        Detects regime changes in which features matter.
        """
        n_samples = len(shap_values)
        dates = X.index[:n_samples]  # SHAP may use a subsample
        window_size = n_samples // n_windows

        regimes = []
        # Track per-feature SHAP by time window for stability analysis
        feature_shap_by_window: dict[str, list[float]] = {f: [] for f in feature_names}

        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, n_samples)
            if end - start < 10:
                continue

            window_shap = shap_values[start:end]
            window_dates = dates[start:end]

            # Identify market regime from target returns
            if start < len(y) and end <= len(y):
                window_y = y.iloc[start:end]
                cum_return = float((1 + window_y).prod() - 1) if len(window_y) > 0 else 0
                volatility = float(window_y.std()) if len(window_y) > 0 else 0
                median_vol = float(y.std())

                if cum_return > 0.05:
                    regime = "bull"
                elif cum_return < -0.05:
                    regime = "bear"
                elif volatility > median_vol * 1.5:
                    regime = "volatile"
                else:
                    regime = "calm"
            else:
                cum_return = 0.0
                regime = "unknown"

            # Top features in this window
            mean_abs = np.mean(np.abs(window_shap), axis=0)
            top_idx = np.argsort(mean_abs)[-10:][::-1]
            top_features = [(feature_names[i], float(mean_abs[i])) for i in top_idx]

            regimes.append(SHAPTemporalRegime(
                period_start=str(window_dates[0].date()) if len(window_dates) > 0 else "",
                period_end=str(window_dates[-1].date()) if len(window_dates) > 0 else "",
                regime_label=regime,
                n_samples=end - start,
                top_features=top_features,
                regime_market_return=cum_return,
            ))

            # Track per-feature importance for stability
            for i, fname in enumerate(feature_names):
                feature_shap_by_window[fname].append(float(mean_abs[i]))

        # Compute stability: correlation of importance between first and second half
        stability = {}
        for fname in feature_names:
            vals = feature_shap_by_window[fname]
            if len(vals) >= 4:
                mid = len(vals) // 2
                first_half = np.array(vals[:mid])
                second_half = np.array(vals[mid:2 * mid])
                if len(first_half) == len(second_half) and len(first_half) > 1:
                    # Stability = 1 - coefficient of variation across windows
                    all_vals = np.array(vals)
                    if np.mean(all_vals) > 0:
                        cv = float(np.std(all_vals) / np.mean(all_vals))
                        stability[fname] = max(0, 1 - cv)  # 1 = perfectly stable
                    else:
                        stability[fname] = 0.0
                else:
                    stability[fname] = 0.0
            else:
                stability[fname] = 0.0

        return regimes, stability

    # ==================================================================
    # Partial dependence
    # ==================================================================

    def _partial_dependence(
        self,
        model,
        X_scaled: np.ndarray,
        feature_names: list[str],
        top_features: list[SHAPFeatureResult],
        n_grid: int = 50,
    ) -> list[PartialDependence]:
        """
        Compute partial dependence plots for top features.

        For each feature, varies it across its range while holding others fixed,
        and records the average model prediction.
        """
        results = []
        # Use a sample for efficiency
        sample_size = min(500, len(X_scaled))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx].copy()

        for fr in top_features[:15]:  # Top 15 features
            feat_idx = feature_names.index(fr.feature_name)
            feat_values = X_scaled[:, feat_idx]
            grid = np.linspace(
                np.percentile(feat_values, 2),
                np.percentile(feat_values, 98),
                n_grid,
            )

            mean_preds = []
            std_preds = []
            for val in grid:
                X_temp = X_sample.copy()
                X_temp[:, feat_idx] = val
                preds = model.predict(X_temp)
                mean_preds.append(float(np.mean(preds)))
                std_preds.append(float(np.std(preds)))

            # Check monotonicity
            diffs = np.diff(mean_preds)
            is_monotonic = bool(np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10))
            direction = ""
            if is_monotonic:
                direction = "increasing" if np.sum(diffs) > 0 else "decreasing"

            results.append(PartialDependence(
                feature_name=fr.feature_name,
                base_index_name=fr.base_index_name,
                grid_values=[float(v) for v in grid],
                mean_predictions=mean_preds,
                std_predictions=std_preds,
                monotonic=is_monotonic,
                direction=direction,
            ))

        return results

    # ==================================================================
    # SHAP-driven feature selection
    # ==================================================================

    def _shap_feature_selection(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        feature_rankings: list[SHAPFeatureResult],
        model_name: str,
        horizon: int,
        min_features: int = 5,
    ) -> SHAPSelectionResult | None:
        """
        Iterative SHAP-guided feature selection.

        Start with all features, then progressively prune the least important
        until we find the minimal feature set that preserves (or improves)
        directional accuracy.
        """
        from scipy import stats as sp_stats
        from sklearn.metrics import r2_score

        splits = walk_forward_split(
            len(X_scaled), self.ml_config.n_splits, self.ml_config.min_train_days
        )
        if not splits:
            return None

        # Evaluate full feature set
        full_acc = self._eval_directional_accuracy(
            X_scaled, y, splits, model_name
        )

        # Rank features by SHAP and try progressively smaller subsets
        ranked_indices = [
            feature_names.index(fr.feature_name)
            for fr in feature_rankings
            if fr.feature_name in feature_names
        ]

        best_acc = full_acc
        best_n = len(feature_names)
        best_features = ranked_indices.copy()

        # Try: top 75%, 50%, 30%, 20%, 15, 10, 5 features
        candidates = set()
        for frac in [0.75, 0.5, 0.3, 0.2]:
            candidates.add(max(min_features, int(len(ranked_indices) * frac)))
        for n in [15, 10, 5]:
            if n < len(ranked_indices):
                candidates.add(n)

        for n_feat in sorted(candidates, reverse=True):
            subset = ranked_indices[:n_feat]
            X_sub = X_scaled[:, subset]
            acc = self._eval_directional_accuracy(X_sub, y, splits, model_name)

            logger.info("  %d features: dir_acc = %.3f", n_feat, acc)

            # Keep if accuracy is at least as good (within 1% tolerance)
            if acc >= best_acc - 0.01:
                best_acc = acc
                best_n = n_feat
                best_features = subset

        selected_names = [feature_names[i] for i in best_features[:best_n]]
        selected_bases = list(dict.fromkeys(
            self._extract_base_name(f) for f in selected_names
        ))

        return SHAPSelectionResult(
            n_original_features=len(feature_names),
            n_selected_features=best_n,
            selected_features=selected_names,
            selected_base_indices=selected_bases,
            full_directional_accuracy=full_acc,
            selected_directional_accuracy=best_acc,
            improvement=best_acc - full_acc,
        )

    def _eval_directional_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        model_name: str,
    ) -> float:
        """Quick directional accuracy evaluation for feature selection."""
        correct = 0
        total = 0
        scaler = StandardScaler()

        for train_idx, test_idx in splits:
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                model = self._build_model(model_name)
                model.fit(X_train, y[train_idx])
                preds = model.predict(X_test)
                correct += np.sum(np.sign(preds) == np.sign(y[test_idx]))
                total += len(test_idx)
            except Exception:
                continue

        return correct / total if total > 0 else 0.0

    # ==================================================================
    # Helpers
    # ==================================================================

    def _build_model(self, model_name: str):
        """Build a fresh model instance by name."""
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        rs = self.ml_config.random_state
        if model_name == "XGBoost" and HAS_XGBOOST:
            return xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=rs, n_jobs=-1, verbosity=0,
            )
        elif model_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=10,
                random_state=rs, n_jobs=-1,
            )
        else:
            return Ridge(alpha=1.0)

    @staticmethod
    def _extract_base_name(feature_name: str) -> str:
        """Extract the base RiskWise index name from an engineered feature name."""
        for suffix in ["_lag", "_ma", "_std", "_mom", "_roc"]:
            if suffix in feature_name:
                return feature_name[:feature_name.index(suffix)]
        return feature_name
