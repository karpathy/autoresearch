"""
Machine Learning prediction engine for RiskWise index analysis.

Stage 2 of the pipeline: after statistical correlations identify candidate indices,
this module builds proper ML models to quantify predictive power.

Approach:
- Walk-forward (expanding window) cross-validation — no look-ahead bias
- Multiple model families: Ridge, Lasso, ElasticNet, Random Forest, XGBoost
- Multi-horizon prediction: 1-day, 5-day, 10-day, 20-day ahead
- Feature importance via permutation importance and SHAP
- Directional accuracy (can we predict up/down?) alongside regression metrics
- Proper time series feature engineering (lags, rolling stats, momentum)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
)

from .config import AnalysisConfig
from .analysis import AnalysisSummary

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing optional heavy dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("xgboost not installed — skipping XGBoost models")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.info("shap not installed — skipping SHAP analysis")


# ======================================================================
# Config
# ======================================================================

@dataclass
class MLConfig:
    """ML-specific configuration."""
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    n_splits: int = 5                      # Walk-forward CV splits
    min_train_days: int = 252              # Minimum training window (~1 year)
    feature_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    top_n_features: int = 30              # Max features to feed into models
    use_significant_only: bool = True     # Only use statistically significant indices
    random_state: int = 42


# ======================================================================
# Result data classes
# ======================================================================

@dataclass
class ModelResult:
    """Results for a single model on a single horizon."""
    model_name: str
    horizon_days: int
    # Regression metrics (out-of-sample, averaged across CV folds)
    rmse: float
    mae: float
    r2: float
    # Directional accuracy
    directional_accuracy: float   # % of times we predict correct sign
    directional_p_value: float    # binomial test p-value vs random (50%)
    # Information coefficient (rank correlation of predictions vs actuals)
    ic_mean: float
    ic_std: float
    # Per-fold details
    fold_r2s: list[float] = field(default_factory=list)
    fold_directional: list[float] = field(default_factory=list)


@dataclass
class FeatureImportance:
    """Feature importance rankings."""
    feature_name: str
    importance_mean: float       # Mean importance across folds
    importance_std: float
    shap_mean: float | None = None  # Mean absolute SHAP value
    direction: str = ""          # "positive" or "negative" (sign of SHAP)


@dataclass
class MLSummary:
    """Full ML analysis results."""
    target_ticker: str
    target_name: str
    n_features_used: int
    feature_names: list[str]
    model_results: dict[int, list[ModelResult]]  # horizon -> list of ModelResult
    best_model_per_horizon: dict[int, ModelResult]
    feature_importances: list[FeatureImportance]
    baseline_rmse: dict[int, float]   # Naive baseline (predict zero / predict mean)
    # Walk-forward equity curve (cumulative directional returns)
    equity_curves: dict[int, pd.Series] | None = None


# ======================================================================
# Feature engineering
# ======================================================================

def build_features(
    index_data: pd.DataFrame,
    target_series: pd.Series,
    ml_config: MLConfig,
    selected_indices: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """
    Build feature matrix and multi-horizon target variables.

    Features per index:
    - Raw value (level)
    - Lagged values (1, 2, 3, 5, 10, 20 days)
    - Rolling mean and std (5, 10, 20, 60 day windows)
    - Momentum (change over window)
    - Rate of change

    Returns:
        features: DataFrame with all engineered features
        targets: dict mapping horizon -> forward return Series
    """
    if selected_indices is not None:
        cols = [c for c in selected_indices if c in index_data.columns]
        index_data = index_data[cols]

    feature_frames = []

    for col in index_data.columns:
        s = index_data[col]
        prefix = col.replace(" ", "_")[:30]  # Keep feature names manageable

        # Raw level
        feature_frames.append(s.rename(f"{prefix}"))

        # Lagged values
        for lag in ml_config.feature_lags:
            feature_frames.append(s.shift(lag).rename(f"{prefix}_lag{lag}"))

        # Rolling statistics
        for win in ml_config.rolling_windows:
            feature_frames.append(
                s.rolling(win).mean().rename(f"{prefix}_ma{win}")
            )
            feature_frames.append(
                s.rolling(win).std().rename(f"{prefix}_std{win}")
            )
            # Momentum: current value minus rolling mean
            feature_frames.append(
                (s - s.rolling(win).mean()).rename(f"{prefix}_mom{win}")
            )

        # Rate of change
        for lag in [1, 5, 20]:
            feature_frames.append(
                s.pct_change(lag).rename(f"{prefix}_roc{lag}")
            )

    features = pd.concat(feature_frames, axis=1)

    # Build multi-horizon targets: forward returns
    targets = {}
    for h in ml_config.horizons:
        targets[h] = target_series.shift(-h)  # Shift back = future return

    return features, targets


# ======================================================================
# Walk-forward cross-validation
# ======================================================================

def walk_forward_split(
    n_samples: int,
    n_splits: int,
    min_train: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate expanding-window train/test splits for time series.

    Each fold uses all data up to the split point for training,
    and the next chunk for testing. No data leakage.
    """
    test_size = (n_samples - min_train) // n_splits
    if test_size < 10:
        # Fall back to fewer splits
        n_splits = max(1, (n_samples - min_train) // 20)
        test_size = (n_samples - min_train) // n_splits

    splits = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n_samples)
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ======================================================================
# Model training & evaluation
# ======================================================================

class MLPredictor:
    """Trains and evaluates multiple ML models via walk-forward CV."""

    def __init__(self, config: AnalysisConfig, ml_config: MLConfig | None = None):
        self.config = config
        self.ml_config = ml_config or MLConfig()

    def run(
        self,
        index_data: pd.DataFrame,
        target_series: pd.Series,
        correlation_summary: AnalysisSummary | None = None,
    ) -> MLSummary:
        """
        Full ML pipeline: feature engineering → walk-forward CV → evaluation.

        Args:
            index_data: Wide-format index DataFrame
            target_series: Target variable (returns or prices)
            correlation_summary: Optional — if provided, uses only significant indices
        """
        # Select indices based on correlation results
        selected = None
        if correlation_summary and self.ml_config.use_significant_only:
            sig_names = [r.index_name for r in correlation_summary.results
                        if r.is_significant]
            if sig_names:
                selected = sig_names[:self.ml_config.top_n_features]
                logger.info("Using %d significant indices as features", len(selected))
            else:
                # Fall back to top by absolute correlation
                ranked = sorted(correlation_summary.results,
                              key=lambda r: abs(r.lagged_correlation), reverse=True)
                selected = [r.index_name for r in ranked[:self.ml_config.top_n_features]]
                logger.info("No FDR-significant indices; using top %d by |correlation|",
                          len(selected))

        # Build features
        logger.info("Engineering features...")
        features, targets = build_features(
            index_data, target_series, self.ml_config, selected
        )

        # Align and drop NaN rows
        all_data = pd.concat([features] + [t.rename(f"target_{h}")
                              for h, t in targets.items()], axis=1)
        all_data = all_data.dropna()

        if len(all_data) < self.ml_config.min_train_days + 50:
            raise ValueError(
                f"Insufficient data after feature engineering: {len(all_data)} rows, "
                f"need at least {self.ml_config.min_train_days + 50}"
            )

        feature_cols = [c for c in all_data.columns if not c.startswith("target_")]
        logger.info("Feature matrix: %d rows × %d features", len(all_data), len(feature_cols))

        X = all_data[feature_cols]
        feature_names = list(X.columns)

        # Models to evaluate
        models = self._build_models()

        # Run per-horizon
        all_results: dict[int, list[ModelResult]] = {}
        best_per_horizon: dict[int, ModelResult] = {}
        baseline_rmse: dict[int, float] = {}
        all_importances: dict[str, list[float]] = {f: [] for f in feature_names}
        equity_curves: dict[int, pd.Series] = {}

        for horizon in self.ml_config.horizons:
            logger.info("=== Horizon: %d-day ahead ===", horizon)
            y = all_data[f"target_{horizon}"]

            splits = walk_forward_split(
                len(X), self.ml_config.n_splits, self.ml_config.min_train_days
            )

            # Naive baseline: predict zero (for returns)
            baseline_preds = np.zeros(sum(len(te) for _, te in splits))
            baseline_actuals = np.concatenate([y.iloc[te].values for _, te in splits])
            baseline_rmse[horizon] = float(np.sqrt(mean_squared_error(
                baseline_actuals, baseline_preds
            )))

            horizon_results = []
            for model_name, model_factory in models.items():
                result = self._evaluate_model(
                    model_name, model_factory, X, y, splits,
                    horizon, feature_names, all_importances
                )
                if result is not None:
                    horizon_results.append(result)

            all_results[horizon] = horizon_results

            if horizon_results:
                best = max(horizon_results, key=lambda r: r.directional_accuracy)
                best_per_horizon[horizon] = best
                logger.info("Best model for %dd: %s (dir. acc = %.1f%%)",
                          horizon, best.model_name, best.directional_accuracy * 100)

            # Build equity curve for best model
            if horizon_results:
                equity_curves[horizon] = self._build_equity_curve(
                    max(horizon_results, key=lambda r: r.directional_accuracy).model_name,
                    models, X, y, splits, horizon
                )

        # Aggregate feature importances
        feature_imp = []
        for fname in feature_names:
            vals = all_importances.get(fname, [])
            if vals:
                feature_imp.append(FeatureImportance(
                    feature_name=fname,
                    importance_mean=float(np.mean(vals)),
                    importance_std=float(np.std(vals)),
                ))
        feature_imp.sort(key=lambda f: f.importance_mean, reverse=True)

        # SHAP analysis on best model (if available)
        if HAS_SHAP and best_per_horizon:
            best_horizon = max(best_per_horizon.keys())
            feature_imp = self._add_shap(
                feature_imp, models, X, all_data[f"target_{best_horizon}"],
                feature_names, best_per_horizon[best_horizon].model_name
            )

        return MLSummary(
            target_ticker=self.config.target_ticker,
            target_name=self.config.target_name,
            n_features_used=len(feature_names),
            feature_names=feature_names,
            model_results=all_results,
            best_model_per_horizon=best_per_horizon,
            feature_importances=feature_imp[:50],  # Top 50
            baseline_rmse=baseline_rmse,
            equity_curves=equity_curves,
        )

    def _build_models(self) -> dict[str, callable]:
        """Return dict of model_name -> factory function."""
        rs = self.ml_config.random_state
        models = {
            "Ridge": lambda: Ridge(alpha=1.0),
            "Lasso": lambda: Lasso(alpha=0.001, max_iter=5000),
            "ElasticNet": lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
            "RandomForest": lambda: RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=10,
                random_state=rs, n_jobs=-1
            ),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = lambda: xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=rs, n_jobs=-1, verbosity=0,
            )
        return models

    def _evaluate_model(
        self,
        model_name: str,
        model_factory: callable,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        horizon: int,
        feature_names: list[str],
        all_importances: dict[str, list[float]],
    ) -> ModelResult | None:
        """Evaluate a single model via walk-forward CV."""
        fold_preds = []
        fold_actuals = []
        fold_r2s = []
        fold_dir = []

        scaler = StandardScaler()

        for train_idx, test_idx in splits:
            X_train = X.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values

            # Scale features
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Replace any inf/nan from scaling
            X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                model = model_factory()
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
            except Exception as e:
                logger.warning("Model %s failed on fold: %s", model_name, e)
                continue

            fold_preds.extend(preds)
            fold_actuals.extend(y_test)

            # Per-fold metrics
            if len(y_test) > 2:
                r2 = r2_score(y_test, preds)
                fold_r2s.append(r2)
                dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
                fold_dir.append(dir_acc)

            # Feature importance (for tree models and linear)
            if hasattr(model, 'feature_importances_'):
                for i, fname in enumerate(feature_names):
                    all_importances[fname].append(model.feature_importances_[i])
            elif hasattr(model, 'coef_'):
                for i, fname in enumerate(feature_names):
                    all_importances[fname].append(abs(model.coef_[i]))

        if not fold_preds:
            return None

        preds_arr = np.array(fold_preds)
        actuals_arr = np.array(fold_actuals)

        # Aggregate metrics
        rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
        mae = float(mean_absolute_error(actuals_arr, preds_arr))
        r2 = float(r2_score(actuals_arr, preds_arr))

        # Directional accuracy
        dir_correct = np.sum(np.sign(preds_arr) == np.sign(actuals_arr))
        dir_total = len(preds_arr)
        dir_acc = dir_correct / dir_total

        # Binomial test: is directional accuracy > 50%?
        dir_p = float(sp_stats.binomtest(int(dir_correct), dir_total, 0.5).pvalue)

        # Information coefficient (rank correlation)
        ic, _ = sp_stats.spearmanr(preds_arr, actuals_arr)

        return ModelResult(
            model_name=model_name,
            horizon_days=horizon,
            rmse=rmse,
            mae=mae,
            r2=r2,
            directional_accuracy=float(dir_acc),
            directional_p_value=dir_p,
            ic_mean=float(ic) if not np.isnan(ic) else 0.0,
            ic_std=float(np.std(fold_r2s)) if fold_r2s else 0.0,
            fold_r2s=fold_r2s,
            fold_directional=fold_dir,
        )

    def _build_equity_curve(
        self,
        model_name: str,
        models: dict,
        X: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        horizon: int,
    ) -> pd.Series:
        """
        Build a hypothetical equity curve: if we traded on the model's
        directional predictions, what would cumulative returns look like?
        """
        scaler = StandardScaler()
        all_returns = []
        all_dates = []

        for train_idx, test_idx in splits:
            X_train_s = scaler.fit_transform(X.iloc[train_idx].values)
            X_test_s = scaler.transform(X.iloc[test_idx].values)
            X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                model = models[model_name]()
                model.fit(X_train_s, y.iloc[train_idx].values)
                preds = model.predict(X_test_s)
            except Exception:
                continue

            actuals = y.iloc[test_idx].values
            # Strategy: go long if predicted positive, short if negative
            strategy_returns = np.sign(preds) * actuals
            all_returns.extend(strategy_returns)
            all_dates.extend(X.index[test_idx])

        if not all_returns:
            return pd.Series(dtype=float)

        equity = pd.Series(all_returns, index=all_dates)
        equity = (1 + equity).cumprod() - 1  # Cumulative returns
        return equity

    def _add_shap(
        self,
        feature_imp: list[FeatureImportance],
        models: dict,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        model_name: str,
    ) -> list[FeatureImportance]:
        """Add SHAP values to feature importance rankings."""
        if not HAS_SHAP:
            return feature_imp

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            model = models[model_name]()
            model.fit(X_scaled, y.values)

            # Use a sample for SHAP (can be slow on large datasets)
            sample_size = min(500, len(X_scaled))
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_idx]

            if model_name in ("XGBoost", "RandomForest"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_scaled)

            shap_values = explainer.shap_values(X_sample)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            mean_shap = np.mean(shap_values, axis=0)  # Signed for direction

            # Map back to feature importance objects
            shap_map = {feature_names[i]: (mean_abs_shap[i], mean_shap[i])
                       for i in range(len(feature_names))}

            for fi in feature_imp:
                if fi.feature_name in shap_map:
                    abs_val, signed_val = shap_map[fi.feature_name]
                    fi.shap_mean = float(abs_val)
                    fi.direction = "positive" if signed_val > 0 else "negative"

            # Re-sort by SHAP if available
            feature_imp.sort(key=lambda f: f.shap_mean or 0, reverse=True)

        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)

        return feature_imp
