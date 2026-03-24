"""Inference pipeline — loads artifacts and replicates predict_fn() exactly.

Every intermediate value is captured for diagnostic logging.
"""

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COUNT_WITH_FUNDING, FEATURE_COUNT_WITHOUT_FUNDING, compute_features

logger = logging.getLogger(__name__)

EXPECTED_ARTIFACT_KEYS = {
    "model", "model_72", "conf_model", "pos_scaler",
    "conf_feat_indices", "conf_train_p5", "conf_train_p95",
    "scaler_feat_mask", "trained_at", "train_data_end", "commit",
    "n_features",
}


@dataclass
class InferenceResult:
    """All intermediate and final values from one inference run."""

    # Raw model outputs
    pred_24_raw: float
    pred_72_raw: float
    pred_72_smoothed: float
    sign_agree: float  # +1 agree, -1 disagree

    # After 72h filter
    pred_after_72h: float

    # Confidence scaler
    conf_prob: float
    conf_smoothed: float
    conf_norm: float
    conf_adj: float
    pred_after_conf: float

    # Position scaler
    pos_scaler_signal: float
    pos_scale: float
    pred_after_pos: float

    # Final
    pred_after_scale: float
    pred_final: float

    # Position sizing
    position: float

    # Context
    btc_price: float
    timestamp: object


def load_artifacts(path: str) -> dict:
    """Load and validate model artifacts from joblib file."""
    artifacts = joblib.load(path)

    missing = EXPECTED_ARTIFACT_KEYS - set(artifacts.keys())
    if missing:
        raise ValueError(f"Artifact missing keys: {missing}")

    n_feat = artifacts["n_features"]
    if n_feat not in (FEATURE_COUNT_WITHOUT_FUNDING, FEATURE_COUNT_WITH_FUNDING):
        raise ValueError(
            f"Artifact n_features={n_feat}, expected "
            f"{FEATURE_COUNT_WITHOUT_FUNDING} or {FEATURE_COUNT_WITH_FUNDING}"
        )

    return artifacts


def validate_artifacts(artifacts: dict) -> bool:
    """Run a smoke test: predict on dummy input, check output is finite."""
    n_feat = artifacts["n_features"]
    dummy = np.zeros((1, n_feat))

    pred = artifacts["model"].predict(dummy)
    if not np.isfinite(pred).all():
        return False

    pred_72 = artifacts["model_72"].predict(dummy)
    if not np.isfinite(pred_72).all():
        return False

    conf_idx = artifacts["conf_feat_indices"]
    proba = artifacts["conf_model"].predict_proba(dummy[:, conf_idx])
    if not np.isfinite(proba).all():
        return False

    scaler_mask = artifacts["scaler_feat_mask"]
    scaler_pred = artifacts["pos_scaler"].predict(dummy[:, scaler_mask])
    if not np.isfinite(scaler_pred).all():
        return False

    return True


def run_inference(df: pd.DataFrame, artifacts: dict) -> InferenceResult:
    """Run the full inference pipeline on an OHLCV DataFrame.

    Replicates train.py predict_fn() lines 492-530 exactly.
    Returns the LAST ROW's values for all intermediate stages.
    """
    model = artifacts["model"]
    model_72 = artifacts["model_72"]
    conf_model = artifacts["conf_model"]
    pos_scaler = artifacts["pos_scaler"]
    conf_feat_indices = artifacts["conf_feat_indices"]
    conf_train_p5 = artifacts["conf_train_p5"]
    conf_train_p95 = artifacts["conf_train_p95"]
    scaler_feat_mask = artifacts["scaler_feat_mask"]

    # Compute features on full history
    feats, ts, vol = compute_features(df)

    # Feature quality check: count NaN before masking
    nan_count = np.isnan(feats).sum()
    nan_frac = nan_count / feats.size if feats.size > 0 else 0
    if nan_frac > 0.05:
        logger.warning(
            f"Feature quality degraded: {nan_count} NaN ({nan_frac:.1%} of values). "
            f"Likely cause: data gap in rolling windows."
        )
    elif nan_count > 0:
        logger.debug(f"Features: {nan_count} NaN masked ({nan_frac:.2%})")

    feats = np.nan_to_num(feats, nan=0.0)

    # Validate feature shape matches artifact
    expected_n = artifacts["n_features"]
    if feats.shape[1] != expected_n:
        raise ValueError(
            f"Feature count mismatch: computed {feats.shape[1]}, "
            f"artifact expects {expected_n}"
        )

    # 24h prediction
    pred_24 = model.predict(feats)

    # 72h prediction + EMA-12 smoothing
    pred_72_raw = model_72.predict(feats)
    pred_72 = pd.Series(pred_72_raw).ewm(span=12, min_periods=1).mean().values

    # Sign matching: dampen on disagreement, no boost on agreement
    sign_match = np.sign(pred_24) * np.sign(pred_72)
    sigma_preds = pred_24 * (1.0 - 0.04 + 0.04 * sign_match)

    # Confidence scaler — asymmetric threshold, EMA-24
    conf_pred = conf_model.predict_proba(feats[:, conf_feat_indices])[:, 1]
    conf_smooth = pd.Series(conf_pred).ewm(span=24, min_periods=1).mean().values

    # Normalize to [0, 1] using train range
    conf_range = max(conf_train_p95 - conf_train_p5, 1e-6)
    conf_norm = np.clip((conf_smooth - conf_train_p5) / conf_range, 0.0, 1.0)

    # Asymmetric threshold: dampen danger tail, boost favorable tail
    dampen = np.clip((conf_norm - 0.05) / 0.10, 0.0, 1.0)
    boost = np.clip((conf_norm - 0.85) / 0.10, 0.0, 1.0)
    conf_adj = 0.70 + 0.30 * dampen + 0.20 * boost
    sigma_preds = sigma_preds * conf_adj

    # Position scaling
    scaler_signal = np.clip(pos_scaler.predict(feats[:, scaler_feat_mask]), 0.0, 1.0)
    pos_scale = 1.08 - 0.72 * scaler_signal
    sigma_preds = sigma_preds * pos_scale

    # Final scaling and smoothing
    sigma_preds = sigma_preds * 0.35
    sigma_smoothed = pd.Series(sigma_preds).ewm(span=20, min_periods=1).mean().values

    # Extract last row values
    i = -1
    btc_price = float(df["close"].iloc[-1])
    pred_final = float(sigma_smoothed[i])
    position = compute_position(pred_final)

    return InferenceResult(
        pred_24_raw=float(pred_24[i]),
        pred_72_raw=float(pred_72_raw[i]),
        pred_72_smoothed=float(pred_72[i]),
        sign_agree=float(sign_match[i]),
        pred_after_72h=float(pred_24[i] * (1.0 - 0.04 + 0.04 * sign_match[i])),
        conf_prob=float(conf_pred[i]),
        conf_smoothed=float(conf_smooth[i]),
        conf_norm=float(conf_norm[i]),
        conf_adj=float(conf_adj[i]),
        pred_after_conf=float(sigma_preds[i] / pos_scale[i] / 0.35),  # before pos_scale
        pos_scaler_signal=float(scaler_signal[i]),
        pos_scale=float(pos_scale[i]),
        pred_after_pos=float(sigma_preds[i] / 0.35),  # before final scale
        pred_after_scale=float(sigma_preds[i]),
        pred_final=pred_final,
        position=position,
        btc_price=btc_price,
        timestamp=ts[i],
    )


@dataclass
class FullInferenceResult:
    """All intermediate arrays from inference over the full dataset."""

    timestamps: np.ndarray       # (N,) aligned timestamps
    pred_24_raw: np.ndarray      # (N,)
    pred_72_raw: np.ndarray      # (N,)
    pred_72_smoothed: np.ndarray # (N,)
    sign_agree: np.ndarray       # (N,)
    pred_after_72h: np.ndarray   # (N,)
    conf_prob: np.ndarray        # (N,)
    conf_smoothed: np.ndarray    # (N,)
    conf_norm: np.ndarray        # (N,)
    conf_adj: np.ndarray         # (N,)
    pos_scaler_signal: np.ndarray # (N,)
    pos_scale: np.ndarray        # (N,)
    pred_after_scale: np.ndarray # (N,) after ×0.35
    pred_final: np.ndarray       # (N,) after EMA-20


def run_inference_full(df: pd.DataFrame, artifacts: dict) -> FullInferenceResult:
    """Run inference on the full DataFrame, returning arrays for ALL hours.

    Same pipeline as run_inference() but returns every intermediate array
    instead of just the last row. Used by replay.py.
    """
    model = artifacts["model"]
    model_72 = artifacts["model_72"]
    conf_model = artifacts["conf_model"]
    pos_scaler = artifacts["pos_scaler"]
    conf_feat_indices = artifacts["conf_feat_indices"]
    conf_train_p5 = artifacts["conf_train_p5"]
    conf_train_p95 = artifacts["conf_train_p95"]
    scaler_feat_mask = artifacts["scaler_feat_mask"]

    feats, ts, vol = compute_features(df)
    feats = np.nan_to_num(feats, nan=0.0)

    expected_n = artifacts["n_features"]
    if feats.shape[1] != expected_n:
        raise ValueError(
            f"Feature count mismatch: computed {feats.shape[1]}, "
            f"artifact expects {expected_n}"
        )

    pred_24 = model.predict(feats)
    pred_72_raw = model_72.predict(feats)
    pred_72 = pd.Series(pred_72_raw).ewm(span=12, min_periods=1).mean().values
    sign_match = np.sign(pred_24) * np.sign(pred_72)
    pred_after_72h = pred_24 * (1.0 - 0.04 + 0.04 * sign_match)

    conf_pred = conf_model.predict_proba(feats[:, conf_feat_indices])[:, 1]
    conf_smooth = pd.Series(conf_pred).ewm(span=24, min_periods=1).mean().values
    conf_range = max(conf_train_p95 - conf_train_p5, 1e-6)
    conf_norm = np.clip((conf_smooth - conf_train_p5) / conf_range, 0.0, 1.0)
    dampen = np.clip((conf_norm - 0.05) / 0.10, 0.0, 1.0)
    boost = np.clip((conf_norm - 0.85) / 0.10, 0.0, 1.0)
    conf_adj = 0.70 + 0.30 * dampen + 0.20 * boost

    scaler_signal = np.clip(pos_scaler.predict(feats[:, scaler_feat_mask]), 0.0, 1.0)
    pos_scale = 1.08 - 0.72 * scaler_signal

    sigma_preds = pred_after_72h * conf_adj * pos_scale
    pred_after_scale = sigma_preds * 0.35
    pred_final = pd.Series(pred_after_scale).ewm(span=20, min_periods=1).mean().values

    return FullInferenceResult(
        timestamps=ts,
        pred_24_raw=pred_24,
        pred_72_raw=pred_72_raw,
        pred_72_smoothed=pred_72,
        sign_agree=sign_match,
        pred_after_72h=pred_after_72h,
        conf_prob=conf_pred,
        conf_smoothed=conf_smooth,
        conf_norm=conf_norm,
        conf_adj=conf_adj,
        pos_scaler_signal=scaler_signal,
        pos_scale=pos_scale,
        pred_after_scale=pred_after_scale,
        pred_final=pred_final,
    )


def compute_position(
    pred_final: float,
    sigma_threshold: float = 0.20,
    sigma_full: float = 0.50,
) -> float:
    """Convert sigma-space prediction to position size [-1, +1]."""
    abs_pred = abs(pred_final)
    if abs_pred < sigma_threshold:
        return 0.0
    size = min((abs_pred - sigma_threshold) / (sigma_full - sigma_threshold), 1.0)
    direction = 1.0 if pred_final > 0 else -1.0
    return direction * size
