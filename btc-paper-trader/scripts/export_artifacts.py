"""Export trained model artifacts for the paper trading system.

Run on the Mac after training. Trains all 4 models on the full dataset,
serializes them via joblib, and saves reference predictions for parity testing.

Usage:
    cd autoresearch
    uv run btc-paper-trader/scripts/export_artifacts.py
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Add autotrader source to path so we can import train.py and prepare.py
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "assets" / "btc_hourly"))

from core.evaluation import compute_decay_weights  # noqa: E402
from prepare import _load_all_data  # noqa: E402
from train import (  # noqa: E402
    MAX_LOOKBACK,
    compute_confidence_targets,
    compute_features,
    compute_targets,
    compute_vol_targets,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def get_git_commit() -> str:
    """Get short git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def export(train_end: str | None = None):
    """Train models on full data and export artifacts.

    Args:
        train_end: Optional cutoff date (YYYY-MM-DD). If None, uses all data.
    """
    print("Loading full dataset...")
    all_data = _load_all_data()

    if train_end:
        mask = all_data["timestamp"] <= pd.Timestamp(train_end)
        train_df = all_data[mask].reset_index(drop=True)
        print(f"  Filtered to {train_end}: {len(train_df)} rows")
    else:
        train_df = all_data
        train_end = str(train_df["timestamp"].max().date())

    print(f"  Data range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  {len(train_df)} rows")

    # --- Compute features and targets (replicates build_model lines 344-358) ---
    print("Computing features...")
    features_all, timestamps, vol_safe = compute_features(train_df)
    features_all = np.nan_to_num(features_all, nan=0.0)
    n_features = features_all.shape[1]
    print(f"  {n_features} features, {len(features_all)} samples")

    targets = compute_targets(train_df)[MAX_LOOKBACK:]
    valid = ~np.isnan(targets)
    features = features_all[valid]
    targets = targets[valid]
    vol_train = vol_safe[valid]

    # Vol-normalize and winsorize
    targets = targets / vol_train
    targets = np.clip(targets, -5.0, 5.0)

    # --- Sample weights: exponential decay matching backtester ---
    # The backtester uses 5-year half-life decay, anchored to the eval start.
    # For the paper trader export, eval_start = day after train_end.
    eval_start = pd.Timestamp(train_end) + pd.Timedelta(days=1)
    sample_weight_full = compute_decay_weights(
        train_df["timestamp"].values, eval_start, half_life_years=5.0,
    )
    sw_trimmed = sample_weight_full[MAX_LOOKBACK:]
    sample_weight = sw_trimmed[valid]
    print(f"  Sample weights: range [{sample_weight.min():.3f}, {sample_weight.max():.3f}] "
          f"(half-life 5yr, anchored to {eval_start.date()})")

    # --- Monotonic constraints ---
    mono_cst = np.zeros(features.shape[1], dtype=int)
    mono_cst[3] = 1  # 48h return
    mono_cst[4] = 1  # 72h return
    mono_cst[5] = 1  # 168h return

    # --- Train 24h return model ---
    print("Training 24h return model...")
    model = HistGradientBoostingRegressor(
        max_iter=1000,
        max_depth=4,
        min_samples_leaf=600,
        learning_rate=0.01,
        max_leaf_nodes=15,
        l2_regularization=1.5,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model.fit(features, targets, sample_weight=sample_weight)
    print("  Done")

    # --- Train 72h auxiliary model ---
    print("Training 72h auxiliary model...")
    targets_72 = compute_targets(train_df, horizon=72)[MAX_LOOKBACK:]
    valid_72 = ~np.isnan(targets_72)
    tgt_72 = targets_72[valid_72] / vol_safe[valid_72]
    tgt_72 = np.clip(tgt_72, -5.0, 5.0)
    sw_72 = sw_trimmed[valid_72]

    model_72 = HistGradientBoostingRegressor(
        max_iter=800,
        max_depth=3,
        min_samples_leaf=1000,
        learning_rate=0.01,
        max_leaf_nodes=10,
        l2_regularization=3.0,
        monotonic_cst=mono_cst.tolist(),
        random_state=42,
    )
    model_72.fit(features_all[valid_72], tgt_72, sample_weight=sw_72)
    print("  Done")

    # --- Train position scaler ---
    print("Training position scaler...")
    scaler_targets_raw = compute_vol_targets(train_df)
    scaler_targets = scaler_targets_raw[MAX_LOOKBACK:]
    scaler_targets = scaler_targets[valid]
    scaler_binary = (scaler_targets > 0.9).astype(np.float64)

    scaler_exclude = {6, 7, 8, 16, 17, 18, 19, 20, 28, 29}
    scaler_feat_mask = [i for i in range(features.shape[1]) if i not in scaler_exclude]
    scaler_features = features[:, scaler_feat_mask]

    pos_scaler = HistGradientBoostingRegressor(
        max_iter=1000,
        max_depth=4,
        min_samples_leaf=400,
        learning_rate=0.02,
        max_leaf_nodes=20,
        l2_regularization=1.5,
        random_state=42,
    )
    scaler_sample_weight = np.where(scaler_binary == 1, 3.0, 1.0) * sample_weight
    pos_scaler.fit(scaler_features, scaler_binary, sample_weight=scaler_sample_weight)
    print(f"  {len(scaler_feat_mask)} features, {scaler_binary.mean()*100:.1f}% positive")

    # --- Train confidence scaler ---
    print("Training confidence scaler...")
    conf_targets_raw = compute_confidence_targets(train_df)
    conf_targets = conf_targets_raw[MAX_LOOKBACK:]
    conf_targets = conf_targets[valid]

    conf_feat_indices = [9, 10, 11, 12, 13, 14, 15, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31]
    conf_features = features[:, conf_feat_indices]

    conf_valid = ~np.isnan(conf_targets)
    conf_binary = conf_targets[conf_valid].astype(int)

    conf_model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=3,
        min_samples_leaf=800,
        learning_rate=0.02,
        max_leaf_nodes=10,
        l2_regularization=2.0,
        random_state=42,
    )
    conf_model.fit(
        conf_features[conf_valid], conf_binary,
        sample_weight=sample_weight[conf_valid],
    )

    rtp = conf_model.predict_proba(conf_features[conf_valid])[:, 1]
    conf_train_p5 = float(np.percentile(rtp, 5))
    conf_train_p95 = float(np.percentile(rtp, 95))
    print(f"  {conf_binary.mean()*100:.1f}% favorable, p5={conf_train_p5:.3f} p95={conf_train_p95:.3f}")

    # --- Generate reference predictions for parity testing ---
    print("Generating reference predictions...")
    ref_n = min(1000, len(features_all))
    ref_feats = features_all[-ref_n:]

    ref_pred_24 = model.predict(ref_feats)
    ref_pred_72_raw = model_72.predict(ref_feats)
    ref_pred_72 = pd.Series(ref_pred_72_raw).ewm(span=12, min_periods=1).mean().values
    ref_sign_match = np.sign(ref_pred_24) * np.sign(ref_pred_72)
    ref_sigma_after_72h = ref_pred_24 * (1.0 - 0.04 + 0.04 * ref_sign_match)

    ref_conf_pred = conf_model.predict_proba(ref_feats[:, conf_feat_indices])[:, 1]
    ref_conf_smooth = pd.Series(ref_conf_pred).ewm(span=24, min_periods=1).mean().values
    conf_range = max(conf_train_p95 - conf_train_p5, 1e-6)
    ref_conf_norm = np.clip((ref_conf_smooth - conf_train_p5) / conf_range, 0.0, 1.0)
    ref_dampen = np.clip((ref_conf_norm - 0.05) / 0.10, 0.0, 1.0)
    ref_boost = np.clip((ref_conf_norm - 0.85) / 0.10, 0.0, 1.0)
    ref_conf_adj = 0.70 + 0.30 * ref_dampen + 0.20 * ref_boost
    ref_after_conf = ref_sigma_after_72h * ref_conf_adj

    ref_scaler_signal = np.clip(pos_scaler.predict(ref_feats[:, scaler_feat_mask]), 0.0, 1.0)
    ref_pos_scale = 1.08 - 0.72 * ref_scaler_signal
    ref_after_pos = ref_after_conf * ref_pos_scale

    ref_after_scale = ref_after_pos * 0.35
    ref_final = pd.Series(ref_after_scale).ewm(span=20, min_periods=1).mean().values

    reference_predictions = {
        "features": ref_feats,
        "pred_24": ref_pred_24,
        "pred_72_raw": ref_pred_72_raw,
        "pred_72_smoothed": ref_pred_72,
        "sign_match": ref_sign_match,
        "pred_after_72h": ref_sigma_after_72h,
        "conf_prob": ref_conf_pred,
        "conf_smoothed": ref_conf_smooth,
        "conf_norm": ref_conf_norm,
        "conf_adj": ref_conf_adj,
        "pred_after_conf": ref_after_conf,
        "scaler_signal": ref_scaler_signal,
        "pos_scale": ref_pos_scale,
        "pred_after_pos": ref_after_pos,
        "pred_after_scale": ref_after_scale,
        "pred_final": ref_final,
    }

    print(f"  Reference: {ref_n} rows, final pred range [{ref_final.min():.4f}, {ref_final.max():.4f}]")

    # --- Serialize ---
    commit = get_git_commit()
    artifacts = {
        "model": model,
        "model_72": model_72,
        "conf_model": conf_model,
        "pos_scaler": pos_scaler,
        "conf_feat_indices": conf_feat_indices,
        "conf_train_p5": conf_train_p5,
        "conf_train_p95": conf_train_p95,
        "scaler_feat_mask": scaler_feat_mask,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_data_end": train_end,
        "commit": commit,
        "n_features": n_features,
        "sklearn_version": sklearn.__version__,
        "reference_predictions": reference_predictions,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"model_{commit}.joblib"
    joblib.dump(artifacts, out_path, compress=3)
    print(f"\nArtifact saved: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Commit: {commit}")
    print(f"  sklearn: {sklearn.__version__}")
    print(f"  Features: {n_features}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model artifacts for paper trading")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Training data cutoff date (YYYY-MM-DD). Default: all data.")
    args = parser.parse_args()
    export(train_end=args.train_end)
