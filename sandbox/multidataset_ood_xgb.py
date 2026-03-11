#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_DATASET_ROOT = Path("/playpen-ssd/smerrill/deception2/Dataset")
DEFAULT_RESULTS_DIR = Path("/playpen-ssd/smerrill/deception2/ood_modeling_multidataset")
DEFAULT_DATASETS = ("AdvisorAudit", "BS", "Gridworld")
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
DEFAULT_SEEDS = (42,)
DEFAULT_SCOPES = ("before", "before_at", "before_at_after")
CLASS_PROB_CUTOFF = 0.5
MODEL_BUNDLE_VERSION = 1
MODEL_FILENAME_PREFIX = "xgb_cls_ablation"
REGRESSION_BUNDLE_VERSION = 1
REGRESSION_MODEL_FILENAME_PREFIX = "xgb_reg_ablation"
VALID_SCOPES = set(DEFAULT_SCOPES)
REQUIRED_SET_NAMES = (
    "baseline_struct",
    "baseline_struct_lex",
    "set1_struct_lex_entropy",
    "set2_struct_lex_entropy_activation",
    "set3_struct_lex_entropy_activation_attention",
)


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset: str
    model_dir: Path
    features_path: Path
    feature_sets_path: Path


def parse_float_csv(value: str) -> list[float]:
    out = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece:
            out.append(float(piece))
    return out


def parse_int_csv(value: str) -> list[int]:
    out = []
    for piece in value.split(","):
        piece = piece.strip()
        if piece:
            out.append(int(piece))
    return out


def ordered_unique(cols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def subtract_cols(full_cols: list[str], cols_to_remove: list[str]) -> list[str]:
    remove = set(cols_to_remove)
    return [col for col in full_cols if col not in remove]


def sanitize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, cutoff: float = CLASS_PROB_CUTOFF) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= cutoff).astype(int)

    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }

    if len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan

    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def detect_xgb_device() -> str:
    x_probe = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y_probe = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    try:
        probe = xgb.XGBRegressor(
            n_estimators=4,
            max_depth=2,
            learning_rate=0.3,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            eval_metric="rmse",
            n_jobs=1,
            random_state=0,
        )
        probe.fit(x_probe, y_probe, verbose=False)
        return "cuda"
    except Exception:
        return "cpu"


def resolve_dataset_artifacts(dataset_root: Path, dataset_name: str) -> DatasetArtifacts:
    dataset_dir = dataset_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    candidates: list[DatasetArtifacts] = []
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        features_path = model_dir / "features.parquet"
        feature_sets_path = model_dir / "feature_sets.json"
        if features_path.exists() and feature_sets_path.exists():
            candidates.append(
                DatasetArtifacts(
                    dataset=dataset_name,
                    model_dir=model_dir,
                    features_path=features_path,
                    feature_sets_path=feature_sets_path,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            f"No model subdirectory with features.parquet + feature_sets.json under {dataset_dir}"
        )

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda item: item.features_path.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    print(
        f"[warn] multiple artifact directories under {dataset_dir}; "
        f"using newest {chosen.model_dir.name}"
    )
    return chosen


def load_dataset_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["example_id"] = df["example_id"].astype(str)
    df["deception_rate"] = pd.to_numeric(df["deception_rate"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df = df.dropna(subset=["example_id", "deception_rate"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[load] dropped {dropped:,} rows with invalid example_id/deception_rate from {path}")
    return df


def load_feature_sets_from_json(path: Path, available_cols: set[str]) -> OrderedDict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: OrderedDict[str, list[str]] = OrderedDict()
    for set_name, raw_cols in payload.items():
        cols = [col for col in raw_cols if col in available_cols]
        cols = ordered_unique(cols)
        if cols:
            out[str(set_name)] = cols
    return out


def build_ablation_feature_sets(base_feature_sets: OrderedDict[str, list[str]]) -> tuple[OrderedDict[str, list[str]], pd.DataFrame]:
    missing = [name for name in REQUIRED_SET_NAMES if name not in base_feature_sets]
    if missing:
        raise KeyError(f"Missing expected base feature set(s): {missing}")

    baseline_struct = ordered_unique(base_feature_sets["baseline_struct"])
    baseline_struct_lex = ordered_unique(base_feature_sets["baseline_struct_lex"])
    set1 = ordered_unique(base_feature_sets["set1_struct_lex_entropy"])
    set2 = ordered_unique(base_feature_sets["set2_struct_lex_entropy_activation"])
    full_all = ordered_unique(base_feature_sets["set3_struct_lex_entropy_activation_attention"])

    lex_delta = [col for col in baseline_struct_lex if col not in set(baseline_struct)]
    entropy_delta = [col for col in set1 if col not in set(baseline_struct_lex)]
    activation_delta = [col for col in set2 if col not in set(set1)]
    attention_delta = [col for col in full_all if col not in set(set2)]
    baseline_lex = ordered_unique(lex_delta)

    out: OrderedDict[str, list[str]] = OrderedDict(
        {
            "full": full_all,
            "full_minus_attention": subtract_cols(full_all, attention_delta),
            "full_minus_entropy": subtract_cols(full_all, entropy_delta),
            "full_minus_activation": subtract_cols(full_all, activation_delta),
            "baseline_struct": baseline_struct,
            "baseline_lex": baseline_lex,
            "baseline_lex_plus_struct": baseline_struct_lex,
        }
    )

    summary = pd.DataFrame(
        [
            {
                "feature_set": "full",
                "n_features": len(out["full"]),
                "dropped_family": "",
                "dropped_count": 0,
            },
            {
                "feature_set": "full_minus_attention",
                "n_features": len(out["full_minus_attention"]),
                "dropped_family": "attention",
                "dropped_count": len(attention_delta),
            },
            {
                "feature_set": "full_minus_entropy",
                "n_features": len(out["full_minus_entropy"]),
                "dropped_family": "entropy",
                "dropped_count": len(entropy_delta),
            },
            {
                "feature_set": "full_minus_activation",
                "n_features": len(out["full_minus_activation"]),
                "dropped_family": "activation",
                "dropped_count": len(activation_delta),
            },
            {
                "feature_set": "baseline_struct",
                "n_features": len(out["baseline_struct"]),
                "dropped_family": "",
                "dropped_count": 0,
            },
            {
                "feature_set": "baseline_lex",
                "n_features": len(out["baseline_lex"]),
                "dropped_family": "",
                "dropped_count": 0,
            },
            {
                "feature_set": "baseline_lex_plus_struct",
                "n_features": len(out["baseline_lex_plus_struct"]),
                "dropped_family": "",
                "dropped_count": 0,
            },
        ]
    )
    return out, summary


def expand_feature_sets_by_scope(
    feature_sets: OrderedDict[str, list[str]],
    scopes: list[str],
) -> tuple[OrderedDict[str, list[str]], pd.DataFrame]:
    unknown = [scope for scope in scopes if scope not in VALID_SCOPES]
    if unknown:
        raise ValueError(f"Unsupported scope(s): {unknown}")

    out: OrderedDict[str, list[str]] = OrderedDict()
    summary_rows: list[dict[str, Any]] = []

    for base_name, cols in feature_sets.items():
        ordered = ordered_unique(cols)
        temporal_cols = [
            col
            for col in ordered
            if col.startswith("before_") or col.startswith("at_") or col.startswith("after_")
        ]
        passthrough_cols = [col for col in ordered if col not in temporal_cols]

        for scope in scopes:
            if scope == "before":
                scoped = [col for col in temporal_cols if col.startswith("before_")]
            elif scope == "before_at":
                scoped = [
                    col
                    for col in temporal_cols
                    if col.startswith("before_") or col.startswith("at_")
                ]
            else:
                scoped = temporal_cols + passthrough_cols

            scoped = ordered_unique(scoped)
            if not scoped:
                continue

            set_name = f"{base_name}__{scope}"
            out[set_name] = scoped
            summary_rows.append(
                {
                    "feature_set": set_name,
                    "base_feature_set": base_name,
                    "temporal_scope": scope,
                    "n_features": len(scoped),
                    "n_before": int(sum(col.startswith("before_") for col in scoped)),
                    "n_at": int(sum(col.startswith("at_") for col in scoped)),
                    "n_after": int(sum(col.startswith("after_") for col in scoped)),
                    "n_other": int(
                        sum(
                            not (
                                col.startswith("before_")
                                or col.startswith("at_")
                                or col.startswith("after_")
                            )
                            for col in scoped
                        )
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(
            ["base_feature_set", "temporal_scope"]
        ).reset_index(drop=True)
    return out, summary_df


def split_feature_set_name(feature_set: str) -> tuple[str, str]:
    txt = str(feature_set)
    base, sep, suffix = txt.rpartition("__")
    if sep and suffix in VALID_SCOPES:
        return base, suffix
    return txt, "original"


def filter_feature_sets(
    feature_sets: OrderedDict[str, list[str]],
    allowed_names: set[str] | None,
) -> OrderedDict[str, list[str]]:
    if not allowed_names:
        return feature_sets
    out = OrderedDict((name, cols) for name, cols in feature_sets.items() if name in allowed_names)
    if not out:
        raise ValueError(f"No feature sets left after filtering. Requested: {sorted(allowed_names)}")
    return out


def run_xgb_classifier_search(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    early_stopping_rounds: int,
    device: str,
) -> dict[str, Any] | None:
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if pos == 0 or neg == 0:
        return None

    scale_pos_weight = float(neg) / float(pos)
    param_grid = [
        {
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.60,
            "reg_lambda": 5.0,
            "reg_alpha": 0.5,
            "scale_pos_weight": scale_pos_weight,
        },
        {
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 12,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
            "scale_pos_weight": scale_pos_weight,
        },
        {
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.90,
            "colsample_bytree": 0.70,
            "reg_lambda": 4.0,
            "reg_alpha": 0.2,
            "scale_pos_weight": scale_pos_weight,
        },
    ]

    best: dict[str, Any] | None = None
    best_val_f1 = -1.0

    for cfg in param_grid:
        model = xgb.XGBClassifier(
            n_estimators=3500,
            objective="binary:logistic",
            random_state=seed,
            tree_method="hist",
            device=device,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            **cfg,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        val_prob = model.predict_proba(x_val)[:, 1]
        val_f1 = float(
            f1_score(y_val, (val_prob >= CLASS_PROB_CUTOFF).astype(int), zero_division=0)
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best = {
                "model": model,
                "config": cfg,
                "val_prob": val_prob,
                "val_f1": val_f1,
                "best_iteration": getattr(model, "best_iteration", None),
            }

    return best


def run_xgb_regressor_search(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    early_stopping_rounds: int,
    device: str,
) -> dict[str, Any] | None:
    param_grid = [
        {
            "objective": "reg:squarederror",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 8,
            "subsample": 0.85,
            "colsample_bytree": 0.60,
            "reg_lambda": 5.0,
            "reg_alpha": 0.5,
        },
        {
            "objective": "reg:squarederror",
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 12,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
        },
        {
            "objective": "reg:pseudohubererror",
            "learning_rate": 0.02,
            "max_depth": 5,
            "min_child_weight": 12,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
        },
        {
            "objective": "reg:absoluteerror",
            "learning_rate": 0.025,
            "max_depth": 4,
            "min_child_weight": 10,
            "subsample": 0.90,
            "colsample_bytree": 0.70,
            "reg_lambda": 4.0,
            "reg_alpha": 0.2,
        },
    ]

    best: dict[str, Any] | None = None
    best_val_rmse = np.inf

    for cfg in param_grid:
        model = xgb.XGBRegressor(
            n_estimators=4000,
            random_state=seed,
            tree_method="hist",
            device=device,
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            **cfg,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        val_pred = model.predict(x_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best = {
                "model": model,
                "config": cfg,
                "val_pred": val_pred,
                "val_rmse": val_rmse,
                "best_iteration": getattr(model, "best_iteration", None),
            }

    return best


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return slug.strip("_") or "value"


def format_threshold_tag(threshold: float) -> str:
    return sanitize_slug(f"thr{threshold:.2f}".replace(".", "p"))


def build_model_path(model_dir: Path, feature_set: str, threshold: float, seed: int) -> Path:
    filename = (
        f"{MODEL_FILENAME_PREFIX}__{sanitize_slug(feature_set)}__"
        f"{format_threshold_tag(threshold)}__seed{int(seed)}.joblib"
    )
    return model_dir / filename


def build_regression_model_path(model_dir: Path, feature_set: str, seed: int) -> Path:
    filename = (
        f"{REGRESSION_MODEL_FILENAME_PREFIX}__{sanitize_slug(feature_set)}__"
        f"seed{int(seed)}.joblib"
    )
    return model_dir / filename


def is_compatible_bundle(
    bundle: dict[str, Any],
    *,
    train_dataset: str,
    feature_set: str,
    threshold: float,
    seed: int,
    feature_names: list[str],
) -> bool:
    if bundle.get("bundle_version") != MODEL_BUNDLE_VERSION:
        return False
    if bundle.get("train_dataset") != train_dataset:
        return False
    if bundle.get("feature_set") != feature_set:
        return False
    if int(bundle.get("seed", -1)) != int(seed):
        return False
    if not math.isclose(float(bundle.get("threshold", -999.0)), float(threshold), rel_tol=0.0, abs_tol=1e-12):
        return False
    if list(bundle.get("feature_names", [])) != list(feature_names):
        return False
    if bundle.get("kind") not in {"xgb", "constant"}:
        return False
    return True


def load_compatible_bundle(
    *,
    model_path: Path,
    train_dataset: str,
    feature_set: str,
    threshold: float,
    seed: int,
    feature_names: list[str],
) -> dict[str, Any] | None:
    if not model_path.exists():
        return None

    bundle = load(model_path)
    if is_compatible_bundle(
        bundle,
        train_dataset=train_dataset,
        feature_set=feature_set,
        threshold=threshold,
        seed=seed,
        feature_names=feature_names,
    ):
        return bundle
    return None


def is_compatible_regression_bundle(
    bundle: dict[str, Any],
    *,
    train_dataset: str,
    feature_set: str,
    seed: int,
    feature_names: list[str],
) -> bool:
    if bundle.get("bundle_version") != REGRESSION_BUNDLE_VERSION:
        return False
    if bundle.get("train_dataset") != train_dataset:
        return False
    if bundle.get("feature_set") != feature_set:
        return False
    if int(bundle.get("seed", -1)) != int(seed):
        return False
    if list(bundle.get("feature_names", [])) != list(feature_names):
        return False
    if bundle.get("kind") not in {"xgb_reg", "constant_reg"}:
        return False
    return True


def load_compatible_regression_bundle(
    *,
    model_path: Path,
    train_dataset: str,
    feature_set: str,
    seed: int,
    feature_names: list[str],
) -> dict[str, Any] | None:
    if not model_path.exists():
        return None

    bundle = load(model_path)
    if is_compatible_regression_bundle(
        bundle,
        train_dataset=train_dataset,
        feature_set=feature_set,
        seed=seed,
        feature_names=feature_names,
    ):
        return bundle
    return None


def fit_and_save_bundle(
    *,
    model_path: Path,
    train_dataset: str,
    feature_set: str,
    threshold: float,
    seed: int,
    feature_names: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    n_jobs: int,
    early_stopping_rounds: int,
    device: str,
) -> dict[str, Any]:

    bundle: dict[str, Any]
    unique = np.unique(y_train)
    if len(unique) < 2:
        constant_prob = float(np.mean(y_train))
        bundle = {
            "bundle_version": MODEL_BUNDLE_VERSION,
            "kind": "constant",
            "train_dataset": train_dataset,
            "feature_set": feature_set,
            "threshold": float(threshold),
            "seed": int(seed),
            "feature_names": list(feature_names),
            "constant_prob": constant_prob,
            "config": {},
            "best_iteration": None,
        }
    else:
        best = run_xgb_classifier_search(
            x_train,
            y_train,
            x_val,
            y_val,
            seed=seed,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
        )

        if best is None:
            constant_prob = float(np.mean(y_train))
            bundle = {
                "bundle_version": MODEL_BUNDLE_VERSION,
                "kind": "constant",
                "train_dataset": train_dataset,
                "feature_set": feature_set,
                "threshold": float(threshold),
                "seed": int(seed),
                "feature_names": list(feature_names),
                "constant_prob": constant_prob,
                "config": {},
                "best_iteration": None,
            }
        else:
            bundle = {
                "bundle_version": MODEL_BUNDLE_VERSION,
                "kind": "xgb",
                "train_dataset": train_dataset,
                "feature_set": feature_set,
                "threshold": float(threshold),
                "seed": int(seed),
                "feature_names": list(feature_names),
                "model": best["model"],
                "config": best["config"],
                "best_iteration": best["best_iteration"],
            }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, model_path)
    return bundle


def fit_and_save_regression_bundle(
    *,
    model_path: Path,
    train_dataset: str,
    feature_set: str,
    seed: int,
    feature_names: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    n_jobs: int,
    early_stopping_rounds: int,
    device: str,
) -> dict[str, Any]:
    bundle: dict[str, Any]
    if len(np.unique(y_train)) < 2:
        constant_value = float(np.mean(y_train))
        bundle = {
            "bundle_version": REGRESSION_BUNDLE_VERSION,
            "kind": "constant_reg",
            "train_dataset": train_dataset,
            "feature_set": feature_set,
            "seed": int(seed),
            "feature_names": list(feature_names),
            "constant_value": constant_value,
            "config": {},
            "best_iteration": None,
        }
    else:
        best = run_xgb_regressor_search(
            x_train,
            y_train,
            x_val,
            y_val,
            seed=seed,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
        )

        if best is None:
            constant_value = float(np.mean(y_train))
            bundle = {
                "bundle_version": REGRESSION_BUNDLE_VERSION,
                "kind": "constant_reg",
                "train_dataset": train_dataset,
                "feature_set": feature_set,
                "seed": int(seed),
                "feature_names": list(feature_names),
                "constant_value": constant_value,
                "config": {},
                "best_iteration": None,
            }
        else:
            bundle = {
                "bundle_version": REGRESSION_BUNDLE_VERSION,
                "kind": "xgb_reg",
                "train_dataset": train_dataset,
                "feature_set": feature_set,
                "seed": int(seed),
                "feature_names": list(feature_names),
                "model": best["model"],
                "config": best["config"],
                "best_iteration": best["best_iteration"],
            }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, model_path)
    return bundle


def is_xgb_memory_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = (
        "cudaerrormemoryallocation",
        "memory allocation error",
        "out of memory",
        "std::bad_alloc",
    )
    return any(needle in text for needle in needles)


def predict_bundle_cpu(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    booster = model.get_booster()
    booster.set_param({"device": "cpu"})

    dmat = xgb.DMatrix(
        np.asarray(x, dtype=np.float32),
        feature_names=[str(col) for col in x.columns],
    )

    predict_kwargs: dict[str, Any] = {}
    best_iteration = bundle.get("best_iteration")
    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)

    pred = booster.predict(dmat, **predict_kwargs)
    return np.asarray(pred, dtype=float).reshape(-1)


def predict_bundle(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    if bundle["kind"] == "constant":
        return np.full(len(x), float(bundle["constant_prob"]), dtype=float)
    model = bundle["model"]
    try:
        return model.predict_proba(x)[:, 1]
    except xgb.core.XGBoostError as exc:
        if not is_xgb_memory_error(exc):
            raise

        print(
            "[warn] XGBoost GPU prediction hit OOM; retrying on CPU "
            f"(feature_set={bundle.get('feature_set')}, threshold={bundle.get('threshold')}, seed={bundle.get('seed')})"
        )
        return predict_bundle_cpu(bundle, x)


def predict_regression_bundle_cpu(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    booster = model.get_booster()
    booster.set_param({"device": "cpu"})

    dmat = xgb.DMatrix(
        np.asarray(x, dtype=np.float32),
        feature_names=[str(col) for col in x.columns],
    )

    predict_kwargs: dict[str, Any] = {}
    best_iteration = bundle.get("best_iteration")
    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)

    pred = booster.predict(dmat, **predict_kwargs)
    return np.asarray(pred, dtype=float).reshape(-1)


def predict_regression_bundle(bundle: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    if bundle["kind"] == "constant_reg":
        return np.full(len(x), float(bundle["constant_value"]), dtype=float)

    model = bundle["model"]
    try:
        pred = model.predict(x)
        return np.asarray(pred, dtype=float).reshape(-1)
    except xgb.core.XGBoostError as exc:
        if not is_xgb_memory_error(exc):
            raise

        print(
            "[warn] XGBoost GPU regression prediction hit OOM; retrying on CPU "
            f"(feature_set={bundle.get('feature_set')}, seed={bundle.get('seed')})"
        )
        return predict_regression_bundle_cpu(bundle, x)


def decode_xgb_feature_name(raw_name: str, model: Any) -> str:
    if (
        isinstance(raw_name, str)
        and raw_name.startswith("f")
        and raw_name[1:].isdigit()
        and hasattr(model, "feature_names_in_")
    ):
        idx = int(raw_name[1:])
        names = list(model.feature_names_in_)
        if 0 <= idx < len(names):
            return names[idx]
    return str(raw_name)


def extract_importance_rows(
    train_dataset: str,
    threshold: float,
    seed: int,
    feature_set: str,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    if bundle.get("kind") != "xgb":
        return []

    model = bundle["model"]
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    keys = set(gain) | set(weight) | set(cover)

    rows = []
    for key in keys:
        rows.append(
            {
                "train_dataset": train_dataset,
                "seed": int(seed),
                "threshold": float(threshold),
                "feature_set": feature_set,
                "feature": decode_xgb_feature_name(key, model),
                "gain": float(gain.get(key, 0.0)),
                "weight": float(weight.get(key, 0.0)),
                "cover": float(cover.get(key, 0.0)),
            }
        )
    return rows


def extract_regression_importance_rows(
    train_dataset: str,
    seed: int,
    feature_set: str,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    if bundle.get("kind") != "xgb_reg":
        return []

    model = bundle["model"]
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")
    keys = set(gain) | set(weight) | set(cover)

    rows = []
    for key in keys:
        rows.append(
            {
                "train_dataset": train_dataset,
                "seed": int(seed),
                "feature_set": feature_set,
                "feature": decode_xgb_feature_name(key, model),
                "gain": float(gain.get(key, 0.0)),
                "weight": float(weight.get(key, 0.0)),
                "cover": float(cover.get(key, 0.0)),
            }
        )
    return rows


def make_eval_row(
    *,
    train_dataset: str,
    eval_dataset: str,
    eval_kind: str,
    seed: int,
    threshold: float,
    feature_set: str,
    feature_names: list[str],
    bundle: dict[str, Any],
    model_path: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true, y_prob, cutoff=CLASS_PROB_CUTOFF)
    base_feature_set, temporal_scope = split_feature_set_name(feature_set)
    cfg = bundle.get("config", {}) or {}
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "eval_kind": eval_kind,
        "seed": int(seed),
        "threshold": float(threshold),
        "feature_set": feature_set,
        "base_feature_set": base_feature_set,
        "temporal_scope": temporal_scope,
        "n_features": len(feature_names),
        "model_kind": bundle.get("kind", "unknown"),
        "model_path": str(model_path),
        "best_iteration": bundle.get("best_iteration"),
        "learning_rate": cfg.get("learning_rate", np.nan),
        "max_depth": cfg.get("max_depth", np.nan),
        "min_child_weight": cfg.get("min_child_weight", np.nan),
        "subsample": cfg.get("subsample", np.nan),
        "colsample_bytree": cfg.get("colsample_bytree", np.nan),
        "reg_lambda": cfg.get("reg_lambda", np.nan),
        "reg_alpha": cfg.get("reg_alpha", np.nan),
        "scale_pos_weight": cfg.get("scale_pos_weight", np.nan),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
    }


def make_regression_eval_row(
    *,
    train_dataset: str,
    eval_dataset: str,
    eval_kind: str,
    seed: int,
    feature_set: str,
    feature_names: list[str],
    bundle: dict[str, Any],
    model_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = regression_metrics(y_true, y_pred)
    base_feature_set, temporal_scope = split_feature_set_name(feature_set)
    cfg = bundle.get("config", {}) or {}
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "eval_kind": eval_kind,
        "seed": int(seed),
        "feature_set": feature_set,
        "base_feature_set": base_feature_set,
        "temporal_scope": temporal_scope,
        "n_features": len(feature_names),
        "model_kind": bundle.get("kind", "unknown"),
        "model_path": str(model_path),
        "best_iteration": bundle.get("best_iteration"),
        "objective": cfg.get("objective", np.nan),
        "learning_rate": cfg.get("learning_rate", np.nan),
        "max_depth": cfg.get("max_depth", np.nan),
        "min_child_weight": cfg.get("min_child_weight", np.nan),
        "subsample": cfg.get("subsample", np.nan),
        "colsample_bytree": cfg.get("colsample_bytree", np.nan),
        "reg_lambda": cfg.get("reg_lambda", np.nan),
        "reg_alpha": cfg.get("reg_alpha", np.nan),
        "target_mean": float(np.mean(y_true)) if len(y_true) else np.nan,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "pearson": metrics["pearson"],
    }


def save_accuracy_threshold_plots(
    train_dataset: str,
    metrics_df: pd.DataFrame,
    feature_sets: OrderedDict[str, list[str]],
    thresholds: list[float],
    output_dir: Path,
    metric: str,
) -> None:
    eval_rows = metrics_df[metrics_df["eval_kind"].isin(["val", "ood"])].copy()
    if eval_rows.empty:
        return

    other_eval_datasets = sorted(
        eval_rows.loc[eval_rows["eval_kind"] == "ood", "eval_dataset"].dropna().unique().tolist()
    )
    eval_panels = [("val", train_dataset, "Validation")] + [
        ("ood", dataset_name, f"OOD: {dataset_name}") for dataset_name in other_eval_datasets
    ]

    scope_order = [scope for scope in DEFAULT_SCOPES if scope in set(eval_rows["temporal_scope"])]
    if not scope_order:
        scope_order = sorted(eval_rows["temporal_scope"].dropna().unique().tolist())

    for scope in scope_order:
        sub_scope = eval_rows[eval_rows["temporal_scope"] == scope].copy()
        if sub_scope.empty:
            continue

        ordered_sets = [name for name in feature_sets.keys() if name.endswith(f"__{scope}")]
        base_order = [split_feature_set_name(name)[0] for name in ordered_sets]
        if not base_order:
            base_order = sorted(sub_scope["base_feature_set"].dropna().unique().tolist())

        fig, axes = plt.subplots(
            1,
            len(eval_panels),
            figsize=(5.5 * len(eval_panels), 4.5),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for ax, (eval_kind, eval_dataset, panel_title) in zip(axes, eval_panels):
            panel_df = sub_scope[
                (sub_scope["eval_kind"] == eval_kind) & (sub_scope["eval_dataset"] == eval_dataset)
            ].copy()
            if panel_df.empty:
                ax.set_title(f"{panel_title} | {scope}\n(no rows)")
                ax.axis("off")
                continue

            pivot = panel_df.pivot_table(
                index="threshold",
                columns="base_feature_set",
                values=metric,
                aggfunc="mean",
            ).sort_index()

            for base_name in base_order:
                if base_name in pivot.columns:
                    ax.plot(
                        pivot.index,
                        pivot[base_name],
                        marker="o",
                        linewidth=2,
                        label=base_name,
                    )

            ax.set_title(f"{panel_title} {metric.upper()} | {scope}")
            ax.set_xlabel("Threshold for y = 1")
            ax.set_ylabel(metric.upper())
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(thresholds)
            ax.legend(loc="best", fontsize=8)

        out_path = output_dir / (
            f"{sanitize_slug(train_dataset)}__{metric}_vs_threshold__{sanitize_slug(scope)}.png"
        )
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_confusion_grids(
    train_dataset: str,
    metrics_df: pd.DataFrame,
    confusion_mats: dict[tuple[str, int, float, str, str], np.ndarray],
    feature_sets: OrderedDict[str, list[str]],
    thresholds: list[float],
    output_dir: Path,
) -> None:
    eval_rows = metrics_df[metrics_df["eval_kind"].isin(["val", "ood"])].copy()
    if eval_rows.empty:
        return

    seeds = sorted(eval_rows["seed"].dropna().astype(int).unique().tolist())
    other_eval_datasets = sorted(
        eval_rows.loc[eval_rows["eval_kind"] == "ood", "eval_dataset"].dropna().unique().tolist()
    )
    eval_panels = [("val", train_dataset, "Validation")] + [
        ("ood", dataset_name, f"OOD: {dataset_name}") for dataset_name in other_eval_datasets
    ]

    scope_order = [scope for scope in DEFAULT_SCOPES if scope in set(eval_rows["temporal_scope"])]
    if not scope_order:
        scope_order = sorted(eval_rows["temporal_scope"].dropna().unique().tolist())

    for seed in seeds:
        seed_df = eval_rows[eval_rows["seed"] == seed].copy()
        for threshold in thresholds:
            for scope in scope_order:
                set_names = [name for name in feature_sets.keys() if name.endswith(f"__{scope}")]
                if not set_names:
                    continue

                n = len(set_names)
                ncols = 3
                nrows = int(np.ceil(n / ncols))

                for eval_kind, eval_dataset, panel_title in eval_panels:
                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        figsize=(5.0 * ncols, 4.0 * nrows),
                        constrained_layout=True,
                    )
                    axes = np.atleast_1d(axes).reshape(nrows, ncols)

                    panel_df = seed_df[
                        (seed_df["threshold"] == float(threshold))
                        & (seed_df["temporal_scope"] == scope)
                        & (seed_df["eval_kind"] == eval_kind)
                        & (seed_df["eval_dataset"] == eval_dataset)
                    ]

                    for idx, set_name in enumerate(set_names):
                        r = idx // ncols
                        c = idx % ncols
                        ax = axes[r, c]

                        cm = confusion_mats.get(
                            (
                                train_dataset,
                                seed,
                                float(threshold),
                                set_name,
                                eval_kind if eval_kind == "val" else eval_dataset,
                            ),
                            np.array([[0, 0], [0, 0]], dtype=int),
                        )
                        ax.imshow(cm, cmap="Blues")
                        for i in range(2):
                            for j in range(2):
                                ax.text(
                                    j,
                                    i,
                                    str(int(cm[i, j])),
                                    ha="center",
                                    va="center",
                                    color="black",
                                    fontsize=11,
                                )

                        row = panel_df[panel_df["feature_set"] == set_name]
                        acc_txt = "nan" if row.empty else f"{float(row['accuracy'].iloc[0]):.3f}"
                        f1_txt = "nan" if row.empty else f"{float(row['f1'].iloc[0]):.3f}"
                        base_name, _ = split_feature_set_name(set_name)
                        ax.set_title(f"{base_name}\nacc={acc_txt} | f1={f1_txt}")
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.set_xticklabels(["Pred 0", "Pred 1"])
                        ax.set_yticklabels(["True 0", "True 1"])

                    for idx in range(n, nrows * ncols):
                        r = idx // ncols
                        c = idx % ncols
                        axes[r, c].axis("off")

                    fig.suptitle(
                        f"{panel_title} confusion matrices | train={train_dataset} | "
                        f"seed={seed} | threshold>{threshold:.1f} | {scope}",
                        fontsize=14,
                    )
                    out_path = output_dir / (
                        f"{sanitize_slug(train_dataset)}__confusion__seed{seed}__"
                        f"{sanitize_slug(scope)}__{format_threshold_tag(threshold)}__"
                        f"{sanitize_slug(eval_kind if eval_kind == 'val' else eval_dataset)}.png"
                    )
                    fig.savefig(out_path, dpi=160, bbox_inches="tight")
                    plt.close(fig)


def save_best_importance_plots(
    train_dataset: str,
    metrics_df: pd.DataFrame,
    bundle_lookup: dict[tuple[str, int, float, str], dict[str, Any]],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    ood_df = metrics_df[metrics_df["eval_kind"] == "ood"].copy()
    if ood_df.empty:
        return

    seeds = sorted(ood_df["seed"].dropna().astype(int).unique().tolist())
    for seed in seeds:
        best_rows = (
            ood_df[ood_df["seed"] == seed]
            .groupby(["train_dataset", "seed", "threshold", "feature_set"], as_index=False)[
                ["accuracy", "f1"]
            ]
            .mean()
            .sort_values(["threshold", "accuracy", "f1"], ascending=[True, False, False])
            .drop_duplicates(["threshold"], keep="first")
        )

        if best_rows.empty:
            continue

        fig, axes = plt.subplots(
            len(best_rows),
            1,
            figsize=(12, max(3, 3 * len(best_rows))),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for ax, (_, row) in zip(axes, best_rows.iterrows()):
            key = (
                str(row["train_dataset"]),
                int(row["seed"]),
                float(row["threshold"]),
                str(row["feature_set"]),
            )
            bundle = bundle_lookup.get(key)
            if not bundle or bundle.get("kind") != "xgb":
                ax.set_title(
                    f"Seed {seed} | threshold > {float(row['threshold']):.1f} | "
                    f"{row['feature_set']} | no xgb importances"
                )
                ax.axis("off")
                continue

            importance_rows = extract_importance_rows(
                train_dataset=str(row["train_dataset"]),
                threshold=float(row["threshold"]),
                seed=int(row["seed"]),
                feature_set=str(row["feature_set"]),
                bundle=bundle,
            )
            imp_df = pd.DataFrame(importance_rows).sort_values("gain", ascending=False).head(top_n)
            if imp_df.empty:
                ax.set_title(
                    f"Seed {seed} | threshold > {float(row['threshold']):.1f} | "
                    f"{row['feature_set']} | no gain rows"
                )
                ax.axis("off")
                continue

            ax.barh(imp_df["feature"].iloc[::-1], imp_df["gain"].iloc[::-1], color="#4c78a8")
            ax.set_xlabel("XGBoost gain importance")
            ax.set_title(
                f"Seed {seed} | threshold > {float(row['threshold']):.1f} | "
                f"{row['feature_set']} | mean OOD acc={float(row['accuracy']):.3f}"
            )

        out_path = output_dir / f"{sanitize_slug(train_dataset)}__seed{seed}__best_model_importance.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cached XGBoost OOD classifiers across AdvisorAudit, BS, and Gridworld."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--feature-set-names", type=str, default="")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-confusion-plots", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_float_csv(args.thresholds)
    seeds = parse_int_csv(args.seeds)
    allowed_feature_sets = {
        piece.strip() for piece in args.feature_set_names.split(",") if piece.strip()
    } or None
    if args.skip_classification and args.skip_regression:
        raise ValueError("At least one of classification or regression must be enabled.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = detect_xgb_device() if args.device == "auto" else args.device
    print(f"[config] XGBoost device: {device}")
    print(f"[config] thresholds: {thresholds}")
    print(f"[config] seeds: {seeds}")

    artifacts = {
        dataset_name: resolve_dataset_artifacts(args.dataset_root, dataset_name)
        for dataset_name in args.datasets
    }
    for dataset_name, item in artifacts.items():
        print(
            f"[data] {dataset_name}: features={item.features_path} | "
            f"feature_sets={item.feature_sets_path}"
        )

    frames = {dataset_name: load_dataset_frame(item.features_path) for dataset_name, item in artifacts.items()}
    common_cols = set.intersection(*(set(df.columns) for df in frames.values()))
    print(f"[data] common columns across datasets: {len(common_cols):,}")

    all_metric_rows: list[dict[str, Any]] = []
    all_importance_rows: list[dict[str, Any]] = []
    all_regression_metric_rows: list[dict[str, Any]] = []
    all_regression_importance_rows: list[dict[str, Any]] = []
    confusion_mats: dict[tuple[str, int, float, str, str], np.ndarray] = {}
    bundle_lookup: dict[tuple[str, int, float, str], dict[str, Any]] = {}

    for train_dataset in args.datasets:
        item = artifacts[train_dataset]
        train_full_df = frames[train_dataset].copy().reset_index(drop=True)

        raw_feature_sets = load_feature_sets_from_json(item.feature_sets_path, common_cols)
        ablation_sets, ablation_summary = build_ablation_feature_sets(raw_feature_sets)
        feature_sets, scope_summary = expand_feature_sets_by_scope(
            ablation_sets, scopes=list(DEFAULT_SCOPES)
        )
        feature_sets = filter_feature_sets(feature_sets, allowed_feature_sets)

        train_report_dir = args.results_dir / train_dataset
        train_report_dir.mkdir(parents=True, exist_ok=True)
        ablation_summary.to_csv(train_report_dir / "ablation_feature_sets.csv", index=False)
        scope_summary.to_csv(train_report_dir / "scope_feature_sets.csv", index=False)

        for seed in seeds:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=seed)
            idx = np.arange(len(train_full_df))
            tr_idx, va_idx = next(gss.split(idx, groups=train_full_df["example_id"].astype(str)))
            train_df = train_full_df.iloc[tr_idx].reset_index(drop=True)
            val_df = train_full_df.iloc[va_idx].reset_index(drop=True)
            model_dir = item.model_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[split] train={train_dataset} seed={seed} | "
                f"train_rows={len(train_df):,} val_rows={len(val_df):,} "
                f"feature_sets={len(feature_sets)}"
            )

            if not args.skip_classification:
                for threshold in thresholds:
                    y_train = (train_df["deception_rate"].to_numpy(dtype=float) > threshold).astype(int)
                    y_val = (val_df["deception_rate"].to_numpy(dtype=float) > threshold).astype(int)
                    print(
                        f"[train] dataset={train_dataset} seed={seed} thr={threshold:.2f} "
                        f"pos_rate_train={y_train.mean():.4f} pos_rate_val={y_val.mean():.4f}"
                    )

                    for feature_set_name, feature_names in feature_sets.items():
                        model_path = build_model_path(model_dir, feature_set_name, threshold, seed)
                        bundle = None if args.force_retrain else load_compatible_bundle(
                            model_path=model_path,
                            train_dataset=train_dataset,
                            feature_set=feature_set_name,
                            threshold=threshold,
                            seed=seed,
                            feature_names=feature_names,
                        )
                        loaded_from_cache = bundle is not None

                        if bundle is None:
                            x_train = sanitize_numeric(train_df, feature_names).astype(np.float32)
                            x_val = sanitize_numeric(val_df, feature_names).astype(np.float32)
                            bundle = fit_and_save_bundle(
                                model_path=model_path,
                                train_dataset=train_dataset,
                                feature_set=feature_set_name,
                                threshold=threshold,
                                seed=seed,
                                feature_names=feature_names,
                                x_train=x_train,
                                y_train=y_train,
                                x_val=x_val,
                                y_val=y_val,
                                n_jobs=args.n_jobs,
                                early_stopping_rounds=args.early_stopping_rounds,
                                device=device,
                            )
                        else:
                            x_train = sanitize_numeric(train_df, feature_names).astype(np.float32)
                            x_val = sanitize_numeric(val_df, feature_names).astype(np.float32)

                        cache_txt = "cache" if loaded_from_cache else "fit"
                        print(
                            f"  [{cache_txt}] {feature_set_name} "
                            f"({len(feature_names)} features) -> {model_path.name}"
                        )

                        bundle_lookup[(train_dataset, seed, float(threshold), feature_set_name)] = bundle
                        all_importance_rows.extend(
                            extract_importance_rows(
                                train_dataset=train_dataset,
                                threshold=threshold,
                                seed=seed,
                                feature_set=feature_set_name,
                                bundle=bundle,
                            )
                        )

                        train_prob = predict_bundle(bundle, x_train)
                        val_prob = predict_bundle(bundle, x_val)

                        train_row = make_eval_row(
                            train_dataset=train_dataset,
                            eval_dataset=train_dataset,
                            eval_kind="train",
                            seed=seed,
                            threshold=threshold,
                            feature_set=feature_set_name,
                            feature_names=feature_names,
                            bundle=bundle,
                            model_path=model_path,
                            y_true=y_train,
                            y_prob=train_prob,
                        )
                        val_row = make_eval_row(
                            train_dataset=train_dataset,
                            eval_dataset=train_dataset,
                            eval_kind="val",
                            seed=seed,
                            threshold=threshold,
                            feature_set=feature_set_name,
                            feature_names=feature_names,
                            bundle=bundle,
                            model_path=model_path,
                            y_true=y_val,
                            y_prob=val_prob,
                        )
                        all_metric_rows.extend([train_row, val_row])
                        confusion_mats[
                            (train_dataset, seed, float(threshold), feature_set_name, "val")
                        ] = binary_metrics(y_val, val_prob)["cm"]

                        for eval_dataset in args.datasets:
                            if eval_dataset == train_dataset:
                                continue
                            eval_df = frames[eval_dataset]
                            x_eval = sanitize_numeric(eval_df, feature_names).astype(np.float32)
                            y_eval = (
                                eval_df["deception_rate"].to_numpy(dtype=float) > threshold
                            ).astype(int)
                            eval_prob = predict_bundle(bundle, x_eval)
                            eval_row = make_eval_row(
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                eval_kind="ood",
                                seed=seed,
                                threshold=threshold,
                                feature_set=feature_set_name,
                                feature_names=feature_names,
                                bundle=bundle,
                                model_path=model_path,
                                y_true=y_eval,
                                y_prob=eval_prob,
                            )
                            all_metric_rows.append(eval_row)
                            confusion_mats[
                                (train_dataset, seed, float(threshold), feature_set_name, eval_dataset)
                            ] = binary_metrics(y_eval, eval_prob)["cm"]

            if not args.skip_regression:
                y_train_reg = train_df["deception_rate"].to_numpy(dtype=np.float32)
                y_val_reg = val_df["deception_rate"].to_numpy(dtype=np.float32)
                print(
                    f"[reg] dataset={train_dataset} seed={seed} "
                    f"train_target_mean={y_train_reg.mean():.4f} val_target_mean={y_val_reg.mean():.4f}"
                )

                for feature_set_name, feature_names in feature_sets.items():
                    model_path = build_regression_model_path(model_dir, feature_set_name, seed)
                    bundle = None if args.force_retrain else load_compatible_regression_bundle(
                        model_path=model_path,
                        train_dataset=train_dataset,
                        feature_set=feature_set_name,
                        seed=seed,
                        feature_names=feature_names,
                    )
                    loaded_from_cache = bundle is not None

                    x_train = sanitize_numeric(train_df, feature_names).astype(np.float32)
                    x_val = sanitize_numeric(val_df, feature_names).astype(np.float32)
                    if bundle is None:
                        bundle = fit_and_save_regression_bundle(
                            model_path=model_path,
                            train_dataset=train_dataset,
                            feature_set=feature_set_name,
                            seed=seed,
                            feature_names=feature_names,
                            x_train=x_train,
                            y_train=y_train_reg,
                            x_val=x_val,
                            y_val=y_val_reg,
                            n_jobs=args.n_jobs,
                            early_stopping_rounds=args.early_stopping_rounds,
                            device=device,
                        )

                    cache_txt = "cache" if loaded_from_cache else "fit"
                    print(
                        f"  [reg-{cache_txt}] {feature_set_name} "
                        f"({len(feature_names)} features) -> {model_path.name}"
                    )

                    all_regression_importance_rows.extend(
                        extract_regression_importance_rows(
                            train_dataset=train_dataset,
                            seed=seed,
                            feature_set=feature_set_name,
                            bundle=bundle,
                        )
                    )

                    train_pred = predict_regression_bundle(bundle, x_train)
                    val_pred = predict_regression_bundle(bundle, x_val)
                    all_regression_metric_rows.extend(
                        [
                            make_regression_eval_row(
                                train_dataset=train_dataset,
                                eval_dataset=train_dataset,
                                eval_kind="train",
                                seed=seed,
                                feature_set=feature_set_name,
                                feature_names=feature_names,
                                bundle=bundle,
                                model_path=model_path,
                                y_true=y_train_reg,
                                y_pred=train_pred,
                            ),
                            make_regression_eval_row(
                                train_dataset=train_dataset,
                                eval_dataset=train_dataset,
                                eval_kind="val",
                                seed=seed,
                                feature_set=feature_set_name,
                                feature_names=feature_names,
                                bundle=bundle,
                                model_path=model_path,
                                y_true=y_val_reg,
                                y_pred=val_pred,
                            ),
                        ]
                    )

                    for eval_dataset in args.datasets:
                        if eval_dataset == train_dataset:
                            continue
                        eval_df = frames[eval_dataset]
                        x_eval = sanitize_numeric(eval_df, feature_names).astype(np.float32)
                        y_eval = eval_df["deception_rate"].to_numpy(dtype=np.float32)
                        eval_pred = predict_regression_bundle(bundle, x_eval)
                        all_regression_metric_rows.append(
                            make_regression_eval_row(
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                eval_kind="ood",
                                seed=seed,
                                feature_set=feature_set_name,
                                feature_names=feature_names,
                                bundle=bundle,
                                model_path=model_path,
                                y_true=y_eval,
                                y_pred=eval_pred,
                            )
                        )

        if not args.skip_classification:
            train_metrics_df = pd.DataFrame(
                [row for row in all_metric_rows if row["train_dataset"] == train_dataset]
            )
            train_metrics_df.to_csv(train_report_dir / "classification_metrics.csv", index=False)
            save_accuracy_threshold_plots(
                train_dataset=train_dataset,
                metrics_df=train_metrics_df,
                feature_sets=feature_sets,
                thresholds=thresholds,
                output_dir=train_report_dir,
                metric="accuracy",
            )
            save_accuracy_threshold_plots(
                train_dataset=train_dataset,
                metrics_df=train_metrics_df,
                feature_sets=feature_sets,
                thresholds=thresholds,
                output_dir=train_report_dir,
                metric="f1",
            )
            if not args.skip_confusion_plots:
                save_confusion_grids(
                    train_dataset=train_dataset,
                    metrics_df=train_metrics_df,
                    confusion_mats=confusion_mats,
                    feature_sets=feature_sets,
                    thresholds=thresholds,
                    output_dir=train_report_dir,
                )
            save_best_importance_plots(
                train_dataset=train_dataset,
                metrics_df=train_metrics_df,
                bundle_lookup=bundle_lookup,
                output_dir=train_report_dir,
            )

        if not args.skip_regression:
            train_regression_df = pd.DataFrame(
                [row for row in all_regression_metric_rows if row["train_dataset"] == train_dataset]
            )
            if not train_regression_df.empty:
                train_regression_df.to_csv(train_report_dir / "regression_metrics.csv", index=False)

    metrics_df = pd.DataFrame(all_metric_rows)
    importance_df = pd.DataFrame(all_importance_rows)
    regression_metrics_df = pd.DataFrame(all_regression_metric_rows)
    regression_importance_df = pd.DataFrame(all_regression_importance_rows)

    if not args.skip_classification:
        metrics_path = args.results_dir / "all_classification_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[save] metrics -> {metrics_path}")

    if not args.skip_classification and not importance_df.empty:
        importance_path = args.results_dir / "all_classification_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"[save] importances -> {importance_path}")

    if not args.skip_regression and not regression_metrics_df.empty:
        regression_metrics_path = args.results_dir / "all_regression_metrics.csv"
        regression_metrics_df.to_csv(regression_metrics_path, index=False)
        print(f"[save] regression metrics -> {regression_metrics_path}")

    if not args.skip_regression and not regression_importance_df.empty:
        regression_importance_path = args.results_dir / "all_regression_importance.csv"
        regression_importance_df.to_csv(regression_importance_path, index=False)
        print(f"[save] regression importances -> {regression_importance_path}")

    if not args.skip_classification:
        summary_df = metrics_df[metrics_df["eval_kind"].isin(["val", "ood"])].copy()
        summary_wide = (
            summary_df.assign(
                eval_label=np.where(
                    summary_df["eval_kind"] == "val",
                    "val",
                    "ood_" + summary_df["eval_dataset"].astype(str),
                )
            )
            .pivot_table(
                index=["train_dataset", "seed", "threshold", "feature_set", "base_feature_set", "temporal_scope"],
                columns="eval_label",
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
        )
        summary_wide.columns = [
            col if isinstance(col, str) else "__".join(str(part) for part in col if part)
            for col in summary_wide.columns.to_flat_index()
        ]
        summary_path = args.results_dir / "accuracy_summary_wide.csv"
        summary_wide.to_csv(summary_path, index=False)
        print(f"[save] accuracy summary -> {summary_path}")


if __name__ == "__main__":
    main()
