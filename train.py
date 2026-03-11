#!/usr/bin/env python3
"""
Sandboxed feature-mining and OOD modeling runner for deception2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"


def parse_float_csv(value: str) -> list[float]:
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def format_float(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def tail_text(path: Path, *, lines: int = 40) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def ensure_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_logged_command(
    cmd: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    print(f"[cmd] {printable}")
    print(f"[log] {log_path}")
    if dry_run:
        return

    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        tail = tail_text(log_path)
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {printable}\n"
            f"Last log lines from {log_path}:\n{tail}"
        )


def load_config(run_tag: str, config_path: Path | None) -> dict[str, Any]:
    path = config_path or (RUNS_DIR / run_tag / "run_config.json")
    ensure_file(path, label="run config")
    return json.loads(path.read_text(encoding="utf-8"))


def select_dataset_entries(
    config: dict[str, Any],
    selected_names: list[str] | None,
) -> list[dict[str, Any]]:
    all_items = list(config["datasets"])
    if not selected_names:
        return all_items

    lookup = {item["name"]: item for item in all_items}
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise KeyError(f"Unknown dataset(s): {missing}")
    return [lookup[name] for name in selected_names]


def build_feature_command(
    config: dict[str, Any],
    dataset_item: dict[str, Any],
    *,
    num_examples: int,
) -> list[str]:
    extractor_path = Path(config["editable_files"]["feature_extractor"])
    ensure_file(extractor_path, label="sandbox feature extractor")

    cmd = [
        sys.executable,
        str(extractor_path),
        "--out-path",
        dataset_item["features_path"],
        "--examples-path",
        dataset_item["examples_path"],
        "--sentences-path",
        dataset_item["sentences_path"],
        "--localization-path",
        dataset_item["localization_path"],
        "--model-name",
        dataset_item["model_name"],
        "--num-examples",
        str(num_examples),
        "--device",
        config["feature_extraction"]["device"],
        "--feature-sets-out",
        dataset_item["feature_sets_path"],
        "--manifest-out",
        dataset_item["manifest_path"],
        "--raw-out-path",
        dataset_item["raw_features_path"],
    ]
    return cmd


def validate_feature_cache(dataset_items: list[dict[str, Any]]) -> None:
    for item in dataset_items:
        for key in ("features_path", "feature_sets_path", "manifest_path"):
            ensure_file(Path(item[key]), label=f"{item['name']} cached {key}")


def build_modeling_command(
    config: dict[str, Any],
    *,
    dataset_names: list[str],
    thresholds: list[float],
    seeds: list[int],
    force_retrain: bool,
) -> list[str]:
    modeling_path = Path(config["editable_files"]["multidataset_ood_xgb"])
    ensure_file(modeling_path, label="sandbox OOD modeling script")

    results_root = config["outputs"]["results_root"]
    feature_cache_root = config["outputs"]["feature_cache_root"]
    modeling_cfg = config["modeling"]

    cmd = [
        sys.executable,
        str(modeling_path),
        "--dataset-root",
        feature_cache_root,
        "--datasets",
        *dataset_names,
        "--thresholds",
        ",".join(str(value) for value in thresholds),
        "--seeds",
        ",".join(str(value) for value in seeds),
        "--n-jobs",
        str(modeling_cfg["n_jobs"]),
        "--early-stopping-rounds",
        str(modeling_cfg["early_stopping_rounds"]),
        "--device",
        modeling_cfg["device"],
        "--results-dir",
        results_root,
    ]
    if force_retrain:
        cmd.append("--force-retrain")
    return cmd


def make_classification_leaderboard(
    metrics_path: Path,
    output_path: Path,
) -> dict[str, Any] | None:
    ensure_file(metrics_path, label="classification metrics")
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    key_cols = [
        "train_dataset",
        "seed",
        "threshold",
        "feature_set",
        "base_feature_set",
        "temporal_scope",
        "n_features",
        "model_kind",
        "model_path",
    ]

    for metric in ("roc_auc", "f1", "accuracy"):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    ood = df[df["eval_kind"] == "ood"].copy()
    if ood.empty:
        return None

    leaderboard = (
        ood.groupby(key_cols, dropna=False)
        .agg(
            mean_ood_auroc=("roc_auc", "mean"),
            mean_ood_f1=("f1", "mean"),
            mean_ood_accuracy=("accuracy", "mean"),
            ood_eval_datasets=("eval_dataset", "nunique"),
        )
        .reset_index()
    )

    val = (
        df[df["eval_kind"] == "val"]
        .groupby(key_cols, dropna=False)
        .agg(
            val_auroc=("roc_auc", "mean"),
            val_f1=("f1", "mean"),
            val_accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )
    leaderboard = leaderboard.merge(val, how="left", on=key_cols)
    leaderboard = leaderboard.sort_values(
        [
            "mean_ood_auroc",
            "mean_ood_f1",
            "mean_ood_accuracy",
            "val_auroc",
            "val_f1",
            "n_features",
            "train_dataset",
            "feature_set",
            "threshold",
        ],
        ascending=[False, False, False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    leaderboard.to_csv(output_path, index=False)
    return leaderboard.iloc[0].to_dict()


def make_regression_leaderboard(
    metrics_path: Path,
    output_path: Path,
) -> dict[str, Any] | None:
    ensure_file(metrics_path, label="regression metrics")
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    key_cols = [
        "train_dataset",
        "seed",
        "feature_set",
        "base_feature_set",
        "temporal_scope",
        "n_features",
        "model_kind",
        "model_path",
    ]

    for metric in ("pearson", "r2", "rmse", "mae"):
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    ood = df[df["eval_kind"] == "ood"].copy()
    if ood.empty:
        return None

    leaderboard = (
        ood.groupby(key_cols, dropna=False)
        .agg(
            mean_ood_pearson=("pearson", "mean"),
            mean_ood_r2=("r2", "mean"),
            mean_ood_rmse=("rmse", "mean"),
            mean_ood_mae=("mae", "mean"),
            ood_eval_datasets=("eval_dataset", "nunique"),
        )
        .reset_index()
    )

    val = (
        df[df["eval_kind"] == "val"]
        .groupby(key_cols, dropna=False)
        .agg(
            val_pearson=("pearson", "mean"),
            val_r2=("r2", "mean"),
            val_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    leaderboard = leaderboard.merge(val, how="left", on=key_cols)
    leaderboard = leaderboard.sort_values(
        [
            "mean_ood_pearson",
            "mean_ood_r2",
            "val_pearson",
            "val_r2",
            "mean_ood_rmse",
            "n_features",
            "train_dataset",
            "feature_set",
        ],
        ascending=[False, False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    leaderboard.to_csv(output_path, index=False)
    return leaderboard.iloc[0].to_dict()


def build_summary(
    config: dict[str, Any],
    *,
    classification_best: dict[str, Any] | None,
    regression_best: dict[str, Any] | None,
    notebook_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_tag": config["run_tag"],
        "gpu_id": config["gpu_id"],
        "binary_label_rule": config["classification"]["label_rule"],
        "classification": classification_best,
        "regression": regression_best,
        "paths": {
            "results_root": config["outputs"]["results_root"],
            "classification_leaderboard": config["outputs"]["classification_leaderboard"],
            "regression_leaderboard": config["outputs"]["regression_leaderboard"],
            "summary_json": config["outputs"]["summary_json"],
            "notebook_path": str(notebook_path) if notebook_path is not None else None,
        },
    }


def patch_notebook_results_dir(
    template_path: Path,
    output_path: Path,
    *,
    results_dir: Path,
) -> None:
    ensure_file(template_path, label="paper notebook template")
    notebook = json.loads(template_path.read_text(encoding="utf-8"))

    replacement = (
        "RESULTS_DIR_CANDIDATES = [\n"
        f"    Path({str(results_dir)!r}),\n"
        "]\n\n"
        "PRIMARY_METRIC ="
    )

    patched = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "RESULTS_DIR_CANDIDATES = [" not in src:
            continue
        src_new, count = re.subn(
            r"RESULTS_DIR_CANDIDATES = \[(?:.|\n)*?\]\n\nPRIMARY_METRIC =",
            replacement,
            src,
            count=1,
        )
        if count:
            cell["source"] = src_new.splitlines(keepends=True)
            patched = True
            break

    if not patched:
        raise RuntimeError(
            f"Could not patch RESULTS_DIR_CANDIDATES in notebook template: {template_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sandboxed deception2 feature-mining and OOD modeling pipeline."
    )
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-modeling", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.run_tag, args.config_path)
    dataset_items = select_dataset_entries(config, args.datasets)
    dataset_names = [item["name"] for item in dataset_items]

    thresholds = (
        parse_float_csv(args.thresholds)
        if args.thresholds
        else list(config["modeling"]["thresholds"])
    )
    seeds = parse_int_csv(args.seeds) if args.seeds else list(config["modeling"]["seeds"])
    num_examples = (
        int(args.num_examples)
        if args.num_examples is not None
        else int(config["feature_extraction"]["num_examples"])
    )

    outputs = config["outputs"]
    results_root = Path(outputs["results_root"])
    logs_root = Path(outputs["logs_root"])
    summary_json_path = Path(outputs["summary_json"])
    classification_leaderboard_path = Path(outputs["classification_leaderboard"])
    regression_leaderboard_path = Path(outputs["regression_leaderboard"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(config["gpu_id"])
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[run] tag={config['run_tag']}")
    print(f"[run] datasets={dataset_names}")
    print(f"[run] thresholds={thresholds}")
    print(f"[run] seeds={seeds}")
    print(f"[run] num_examples={num_examples}")
    print(f"[run] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    if not args.skip_features:
        for item in dataset_items:
            cmd = build_feature_command(
                config,
                item,
                num_examples=num_examples,
            )
            run_logged_command(
                cmd,
                log_path=logs_root / f"feature_{item['name']}.log",
                env=env,
                dry_run=args.dry_run,
            )
    else:
        validate_feature_cache(dataset_items)
        print("[skip] reusing existing feature cache")

    if not args.skip_modeling:
        cmd = build_modeling_command(
            config,
            dataset_names=dataset_names,
            thresholds=thresholds,
            seeds=seeds,
            force_retrain=args.force_retrain,
        )
        run_logged_command(
            cmd,
            log_path=logs_root / "modeling.log",
            env=env,
            dry_run=args.dry_run,
        )
    else:
        print("[skip] reusing existing modeling outputs")

    if args.dry_run:
        print("[dry-run] commands printed, no artifacts were generated")
        return

    classification_metrics_path = results_root / "all_classification_metrics.csv"
    regression_metrics_path = results_root / "all_regression_metrics.csv"

    classification_best = None
    regression_best = None

    if classification_metrics_path.exists():
        classification_best = make_classification_leaderboard(
            classification_metrics_path,
            classification_leaderboard_path,
        )
    if regression_metrics_path.exists():
        regression_best = make_regression_leaderboard(
            regression_metrics_path,
            regression_leaderboard_path,
        )

    notebook_path: Path | None = None
    if not args.skip_notebook:
        notebook_path = Path(outputs["notebooks_root"]) / "paper_ood_figures.ipynb"
        patch_notebook_results_dir(
            Path(config["notebook_template"]),
            notebook_path,
            results_dir=results_root,
        )

    summary = build_summary(
        config,
        classification_best=classification_best,
        regression_best=regression_best,
        notebook_path=notebook_path,
    )
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("---")
    print(
        "classification_mean_ood_auroc: "
        + format_float(
            None if classification_best is None else classification_best["mean_ood_auroc"]
        )
    )
    print(
        "classification_train_dataset: "
        + (
            "none"
            if classification_best is None
            else str(classification_best["train_dataset"])
        )
    )
    print(
        "classification_feature_set: "
        + (
            "none"
            if classification_best is None
            else str(classification_best["feature_set"])
        )
    )
    print(
        "classification_threshold: "
        + (
            "none"
            if classification_best is None
            else f"{float(classification_best['threshold']):.2f}"
        )
    )
    print(
        "regression_mean_ood_pearson: "
        + format_float(
            None if regression_best is None else regression_best["mean_ood_pearson"]
        )
    )
    print(
        "regression_train_dataset: "
        + ("none" if regression_best is None else str(regression_best["train_dataset"]))
    )
    print(
        "regression_feature_set: "
        + ("none" if regression_best is None else str(regression_best["feature_set"]))
    )
    print(f"results_dir: {results_root}")
    print(f"summary_json: {summary_json_path}")
    print(f"notebook_path: {notebook_path if notebook_path is not None else 'none'}")


if __name__ == "__main__":
    main()
