#!/usr/bin/env python3
"""
Initialize a sandboxed autoresearch run for deception2.

This script:
1. validates the fixed 7B dataset layout
2. seeds editable sandbox copies of the baseline scripts
3. creates runs/<tag>/run_config.json and results.tsv
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
SANDBOX_DIR = REPO_ROOT / "sandbox"
RUNS_DIR = REPO_ROOT / "runs"

DEFAULT_DATASET_ROOT = Path("/playpen-ssd/smerrill/deception2/Dataset")
DEFAULT_NOTEBOOK_TEMPLATE = Path(
    "/playpen-ssd/smerrill/deception2/Notebooks/paper_ood_figures.ipynb"
)
DEFAULT_BASELINE_SOURCES = {
    "feature_extractor": Path(
        "/playpen-ssd/smerrill/deception2/src/feature_extractor.py"
    ),
    "multidataset_ood_xgb": Path(
        "/playpen-ssd/smerrill/deception2/src/multidataset_ood_xgb.py"
    ),
}

DEFAULT_DATASETS = ("AdvisorAudit", "BS", "Gridworld")
MODEL_DIR_NAME = "DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
DEFAULT_SEEDS = (42,)
DEFAULT_SCOPES = ("before", "before_at", "before_at_after")
DEFAULT_NUM_EXAMPLES = 100000
DEFAULT_N_JOBS = 8
DEFAULT_EARLY_STOPPING_ROUNDS = 120
DEFAULT_GPU_ID = 7

REQUIRED_SOURCE_FILES = (
    "examples.jsonl",
    "sentences.jsonl",
    "localization.jsonl",
)

RESULTS_TSV_HEADER = (
    "commit\tclassification_mean_ood_auroc\tregression_mean_ood_pearson\tstatus\tdescription\n"
)


def default_run_tag() -> str:
    now = datetime.now()
    return now.strftime("%b").lower() + str(now.day)


def ensure_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def copy_if_needed(src: Path, dst: Path, *, refresh: bool) -> str:
    if dst.exists() and not refresh:
        return "kept"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def dataset_entry(dataset_root: Path, run_root: Path, dataset_name: str) -> dict[str, Any]:
    source_root = dataset_root / dataset_name / MODEL_DIR_NAME
    ensure_exists(source_root, label=f"{dataset_name} source directory")

    source_paths = {}
    for filename in REQUIRED_SOURCE_FILES:
        src = source_root / filename
        ensure_exists(src, label=f"{dataset_name} {filename}")
        source_paths[filename] = str(src)

    feature_cache_root = run_root / "feature_cache" / dataset_name / MODEL_DIR_NAME
    return {
        "name": dataset_name,
        "model_dir_name": MODEL_DIR_NAME,
        "model_name": MODEL_NAME,
        "source_root": str(source_root),
        "examples_path": source_paths["examples.jsonl"],
        "sentences_path": source_paths["sentences.jsonl"],
        "localization_path": source_paths["localization.jsonl"],
        "feature_cache_root": str(feature_cache_root),
        "features_path": str(feature_cache_root / "features.parquet"),
        "feature_sets_path": str(feature_cache_root / "feature_sets.json"),
        "manifest_path": str(feature_cache_root / "features_manifest.json"),
        "raw_features_path": str(feature_cache_root / "raw_features.parquet"),
    }


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    run_root = RUNS_DIR / args.run_tag
    datasets = [
        dataset_entry(args.dataset_root, run_root, dataset_name)
        for dataset_name in args.datasets
    ]

    sandbox_paths = {
        name: str(SANDBOX_DIR / f"{name}.py")
        for name in DEFAULT_BASELINE_SOURCES.keys()
    }

    results_root = run_root / "results"
    config = {
        "run_tag": args.run_tag,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "sandbox_root": str(SANDBOX_DIR),
        "run_root": str(run_root),
        "gpu_id": int(args.gpu_id),
        "visible_cuda_devices": str(args.gpu_id),
        "dataset_root": str(args.dataset_root),
        "editable_files": sandbox_paths,
        "notebook_template": str(args.notebook_template),
        "datasets": datasets,
        "classification": {
            "label_rule": "binary label is 1 iff deception_rate > threshold",
            "thresholds": list(DEFAULT_THRESHOLDS),
            "primary_metric": "mean_ood_auroc",
        },
        "regression": {
            "target": "deception_rate",
            "primary_metric": "mean_ood_pearson",
        },
        "feature_extraction": {
            "num_examples": int(args.num_examples),
            "device": "cuda",
            "max_tokens": 3000,
        },
        "modeling": {
            "datasets": list(args.datasets),
            "thresholds": list(DEFAULT_THRESHOLDS),
            "seeds": list(DEFAULT_SEEDS),
            "scopes": list(DEFAULT_SCOPES),
            "n_jobs": int(args.n_jobs),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "device": "cuda",
        },
        "outputs": {
            "feature_cache_root": str(run_root / "feature_cache"),
            "results_root": str(results_root),
            "logs_root": str(run_root / "logs"),
            "notebooks_root": str(run_root / "notebooks"),
            "summary_json": str(run_root / "summary.json"),
            "classification_leaderboard": str(run_root / "classification_leaderboard.csv"),
            "regression_leaderboard": str(run_root / "regression_leaderboard.csv"),
            "results_tsv": str(run_root / "results.tsv"),
        },
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a sandboxed deception2 autoresearch run."
    )
    parser.add_argument("--run-tag", default=default_run_tag())
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
    )
    parser.add_argument(
        "--notebook-template",
        type=Path,
        default=DEFAULT_NOTEBOOK_TEMPLATE,
    )
    parser.add_argument(
        "--refresh-sandbox",
        action="store_true",
        help="Overwrite sandbox working copies from the baseline sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_exists(args.dataset_root, label="dataset root")
    ensure_exists(args.notebook_template, label="paper notebook template")
    for src in DEFAULT_BASELINE_SOURCES.values():
        ensure_exists(src, label="baseline source")

    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    run_root = RUNS_DIR / args.run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    sandbox_status = {}
    for name, src in DEFAULT_BASELINE_SOURCES.items():
        dst = SANDBOX_DIR / f"{name}.py"
        sandbox_status[name] = copy_if_needed(
            src, dst, refresh=args.refresh_sandbox
        )

    config = build_run_config(args)
    outputs = config["outputs"]
    for key in ("feature_cache_root", "results_root", "logs_root", "notebooks_root"):
        Path(outputs[key]).mkdir(parents=True, exist_ok=True)

    config_path = run_root / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    results_tsv = Path(outputs["results_tsv"])
    if not results_tsv.exists():
        results_tsv.write_text(RESULTS_TSV_HEADER, encoding="utf-8")

    print(f"[ok] run tag: {args.run_tag}")
    print(f"[ok] config: {config_path}")
    print(f"[ok] notebook template: {args.notebook_template}")
    print("[ok] sandbox files:")
    for name, status in sandbox_status.items():
        print(f"  - {name}: {status} -> {SANDBOX_DIR / (name + '.py')}")
    print("[ok] dataset inputs:")
    for item in config["datasets"]:
        print(f"  - {item['name']}: {item['source_root']}")
    print("[ok] binary classification label:")
    print("  - y_binary = (deception_rate > threshold).astype(int)")
    print(f"[ok] gpu lock: CUDA_VISIBLE_DEVICES={args.gpu_id}")


if __name__ == "__main__":
    main()
