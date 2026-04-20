"""Synthetic benchmark suite for harness quality measurement.

Usage:
    python tests/benchmark.py --all          # Run all tasks
    python tests/benchmark.py --quick        # Run subset
    python tests/benchmark.py --all --json   # JSON output
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from observe import TelemetryCollector

TASKS_DIR = Path(__file__).parent / "tasks"

REQUIRED_FIELDS = {"name", "description", "category", "input", "expected_outcome", "metric_annotations"}
REQUIRED_INPUT_FIELDS = {"prompt"}
REQUIRED_ANNOTATION_FIELDS = {"exercises"}

VALID_METRICS = {
    "task_success_rate",
    "quality_gate_pass_rate",
    "rework_rate",
    "avg_token_consumption",
    "time_per_turn",
}


def discover_tasks(tasks_dir: Path) -> list[Path]:
    """Find all YAML task definitions under the tasks directory."""
    return sorted(tasks_dir.rglob("*.yaml"))


def validate_task(task_path: Path) -> tuple[dict | None, list[str]]:
    """Load and validate a task definition. Returns (parsed_task, errors)."""
    errors: list[str] = []
    try:
        data = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [f"YAML parse error: {exc}"]

    if not isinstance(data, dict):
        return None, ["Task file must contain a YAML mapping"]

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        errors.append(f"Missing required fields: {', '.join(sorted(missing))}")

    inp = data.get("input", {})
    if isinstance(inp, dict):
        missing_input = REQUIRED_INPUT_FIELDS - inp.keys()
        if missing_input:
            errors.append(f"input missing fields: {', '.join(sorted(missing_input))}")
    else:
        errors.append("input must be a mapping")

    annotations = data.get("metric_annotations", {})
    if isinstance(annotations, dict):
        missing_ann = REQUIRED_ANNOTATION_FIELDS - annotations.keys()
        if missing_ann:
            errors.append(f"metric_annotations missing: {', '.join(sorted(missing_ann))}")
        exercises = annotations.get("exercises", [])
        if isinstance(exercises, list):
            invalid = set(exercises) - VALID_METRICS
            if invalid:
                errors.append(f"Unknown metrics: {', '.join(sorted(invalid))}")
    else:
        errors.append("metric_annotations must be a mapping")

    return data, errors


def placeholder_metrics() -> dict:
    """Return placeholder metric values (skeleton -- real evaluation comes later)."""
    return {
        "task_success_rate": 1.0,
        "quality_gate_pass_rate": 1.0,
        "rework_rate": 0.0,
        "avg_token_consumption": 0.0,
        "time_per_turn": 0.0,
    }


def run_benchmark(tasks: list[Path], collector: TelemetryCollector | None = None) -> dict:
    """Run all tasks and collect results."""
    results: list[dict] = []
    passed = 0
    failed = 0

    for task_path in tasks:
        rel = task_path.relative_to(TASKS_DIR)
        start = time.time()
        data, errors = validate_task(task_path)
        end = time.time()

        if collector:
            with collector.task(str(rel)) as t:
                t.record_timing(start, end)
                t.record_tokens(input=0, output=0)
                if errors:
                    t.status = "fail"
                    t.error_category = "constraint_violation"

        if errors:
            failed += 1
            results.append({"task": str(rel), "status": "FAIL", "errors": errors})
        else:
            passed += 1
            results.append({"task": str(rel), "status": "PASS", "name": data["name"]})

    metrics = placeholder_metrics()
    if tasks:
        metrics["task_success_rate"] = passed / len(tasks)
        metrics["quality_gate_pass_rate"] = passed / len(tasks)
        metrics["rework_rate"] = failed / len(tasks) if len(tasks) > 0 else 0.0

    composite = composite_score_local(metrics)
    return {
        "total": len(tasks),
        "passed": passed,
        "failed": failed,
        "results": results,
        "metrics": metrics,
        "harness_score": round(composite, 3),
    }


def composite_score_local(metrics: dict) -> float:
    """Weighted harmonic mean. Tier weights: T1=0.5, T2=0.3, T3=0.2."""
    t1 = [metrics["task_success_rate"], metrics["quality_gate_pass_rate"]]
    t2 = [1.0 - metrics["rework_rate"]]  # invert so higher is better
    t3_raw = [metrics["avg_token_consumption"], metrics["time_per_turn"]]
    # For T3, lower is better; normalize with 1/(1+x) so result is in (0,1]
    t3 = [1.0 / (1.0 + v) for v in t3_raw]

    groups = [(0.5, t1), (0.3, t2), (0.2, t3)]
    weighted_sum = 0.0
    weight_total = 0.0
    for weight, values in groups:
        avg = sum(values) / len(values) if values else 0.0
        if avg <= 0:
            return 0.0
        weighted_sum += weight / avg
        weight_total += weight

    if weighted_sum <= 0:
        return 0.0
    return weight_total / weighted_sum


def print_human(summary: dict) -> None:
    """Print human-readable results."""
    print(f"Benchmark: {summary['passed']}/{summary['total']} tasks passed\n")
    for r in summary["results"]:
        status = r["status"]
        task = r["task"]
        if status == "FAIL":
            print(f"  FAIL  {task}")
            for err in r.get("errors", []):
                print(f"        {err}")
        else:
            print(f"  PASS  {task}")

    print(f"\nMetrics:")
    for k, v in summary["metrics"].items():
        print(f"  {k}: {v:.3f}")
    print(f"\nharness_score: {summary['harness_score']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic benchmark suite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all tasks")
    group.add_argument("--quick", action="store_true", help="Run a subset of tasks")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--telemetry", action="store_true", help="Collect telemetry to results/telemetry.jsonl")
    args = parser.parse_args()

    tasks = discover_tasks(TASKS_DIR)
    if args.quick:
        tasks = tasks[:2]

    results_dir = Path(__file__).parent / "results"
    collector = TelemetryCollector(results_dir) if args.telemetry else None

    summary = run_benchmark(tasks, collector=collector)

    if collector:
        out = collector.flush()
        print(f"Telemetry written to {out}")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human(summary)


if __name__ == "__main__":
    main()
