"""Synthetic benchmark suite for harness quality measurement.

Runs tasks against a real LLM agent (Copilot SDK) and evaluates responses
against expected outcomes. Falls back to dry-run (structural validation)
when --dry-run is passed or the Copilot SDK is unavailable.

Usage:
    python benchmark.py --all              # Run all tasks with agent
    python benchmark.py --quick            # Run subset with agent
    python benchmark.py --all --dry-run    # Validate structure only
    python benchmark.py --all --json       # JSON output
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

from agent_runner import AgentRunner, load_context_files
from checker import CheckResult, check_response, format_audit_entry
from observe import TelemetryCollector

TASKS_DIR = Path(__file__).parent / "tasks"
AUDIT_DIR = Path(__file__).parent / "results" / "audit"

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
    """Return placeholder metric values for dry-run mode."""
    return {
        "task_success_rate": 1.0,
        "quality_gate_pass_rate": 1.0,
        "rework_rate": 0.0,
        "avg_token_consumption": 0.0,
        "time_per_turn": 0.0,
    }


async def run_task_with_agent(
    runner: AgentRunner,
    task_data: dict,
    task_path: Path,
    collector: TelemetryCollector | None = None,
) -> dict:
    """Run a single task against the agent and evaluate the response."""
    rel = task_path.relative_to(TASKS_DIR)
    name = task_data.get("name", str(rel))
    prompt = task_data["input"]["prompt"]
    context_file_paths = task_data["input"].get("context_files", [])
    expected = task_data.get("expected_outcome", {})

    # Load context files
    context_texts = load_context_files(context_file_paths)

    # Run agent
    start = time.time()
    task_result = await runner.run_task(prompt, context_texts, timeout=120)
    end = time.time()

    # Evaluate response
    if task_result.error:
        check_result = CheckResult(mode="error")
        check_result.add_check("agent_execution", False, task_result.error)
        check_result.compute_score()
    else:
        check_result = check_response(task_result.response, expected)

    # Record telemetry
    if collector:
        with collector.task(str(rel)) as t:
            t.record_timing(start, end)
            t.record_tokens(input=task_result.input_tokens, output=task_result.output_tokens)
            t.status = "pass" if check_result.passed else "fail"
            if not check_result.passed:
                t.error_category = "quality_gate_fail"
            t.record_evaluation(
                response=task_result.response,
                expected=expected,
                checks=check_result.checks,
                mode=check_result.mode,
                score=check_result.score,
                prompt_hash=task_result.prompt_hash,
                harness_version=task_result.harness_version,
                model=task_result.model,
            )

    # Write audit file
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = AUDIT_DIR / f"{name}.md"
    audit_path.write_text(
        format_audit_entry(
            task_name=name,
            prompt=prompt,
            context_files=context_file_paths,
            response=task_result.response,
            check_result=check_result,
            harness_version=task_result.harness_version,
        ),
        encoding="utf-8",
    )

    return {
        "task": str(rel),
        "name": name,
        "status": "PASS" if check_result.passed else "FAIL",
        "score": check_result.score,
        "mode": check_result.mode,
        "checks": check_result.checks,
        "tokens": task_result.input_tokens + task_result.output_tokens,
        "duration_s": round(end - start, 3),
        "error": task_result.error,
    }


async def run_benchmark_with_agent(
    tasks: list[Path],
    model: str = "claude-sonnet-4-6",
    collector: TelemetryCollector | None = None,
) -> dict:
    """Run all tasks against a real agent and collect results."""
    results: list[dict] = []
    passed = 0
    failed = 0
    total_tokens = 0
    total_duration = 0.0

    async with AgentRunner(backend="copilot", model=model) as runner:
        for task_path in tasks:
            data, errors = validate_task(task_path)
            if errors:
                failed += 1
                results.append({"task": str(task_path.relative_to(TASKS_DIR)),
                                "status": "FAIL", "errors": errors, "score": 0.0})
                continue

            result = await run_task_with_agent(runner, data, task_path, collector)
            results.append(result)
            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1
            total_tokens += result.get("tokens", 0)
            total_duration += result.get("duration_s", 0.0)

    total = len(tasks)
    metrics = {
        "task_success_rate": passed / total if total else 0.0,
        "quality_gate_pass_rate": passed / total if total else 0.0,
        "rework_rate": failed / total if total else 0.0,
        "avg_token_consumption": total_tokens / total if total else 0.0,
        "time_per_turn": total_duration / total if total else 0.0,
    }
    composite = composite_score_local(metrics)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": results,
        "metrics": metrics,
        "harness_score": round(composite, 3),
        "mode": "agent",
    }


def run_benchmark(tasks: list[Path], collector: TelemetryCollector | None = None) -> dict:
    """Run all tasks in dry-run mode (structural validation only)."""
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
    mode = summary.get("mode", "dry-run")
    print(f"Benchmark ({mode}): {summary['passed']}/{summary['total']} tasks passed\n")
    for r in summary["results"]:
        status = r["status"]
        task = r["task"]
        if status == "FAIL":
            print(f"  FAIL  {task}")
            for err in r.get("errors", []):
                print(f"        {err}")
            # Show check details in agent mode
            for check in r.get("checks", []):
                if not check.get("passed"):
                    print(f"        [{check['name']}] {check.get('detail', '')}")
        else:
            score = r.get("score", "")
            score_str = f" (score: {score:.2f})" if isinstance(score, float) else ""
            print(f"  PASS  {task}{score_str}")

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
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate task structure only (no agent invocation)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Model for agent tasks (default: claude-sonnet-4-6)")
    parser.add_argument("--telemetry", action="store_true",
                        help="Collect telemetry to results/telemetry.jsonl")
    args = parser.parse_args()

    tasks = discover_tasks(TASKS_DIR)
    if args.quick:
        tasks = tasks[:3]

    results_dir = Path(__file__).parent / "results"
    collector = TelemetryCollector(results_dir) if args.telemetry else None

    if args.dry_run:
        summary = run_benchmark(tasks, collector=collector)
        summary["mode"] = "dry-run"
    else:
        try:
            summary = asyncio.run(
                run_benchmark_with_agent(tasks, model=args.model, collector=collector)
            )
        except RuntimeError as e:
            if "Copilot SDK" in str(e):
                print(f"Warning: {e}", file=sys.stderr)
                print("Falling back to dry-run mode.\n", file=sys.stderr)
                summary = run_benchmark(tasks, collector=collector)
                summary["mode"] = "dry-run (fallback)"
            else:
                raise

    if collector:
        out = collector.flush()
        print(f"Telemetry written to {out}", file=sys.stderr)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human(summary)


if __name__ == "__main__":
    main()
