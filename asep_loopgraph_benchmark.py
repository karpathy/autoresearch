#!/usr/bin/env python3
"""ASEP-compatible benchmark wrapper for LoopGraph/ResearchFS.

Runs a LoopGraph agent sweep for a single retrieval mode and emits ASEP-style
JSON metrics. Designed to be called from an ASEP ``program.md`` experiment
command so that ASEP can compare retrieval strategies (baseline vs loopgraph)
as categorical parameters.

Example ASEP command:
    {python_bin} asep_loopgraph_benchmark.py
        --mode {mode}
        --metric val_bpb
        --max-experiments 2
        --steps-budget 50
        --time-budget 60
        --api-provider openrouter
        --model z-ai/glm-4.5-air:free
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the LoopGraph repo root is on sys.path so we can import its modules.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import requests

from agents.harness_agent import run_agent_mode
from evals.nanochat_bpb.metrics import load_run
from run_agent import call_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASEP benchmark wrapper for LoopGraph agent runs"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "loopgraph"],
        required=True,
        help="LoopGraph agent retrieval mode",
    )
    parser.add_argument(
        "--metric",
        default="val_bpb",
        choices=["val_bpb", "tokens", "time_seconds", "cost"],
        help="Primary metric to emit for ASEP optimization",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=2,
        help="Number of LoopGraph propose-train-evaluate iterations",
    )
    parser.add_argument(
        "--steps-budget",
        type=int,
        default=0,
        help="Training steps budget per experiment (0 = use time budget)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=30,
        help="Training time budget per experiment in seconds",
    )
    parser.add_argument(
        "--api-provider",
        choices=["openai", "openrouter", "venice"],
        default="openai",
        help="LLM API provider",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model name",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="PyTorch device passed to train.py",
    )
    parser.add_argument(
        "--max-method-chars",
        type=int,
        default=None,
        help="Maximum characters for the retrieved method pack (default: no limit)",
    )
    return parser.parse_args()


def set_fast_train_env(steps_budget: int, time_budget: int, device: str) -> None:
    """Configure environment variables for cheap, fast LoopGraph runs."""
    os.environ.setdefault("DEVICE", device)
    os.environ.setdefault("MAX_SEQ_LEN", "512")
    os.environ.setdefault("DEPTH", "4")
    os.environ.setdefault("DEVICE_BATCH_SIZE", "16")
    os.environ.setdefault("TOTAL_BATCH_SIZE", "65536")
    os.environ.setdefault("EVAL_TOKENS", "524288")
    os.environ.setdefault("TIME_BUDGET", str(time_budget))
    os.environ.setdefault("STEPS_BUDGET", str(steps_budget))


def current_git_branch() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    if branch != "HEAD":
        return branch
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def autoresearch_branches() -> set[str]:
    """Return local autoresearch experiment branch names."""
    result = subprocess.run(
        ["git", "branch", "--list", "autoresearch/exp_*"],
        capture_output=True,
        text=True,
        check=False,
    )
    branches = set()
    for branch in result.stdout.splitlines():
        clean = branch.strip().lstrip("* ").strip()
        if clean:
            branches.add(clean)
    return branches


def _temporary_branch_from_this_run(
    current_branch: str | None,
    original_branch: str | None,
    branches_before: set[str],
) -> str | None:
    if (
        current_branch
        and current_branch != original_branch
        and current_branch.startswith("autoresearch/exp_")
        and current_branch not in branches_before
    ):
        return current_branch
    return None


def cleanup_git_state(original_branch: str | None, branches_before: set[str]) -> None:
    """Return to the original checkout target and only delete a known temporary branch."""
    current_branch = current_git_branch()
    temporary_branch = _temporary_branch_from_this_run(
        current_branch,
        original_branch,
        branches_before,
    )

    checkout_succeeded = False
    if original_branch and original_branch != "HEAD":
        if current_branch == original_branch:
            checkout_succeeded = True
        else:
            result = subprocess.run(
                ["git", "checkout", original_branch],
                check=False,
                capture_output=True,
                text=True,
            )
            checkout_succeeded = result.returncode == 0
            if not checkout_succeeded:
                detail = (result.stderr or result.stdout).strip()
                print(
                    f"Warning: could not checkout original target {original_branch}; "
                    f"leaving current branch unchanged. {detail}",
                    file=sys.stderr,
                )

    if temporary_branch and checkout_succeeded:
        result = subprocess.run(
            ["git", "branch", "-D", temporary_branch],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            print(
                f"Warning: could not delete temporary branch {temporary_branch}. {detail}",
                file=sys.stderr,
            )
    elif temporary_branch:
        print(
            f"Warning: leaving temporary branch {temporary_branch} in place because checkout cleanup did not complete.",
            file=sys.stderr,
        )


def build_llm_fn(provider: str, model: str) -> Any:
    def _llm(prompt: str) -> str:
        max_retries = 5
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = call_llm(
                    provider,
                    model,
                    [{"role": "user", "content": prompt}],
                )
                return response["content"]
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited (429); retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                raise

    return _llm


def main() -> int:
    args = parse_args()
    set_fast_train_env(args.steps_budget, args.time_budget, args.device)

    original_branch = current_git_branch()
    branches_before = autoresearch_branches()
    start_time = time.monotonic()

    try:
        llm_fn = build_llm_fn(args.api_provider, args.model)
        agent_run_id = run_agent_mode(
            mode=args.mode,
            model=args.model,
            max_experiments=args.max_experiments,
            time_budget=args.time_budget,
            llm_fn=llm_fn,
            steps_budget=args.steps_budget,
            max_method_chars=args.max_method_chars,
        )
        summary = load_run(agent_run_id)
    finally:
        cleanup_git_state(original_branch, branches_before)

    elapsed = time.monotonic() - start_time

    best_bpb = summary.get("best_bpb")
    total_tokens = summary.get("total_tokens", 0)
    cost = summary.get("cost", 0.0)

    # ASEP minimizes the primary metric. For tokens/time/cost, lower is better.
    # For val_bpb, lower is also better.
    metric_value: float | None
    if args.metric == "val_bpb":
        metric_value = best_bpb
    elif args.metric == "tokens":
        metric_value = float(total_tokens)
    elif args.metric == "time_seconds":
        metric_value = elapsed
    elif args.metric == "cost":
        metric_value = cost
    else:
        metric_value = None

    # ASEP expects the metric field to be present and numeric. If the run
    # produced no valid BPB, emit a large finite penalty so JSON consumers do
    # not receive null/Infinity and ASEP treats the run as a failure.
    if metric_value is None:
        metric_value = 1e9

    payload = {
        "mode": args.mode,
        "agent_run_id": agent_run_id,
        "best_bpb": best_bpb,
        "total_tokens": total_tokens,
        "cost": cost,
        "time_seconds": round(elapsed, 3),
        args.metric: round(metric_value, 6),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
