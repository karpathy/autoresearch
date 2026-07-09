#!/usr/bin/env python3
"""Orchestrate a baseline vs ResearchFS agent comparison and emit a report."""

import argparse
import os
import re
import subprocess
import sys

from compare_runs import build_comparison_report

from researchfs.db import initialize_db
from seed import ensure_seed


def _run_agent(
    mode: str,
    model: str,
    max_experiments: int,
    time_budget: int,
    steps_budget: int,
    api_provider: str,
    search: bool,
    env: dict,
) -> int:
    """Run run_agent.py in the given mode and return the agent_run_id."""
    cmd = [
        "uv", "run", "python", "run_agent.py",
        "--mode", mode,
        "--model", model,
        "--max-experiments", str(max_experiments),
        "--time-budget", str(time_budget),
        "--steps-budget", str(steps_budget),
        "--api-provider", api_provider,
    ]
    if search:
        cmd.append("--search")

    print(f"\n{'='*60}")
    print(f"Starting {mode} agent run")
    print(f"{'='*60}")

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    # Echo captured output after the child exits; capture is needed to parse agent_run_id.
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)

    if res.returncode != 0:
        detail = (res.stderr or res.stdout[-4000:]).strip()
        raise RuntimeError(
            f"{mode} agent run failed with exit code {res.returncode}"
            + (f": {detail}" if detail else "")
        )

    # Parse agent_run_id from the summary line.
    match = re.search(r"agent_run_id:\s*(\d+)", res.stdout)
    if not match:
        raise RuntimeError(f"Could not parse agent_run_id from {mode} run output")

    return int(match.group(1))


def _current_branch(env: dict) -> str:
    """Return a checkout target for the current branch, or commit when detached."""
    res = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    branch = res.stdout.strip()
    if branch != "HEAD":
        return branch
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return res.stdout.strip()


def _checkout(branch: str, env: dict) -> None:
    """Checkout a git branch and raise if the checkout cannot be completed."""
    res = subprocess.run(
        ["git", "checkout", branch],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        if detail:
            raise RuntimeError(f"Could not checkout {branch}: {detail}")
        raise RuntimeError(f"Could not checkout {branch} (git exited {res.returncode})")


def _prepare_data(env: dict) -> None:
    """Run prepare.py if the cached data directory is missing."""
    from prepare import DATA_DIR

    if os.path.isdir(DATA_DIR):
        return
    print(f"Data directory not found at {DATA_DIR}. Running prepare.py...")
    res = subprocess.run(
        ["uv", "run", "python", "prepare.py"],
        env=env,
    )
    if res.returncode != 0:
        raise RuntimeError("Data preparation failed")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline and ResearchFS agents back-to-back and compare."
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM model name"
    )
    parser.add_argument(
        "--max-experiments", type=int, default=5, help="Experiments per agent run"
    )
    parser.add_argument(
        "--time-budget", type=int, default=300, help="Training budget per experiment in seconds"
    )
    parser.add_argument(
        "--steps-budget", type=int, default=0, help="Training budget per experiment in steps"
    )
    parser.add_argument(
        "--api-provider", choices=["openai", "openrouter", "codex"], default="codex",
        help="API provider to use"
    )
    parser.add_argument(
        "--search", action="store_true", help="Enable web search for both agents"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick test mode: 60s budget, 2 experiments, faster model"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="Run prepare.py if data is missing"
    )
    parser.add_argument(
        "--output", type=str, help="Write the comparison report to this file"
    )

    args = parser.parse_args()

    model = args.model
    max_experiments = args.max_experiments
    time_budget = args.time_budget
    steps_budget = args.steps_budget

    if args.fast:
        model = "gpt-4o-mini"
        max_experiments = 2
        time_budget = 60
        steps_budget = 50
        print(f"Fast mode enabled: model={model}, max-experiments={max_experiments}, time-budget={time_budget}, steps-budget={steps_budget}")

    env = os.environ.copy()
    if "GIT_AUTHOR_NAME" not in env:
        env["GIT_AUTHOR_NAME"] = "Autoresearch Agent"
    if "GIT_AUTHOR_EMAIL" not in env:
        env["GIT_AUTHOR_EMAIL"] = "agent@autoresearch.local"
    if "GIT_COMMITTER_NAME" not in env:
        env["GIT_COMMITTER_NAME"] = "Autoresearch Agent"
    if "GIT_COMMITTER_EMAIL" not in env:
        env["GIT_COMMITTER_EMAIL"] = "agent@autoresearch.local"

    # Ensure ResearchFS is ready.
    initialize_db()
    ensure_seed()

    if args.prepare:
        _prepare_data(env)

    original_branch = _current_branch(env)
    print(f"Original checkout target: {original_branch}")

    baseline_run_id = None
    loopgraph_run_id = None

    try:
        baseline_run_id = _run_agent(
            "baseline", model, max_experiments, time_budget, steps_budget, args.api_provider, args.search, env
        )
        _checkout(original_branch, env)
        loopgraph_run_id = _run_agent(
            "loopgraph", model, max_experiments, time_budget, steps_budget, args.api_provider, args.search, env
        )
    finally:
        active_exception = sys.exc_info()[1]
        try:
            _checkout(original_branch, env)
        except RuntimeError as cleanup_error:
            if active_exception is None:
                raise
            if hasattr(active_exception, "add_note"):
                active_exception.add_note(f"Cleanup checkout failed: {cleanup_error}")
            else:
                print(f"Cleanup checkout failed: {cleanup_error}", file=sys.stderr)

    # Generate comparison report.
    print(f"\n{'='*60}")
    print("Generating comparison report")
    print(f"{'='*60}")

    report = build_comparison_report(baseline_run_id, loopgraph_run_id)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
