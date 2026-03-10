#!/usr/bin/env python3
"""
Experiment runner for autoresearch.
Wraps `uv run train.py` with environment setup, result parsing, and knowledge base logging.

Usage:
    python run_experiment.py --scale quick --description "try GLU activation"
    python run_experiment.py --scale standard --description "baseline" --agent-id explorer-0
    python run_experiment.py --scale long --resume-from checkpoints/abc1234_standard.pt
    python run_experiment.py --briefing  # print research briefing and exit
"""

import argparse
import os
import re
import subprocess
import sys
import time

from config import TIME_BUDGETS, get_results_dir
from knowledge import (
    ExperimentRecord,
    LessonRecord,
    append_experiment,
    append_lesson,
    build_research_briefing,
    sync_to_legacy_tsv,
)


def get_git_info():
    """Get current branch and short commit hash."""
    branch = "unknown"
    commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    return branch, commit


def parse_results(log_path: str) -> dict:
    """Parse the training output summary from run.log."""
    results = {}
    patterns = {
        "val_bpb": r"^val_bpb:\s+([\d.]+)",
        "peak_vram_mb": r"^peak_vram_mb:\s+([\d.]+)",
        "mfu_percent": r"^mfu_percent:\s+([\d.]+)",
        "num_params_M": r"^num_params_M:\s+([\d.]+)",
        "depth": r"^depth:\s+(\d+)",
        "total_batch_size": r"^total_batch_size:\s+(\d+)",
        "matrix_lr": r"^matrix_lr:\s+([\d.]+)",
        "training_seconds": r"^training_seconds:\s+([\d.]+)",
        "num_steps": r"^num_steps:\s+(\d+)",
        "loss_trajectory": r"^loss_trajectory:\s+(.+)",
    }

    if not os.path.exists(log_path):
        return results

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            for key, pattern in patterns.items():
                m = re.match(pattern, line)
                if m:
                    val = m.group(1)
                    if key in ("depth", "total_batch_size", "num_steps"):
                        results[key] = int(val)
                    elif key == "loss_trajectory":
                        results[key] = val
                    else:
                        results[key] = float(val)
    return results


def get_log_tail(log_path: str, n: int = 50) -> str:
    """Get last n lines of log file for crash diagnosis."""
    if not os.path.exists(log_path):
        return "(no log file)"
    with open(log_path, "r") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def run_experiment(
    scale: str,
    description: str,
    agent_id: str = "solo-0",
    agent_role: str = "solo",
    resume_from: str = None,
    save_checkpoint: bool = False,
    use_queue: bool = False,
    timeout_multiplier: float = 2.5,
) -> ExperimentRecord:
    """Run a training experiment and log results to the knowledge base."""

    time_budget = TIME_BUDGETS.get(scale, 300)
    branch, commit = get_git_info()
    log_path = "run.log"

    # Set environment
    env = os.environ.copy()
    env["AR_TIME_BUDGET"] = str(time_budget)
    env["AR_SCALE"] = scale
    if save_checkpoint or scale in ("long", "deep"):
        env["AR_SAVE_CHECKPOINT"] = "1"
    if resume_from:
        env["AR_RESUME_CHECKPOINT"] = resume_from

    # Build command
    if use_queue:
        # Submit to GPU queue and wait
        cmd = ["python", "gpu_queue.py", "submit",
               "--agent-id", agent_id,
               "--scale", scale,
               "--command", "uv run train.py"]
    else:
        cmd = ["uv", "run", "train.py"]

    timeout = int(time_budget * timeout_multiplier) + 60  # generous timeout

    print(f"[{agent_id}] Running experiment: {description}")
    print(f"[{agent_id}] Scale: {scale} ({time_budget}s), branch: {branch}, commit: {commit}")

    crashed = False
    exit_code = 0

    try:
        if use_queue:
            # Queue mode: submit and wait
            result = subprocess.run(
                cmd, env=env, timeout=timeout,
                capture_output=True, text=True, cwd=os.getcwd()
            )
            exit_code = result.returncode
            # The queue worker writes to run.log in the worktree
        else:
            # Direct mode: redirect all output to run.log
            with open(log_path, "w") as logf:
                result = subprocess.run(
                    cmd, env=env, timeout=timeout,
                    stdout=logf, stderr=subprocess.STDOUT, cwd=os.getcwd()
                )
                exit_code = result.returncode
    except subprocess.TimeoutExpired:
        print(f"[{agent_id}] TIMEOUT after {timeout}s")
        crashed = True
        exit_code = -1
    except Exception as e:
        print(f"[{agent_id}] ERROR: {e}")
        crashed = True
        exit_code = -1

    # Parse results
    parsed = parse_results(log_path)

    if not parsed.get("val_bpb") or exit_code != 0:
        crashed = True

    # Build record
    record = ExperimentRecord(
        agent_role=agent_role,
        agent_id=agent_id,
        branch=branch,
        commit=commit,
        scale=scale,
        time_budget=time_budget,
        val_bpb=parsed.get("val_bpb", 0.0),
        peak_vram_mb=parsed.get("peak_vram_mb", 0.0),
        mfu_percent=parsed.get("mfu_percent", 0.0),
        num_params_M=parsed.get("num_params_M", 0.0),
        depth=parsed.get("depth", 0),
        total_batch_size=parsed.get("total_batch_size", 0),
        matrix_lr=parsed.get("matrix_lr", 0.0),
        loss_trajectory=parsed.get("loss_trajectory", ""),
        status="crash" if crashed else "pending",  # agent decides keep/discard
        description=description,
        parent_commit=commit,
        escalated_from=resume_from,
    )

    # Log to knowledge base
    append_experiment(record)
    sync_to_legacy_tsv()

    # Print summary
    if crashed:
        print(f"\n[{agent_id}] CRASHED (exit code {exit_code})")
        tail = get_log_tail(log_path)
        print(f"[{agent_id}] Last 50 lines of log:\n{tail}")
    else:
        print(f"\n[{agent_id}] RESULT:")
        print(f"  val_bpb:      {record.val_bpb:.6f}")
        print(f"  peak_vram_mb: {record.peak_vram_mb:.1f}")
        print(f"  mfu_percent:  {record.mfu_percent:.2f}")
        print(f"  num_params_M: {record.num_params_M:.1f}")
        print(f"  depth:        {record.depth}")
        print(f"  scale:        {record.scale} ({record.time_budget}s)")

    return record


def main():
    parser = argparse.ArgumentParser(description="Run an autoresearch experiment")
    parser.add_argument("--scale", default="standard", choices=TIME_BUDGETS.keys(),
                        help="Experiment time scale")
    parser.add_argument("--description", default="experiment", help="What this experiment tries")
    parser.add_argument("--agent-id", default="solo-0", help="Agent identifier")
    parser.add_argument("--agent-role", default="solo", help="Agent role")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save checkpoint after training")
    parser.add_argument("--use-queue", action="store_true", help="Submit to GPU queue instead of running directly")
    parser.add_argument("--briefing", action="store_true", help="Print research briefing and exit")
    parser.add_argument("--lesson", nargs=3, metavar=("CATEGORY", "CONFIDENCE", "TEXT"),
                        help="Log a lesson: category confidence text")

    args = parser.parse_args()

    if args.briefing:
        print(build_research_briefing())
        return

    if args.lesson:
        category, confidence, text = args.lesson
        lesson = LessonRecord(
            agent_role=args.agent_role,
            agent_id=args.agent_id,
            category=category,
            lesson=text,
            confidence=confidence,
        )
        append_lesson(lesson)
        print(f"Lesson logged: [{confidence}] {category}: {text}")
        return

    record = run_experiment(
        scale=args.scale,
        description=args.description,
        agent_id=args.agent_id,
        agent_role=args.agent_role,
        resume_from=args.resume_from,
        save_checkpoint=args.save_checkpoint,
        use_queue=args.use_queue,
    )

    sys.exit(0 if record.status != "crash" else 1)


if __name__ == "__main__":
    main()
