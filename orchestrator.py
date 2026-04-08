"""
Autoresearch Orchestrator — drives autonomous experiment loops via local LLM.

Calls the local Nemotron model (via vLLM) to propose train.py modifications,
runs experiments, tracks results, and keeps/discards changes via git.

Usage:
    Inside crsai-pytorch container:
        AUTORESEARCH_PROFILE=rtx5060 python3 orchestrator.py --tag apr8

    Or from host via docker exec:
        docker exec -w /workspace/autoresearch crsai-pytorch \
            env AUTORESEARCH_PROFILE=rtx5060 python3 orchestrator.py --tag apr8
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_URL = os.getenv("LLM_URL", "http://crsai-vllm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "primary")
MAX_EXPERIMENT_SECONDS = 600  # kill if > 10 min
TRAIN_CMD = "python3 train.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: str, timeout: int | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a shell command, return CompletedProcess."""
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=cwd,
    )


def git_short_hash() -> str:
    r = run("git rev-parse --short HEAD")
    return r.stdout.strip()


def git_commit(msg: str) -> str:
    run("git add train.py")
    run(f'git commit -m "{msg}"')
    return git_short_hash()


def git_reset_hard(ref: str = "HEAD~1") -> None:
    run(f"git reset --hard {ref}")


def read_train_py() -> str:
    return Path("train.py").read_text()


def write_train_py(content: str) -> None:
    Path("train.py").write_text(content)


def init_results_tsv() -> None:
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def append_result(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    line = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(line)


def read_results() -> str:
    if Path(RESULTS_FILE).exists():
        return Path(RESULTS_FILE).read_text()
    return ""


def parse_run_log(log_path: str = RUN_LOG) -> dict:
    """Extract val_bpb and peak_vram_mb from run.log."""
    result = {"val_bpb": None, "peak_vram_mb": None, "crashed": False}
    try:
        text = Path(log_path).read_text()
    except FileNotFoundError:
        result["crashed"] = True
        return result

    # Check for FAIL or crash
    if "FAIL" in text or "Error" in text or "Traceback" in text:
        result["crashed"] = True

    for line in text.splitlines():
        if line.startswith("val_bpb:"):
            result["val_bpb"] = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            result["peak_vram_mb"] = float(line.split(":")[1].strip())

    if result["val_bpb"] is None:
        result["crashed"] = True

    return result


def best_val_bpb() -> float | None:
    """Return best val_bpb from results.tsv (keeps only)."""
    results = read_results()
    best = None
    for line in results.strip().splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            bpb = float(parts[1])
            if best is None or bpb < best:
                best = bpb
    return best


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def call_llm(system: str, user: str, max_tokens: int = 4096) -> str:
    """Call local vLLM endpoint. Uses requests (already installed)."""
    import requests

    resp = requests.post(
        f"{LLM_URL}/chat/completions",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


SYSTEM_PROMPT = """You are an expert ML researcher running autonomous training experiments.
You modify train.py to improve val_bpb (validation bits per byte — lower is better).
The training runs for exactly 5 minutes. You can change anything in train.py:
architecture, optimizer, hyperparameters, batch size, model size, etc.

CONSTRAINTS:
- Do NOT modify prepare.py or add packages
- Keep changes focused — one idea per experiment
- Simpler is better at equal performance
- VRAM is limited (~5-8 GB available)
- The AUTORESEARCH_PROFILE override block near line 440 MUST be preserved

RESPONSE FORMAT:
Return ONLY a JSON object with these keys:
{
  "description": "brief description of what this experiment tries",
  "replacements": [
    {"old": "exact lines to find in train.py", "new": "replacement lines"}
  ]
}

Each replacement is a search-and-replace pair. Use the EXACT text from train.py
including whitespace. Keep replacements minimal — only the lines that change.
Do NOT include markdown code fences. Return raw JSON only."""


def apply_replacements(train_py: str, replacements: list[dict]) -> str:
    """Apply a list of {old, new} replacements to train.py content."""
    result = train_py
    for r in replacements:
        old = r.get("old", "")
        new = r.get("new", "")
        if old not in result:
            raise ValueError(f"Replacement target not found in train.py: {old[:80]!r}")
        result = result.replace(old, new, 1)
    return result


def propose_experiment(train_py: str, results_history: str) -> dict:
    """Ask the LLM to propose a train.py modification."""
    user_msg = f"""Current train.py:
```python
{train_py}
```

Experiment history (TSV):
```
{results_history}
```

Current best val_bpb: {best_val_bpb() or 'no results yet — this is the first run'}

Propose a single focused modification to improve val_bpb.
Return search-and-replace pairs (exact text from the file) and a description."""

    response = call_llm(SYSTEM_PROMPT, user_msg, max_tokens=4096)

    # Strip thinking tags (reasoning model outputs <think>...</think> before JSON)
    response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
    # If <think> started but never closed, strip everything before the first {
    if "<think>" in response:
        idx = response.find("{")
        if idx >= 0:
            response = response[idx:]

    # Strip markdown fences if present
    response = re.sub(r"^```(?:json)?\n?", "", response.strip())
    response = re.sub(r"\n?```$", "", response.strip())

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse LLM response as JSON:\n{response[:500]}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Run train.py and return parsed results."""
    print(f"  Running training ({MAX_EXPERIMENT_SECONDS}s timeout)...", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.run(
                TRAIN_CMD.split(),
                stdout=log_file, stderr=subprocess.STDOUT,
                timeout=MAX_EXPERIMENT_SECONDS,
                env=env,
            )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT — experiment killed", flush=True)
        return {"val_bpb": None, "peak_vram_mb": None, "crashed": True}

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.0f}s", flush=True)

    return parse_run_log()


def main():
    parser = argparse.ArgumentParser(description="Autoresearch orchestrator")
    parser.add_argument("--tag", required=True, help="Experiment run tag (e.g. apr8)")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Max experiments to run (0 = infinite)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the baseline, then exit")
    args = parser.parse_args()

    branch = f"autoresearch/{args.tag}"
    print(f"=== Autoresearch Orchestrator ===", flush=True)
    print(f"Branch: {branch}", flush=True)
    print(f"LLM: {LLM_URL} / {LLM_MODEL}", flush=True)
    print(f"Profile: {os.getenv('AUTORESEARCH_PROFILE', 'default')}", flush=True)
    print(flush=True)

    # Setup branch
    r = run(f"git rev-parse --verify {branch}")
    if r.returncode != 0:
        run(f"git checkout -b {branch}")
        print(f"Created branch: {branch}", flush=True)
    else:
        run(f"git checkout {branch}")
        print(f"Checked out existing branch: {branch}", flush=True)

    init_results_tsv()

    # Baseline run
    if best_val_bpb() is None:
        print("\n--- Baseline Run ---", flush=True)
        result = run_experiment()
        commit = git_short_hash()
        if result["crashed"]:
            print("  BASELINE CRASHED — check run.log", flush=True)
            append_result(commit, 0.0, 0.0, "crash", "baseline")
            sys.exit(1)

        bpb = result["val_bpb"]
        mem = result["peak_vram_mb"] / 1024 if result["peak_vram_mb"] else 0
        append_result(commit, bpb, mem, "keep", "baseline")
        print(f"  Baseline: val_bpb={bpb:.6f}, memory={mem:.1f}GB", flush=True)

    if args.baseline_only:
        print("\nBaseline complete. Exiting.", flush=True)
        return

    # Experiment loop
    experiment_num = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    while True:
        experiment_num += 1
        if args.max_experiments and experiment_num > args.max_experiments:
            print(f"\nReached max experiments ({args.max_experiments}). Stopping.", flush=True)
            break

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n{MAX_CONSECUTIVE_FAILURES} consecutive LLM failures. Stopping.", flush=True)
            break

        current_best = best_val_bpb()
        print(f"\n--- Experiment {experiment_num} (best so far: {current_best:.6f}) ---", flush=True)

        # Get current state
        train_py = read_train_py()
        results_history = read_results()

        # Ask LLM for proposal
        try:
            print("  Querying LLM for proposal...", flush=True)
            proposal = propose_experiment(train_py, results_history)
            description = proposal.get("description", "unknown modification")
            replacements = proposal.get("replacements", [])

            if not replacements:
                print("  LLM returned no replacements, skipping", flush=True)
                continue

            print(f"  Proposal: {description}", flush=True)
            print(f"  Replacements: {len(replacements)}", flush=True)
        except Exception as e:
            print(f"  LLM error: {e}", flush=True)
            print("  Retrying in 30s...", flush=True)
            consecutive_failures += 1
            time.sleep(30)
            continue

        # Apply modification
        try:
            new_train_py = apply_replacements(train_py, replacements)
            write_train_py(new_train_py)
        except ValueError as e:
            print(f"  Replacement failed: {e}", flush=True)
            consecutive_failures += 1
            continue

        consecutive_failures = 0  # reset on successful proposal + apply
        commit = git_commit(f"exp{experiment_num}: {description[:60]}")

        # Run experiment
        result = run_experiment()

        if result["crashed"]:
            mem = 0.0
            append_result(commit, 0.0, mem, "crash", description)
            print(f"  CRASHED — reverting", flush=True)
            git_reset_hard()
            continue

        bpb = result["val_bpb"]
        mem = result["peak_vram_mb"] / 1024 if result["peak_vram_mb"] else 0

        if bpb < current_best:
            append_result(commit, bpb, mem, "keep", description)
            improvement = current_best - bpb
            print(f"  KEEP — val_bpb={bpb:.6f} (improved by {improvement:.6f})", flush=True)
        else:
            append_result(commit, bpb, mem, "discard", description)
            regression = bpb - current_best
            print(f"  DISCARD — val_bpb={bpb:.6f} (worse by {regression:.6f})", flush=True)
            git_reset_hard()

    # Summary
    print(f"\n=== Summary ===", flush=True)
    print(f"Total experiments: {experiment_num}", flush=True)
    print(f"Best val_bpb: {best_val_bpb():.6f}", flush=True)
    print(f"Results: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
