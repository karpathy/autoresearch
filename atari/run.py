"""
Autonomous Atari autoresearch runner.

Runs the full experiment loop: asks an LLM to modify agent.py,
evaluates the result, keeps improvements, discards failures, repeats.

Usage:
    # Default: uses ANTHROPIC_AUTH_TOKEN env var + Scale litellm proxy
    uv run run.py

    # Explicit key
    uv run run.py --api-key <token>

    # Direct Anthropic API (no proxy)
    uv run run.py --api-key sk-ant-... --api-base https://api.anthropic.com

    # Customize
    uv run run.py --model anthropic/claude-opus-4-6 --max-experiments 50
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ATARI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ATARI_DIR)
AGENT_FILE = os.path.join(ATARI_DIR, "agent.py")
PREPARE_FILE = os.path.join(ATARI_DIR, "prepare.py")
RESULTS_FILE = os.path.join(ATARI_DIR, "results.tsv")
RUN_LOG = os.path.join(ATARI_DIR, "run.log")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args):
    """Run a git command in the repo directory, return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_short_hash():
    return git("rev-parse", "--short", "HEAD")


def git_commit(message):
    git("add", AGENT_FILE)
    git("commit", "-m", message)
    return git_short_hash()


def git_reset_hard(commit):
    git("reset", "--hard", commit)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    """Run agent.py and return (mean_reward, raw_output) or (None, error_output) on crash."""
    result = subprocess.run(
        [sys.executable, "agent.py"],
        cwd=ATARI_DIR, capture_output=True, text=True, timeout=600,
    )

    output = result.stdout + result.stderr

    # Write run.log
    with open(RUN_LOG, "w") as f:
        f.write(output)

    if result.returncode != 0:
        return None, output

    match = re.search(r"^mean_reward:\s+([\d.]+)", output, re.MULTILINE)
    if not match:
        return None, output

    return float(match.group(1)), output


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tmean_reward\tstatus\tdescription\n")


def log_result(commit, mean_reward, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{mean_reward:.4f}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous AI researcher optimizing an Atari Breakout agent.

Your goal: maximize mean_reward on a fixed 30-episode evaluation.

You modify agent.py — the only file you can edit. The evaluation harness \
(prepare.py) is fixed and cannot be changed.

Constraints:
- Only use imports already available: numpy, time, and anything from prepare.py
- The Agent class must implement act(obs) -> int and reset() -> None
- act() receives a (210, 160, 3) uint8 RGB observation
- Breakout actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
- The train() function can use up to 300s for learning (optional)

When asked to suggest a modification, respond with ONLY the complete new \
agent.py file contents between ```python and ``` markers. Include a one-line \
comment at the top of the Agent class describing your change. No other text."""


def build_prompt(agent_code, results_history, best_reward, attempt_num):
    """Build the prompt for the LLM."""
    prompt = f"""Here is the current agent.py (best so far, mean_reward={best_reward:.4f}):

```python
{agent_code}
```

Experiment history (most recent last):
{results_history if results_history else "(no experiments yet — this will be the first modification)"}

This is experiment #{attempt_num}. Suggest a modification to agent.py that will \
improve mean_reward. Think about what has worked and what hasn't based on the \
history. Try something different from previous failed attempts.

Respond with the complete new agent.py file between ```python and ``` markers."""
    return prompt


def call_llm(api_key, model, prompt, api_base=None):
    """Call the LLM via litellm and return the response text."""
    from litellm import completion

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
        "api_key": api_key,
        "api_base": api_base,
    }

    response = completion(**kwargs)
    return response.choices[0].message.content


def extract_code(response):
    """Extract Python code from markdown code block in LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try without language tag
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Atari autoresearch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uses ANTHROPIC_AUTH_TOKEN env var + Scale litellm proxy by default
  uv run run.py

  # Explicit key
  uv run run.py --api-key <token>

  # Direct Anthropic API (no proxy)
  uv run run.py --api-key sk-ant-... --api-base https://api.anthropic.com

  # Custom model
  uv run run.py --model anthropic/claude-opus-4-6
        """,
    )
    parser.add_argument("--api-key",
                        default=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("LITELLM_API_KEY"),
                        help="API key (or set ANTHROPIC_AUTH_TOKEN / LITELLM_API_KEY env var)")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514",
                        help="LiteLLM model name (default: anthropic/claude-sonnet-4-20250514)")
    parser.add_argument("--api-base",
                        default=os.environ.get("LITELLM_API_BASE", "https://litellm-proxy.ml-serving-internal.scale.com"),
                        help="API base URL (default: Scale litellm proxy, or set LITELLM_API_BASE env var)")
    parser.add_argument("--tag", default=None,
                        help="Branch tag (default: today's date, e.g. apr4)")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Max experiments to run (0 = unlimited)")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip git operations (useful for testing)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key required (or set ANTHROPIC_AUTH_TOKEN env var)")
        sys.exit(1)

    # --- Setup ---
    tag = args.tag or datetime.now().strftime("%b%d").lower()
    branch = f"atari-research/{tag}"

    print("=" * 60)
    print("  ATARI AUTORESEARCH")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Branch:  {branch}")
    print(f"  Limit:   {'unlimited' if args.max_experiments == 0 else args.max_experiments}")
    print("=" * 60)
    print()

    if not args.no_git:
        git("checkout", "-b", branch)

    init_results()

    # --- Baseline ---
    print("[baseline] Running baseline evaluation...")
    with open(AGENT_FILE) as f:
        best_code = f.read()

    mean_reward, output = run_evaluation()
    if mean_reward is None:
        print("[baseline] CRASHED! Fix agent.py and retry.")
        print(output[-500:])
        sys.exit(1)

    best_reward = mean_reward
    baseline_commit = git_short_hash()
    log_result(baseline_commit, best_reward, "keep", "baseline heuristic")
    print(f"[baseline] mean_reward={best_reward:.4f}")
    print()

    # --- Experiment loop ---
    experiment = 0
    kept = 0
    discarded = 0

    while True:
        experiment += 1
        if args.max_experiments > 0 and experiment > args.max_experiments:
            break

        print(f"{'='*60}")
        print(f"  EXPERIMENT {experiment}")
        print(f"  Best so far: {best_reward:.4f} | Kept: {kept} | Discarded: {discarded}")
        print(f"{'='*60}")

        # Read current results history
        results_history = ""
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                results_history = f.read()

        # Ask LLM for a modification
        print("[llm] Asking for modification...")
        prompt = build_prompt(best_code, results_history, best_reward, experiment)

        try:
            response = call_llm(args.api_key, args.model, prompt, args.api_base)
        except Exception as e:
            print(f"[llm] Error: {e}")
            print("[llm] Retrying in 10s...")
            time.sleep(10)
            continue

        new_code = extract_code(response)
        if not new_code:
            print("[llm] Failed to extract code from response. Skipping.")
            continue

        # Extract description from the agent class docstring or first comment
        desc_match = re.search(r"#\s*(.+)", new_code.split("class Agent")[0][-200:]) if "class Agent" in new_code else None
        description = desc_match.group(1).strip() if desc_match else f"experiment {experiment}"
        description = description[:80]

        # Write new agent.py
        with open(AGENT_FILE, "w") as f:
            f.write(new_code)

        # Commit
        pre_commit = git_short_hash()
        if not args.no_git:
            commit_hash = git_commit(f"try: {description}")
        else:
            commit_hash = "nogit"

        # Evaluate
        print(f"[eval] Running: {description}")
        try:
            mean_reward, output = run_evaluation()
        except subprocess.TimeoutExpired:
            mean_reward = None
            output = "TIMEOUT: evaluation exceeded 10 minutes"

        if mean_reward is None:
            # Crash
            print(f"[eval] CRASHED")
            error_tail = output[-300:] if output else "no output"
            print(f"  {error_tail}")
            log_result(commit_hash, 0.0, "crash", description)
            if not args.no_git:
                git_reset_hard(pre_commit)
            else:
                with open(AGENT_FILE, "w") as f:
                    f.write(best_code)
            discarded += 1

        elif mean_reward > best_reward:
            # Improvement!
            print(f"[eval] IMPROVED: {best_reward:.4f} -> {mean_reward:.4f} (+{mean_reward - best_reward:.4f})")
            log_result(commit_hash, mean_reward, "keep", description)
            best_reward = mean_reward
            best_code = new_code
            kept += 1

        else:
            # No improvement
            print(f"[eval] No improvement: {mean_reward:.4f} <= {best_reward:.4f}")
            log_result(commit_hash, mean_reward, "discard", description)
            if not args.no_git:
                git_reset_hard(pre_commit)
            else:
                with open(AGENT_FILE, "w") as f:
                    f.write(best_code)
            discarded += 1

        print()

    # --- Summary ---
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Experiments: {experiment - 1}")
    print(f"  Kept:        {kept}")
    print(f"  Discarded:   {discarded}")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Results:     {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
