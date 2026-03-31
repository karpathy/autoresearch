"""
Run YC-bench with the prompt from prompt.txt and extract results.
Usage: uv run run_bench.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent / "yc-bench-main"
PROMPT_FILE = Path(__file__).parent / "prompt.txt"
CONFIG_FILE = Path(__file__).parent / "custom_config.toml"
TIMEOUT = 900  # 15 minutes max

def main():
    # Read the prompt
    prompt_text = PROMPT_FILE.read_text()

    # Write TOML config with the prompt
    # Use triple quotes for multi-line TOML string
    toml_content = 'extends = "medium"\n\n[agent]\n'
    # Escape the prompt for TOML multi-line literal string
    # Use triple-quoted basic string to handle special chars
    toml_content += 'system_prompt = """\n'
    toml_content += prompt_text.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    toml_content += '\n"""\n'
    CONFIG_FILE.write_text(toml_content)

    # Set API key
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "REDACTED_KEY"

    # Run the benchmark
    cmd = [
        "uv", "run", "yc-bench", "run",
        "--model", "anthropic/claude-sonnet-4-6",
        "--seed", "1",
        "--config", str(CONFIG_FILE.resolve()),
        "--no-live",
    ]

    print(f"Running YC-bench...")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=BENCH_DIR, env=env,
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.0f}s")
        print("---")
        print("final_funds:   TIMEOUT")
        print("tasks_done:    0")
        print("tasks_failed:  0")
        print(f"elapsed:       {elapsed:.0f}")
        print("outcome:       timeout")
        sys.exit(1)

    # Find the result JSON
    results_dir = BENCH_DIR / "results"
    result_files = sorted(results_dir.glob("yc_bench_result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not result_files:
        print("ERROR: No result JSON found")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        sys.exit(1)

    # Parse the most recent result
    data = json.loads(result_files[0].read_text())

    # Extract final funds from time_series
    funds_series = data.get("time_series", {}).get("funds", [])
    if funds_series:
        final_funds_cents = funds_series[-1]["funds_cents"]
    else:
        final_funds_cents = 0

    # Extract task stats
    tasks = data.get("time_series", {}).get("tasks", [])
    tasks_done = sum(1 for t in tasks if t.get("status") == "completed_success")
    tasks_failed = sum(1 for t in tasks if t.get("status") == "completed_fail")

    terminal_reason = data.get("terminal_reason", "unknown")
    turns = data.get("turns_completed", 0)
    cost = data.get("total_cost_usd", 0)

    # Print summary in greppable format
    print("---")
    print(f"final_funds:   ${final_funds_cents / 100:,.2f}")
    print(f"funds_cents:   {final_funds_cents}")
    print(f"tasks_done:    {tasks_done}")
    print(f"tasks_failed:  {tasks_failed}")
    print(f"turns:         {turns}")
    print(f"api_cost:      ${cost:.4f}")
    print(f"elapsed:       {elapsed:.0f}")
    print(f"outcome:       {terminal_reason}")

if __name__ == "__main__":
    main()
