"""Scoring — run scoring/score.py and parse JSON output.

Runs the problem's score() function via subprocess and extracts
the metric value from the JSON output.
"""

import json
import subprocess
import sys
import time


def parse_score_output(stdout: str, score_name: str):
    """Extract metric value from scoring subprocess stdout.

    Searches from the last line backward for a JSON object containing
    the named metric.

    Returns:
        (score, metrics) — score is a float or None, metrics is a dict or None.
    """
    if not stdout or not stdout.strip():
        return None, None

    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                metrics = json.loads(line)
                raw = metrics.get(score_name)
                if raw is not None:
                    return float(raw), metrics
                return None, metrics
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return None, None


def run_score(problem_dir: str, score_name: str, timeout: int):
    """Run scoring/score.py and return results.

    Invokes the score() function from scoring/score.py in a subprocess,
    with the problem directory as cwd so that imports from state/ and
    context/ resolve correctly.

    Args:
        problem_dir: Path to the problem directory.
        score_name: Metric key to extract from JSON output.
        timeout: Seconds before scoring is killed.

    Returns:
        (score, metrics, duration_seconds, error_message)
    """
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; from scoring.score import score; print(json.dumps(score()))"],
            capture_output=True, text=True, cwd=problem_dir,
            timeout=timeout,
        )
        duration = time.time() - t0

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            stdout_tail = result.stdout[-2000:] if result.stdout else ""
            return None, None, duration, (
                f"Exit code {result.returncode}\n{stderr_tail}\n{stdout_tail}"
            )

        score, metrics = parse_score_output(result.stdout, score_name)
        if score is not None:
            return score, metrics, duration, None

        return None, None, duration, (
            f"No JSON metrics in output\nstdout tail: {result.stdout[-500:]}"
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - t0
        return None, None, duration, f"Evaluation timed out (>{timeout}s)"


def is_better(new_score: float, old_score: float, direction: str = "minimize") -> bool:
    """Check if new_score beats old_score."""
    if direction == "minimize":
        return new_score < old_score
    return new_score > old_score
