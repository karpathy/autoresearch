"""Evaluator — polling evaluation loop.

Orchestrates scoring: finds pending proposals, scores them, merges
improvements, and updates the leaderboard.
"""

from autoanything.history import (
    get_incumbent,
    update_incumbent,
    record_evaluation,
)
from autoanything.leaderboard import export_leaderboard
from autoanything.scoring import run_score, is_better

import subprocess as _subprocess


def git(*args, cwd: str, check: bool = True):
    """Run a git command. Defined locally so tests can patch it."""
    result = _subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=cwd,
    )
    if check and result.returncode != 0:
        raise __subprocess.CalledProcessError(
            result.returncode, ["git"] + list(args),
            output=result.stdout, stderr=result.stderr,
        )
    return result

import os


def establish_baseline(conn, problem_dir: str, config):
    """Run the baseline (current main) and record it.

    Args:
        conn: SQLite connection.
        problem_dir: Path to the problem directory.
        config: ProblemConfig (or mock with .score and .git attributes).

    Returns:
        True if baseline was established, False on failure.
    """
    base_branch = config.git.base_branch
    score_name = config.score.name
    script = os.path.join(problem_dir, config.score.script)
    timeout = config.score.timeout
    leaderboard_path = os.path.join(problem_dir, "leaderboard.md")

    print("=" * 60)
    print("ESTABLISHING BASELINE")
    print("=" * 60)

    git("checkout", base_branch, cwd=problem_dir)
    commit_sha = git("rev-parse", "HEAD", cwd=problem_dir).stdout.strip()

    print(f"Commit: {commit_sha[:7]}")
    print("Running score.sh...")

    score, metrics, duration, error = run_score(
        script, score_name=score_name, timeout=timeout, cwd=problem_dir,
    )

    if score is not None:
        record_evaluation(
            conn, commit_sha, base_branch, score, "baseline",
            "initial baseline", duration, metrics=metrics,
        )
        update_incumbent(conn, commit_sha, score)
        export_leaderboard(conn, leaderboard_path, direction=config.score.direction)
        git("add", "leaderboard.md", cwd=problem_dir)
        git("commit", "-m", "Initialize leaderboard with baseline score",
            cwd=problem_dir, check=False)
        print(f"Baseline established: {score_name} = {score:.6f} ({duration:.0f}s)")
        return True
    else:
        print(f"Baseline FAILED: {error}")
        return False


def evaluate_proposal(conn, branch: str, commit_sha: str, direction: str,
                      problem_dir: str, config):
    """Evaluate a single proposal branch.

    Args:
        conn: SQLite connection.
        branch: Proposal branch name.
        commit_sha: Commit SHA to evaluate.
        direction: "minimize" or "maximize".
        problem_dir: Path to the problem directory.
        config: ProblemConfig (or mock with .score and .git attributes).
    """
    base_branch = config.git.base_branch
    score_name = config.score.name
    script = os.path.join(problem_dir, config.score.script)
    timeout = config.score.timeout
    leaderboard_path = os.path.join(problem_dir, "leaderboard.md")

    description = git("log", "-1", "--format=%s", commit_sha, cwd=problem_dir).stdout.strip()
    incumbent = get_incumbent(conn)

    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {branch}")
    print(f"  Commit:      {commit_sha[:7]}")
    print(f"  Description: {description}")
    print(f"  Incumbent:   {incumbent['score']:.6f}")
    print("=" * 60)

    # Detach HEAD at the proposal commit
    try:
        git("checkout", commit_sha, "--detach", cwd=problem_dir)
    except _subprocess.CalledProcessError as e:
        print(f"  Failed to checkout: {e.stderr}")
        return

    score, metrics, duration, error = run_score(
        script, score_name=score_name, timeout=timeout, cwd=problem_dir,
    )

    # Return to main branch
    git("checkout", base_branch, cwd=problem_dir)

    if error or score is None:
        print(f"  CRASH ({duration:.0f}s): {(error or 'unknown')[:200]}")
        record_evaluation(
            conn, commit_sha, branch, None, "crash",
            description, duration, error_message=error, metrics=metrics,
        )
    elif is_better(score, incumbent["score"], direction):
        print(f"  ACCEPTED: {score:.6f} (was {incumbent['score']:.6f}, "
              f"delta={score - incumbent['score']:.6f})")
        record_evaluation(
            conn, commit_sha, branch, score, "accepted",
            description, duration, metrics=metrics,
        )
        try:
            git("merge", f"origin/{branch}", "--no-ff",
                "-m", f"Merge {branch}: score improved", cwd=problem_dir)
            update_incumbent(conn, commit_sha, score)
            print("  Merged to main.")
        except _subprocess.CalledProcessError as e:
            print(f"  Merge failed (score still recorded): {e.stderr}")
    else:
        print(f"  REJECTED: {score:.6f} (incumbent: {incumbent['score']:.6f})")
        record_evaluation(
            conn, commit_sha, branch, score, "rejected",
            description, duration, metrics=metrics,
        )

    export_leaderboard(conn, leaderboard_path, direction=direction)
    score_str = f"{score:.6f}" if score else "crash"
    git("add", "leaderboard.md", cwd=problem_dir)
    git("commit", "-m",
        f"Update leaderboard: {branch} ({score_str})",
        cwd=problem_dir, check=False)
