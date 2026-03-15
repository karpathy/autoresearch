"""Evaluator — polling evaluation loop.

Orchestrates scoring: finds pending proposals, scores them, merges
improvements, and updates the leaderboard.

Provides both unit functions (establish_baseline, evaluate_proposal)
and the full polling loop (run_evaluator).
"""

import os
import time

from autoanything.git import (
    git,
    get_proposal_branches,
    get_branch_commit,
    get_head_commit,
    get_commit_message,
)
from autoanything.history import (
    init_db,
    get_incumbent,
    update_incumbent,
    record_evaluation,
    is_evaluated,
)
from autoanything.leaderboard import export_leaderboard, export_history
from autoanything.scoring import run_score, is_better


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
    timeout = config.score.timeout
    leaderboard_path = os.path.join(problem_dir, "leaderboard.md")

    print("=" * 60)
    print("ESTABLISHING BASELINE")
    print("=" * 60)

    git("checkout", base_branch, cwd=problem_dir)
    commit_sha = get_head_commit(cwd=problem_dir)

    print(f"Commit: {commit_sha[:7]}")
    print("Running scoring...")

    score, metrics, duration, error = run_score(
        problem_dir, score_name=score_name, timeout=timeout,
    )

    if score is not None:
        record_evaluation(
            conn, commit_sha, base_branch, score, "baseline",
            "initial baseline", duration, metrics=metrics,
        )
        update_incumbent(conn, commit_sha, score)
        export_leaderboard(conn, leaderboard_path, direction=config.score.direction)
        history_path = os.path.join(problem_dir, "history.md")
        export_history(conn, history_path)
        git("add", "leaderboard.md", "history.md", cwd=problem_dir)
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
    timeout = config.score.timeout
    leaderboard_path = os.path.join(problem_dir, "leaderboard.md")

    description = get_commit_message(commit_sha, cwd=problem_dir)
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
    except Exception as e:
        print(f"  Failed to checkout: {e}")
        return

    score, metrics, duration, error = run_score(
        problem_dir, score_name=score_name, timeout=timeout,
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
        except Exception as e:
            print(f"  Merge failed (score still recorded): {e}")
    else:
        print(f"  REJECTED: {score:.6f} (incumbent: {incumbent['score']:.6f})")
        record_evaluation(
            conn, commit_sha, branch, score, "rejected",
            description, duration, metrics=metrics,
        )

    export_leaderboard(conn, leaderboard_path, direction=direction)
    history_path = os.path.join(problem_dir, "history.md")
    export_history(conn, history_path)
    score_str = f"{score:.6f}" if score else "crash"
    git("add", "leaderboard.md", "history.md", cwd=problem_dir)
    git("commit", "-m",
        f"Update leaderboard: {branch} ({score_str})",
        cwd=problem_dir, check=False)


def run_evaluator(problem_dir: str, config, db_path: str,
                  baseline_only: bool = False, push: bool = False,
                  poll_interval: int = 30):
    """Run the full polling evaluation loop.

    Args:
        problem_dir: Path to the problem directory.
        config: ProblemConfig.
        db_path: Path to the SQLite database.
        baseline_only: If True, establish baseline and exit.
        push: If True, push leaderboard and merge results to origin.
        poll_interval: Seconds between polls.
    """
    import sys

    direction = config.score.direction
    base_branch = config.git.base_branch
    pattern = config.git.proposal_pattern

    conn = init_db(db_path)

    # Establish baseline if needed
    incumbent = get_incumbent(conn)
    if incumbent is None:
        if not establish_baseline(conn, problem_dir, config):
            sys.exit(1)
        incumbent = get_incumbent(conn)
        if push:
            git("push", "origin", base_branch, cwd=problem_dir)

    if baseline_only:
        print(f"\nBaseline: {incumbent['score']:.6f}")
        conn.close()
        return

    print(f"\nIncumbent: {incumbent['score']:.6f} ({incumbent['commit_sha'][:7]})")
    print(f"Direction: {direction}")
    print(f"Polling every {poll_interval}s for {pattern} branches...")
    print()

    while True:
        # Fetch latest
        git("fetch", "--all", "--prune", cwd=problem_dir, check=False)

        # Find unevaluated proposals
        branches = get_proposal_branches(cwd=problem_dir, pattern=pattern)
        pending = []
        for branch in branches:
            commit = get_branch_commit(branch, cwd=problem_dir)
            if not is_evaluated(conn, commit):
                pending.append((branch, commit))

        if not pending:
            time.sleep(poll_interval)
            continue

        print(f"Found {len(pending)} pending proposal(s)")

        for branch, commit_sha in pending:
            evaluate_proposal(conn, branch, commit_sha, direction,
                              problem_dir, config)

        if push:
            git("push", "origin", base_branch, cwd=problem_dir, check=False)
