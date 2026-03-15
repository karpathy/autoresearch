"""Local optimization loop — run an agent command repeatedly, scoring each attempt.

The agent command is a black box: it runs in the problem directory, modifies
state files, and exits. The framework handles branching, scoring, merging,
and leaderboard updates. Scoring is hidden from the agent by temporarily
moving the scoring directory out of sight during agent execution.
"""

import os
import shutil
import subprocess
import sys

from autoanything.git import git, get_head_commit, get_commit_message
from autoanything.history import (
    init_db,
    get_incumbent,
    update_incumbent,
    record_evaluation,
)
from autoanything.leaderboard import export_leaderboard, export_history
from autoanything.scoring import run_score, is_better


def _scoring_dir(problem_dir):
    """The scoring directory path."""
    return os.path.join(problem_dir, "scoring")


def _hidden_path(problem_dir):
    """Where scoring gets stashed during agent runs."""
    return os.path.join(problem_dir, ".autoanything", "_scoring")


def _hide_scoring(problem_dir):
    """Move scoring directory into .autoanything/_scoring to hide from agents."""
    src = _scoring_dir(problem_dir)
    if not os.path.isdir(src):
        return False
    dst = _hidden_path(problem_dir)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    return True


def _restore_scoring(problem_dir):
    """Restore scoring directory from hidden location."""
    dst = _scoring_dir(problem_dir)
    src = _hidden_path(problem_dir)
    if not os.path.exists(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)


def _recover_scoring(problem_dir):
    """Recover scoring if left hidden from a previous interrupted run."""
    hidden = _hidden_path(problem_dir)
    scoring = _scoring_dir(problem_dir)
    if os.path.exists(hidden) and not os.path.isdir(scoring):
        shutil.move(hidden, scoring)
        print("Recovered scoring directory from previous interrupted run.")


def run_local(problem_dir, config, db_path, agent_command,
              max_iterations=None, max_consecutive_crashes=5):
    """Run the local optimization loop.

    Args:
        problem_dir: Path to the problem directory.
        config: ProblemConfig.
        db_path: Path to the SQLite database.
        agent_command: Shell command to run as the agent.
        max_iterations: Stop after this many iterations (None = unlimited).
        max_consecutive_crashes: Stop after this many consecutive crashes.
    """
    base_branch = config.git.base_branch
    score_name = config.score.name
    timeout = config.score.timeout
    direction = config.score.direction
    leaderboard_path = os.path.join(problem_dir, "leaderboard.md")
    history_path = os.path.join(problem_dir, "history.md")

    # Recover scoring if hidden from a previous interrupted run
    _recover_scoring(problem_dir)

    conn = init_db(db_path)

    # Ensure we're on the base branch
    git("checkout", base_branch, cwd=problem_dir)

    # Establish baseline if needed
    incumbent = get_incumbent(conn)
    if incumbent is None:
        print("=" * 60)
        print("ESTABLISHING BASELINE")
        print("=" * 60)

        commit_sha = get_head_commit(cwd=problem_dir)
        score_val, metrics, duration, error = run_score(
            problem_dir, score_name=score_name, timeout=timeout,
        )

        if error or score_val is None:
            print(f"Baseline FAILED: {error}")
            conn.close()
            sys.exit(1)

        record_evaluation(conn, commit_sha, base_branch, score_val, "baseline",
                          "initial baseline", duration, metrics=metrics)
        update_incumbent(conn, commit_sha, score_val)
        export_leaderboard(conn, leaderboard_path, direction=direction)
        export_history(conn, history_path)
        git("add", "leaderboard.md", "history.md", cwd=problem_dir)
        git("commit", "-m", "Initialize leaderboard with baseline score",
            cwd=problem_dir, check=False)

        incumbent = get_incumbent(conn)
        print(f"Baseline: {score_name} = {score_val:.6f}")

    print()
    print(f"Incumbent: {incumbent['score']:.6f}")
    print(f"Direction: {direction}")
    print(f"Agent:     {agent_command}")
    if max_iterations:
        print(f"Max iterations: {max_iterations}")
    print()

    iteration = 0
    consecutive_crashes = 0

    try:
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                print(f"\nReached max iterations ({max_iterations})")
                break

            if consecutive_crashes >= max_consecutive_crashes:
                print(f"\nStopping after {consecutive_crashes} consecutive crashes")
                break

            iteration += 1
            branch = f"proposals/local/attempt-{iteration}"

            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration}")
            print(f"  Incumbent: {incumbent['score']:.6f}")
            print("=" * 60)

            # Clean up branch if it exists from a previous interrupted run
            git("branch", "-D", branch, cwd=problem_dir, check=False)

            # Create proposal branch
            git("checkout", "-b", branch, cwd=problem_dir)

            # Hide scoring, run agent, restore scoring
            hidden = _hide_scoring(problem_dir)
            try:
                agent_env = {
                    **os.environ,
                    "AUTOANYTHING_ITERATION": str(iteration),
                    "AUTOANYTHING_SCORE": str(incumbent["score"]),
                    "AUTOANYTHING_DIRECTION": direction,
                    "AUTOANYTHING_METRIC": score_name,
                    "AUTOANYTHING_PROBLEM": config.name,
                }

                print(f"Running agent...")
                subprocess.run(
                    agent_command, shell=True,
                    cwd=problem_dir, env=agent_env,
                )
            finally:
                if hidden:
                    _restore_scoring(problem_dir)

            # --- Detect what the agent changed ---

            # Uncommitted changes to tracked files only
            diff_out = git("diff", "--name-only", cwd=problem_dir).stdout.strip()
            staged_out = git("diff", "--cached", "--name-only", cwd=problem_dir).stdout.strip()
            uncommitted = set()
            if diff_out:
                uncommitted.update(diff_out.splitlines())
            if staged_out:
                uncommitted.update(staged_out.splitlines())

            # Did the agent commit?
            base_sha = git("rev-parse", base_branch, cwd=problem_dir).stdout.strip()
            head_sha = git("rev-parse", "HEAD", cwd=problem_dir).stdout.strip()
            agent_committed = (base_sha != head_sha)

            if not uncommitted and not agent_committed:
                print("  No changes — skipping")
                git("checkout", base_branch, cwd=problem_dir)
                git("branch", "-D", branch, cwd=problem_dir, check=False)
                continue

            # Collect all changed files for validation
            all_changes = set(uncommitted)
            if agent_committed:
                committed_out = git(
                    "diff", "--name-only", f"{base_branch}...HEAD", cwd=problem_dir,
                ).stdout.strip()
                if committed_out:
                    all_changes.update(committed_out.splitlines())

            # Validate only state/ files were touched (path-prefix check)
            invalid = {f for f in all_changes if not f.startswith("state/")}
            if invalid:
                print(f"  INVALID: modified non-state files: {invalid}")
                git("checkout", ".", cwd=problem_dir, check=False)
                git("checkout", base_branch, cwd=problem_dir)
                git("branch", "-D", branch, cwd=problem_dir, check=False)
                consecutive_crashes += 1
                continue

            # Auto-commit if agent left uncommitted changes
            if uncommitted:
                for f in uncommitted:
                    git("add", f, cwd=problem_dir)
                git("commit", "-m", f"local attempt #{iteration}", cwd=problem_dir)

            # --- Score ---
            commit_sha = get_head_commit(cwd=problem_dir)
            description = get_commit_message(commit_sha, cwd=problem_dir)

            print("  Scoring...")
            score_val, metrics, duration, error = run_score(
                problem_dir, score_name=score_name, timeout=timeout,
            )

            if error or score_val is None:
                status = "crash"
                print(f"  CRASH ({duration:.0f}s): {(error or 'unknown')[:200]}")
                consecutive_crashes += 1
            elif is_better(score_val, incumbent["score"], direction):
                status = "accepted"
                consecutive_crashes = 0
                delta = score_val - incumbent["score"]
                print(f"  ACCEPTED: {score_val:.6f} "
                      f"(was {incumbent['score']:.6f}, delta={delta:.6f})")
            else:
                status = "rejected"
                consecutive_crashes = 0
                print(f"  REJECTED: {score_val:.6f} "
                      f"(incumbent: {incumbent['score']:.6f})")

            record_evaluation(
                conn, commit_sha, branch, score_val, status,
                description, duration, error_message=error, metrics=metrics,
            )

            # Return to base branch
            git("checkout", base_branch, cwd=problem_dir)

            if status == "accepted":
                git("merge", branch, "--no-ff",
                    "-m", f"Accept local attempt #{iteration}: {score_val:.6f}",
                    cwd=problem_dir)
                update_incumbent(conn, commit_sha, score_val)
                incumbent = get_incumbent(conn)
                print("  Merged.")

            # Clean up proposal branch
            git("branch", "-D", branch, cwd=problem_dir, check=False)

            # Update leaderboard and history
            export_leaderboard(conn, leaderboard_path, direction=direction)
            export_history(conn, history_path)
            git("add", "leaderboard.md", "history.md", cwd=problem_dir)
            score_str = f"{score_val:.6f}" if score_val is not None else "crash"
            git("commit", "-m",
                f"Update leaderboard: attempt #{iteration} ({score_str})",
                cwd=problem_dir, check=False)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        # Try to return to base branch
        try:
            git("checkout", ".", cwd=problem_dir, check=False)
            git("checkout", base_branch, cwd=problem_dir, check=False)
        except Exception:
            pass
        _recover_scoring(problem_dir)
    finally:
        incumbent = get_incumbent(conn)
        if incumbent:
            print(f"\nFinal score: {incumbent['score']:.6f}")
        conn.close()
