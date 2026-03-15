"""Webhook server — FastAPI application for PR-based evaluation.

Factory function `create_app` returns a configured FastAPI instance.
Includes the evaluation worker thread, PR validation, GitHub interaction
via `gh` CLI, and startup scan for unevaluated open PRs.
"""

import hashlib
import hmac
import json
import logging
import os
import subprocess
import threading
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from autoanything.history import init_db, get_incumbent, update_incumbent, record_evaluation, is_evaluated
from autoanything.leaderboard import export_leaderboard, export_history
from autoanything.problem import load_problem
from autoanything.scoring import run_score, is_better

logger = logging.getLogger("autoanything.server")


def validate_pr_files(modified: list[str]) -> tuple[bool, str]:
    """Check that only state/ files were modified.

    Args:
        modified: List of file paths modified in the PR.

    Returns:
        (ok, message) — ok is True if all files are in state/.
    """
    disallowed = [f for f in modified if not f.startswith("state/")]
    if disallowed:
        return False, (
            "This PR modifies files outside of `state/`:\n"
            f"```\n{chr(10).join(disallowed)}\n```\n"
            "Only files in `state/` may be modified."
        )
    return True, ""


# ---------------------------------------------------------------------------
# GitHub interaction via `gh` CLI
# ---------------------------------------------------------------------------


def gh(*args, cwd: str, check: bool = True):
    """Run a gh CLI command in the specified directory."""
    result = subprocess.run(
        ["gh"] + list(args),
        capture_output=True, text=True, cwd=cwd,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["gh"] + list(args),
            output=result.stdout, stderr=result.stderr,
        )
    return result


def pr_comment(pr_number: int, body: str, cwd: str):
    gh("pr", "comment", str(pr_number), "--body", body, cwd=cwd)


def pr_merge(pr_number: int, cwd: str):
    gh("pr", "merge", str(pr_number), "--merge", cwd=cwd)


def pr_close(pr_number: int, cwd: str):
    gh("pr", "close", str(pr_number), cwd=cwd)


def pr_diff_files(pr_number: int, cwd: str) -> list[str]:
    result = gh("pr", "diff", str(pr_number), "--name-only", cwd=cwd)
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


# ---------------------------------------------------------------------------
# PR comment formatting
# ---------------------------------------------------------------------------


def format_results_comment(
    score: float | None,
    incumbent_score: float | None,
    duration: float,
    metrics: dict | None,
    status: str,
    score_name: str = "score",
    error: str | None = None,
) -> str:
    """Format evaluation results as a Markdown PR comment."""
    if status == "crash":
        error_short = (error or "Unknown error")[:1500]
        return (
            "## AutoAnything Evaluation\n\n"
            "**Result:** \U0001f4a5 Crash \u2014 closing\n\n"
            f"```\n{error_short}\n```"
        )

    delta = score - incumbent_score
    delta_str = f"+{delta:.6f}" if delta >= 0 else f"{delta:.6f}"

    if status == "accepted":
        result_str = "\u2705 Accepted \u2014 merging"
    else:
        result_str = "\u274c Rejected \u2014 closing"

    comment = (
        "## AutoAnything Evaluation\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| **Score** | {score:.6f} |\n"
        f"| **Incumbent** | {incumbent_score:.6f} |\n"
        f"| **Delta** | {delta_str} |\n"
        f"| **Result** | {result_str} |\n"
        f"| **Duration** | {duration:.0f}s |\n"
    )

    if metrics:
        extra = {k: v for k, v in metrics.items() if k != score_name}
        if extra:
            comment += (
                "\n<details>\n<summary>Additional metrics</summary>\n\n"
                "| Metric | Value |\n"
                "|--------|-------|\n"
            )
            for k, v in extra.items():
                comment += f"| {k} | {v} |\n"
            comment += "\n</details>\n"

    return comment


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(problem_dir: str, webhook_secret: str = None,
               db_path: str = None, push: bool = False) -> FastAPI:
    """Create a configured FastAPI app for webhook evaluation.

    Args:
        problem_dir: Path to the problem directory.
        webhook_secret: Optional GitHub webhook secret for signature verification.
        db_path: Path to the SQLite database. Defaults to .autoanything/history.db.
        push: Whether to push leaderboard and merge results to origin.

    Returns:
        FastAPI application instance.
    """
    from autoanything.git import git

    # Load problem config
    try:
        config = load_problem(problem_dir)
        base_branch = config.git.base_branch
        direction = config.score.direction
        score_name = config.score.name
        score_timeout = config.score.timeout
    except Exception:
        base_branch = "main"
        direction = "minimize"
        score_name = "score"
        score_timeout = 900

    if db_path is None:
        db_path = os.path.join(problem_dir, ".autoanything", "history.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Evaluation queue
    eval_queue: deque = deque()
    queue_lock = threading.Lock()
    queue_event = threading.Event()
    current_eval = {"ref": None}

    def verify_signature(payload: bytes, signature: str) -> bool:
        if not webhook_secret:
            return True
        expected = "sha256=" + hmac.new(
            webhook_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    # -- Evaluation worker --

    def _evaluate_one_pr(conn, pr_info, pr_number, head_sha, branch):
        """Handle a single PR evaluation."""
        # Already evaluated?
        if is_evaluated(conn, head_sha):
            logger.info("PR #%d (%s) already evaluated, skipping", pr_number, head_sha[:7])
            return

        # Comment "evaluating..."
        try:
            pr_comment(pr_number, "\u23f3 Evaluation started\u2026", cwd=problem_dir)
        except Exception as e:
            logger.warning("Could not comment on PR #%d: %s", pr_number, e)

        # Validate modified files
        try:
            modified = pr_diff_files(pr_number, cwd=problem_dir)
        except subprocess.CalledProcessError:
            modified = None

        if modified is not None:
            ok, msg = validate_pr_files(modified)
            if not ok:
                comment = (
                    "## AutoAnything Evaluation\n\n"
                    "**Result:** \U0001f6ab Rejected \u2014 disallowed file changes\n\n" + msg
                )
                try:
                    pr_comment(pr_number, comment, cwd=problem_dir)
                    pr_close(pr_number, cwd=problem_dir)
                except Exception as e:
                    logger.error("Could not close PR #%d: %s", pr_number, e)
                record_evaluation(
                    conn, head_sha, branch, None, "rejected",
                    f"Disallowed files: {msg[:200]}", 0, error_message=msg,
                )
                return

        # Checkout the PR
        try:
            gh("pr", "checkout", str(pr_number), cwd=problem_dir)
        except subprocess.CalledProcessError as e:
            comment = (
                "## AutoAnything Evaluation\n\n"
                "**Result:** \U0001f4a5 Could not checkout PR\n\n"
                f"```\n{e.stderr[:500]}\n```"
            )
            try:
                pr_comment(pr_number, comment, cwd=problem_dir)
                pr_close(pr_number, cwd=problem_dir)
            except Exception:
                pass
            return

        # Score
        incumbent = get_incumbent(conn)
        incumbent_score = incumbent["score"] if incumbent else None
        description = pr_info.get("title", f"PR #{pr_number}")

        score, metrics, duration, error = run_score(
            problem_dir, score_name=score_name, timeout=score_timeout,
        )

        # Return to base branch
        git("checkout", base_branch, cwd=problem_dir)

        if error or score is None:
            logger.info("PR #%d: CRASH (%.0fs)", pr_number, duration)
            record_evaluation(
                conn, head_sha, branch, None, "crash",
                description, duration, error_message=error, metrics=metrics,
            )
            comment = format_results_comment(
                None, incumbent_score, duration, metrics, "crash",
                score_name=score_name, error=error,
            )
            try:
                pr_comment(pr_number, comment, cwd=problem_dir)
                pr_close(pr_number, cwd=problem_dir)
            except Exception as e:
                logger.error("Could not close PR #%d: %s", pr_number, e)

        elif incumbent_score is not None and is_better(score, incumbent_score, direction):
            logger.info(
                "PR #%d: ACCEPTED %.6f (was %.6f)", pr_number, score, incumbent_score,
            )
            record_evaluation(
                conn, head_sha, branch, score, "accepted",
                description, duration, metrics=metrics,
            )
            comment = format_results_comment(
                score, incumbent_score, duration, metrics, "accepted",
                score_name=score_name,
            )
            try:
                pr_comment(pr_number, comment, cwd=problem_dir)
                pr_merge(pr_number, cwd=problem_dir)
            except Exception as e:
                logger.error("Could not merge PR #%d: %s", pr_number, e)

            update_incumbent(conn, head_sha, score)
            _update_leaderboard(conn, pr_number, score)

        else:
            logger.info(
                "PR #%d: REJECTED %.6f (incumbent: %.6f)",
                pr_number, score, incumbent_score or 0,
            )
            record_evaluation(
                conn, head_sha, branch, score, "rejected",
                description, duration, metrics=metrics,
            )
            comment = format_results_comment(
                score, incumbent_score, duration, metrics, "rejected",
                score_name=score_name,
            )
            try:
                pr_comment(pr_number, comment, cwd=problem_dir)
                pr_close(pr_number, cwd=problem_dir)
            except Exception as e:
                logger.error("Could not close PR #%d: %s", pr_number, e)

            _update_leaderboard(conn, pr_number, score)

    def _update_leaderboard(conn, pr_number: int, score: float | None):
        """Export leaderboard and history, commit, and optionally push."""
        leaderboard_path = os.path.join(problem_dir, "leaderboard.md")
        history_path = os.path.join(problem_dir, "history.md")
        export_leaderboard(conn, leaderboard_path, direction=direction)
        export_history(conn, history_path)
        git("add", "leaderboard.md", "history.md", cwd=problem_dir)
        score_str = f"{score:.6f}" if score is not None else "crash"
        git(
            "commit", "-m",
            f"Update leaderboard: PR #{pr_number} ({score_str})",
            cwd=problem_dir, check=False,
        )
        if push:
            git("push", "origin", base_branch, cwd=problem_dir, check=False)

    def evaluation_worker():
        """Drain the evaluation queue one PR at a time."""
        conn = init_db(db_path)

        while True:
            queue_event.wait()

            while True:
                with queue_lock:
                    if not eval_queue:
                        queue_event.clear()
                        break
                    pr_info = eval_queue.popleft()

                current_eval["ref"] = pr_info.get("number")
                pr_number = pr_info["number"]
                head_sha = pr_info["head_sha"]
                branch = pr_info.get("branch", f"pr-{pr_number}")
                author = pr_info.get("author", "unknown")

                logger.info("Evaluating PR #%d (%s) by %s", pr_number, head_sha[:7], author)

                try:
                    _evaluate_one_pr(conn, pr_info, pr_number, head_sha, branch)
                except Exception as e:
                    logger.exception("Unexpected error evaluating PR #%d", pr_number)
                    try:
                        pr_comment(
                            pr_number,
                            "## AutoAnything Evaluation\n\n"
                            f"**Result:** \U0001f4a5 Internal error\n\n```\n{str(e)[:500]}\n```",
                            cwd=problem_dir,
                        )
                        pr_close(pr_number, cwd=problem_dir)
                    except Exception:
                        pass
                finally:
                    current_eval["ref"] = None

    def startup_scan():
        """Scan for open PRs and enqueue any that haven't been evaluated."""
        try:
            result = gh(
                "pr", "list", "--base", base_branch, "--state", "open",
                "--json", "number,headRefOid,headRefName,author,title",
                cwd=problem_dir, check=False,
            )
            if result.returncode != 0:
                logger.warning("Could not list open PRs: %s", result.stderr)
                return

            prs = json.loads(result.stdout)
            conn = init_db(db_path)
            enqueued = 0
            for pr in prs:
                head_sha = pr.get("headRefOid", "")
                if not head_sha or is_evaluated(conn, head_sha):
                    continue

                author_raw = pr.get("author")
                if isinstance(author_raw, dict):
                    author = author_raw.get("login", "unknown")
                else:
                    author = str(author_raw) if author_raw else "unknown"

                pr_info = {
                    "number": pr["number"],
                    "head_sha": head_sha,
                    "branch": pr.get("headRefName", f"pr-{pr['number']}"),
                    "author": author,
                    "title": pr.get("title", ""),
                }
                with queue_lock:
                    eval_queue.append(pr_info)
                enqueued += 1

            conn.close()
            if enqueued:
                queue_event.set()
                logger.info("Startup scan: enqueued %d unevaluated open PR(s)", enqueued)
            else:
                logger.info("Startup scan: no unevaluated open PRs found")
        except Exception as e:
            logger.warning("Startup scan failed: %s", e)

    # -- FastAPI app --

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = threading.Thread(target=evaluation_worker, daemon=True)
        worker.start()
        logger.info("Evaluation worker started")
        startup_scan()
        yield

    app = FastAPI(title="AutoAnything Web Evaluator", lifespan=lifespan)

    @app.get("/health")
    async def health():
        conn = init_db(db_path)
        incumbent = get_incumbent(conn)
        conn.close()
        with queue_lock:
            queue_len = len(eval_queue)
        return {
            "status": "ok",
            "queue_length": queue_len,
            "currently_evaluating": current_eval["ref"],
            "incumbent_score": incumbent["score"] if incumbent else None,
            "incumbent_commit": incumbent["commit_sha"][:7] if incumbent else None,
        }

    @app.post("/webhook")
    async def webhook(request: Request):
        body = await request.body()
        if webhook_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not verify_signature(body, signature):
                return Response(status_code=401, content="Invalid signature")

        payload = json.loads(body)

        event = request.headers.get("X-GitHub-Event", "")
        if event != "pull_request":
            return {"status": "ignored", "reason": f"event type: {event}"}

        action = payload.get("action", "")
        if action not in ("opened", "synchronize"):
            return {"status": "ignored", "reason": f"action: {action}"}

        pr = payload.get("pull_request", {})
        pr_base = pr.get("base", {}).get("ref", "")
        if pr_base != base_branch:
            return {"status": "ignored", "reason": f"targets {pr_base}, not {base_branch}"}

        pr_number = pr.get("number")
        head_sha = pr.get("head", {}).get("sha", "")
        branch = pr.get("head", {}).get("ref", f"pr-{pr_number}")
        author = pr.get("user", {}).get("login", "unknown")
        title = pr.get("title", "")

        pr_info = {
            "number": pr_number,
            "head_sha": head_sha,
            "branch": branch,
            "author": author,
            "title": title,
        }

        with queue_lock:
            eval_queue.append(pr_info)
            position = len(eval_queue)
        queue_event.set()

        logger.info(
            "Enqueued PR #%d (%s) by %s, position %d",
            pr_number, head_sha[:7], author, position,
        )
        return {"status": "queued", "pr": pr_number, "position": position}

    return app
