"""Webhook server — FastAPI application for PR-based evaluation.

Factory function `create_app` returns a configured FastAPI instance.
"""

import hashlib
import hmac
import json
import os
import threading
from collections import deque

from fastapi import FastAPI, Request, Response

from autoanything.history import init_db, get_incumbent
from autoanything.problem import load_problem


def validate_pr_files(modified: list[str], mutable_files: list[str]) -> tuple[bool, str]:
    """Check that only allowed files were modified.

    Args:
        modified: List of file paths modified in the PR.
        mutable_files: List of allowed mutable file paths.

    Returns:
        (ok, message) — ok is True if all files are allowed.
    """
    disallowed = [f for f in modified if f not in mutable_files]
    if disallowed:
        return False, (
            "This PR modifies files outside the allowed set:\n"
            f"```\n{chr(10).join(disallowed)}\n```\n"
            f"Only these files may be modified: {', '.join(mutable_files)}"
        )
    return True, ""


def create_app(problem_dir: str, webhook_secret: str = None) -> FastAPI:
    """Create a configured FastAPI app for webhook evaluation.

    Args:
        problem_dir: Path to the problem directory.
        webhook_secret: Optional GitHub webhook secret for signature verification.

    Returns:
        FastAPI application instance.
    """
    # Load problem config
    try:
        config = load_problem(problem_dir)
        base_branch = config.git.base_branch
    except Exception:
        base_branch = "main"

    db_path = os.path.join(problem_dir, ".autoanything", "history.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Evaluation queue
    eval_queue: deque = deque()
    queue_lock = threading.Lock()
    current_eval = {"ref": None}

    def verify_signature(payload: bytes, signature: str) -> bool:
        if not webhook_secret:
            return True
        expected = "sha256=" + hmac.new(
            webhook_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    app = FastAPI(title="AutoAnything Web Evaluator")

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

        return {"status": "queued", "pr": pr_number, "position": position}

    return app
