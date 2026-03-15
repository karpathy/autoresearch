"""Tests for autoanything.server — webhook server.

Uses FastAPI's TestClient so no actual HTTP server is started.
"""

import hashlib
import hmac
import json

import pytest

from autoanything.server import create_app


@pytest.fixture
def app(problem_dir):
    """A FastAPI app configured for a test problem directory."""
    return create_app(problem_dir=str(problem_dir))


@pytest.fixture
def client(app):
    """TestClient wrapping the app."""
    from starlette.testclient import TestClient
    return TestClient(app)


def _make_pr_payload(pr_number=1, action="opened", base="main", head_sha="abc123",
                     branch="proposals/agent/test", author="testbot", title="Test PR"):
    return {
        "action": action,
        "pull_request": {
            "number": pr_number,
            "title": title,
            "base": {"ref": base},
            "head": {"sha": head_sha, "ref": branch},
            "user": {"login": author},
        },
    }


class TestHealthEndpoint:
    """GET /health returns server status."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "queue_length" in data

    def test_health_includes_incumbent(self, client, problem_dir):
        # Pre-populate incumbent
        from autoanything.history import init_db, update_incumbent
        db_path = str(problem_dir / ".autoanything" / "history.db")
        conn = init_db(db_path)
        update_incumbent(conn, "abc123", 42.5)
        conn.close()

        response = client.get("/health")
        data = response.json()
        assert data["incumbent_score"] == 42.5


class TestWebhookEndpoint:
    """POST /webhook handles GitHub PR events."""

    def test_ignores_non_pr_events(self, client):
        response = client.post(
            "/webhook",
            json={"action": "created"},
            headers={"X-GitHub-Event": "issue_comment"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignores_non_open_actions(self, client):
        payload = _make_pr_payload(action="closed")
        response = client.post(
            "/webhook",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_ignores_wrong_base_branch(self, client):
        payload = _make_pr_payload(base="develop")
        response = client.post(
            "/webhook",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_enqueues_valid_pr(self, client):
        payload = _make_pr_payload()
        response = client.post(
            "/webhook",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["pr"] == 1

    def test_synchronize_action_accepted(self, client):
        payload = _make_pr_payload(action="synchronize")
        response = client.post(
            "/webhook",
            json=payload,
            headers={"X-GitHub-Event": "pull_request"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"


class TestWebhookSignature:
    """Webhook signature verification when WEBHOOK_SECRET is set."""

    def test_valid_signature_accepted(self, problem_dir):
        secret = "test-secret-123"
        app = create_app(problem_dir=str(problem_dir), webhook_secret=secret)
        from starlette.testclient import TestClient
        client = TestClient(app)

        payload = json.dumps(_make_pr_payload()).encode()
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        response = client.post(
            "/webhook",
            content=payload,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200

    def test_invalid_signature_rejected(self, problem_dir):
        secret = "test-secret-123"
        app = create_app(problem_dir=str(problem_dir), webhook_secret=secret)
        from starlette.testclient import TestClient
        client = TestClient(app)

        payload = json.dumps(_make_pr_payload()).encode()

        response = client.post(
            "/webhook",
            content=payload,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-Hub-Signature-256": "sha256=invalid",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 401


class TestPRValidation:
    """PR file validation — only mutable files should be modified."""

    def test_validate_allowed_files(self):
        from autoanything.server import validate_pr_files
        ok, msg = validate_pr_files(
            modified=["state/solution.py"],
            mutable_files=["state/solution.py", "state/config.py"],
        )
        assert ok is True

    def test_reject_disallowed_files(self):
        from autoanything.server import validate_pr_files
        ok, msg = validate_pr_files(
            modified=["state/solution.py", "context/background.py"],
            mutable_files=["state/solution.py"],
        )
        assert ok is False
        assert "context/background.py" in msg
