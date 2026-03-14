"""Tests for PDCA web routes and HTTP API."""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from pdca_system.domain.models import RunStatus, SeedStatus
from pdca_system.task import LOG_ROOT, run_path, save_run
from pdca_system.web.app import app


class WebTests(unittest.TestCase):
    def test_dashboard_route_renders(self) -> None:
        client = TestClient(app)
        response = client.get("/pdca-system")
        self.assertEqual(response.status_code, 200)
        self.assertIn("PDCA System", response.text)
        self.assertIn("Create Seed", response.text)
        self.assertIn("Direct Code Agent", response.text)
        self.assertIn("Run Direct Code Agent", response.text)
        self.assertIn('id="dashboard-board"', response.text)
        self.assertIn("data-dashboard-partial-url", response.text)
        self.assertIn("data-seed-detail-url-template", response.text)

    def test_run_log_api_missing_run_returns_404(self) -> None:
        client = TestClient(app)
        response = client.get("/pdca-system/api/runs/unknown-run/log?stream=stdout&offset=0")
        self.assertEqual(response.status_code, 404)

    def test_run_log_api_reads_run_named_log_without_run_state(self) -> None:
        client = TestClient(app)
        run_id = "p-20990101-000001-deadbeef"
        log_path = LOG_ROOT / f"{run_id}.stdout.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("hello log\n", encoding="utf-8")
        try:
            response = client.get(f"/pdca-system/api/runs/{run_id}/log?stream=stdout&offset=0")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["chunk"], "hello log\r\n")
            self.assertEqual(payload["next_offset"], len(payload["chunk"]))
        finally:
            if log_path.exists():
                log_path.unlink()

    def test_run_log_api_running_run_with_missing_log_returns_empty_chunk(self) -> None:
        client = TestClient(app)
        run_id = "ca-20990101-000001-deadbeef"
        run_state_path = run_path(run_id)
        save_run(
            {
                "run_id": run_id,
                "seed_id": "seed-test",
                "stage": "ca",
                "status": "running",
                "task_id": "task-ca-20990101-000001-cafebabe",
                "created_at": 0.0,
                "updated_at": 0.0,
                "log_path": None,
                "stderr_log_path": None,
                "summary": {},
                "metrics": {},
                "signal": None,
                "worktree_path": None,
                "branch_name": None,
                "error": None,
            }
        )
        try:
            response = client.get(f"/pdca-system/api/runs/{run_id}/log?stream=stdout&offset=0")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["chunk"], "")
            self.assertEqual(payload["next_offset"], 0)
            self.assertEqual(payload["size"], 0)
            self.assertFalse(payload["complete"])
        finally:
            if run_state_path.exists():
                run_state_path.unlink()

    def test_seed_detail_page_exposes_seed_detail_refresh_target(self) -> None:
        service = app.state.workflow
        seed = service.create_seed("seed detail refresh target", baseline_branch="master")
        client = TestClient(app)

        response = client.get(f"/pdca-system/seeds/{seed.seed_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="seed-detail"', response.text)
        self.assertIn("data-seed-detail-url-template", response.text)

    def test_direct_code_agent_route_creates_seed_and_redirects(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/pdca-system/actions/direct-code-agent",
                data={"prompt": "make a direct code edit"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        self.assertIn("/pdca-system/", location)
        self.assertIn("?seed_id=", location)
        seed_id = location.rsplit("=", 1)[-1]
        seed = app.state.workflow.require_seed(seed_id)
        run = app.state.workflow.require_run(seed.latest_run_id)

        self.assertEqual(seed.status, SeedStatus.adapting)
        self.assertEqual(run.status, RunStatus.queued)
        self.assertEqual(seed.prompt, "make a direct code edit")


if __name__ == "__main__":
    unittest.main()
