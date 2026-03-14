"""Tests for PDCA web routes and HTTP API."""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from pdca_system.domain.models import RunStatus, SeedStatus, StageName, StageRun
from pdca_system.task import LOG_ROOT, run_path, save_run, write_task
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

    def test_get_daemon_settings_api_returns_config(self) -> None:
        client = TestClient(app)
        response = client.get("/pdca-system/api/settings/daemon")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("stuck_check_timeout_seconds", data)
        self.assertIn("default_timeouts", data)
        self.assertIsInstance(data["default_timeouts"], dict)
        self.assertIn("pd", data["default_timeouts"])
        self.assertIn("ca", data["default_timeouts"])
        self.assertIn("direct", data["default_timeouts"])

    def test_patch_daemon_settings_persists_and_returns_config(self) -> None:
        from pdca_system.task import get_daemon_config, write_daemon_config

        client = TestClient(app)
        # Restore after test so we don't change real config
        orig = get_daemon_config()
        try:
            response = client.patch(
                "/pdca-system/api/settings/daemon",
                json={"stuck_check_timeout_seconds": 99, "default_timeouts": {"pd": 111}},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["stuck_check_timeout_seconds"], 99)
            self.assertEqual(data["default_timeouts"]["pd"], 111)
            self.assertEqual(get_daemon_config()["stuck_check_timeout_seconds"], 99)
        finally:
            write_daemon_config(orig)

    def test_post_daemon_settings_form_saves_and_returns_partial(self) -> None:
        from pdca_system.task import get_daemon_config, write_daemon_config

        client = TestClient(app)
        orig = get_daemon_config()
        try:
            response = client.post(
                "/pdca-system/actions/settings/daemon",
                data={
                    "stuck_check_timeout_seconds": "88",
                    "timeout_pd": "222",
                    "timeout_ca": "3333",
                    "timeout_direct": "4444",
                },
                headers={"HX-Request": "true"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Ralph max loop", response.text)
            self.assertIn('value="88"', response.text)
            cfg = get_daemon_config()
            self.assertEqual(cfg["stuck_check_timeout_seconds"], 88)
            self.assertEqual(cfg["default_timeouts"]["pd"], 222)
            self.assertEqual(cfg["default_timeouts"]["ca"], 3333)
            self.assertEqual(cfg["default_timeouts"]["direct"], 4444)
        finally:
            write_daemon_config(orig)

    def test_dashboard_includes_daemon_settings_popup_trigger(self) -> None:
        client = TestClient(app)
        response = client.get("/pdca-system")
        self.assertEqual(response.status_code, 200)
        self.assertIn("daemon-settings-modal-body", response.text)
        self.assertIn("Settings", response.text)

    def test_terminate_tasks_redirects_and_marks_runs_failed(self) -> None:
        service = app.state.workflow
        seed = service.create_seed("web terminate test", baseline_branch="master")
        run = StageRun(
            run_id="pd-20990101-000099-webterm",
            seed_id=seed.seed_id,
            stage=StageName.pd,
            status=RunStatus.queued,
            task_id="task-pd-webterm",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        write_task("pd", {"seed_id": seed.seed_id, "run_id": run.run_id}, task_id=run.task_id)
        client = TestClient(app)

        response = client.post(
            f"/pdca-system/actions/seeds/{seed.seed_id}/terminate",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn(f"?seed_id={seed.seed_id}", response.headers.get("location", ""))
        updated_run = service.require_run(run.run_id)
        updated_seed = service.require_seed(seed.seed_id)
        self.assertEqual(updated_run.status, RunStatus.failed)
        self.assertEqual(updated_run.error, "Terminated by user.")
        self.assertEqual(updated_seed.status, SeedStatus.failed)

    def test_terminate_tasks_unknown_seed_returns_404(self) -> None:
        client = TestClient(app)
        response = client.post("/pdca-system/actions/seeds/nonexistent-seed-id/terminate")
        self.assertEqual(response.status_code, 404)

    def test_terminate_tasks_no_non_completed_redirects_with_no_tasks_param(self) -> None:
        service = app.state.workflow
        seed = service.create_seed("web terminate no tasks", baseline_branch="master")
        run = StageRun(
            run_id="ca-20990101-000100-done",
            seed_id=seed.seed_id,
            stage=StageName.ca,
            status=RunStatus.succeeded,
            task_id="task-ca-done-100",
            created_at=0.0,
            updated_at=0.0,
        )
        service.run_repo.save(run)
        client = TestClient(app)
        response = client.post(
            f"/pdca-system/actions/seeds/{seed.seed_id}/terminate",
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertIn("terminate_result=no_tasks", response.headers.get("location", ""))
        get_response = client.get(
            f"/pdca-system/?seed_id={seed.seed_id}&terminate_result=no_tasks",
            follow_redirects=True,
        )
        self.assertIn("Completed tasks cannot be terminated", get_response.text)


if __name__ == "__main__":
    unittest.main()
