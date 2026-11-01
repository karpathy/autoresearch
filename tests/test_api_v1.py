from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from types import SimpleNamespace

from autoresearch_runtime.dash_http import create_server
from autoresearch_runtime import runtimeops
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.commands import common as common_commands
from autoresearch_runtime.config import load_config
from autoresearch_runtime.harness_policy import contract_fingerprint


class ApiV1SecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dashboard"
        self.home = Path(self.temp.name) / "home"
        self.old_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.server = create_server(self.root, port=0, stop_token="local-test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), self.thread.join(timeout=2)))
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def _restore_home(self) -> None:
        if self.old_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.old_home

    def _install_offline_harness_profile(self, lab_id: str) -> None:
        config = load_config(lab_id)
        profile_path = self.root / f"harness-profile-{lab_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "api-test-harness", "version": "1"},
            "approval": {"approved": True, "contract_fingerprint": contract_fingerprint(config)},
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
            "spend": None,
        }), encoding="utf-8")
        old_profile = os.environ.get("AUTORESEARCH_HARNESS_PROFILE")
        os.environ["AUTORESEARCH_HARNESS_PROFILE"] = str(profile_path)

        def restore_profile() -> None:
            if old_profile is None:
                os.environ.pop("AUTORESEARCH_HARNESS_PROFILE", None)
            else:
                os.environ["AUTORESEARCH_HARNESS_PROFILE"] = old_profile

        self.addCleanup(restore_profile)

    def test_windows_identity_bound_worker_holds_job_after_nonblocking_launch(self) -> None:
        waiting = threading.Event()
        release = threading.Event()
        reaped = threading.Event()

        class FakeProcess:
            pid = 4321

            def wait(self):
                waiting.set()
                release.wait(2)
                reaped.set()
                return 0

        process = FakeProcess()
        job_handle = object()
        identity = {
            "kind": "windows_job", "start_token": "captured-process-token",
            "job_name": "Local\\AutoResearch-47f5c3e7-76b4-4ae5-a16b-3391c3eefac1",
        }
        started = time.monotonic()
        duplicated = []
        # Isolate the simulated platform flag so pathlib keeps the host Path type.
        simulated_windows_os = SimpleNamespace(name="nt", environ=os.environ, pathsep=os.pathsep)
        with patch.object(common_commands, "os", simulated_windows_os), \
             patch.object(common_commands, "hidden_process_kwargs", return_value={"creationflags": 0}), \
             patch.object(common_commands.subprocess, "Popen", return_value=process) as popen, \
             patch("autoresearch_runtime.runtimeops.capture_worker_identity", return_value=(identity, job_handle)), \
             patch("autoresearch_runtime.runtimeops.duplicate_job_handle_to_process", create=True,
                   side_effect=lambda handle, pid: duplicated.append((handle, pid)) or 8765), \
             patch("autoresearch_runtime.runtimeops._windows_job_active_processes", return_value=0), \
             patch("autoresearch_runtime.runtimeops.wait_for_job_empty") as wait_job, \
             patch("autoresearch_runtime.runtimeops.close_worker_identity_resource") as close_job:
            launch = common_commands.spawn_operation_worker("47f5c3e7-76b4-4ae5-a16b-3391c3eefac1")
            elapsed = time.monotonic() - started
            self.assertEqual(launch["pid"], process.pid)
            self.assertEqual(launch["worker_identity"], identity)
            self.assertEqual(launch["worker_job_handle"], 8765)
            self.assertEqual(popen.call_args.kwargs["creationflags"] & 0x00000004, 0,
                             "the launcher lock gates dispatch without a suspended-process crash gap")
            self.assertEqual(popen.call_args.kwargs["creationflags"] & 0x00000008, 0x00000008,
                             "detached workers must not create a console host in their Job")
            self.assertEqual(popen.call_args.kwargs["creationflags"] & 0x00000200, 0,
                             "Job Object cancellation replaces console process groups")
            self.assertEqual(duplicated, [(job_handle, process.pid)])
            self.assertLess(elapsed, 0.5, "HTTP launch must not wait for worker completion")
            launch["resume"]()
            self.assertTrue(waiting.wait(1), "a background reaper should wait on the Popen handle")
            release.set()
            self.assertTrue(reaped.wait(1))
            wait_job.assert_called_once_with(job_handle)
            close_job.assert_called_once_with(job_handle)

    @unittest.skipUnless(os.name == "nt", "requires Windows process and Job Object APIs")
    def test_windows_suspended_child_is_bound_before_resume(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000 | 0x00000200 | 0x00000004,
        )
        identity, job_handle = runtimeops.capture_worker_identity(process, str(uuid.uuid4()))
        try:
            self.assertEqual(identity["kind"], "windows_job")
            self.assertTrue(runtimeops.resume_suspended_worker(process))
            self.assertEqual(process.wait(timeout=5), 0)
            runtimeops.wait_for_job_empty(job_handle)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            runtimeops.close_worker_identity_resource(job_handle)

    def _request(self, path: str, *, method: str = "GET", headers=None, body: bytes | None = None):
        request = Request(self.origin + path, data=body, method=method, headers=headers or {})
        try:
            response = urlopen(request, timeout=3)
            return response.status, response.headers, response.read()
        except HTTPError as error:
            return error.code, error.headers, error.read()

    def _seed_lab(self, name: str) -> tuple[str, Path]:
        """Create a SQLite lab with an accepted baseline through the store; a durable fixture for reads and refusals."""
        state_dir = Path(self.temp.name) / f"state-{name}"
        state_dir.mkdir()
        candidate = support.git_project(state_dir / "candidate", {"value.txt": "1"})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True,
                                         **support.hidden_process_kwargs()).strip()
        config = {
            "confirmed": True,
            "source_root": str(candidate),
            "state_dir": str(state_dir),
            "config_path": str(state_dir / "lab.json"),
            "goal": "test a durable API operation",
            "repeats": 1,
            "timeout_seconds": 30,
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "evaluator": ["python", "local-evaluator.py"],
            "budget": {"max_trials": 1, "max_cost_usd": 0},
            "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
            "_passport": {
                "confirmed": True, "goal": "test a durable API operation",
                "constraints": [], "limits": {"max_trials": 1},
                "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
            },
        }
        store = SQLiteStore(self.home)
        lab_id = store.create_lab(config, {"best_commit": commit, "current_epoch": 1, "inputs": [], "source": []},
                                  candidate)
        config_bytes = store.load_config_bytes(lab_id)
        (state_dir / "lab.json").write_bytes(config_bytes)
        state = store.load_state(lab_id)
        state.update({
            "best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
            "baseline_verified": True, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "inputs": [], "source": [],
        })
        store.save_state(lab_id, state)
        baseline_id = store.create_run(lab_id, "baseline", "baseline", 1, {})
        store.save_manifest(lab_id, baseline_id, {
            "id": "baseline", "status": "accepted", "runs": [{"status": "ok", "metrics": {"score": 1}}],
        })
        self._install_offline_harness_profile(str(lab_id))
        return str(lab_id), candidate

    def _start_operation(self, lab_id: str, command: str, payload: dict[str, object], key: str) -> dict[str, object]:
        """Reserve and launch a worker operation the way the CLI does; the HTTP write route no longer exists."""
        from autoresearch_runtime.commands import common as command_common
        from autoresearch_runtime.runtimeops import safe_operation_context
        from autoresearch_runtime.trial import validate_trial_request

        params: dict[str, object] = dict(payload)
        if command == "trial":
            config = load_config(lab_id)
            safe_operation_context(config)
            params = {"hypothesis": validate_trial_request(
                config, command_common.normalize_trial_hypothesis(payload["hypothesis"]))}
        params["confirmed"] = True
        return command_common.enqueue_and_start_operation(lab_id, command, params, key)

    def test_public_health_is_minimal_and_reads_do_not_create_sqlite(self):
        status, _, body = self._request("/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(set(data), {"ok", "root_id", "store"})
        self.assertEqual(data["store"], "absent")
        self.assertFalse((self.home / "autoresearch.sqlite3").exists())

    def test_v1_allows_local_calls_without_credentials(self):
        status, _, body = self._request("/api/v1/labs")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["items"], [])
        self.assertFalse((self.home / "autoresearch.sqlite3").exists())

        status, _, _ = self._request(
            "/api/v1/labs", headers={"Origin": "http://evil.example", "Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(status, 403)

        status, rev_headers, rev_body = self._request("/api/v1/revision")
        self.assertEqual(status, 200)
        rev_payload = json.loads(rev_body)
        etag = rev_headers.get("ETag")
        self.assertEqual(etag, f'"{rev_payload["data"]["revision"]}"')

        status_304, headers_304, body_304 = self._request(
            "/api/v1/revision", headers={"If-None-Match": etag}
        )
        self.assertEqual(status_304, 304)
        self.assertEqual(body_304, b"")
        self.assertEqual(headers_304.get("ETag"), etag)

    def test_bootstrap_route_is_removed(self):
        status, headers, body = self._request(
            "/api/v1/auth/bootstrap", method="POST", headers={"Origin": self.origin}, body=b"{}",
        )
        self.assertEqual(status, 405, body.decode("utf-8", errors="replace"))
        self.assertEqual(json.loads(body)["error"]["code"], "method_not_allowed")
        self.assertNotIn("Set-Cookie", headers)

    def test_local_api_write_may_omit_origin_and_is_refused_after_authentication(self):
        status, _, body = self._request(
            "/api/v1/labs", method="POST",
            headers={"Content-Type": "application/json"},
            body=b"{}",
        )
        self.assertEqual(status, 405)
        self.assertEqual(json.loads(body)["error"]["code"], "method_not_allowed")

    def test_local_api_rejects_wrong_host(self):
        status, _, _ = self._request(
            "/api/v1/labs", headers={"Host": "evil.example"}
        )
        self.assertEqual(status, 403)

    def test_every_write_method_on_v1_is_refused_with_the_stop_token_and_changes_nothing(self):
        lab_id, _candidate = self._seed_lab("refusal")
        store = SQLiteStore(self.home)
        revision_before = store.get_revision()
        lifecycle_before = store.get_lab(lab_id)["lifecycle"]
        headers = {"X-AutoResearch-Stop": "local-test-token", "Content-Type": "application/json",
                   "Idempotency-Key": "refused-write"}
        writes = (
            ("POST", f"/api/v1/labs/{lab_id}/lifecycle", b'{"action": "archive"}'),
            ("POST", f"/api/v1/labs/{lab_id}/commands",
             b'{"command": "baseline", "confirmed": true, "payload": {}}'),
            ("PUT", f"/api/v1/labs/{lab_id}", b"{}"),
            ("PATCH", f"/api/v1/labs/{lab_id}", b"{}"),
            ("DELETE", f"/api/v1/labs/{lab_id}", None),
        )
        for method, path_text, body in writes:
            with self.subTest(method=method, path=path_text):
                status, response_headers, response = self._request(path_text, method=method, headers=headers, body=body)
                self.assertEqual(status, 405, response.decode("utf-8", errors="replace"))
                self.assertEqual(json.loads(response)["error"]["code"], "method_not_allowed")
                self.assertEqual(response_headers.get("Allow"), "GET")
        self.assertEqual(store.get_lab(lab_id)["lifecycle"], lifecycle_before)
        self.assertEqual(store.get_revision(), revision_before)
        self.assertIsNone(store.get_latest_operation(lab_id))

    def test_job_evidence_compare_and_run_reads_are_served_from_the_store(self):
        lab_id, candidate = self._seed_lab("reads")
        store = SQLiteStore(self.home)
        status, _, body = self._request(
            f"/api/v1/labs/{lab_id}/guide", headers={"Authorization": "Bearer local-test-token"},
        )
        guide = json.loads(body)["data"]
        self.assertEqual(status, 200, body.decode())
        self.assertTrue(guide["next_steps"])
        guide_argv = guide["next_steps"][0]["argv"]
        self.assertTrue(Path(guide_argv[0]).is_absolute())
        self.assertTrue(Path(guide_argv[1]).is_absolute())
        self.assertEqual(guide_argv[2:4], ["--lab", str(lab_id)])
        self.assertNotIn("manifest.json", guide_argv)

        (candidate / "value.txt").write_text("2", encoding="utf-8")
        operation = store.enqueue_operation(lab_id, "trial", "trial-1", {"command": "trial", "params": {}})
        status, _, body = self._request(
            f"/api/v1/jobs/{operation['operation_id']}", headers={"Authorization": "Bearer local-test-token"},
        )
        job = json.loads(body)["data"]
        self.assertEqual(status, 200)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["command"], "trial")
        self.assertNotIn("payload", job)
        self.assertNotIn("idempotency_key", job)

        artifact = store.register_artifact(
            lab_id, "evidence.txt", b"registered evidence", evidence_key="result",
            media_type="text/plain", metadata={"note": "safe"},
        )
        status, _, body = self._request(
            f"/api/v1/labs/{lab_id}/evidence/{artifact['artifact_id']}",
            headers={"Authorization": "Bearer local-test-token"},
        )
        evidence = json.loads(body)["data"]
        self.assertEqual(status, 200)
        self.assertEqual(evidence["encoding"], "base64")
        self.assertEqual(evidence["content_base64"], "cmVnaXN0ZXJlZCBldmlkZW5jZQ==")
        self.assertNotIn("metadata", evidence)
        self.assertNotIn("object_path", evidence)

        left_id = store.create_run(lab_id, "baseline", "baseline", 1, payload={})
        right_id = store.create_run(lab_id, "trial", "trial-0001", 1, payload={})
        base_manifest = {
            "config_sha256": "cfg-hash", "input_hashes": {"brief": "brief-hash"},
            "evaluator": ["evaluator-fingerprint"], "pin_hashes": {"program": "pin-hash"},
            "epoch": 1, "status": "accepted", "runs": [
                {"status": "ok", "metrics": {"score": 0.4}},
                {"status": "ok", "metrics": {"score": 0.6}},
            ],
        }
        trial_manifest = {**base_manifest, "id": "trial-0001", "runs": [
            {"status": "ok", "metrics": {"score": 0.7}},
            {"status": "ok", "metrics": {"score": 0.8}},
        ]}
        store.save_manifest(lab_id, left_id, {**base_manifest, "id": "baseline"})
        store.save_manifest(lab_id, right_id, trial_manifest)
        with patch("autoresearch_runtime.config.load_config", return_value=SimpleNamespace(
            primary="score", direction="max", guards=[], repeats=2,
        )):
            status, _, body = self._request(
                f"/api/v1/labs/{lab_id}/runs/{left_id}",
                headers={"Authorization": "Bearer local-test-token"},
            )
            run = json.loads(body)["data"]
            self.assertEqual(status, 200)
            self.assertEqual(run["metrics"]["score"]["decision_worst"], 0.4)
            self.assertEqual(run["metrics"]["score"]["mean"], 0.5)
            self.assertEqual(run["metrics"]["score"]["completed_repeats"], 2)

            status, _, body = self._request(
                f"/api/v1/labs/{lab_id}/compare?left=baseline&right=trial-0001",
                headers={"Authorization": "Bearer local-test-token"},
            )
            comparison = json.loads(body)["data"]
            self.assertEqual(status, 200)
            self.assertTrue(comparison["comparable"])
            self.assertEqual(comparison["metrics"]["score"]["delta_mean"], 0.25)

        store.save_manifest(lab_id, right_id, {**trial_manifest, "pin_hashes": {"program": "other-pin"}})
        with patch("autoresearch_runtime.config.load_config", return_value=SimpleNamespace(
            primary="score", direction="max", guards=[], repeats=2,
        )):
            status, _, body = self._request(
                f"/api/v1/labs/{lab_id}/compare?left=baseline&right=trial-0001",
                headers={"Authorization": "Bearer local-test-token"},
            )
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["data"]["comparable"])

    def test_local_worker_runs_baseline_trial_and_resumes_saved_repeat_without_evaluation(self):
        project = support.git_project(Path(self.temp.name) / "local-project", {"score.txt": "1"})
        self.root.mkdir(parents=True, exist_ok=True)
        program = self.root / "program.md"
        program.write_text("Project: local-project\nGoal: improve score\n", encoding="utf-8")
        calls_path = self.root / "evaluation-calls.txt"
        evaluator = self.root / "evaluator.py"
        late_writer_trigger = self.root / "spawn-late-writer"
        late_writer_release = self.root / "release-late-writer"
        late_writer_started = self.root / "late-writer-started"
        late_writer_done = self.root / "late-writer-done"
        self.addCleanup(lambda: late_writer_release.touch(exist_ok=True))
        late_writer_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "score, release, started, done = map(Path, sys.argv[1:])\n"
            "started.write_text('ready')\n"
            "while not release.exists():\n"
            "    time.sleep(.01)\n"
            "score.write_text('999')\n"
            "done.write_text('written')\n"
        )
        evaluator.write_text(
            "import json, os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            f"Path({str(calls_path)!r}).open('a', encoding='utf-8').write('call\\n')\n"
            "time.sleep(1)\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "score = int((candidate / 'score.txt').read_text())\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': score}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n"
            f"if os.name == 'nt' and Path({str(late_writer_trigger)!r}).exists():\n"
            f"    subprocess.Popen([sys.executable, '-c', {late_writer_code!r}, str(candidate / 'score.txt'), "
            f"{str(late_writer_release)!r}, {str(late_writer_started)!r}, {str(late_writer_done)!r}], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
            encoding="utf-8",
        )
        known_case = self.root / "known-case.json"
        known_case.write_text(json.dumps({
            "schema": 1, "id": "api-local-score", "reproducible": True,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "input_hashes": {}, "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        spec = {
            "project": str(project),
            "evaluator": ["{python}", str(evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {"mode": "fresh", "known_case": {
                "path": str(known_case),
                "sha256": hashlib.sha256(known_case.read_bytes()).hexdigest(),
            }},
            "repeats": 1, "timeout_seconds": 20,
        }
        from autoresearch_runtime.bootstrap import prepare_lab_from_dict

        prepared = prepare_lab_from_dict(
            program, spec, confirmed=True,
            passport={"goal": "improve score", "constraints": ["local deterministic evaluator"]},
        )
        lab_id = str(prepared["lab_id"])
        self._install_offline_harness_profile(lab_id)

        def wait_job(operation_id: str) -> dict[str, object]:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status_code, _, response_body = self._request(
                    f"/api/v1/jobs/{operation_id}",
                    headers={"Authorization": "Bearer local-test-token"},
                )
                self.assertEqual(status_code, 200, response_body.decode())
                job = json.loads(response_body)["data"]
                if job["status"] in {"succeeded", "failed", "recovery_required"}:
                    return job
                time.sleep(0.05)
            self.fail(f"Job {operation_id} did not finish")

        # Start the read-only dashboard service, launch the local evaluator, then stop
        # the service while the evaluator is active. The worker must finish from its
        # durable operation without depending on the web process.
        from autoresearch_runtime.dash_process import start_dashboard, stop_dashboard

        start_dashboard(self.home, open_browser=False)
        self.addCleanup(lambda: stop_dashboard(self.home))
        baseline_job = self._start_operation(lab_id, "baseline", {}, "local-baseline")
        deadline = time.monotonic() + 10
        while not calls_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(calls_path.exists(), "local evaluator should start before stopping HTTP")
        stopped = stop_dashboard(self.home)
        self.assertTrue(stopped["stopped"], stopped)

        store = SQLiteStore(self.home)
        operation_deadline = time.monotonic() + 30
        while time.monotonic() < operation_deadline:
            baseline_result = store.get_operation(str(baseline_job["operation_id"]))
            if baseline_result["status"] in {"succeeded", "failed", "recovery_required"}:
                break
            time.sleep(0.05)
        self.assertEqual(baseline_result["status"], "succeeded", baseline_result)
        self.assertEqual(len(calls_path.read_text(encoding="utf-8").splitlines()), 1)

        lab = store.get_lab(lab_id)
        candidate = Path(lab["candidate_path"])
        (candidate / "score.txt").write_text("0", encoding="utf-8")
        hypothesis = {"title": "Lower score", "observation": "The local score was lowered",
                      "expected_effect": "This deterministic trial should be rejected"}
        if os.name == "nt":
            late_writer_trigger.touch()
        trial_job = self._start_operation(lab_id, "trial", {"hypothesis": hypothesis}, "local-trial")
        if os.name == "nt":
            deadline = time.monotonic() + 30
            while not late_writer_started.is_file() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(late_writer_started.is_file(), "evaluator child should be part of this regression")
            pending_deadline = time.monotonic() + 0.5
            while time.monotonic() < pending_deadline:
                pending = store.get_operation(str(trial_job["operation_id"]))
                candidate_before_release = (candidate / "score.txt").read_text(encoding="utf-8")
                if pending["status"] != "running" or candidate_before_release != "0":
                    break
                time.sleep(0.02)
            pending = store.get_operation(str(trial_job["operation_id"]))
            candidate_before_release = (candidate / "score.txt").read_text(encoding="utf-8")
            late_writer_release.touch()
            trial_result = wait_job(str(trial_job["operation_id"]))
            self.assertEqual(pending["status"], "running", pending)
            self.assertEqual(candidate_before_release, "0")
            self.assertTrue(late_writer_done.exists(), "the evaluator descendant must be released and finish")
        else:
            trial_result = wait_job(str(trial_job["operation_id"]))
        self.assertEqual(trial_result["status"], "succeeded", trial_result)
        self.assertEqual(len(calls_path.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual((candidate / "score.txt").read_text(encoding="utf-8"), "1")

        run_page = store.list_runs(lab_id, limit=20)
        self.assertEqual({item["alias"] for item in run_page["items"]}, {"baseline", "trial-0001"})
        status, _, comparison_body = self._request(
            f"/api/v1/labs/{lab_id}/compare?left=baseline&right=trial-0001",
            headers={"Authorization": "Bearer local-test-token"},
        )
        comparison = json.loads(comparison_body)["data"]
        self.assertEqual(status, 200)
        self.assertEqual(comparison["metrics"]["score"]["left"]["mean"], 1.0)
        self.assertEqual(comparison["metrics"]["score"]["right"]["mean"], 0.0)

        interrupted = store.get_run(lab_id, "trial-0001")["manifest"]
        interrupted["status"] = "interrupted"
        store.save_manifest(lab_id, "trial-0001", interrupted)
        before_resume = len(calls_path.read_text(encoding="utf-8").splitlines())
        resume_job = self._start_operation(lab_id, "resume", {"run_id": "trial-0001"}, "local-resume")
        resumed = wait_job(str(resume_job["operation_id"]))
        self.assertEqual(resumed["status"], "succeeded", resumed)
        self.assertEqual(len(calls_path.read_text(encoding="utf-8").splitlines()), before_resume)


if __name__ == "__main__":
    unittest.main()
