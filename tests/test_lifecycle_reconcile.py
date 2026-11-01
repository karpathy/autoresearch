from __future__ import annotations

import io
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime import api_v1, cli
from autoresearch_runtime import runtimeops
from autoresearch_runtime.commands import common as common_commands
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.storage_ops import reconcile_storage
from tests.durable_workflow_helpers import approved_offline_profile


def _lab_config(project: Path, state_dir: Path) -> dict:
    return {
        "confirmed": True,
        "source_root": str(project),
        "state_dir": str(state_dir),
        "config_path": str(state_dir / "lab.json"),
        "goal": "test lifecycle reconciliation",
        "repeats": 1,
        "timeout_seconds": 30,
        "metrics": {"primary": "score", "direction": "max", "guards": []},
        "evaluator": ["python", "local-evaluator.py"],
        "budget": {"max_trials": 1, "max_cost_usd": 0},
        "pins": {"program": {"path": str(project / "program.md"), "sha256": "a" * 64}},
        "_passport": {
            "confirmed": True,
            "goal": "test lifecycle reconciliation",
            "constraints": [],
            "limits": {"max_trials": 1},
            "pins": {"program": {"path": str(project / "program.md"), "sha256": "a" * 64}},
        },
    }


class LifecycleReconcileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "program.md").write_text("goal\n", encoding="utf-8")
        self.state_dir = self.root / "lab"
        self.old_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.state_dir.mkdir()
        self.candidate = support.git_project(self.state_dir / "candidate", {"value.txt": "1"})
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.candidate, text=True,
            **support.hidden_process_kwargs(),
        ).strip()
        self.config_path = self.state_dir / "lab.json"
        config_bytes = (json.dumps(_lab_config(self.project, self.state_dir), indent=2) + "\n").encode()
        self.config_path.write_bytes(config_bytes)
        self.store = SQLiteStore(self.home)
        self.lab_id = self.store.create_lab(
            _lab_config(self.project, self.state_dir),
            {"best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
             "baseline_verified": True, "current_epoch": 1, "setup_pending": False,
             "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "inputs": [], "source": []},
            self.candidate,
        )
        canonical_config = self.store.load_config_bytes(self.lab_id)
        self.config_path.write_bytes(canonical_config)
        recovered_state = self.store.load_state(self.lab_id)
        recovered_state["config_sha256"] = hashlib.sha256(canonical_config).hexdigest()
        self.store.save_state(self.lab_id, recovered_state)

    def _restore_home(self):
        if self.old_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.old_home

    def test_cancel_request_is_durable_and_does_not_store_secrets(self):
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cancel-me",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        request_cancel = getattr(self.store, "request_operation_cancel", None)
        self.assertTrue(callable(request_cancel), "SQLiteStore must persist a cancellation request")

        request_cancel(operation["operation_id"])

        recorded = self.store.get_operation(operation["operation_id"])
        self.assertTrue(recorded["progress"]["cancel_requested"])
        self.assertNotIn("secret", json.dumps(recorded["progress"]).casefold())
        self.assertEqual(recorded["status"], "queued")

    def test_cancel_a_queued_operation_interrupts_its_run_and_restores_the_candidate(self):
        candidate_file = self.candidate / "value.txt"
        candidate_file.write_text("uncommitted", encoding="utf-8")
        run_id = self.store.create_run(self.lab_id, "trial", "trial-0001", 1, {})
        self.store.save_manifest(self.lab_id, run_id, {"id": "trial-0001", "status": "running", "runs": []})
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "api-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )

        runtimeops.cancel_run(str(self.lab_id), "trial-0001")

        recorded = self.store.get_operation(operation["operation_id"])
        self.assertEqual(recorded["status"], "failed")
        self.assertEqual(recorded["error"]["code"], "operation_cancelled")
        run = self.store.get_run(self.lab_id, run_id)
        self.assertEqual(run["status"], "interrupted")
        self.assertEqual(run["manifest"]["reason"], "Operation cancelled by request")
        self.assertEqual(run["manifest"]["cancelled_operation_id"], str(operation["operation_id"]))
        self.assertEqual(candidate_file.read_text(encoding="utf-8"), "1")

    def test_cancel_cli_command_is_registered(self):
        run_id = self.store.create_run(self.lab_id, "trial", "trial-0001", 1, {})
        self.store.save_manifest(self.lab_id, run_id, {"id": "trial-0001", "status": "running", "runs": []})
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cli-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "unused"}}},
        )
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = cli.main(["--lab", str(self.lab_id), "cancel", "trial-0001"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["data"]["status"], "failed")
        self.assertEqual(payload["data"]["error_code"], "operation_cancelled")
        self.assertEqual(payload["data"]["run_id"], "trial-0001")

    def test_cancel_refuses_a_terminal_run_without_touching_other_active_operation(self):
        run_id = self.store.create_run(self.lab_id, "trial", "trial-done", 1, {})
        self.store.save_manifest(self.lab_id, run_id, {"id": "trial-done", "status": "accepted", "runs": []})
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "do-not-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )

        with self.assertRaises(LabError) as raised:
            runtimeops.cancel_run(str(self.lab_id), "trial-done")
        self.assertEqual(raised.exception.code, "operation_not_cancellable")

        unchanged = self.store.get_operation(operation["operation_id"])
        self.assertEqual(unchanged["status"], "queued")
        self.assertIsNone(unchanged["progress"])

    def test_cancel_with_missing_worker_identity_stays_pending_without_signaling_pid(self):
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cancel-no-identity",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        self.store.record_operation_spawn(operation["operation_id"], 23456)
        self.store.claim_operation(operation["operation_id"], 23456)

        with patch.object(runtimeops, "_pid_alive", return_value=True), \
             patch.object(runtimeops, "_terminate_worker_tree") as terminate, \
             patch.object(runtimeops, "_wait_worker_exit") as wait:
            pending = runtimeops.cancel_operation(operation["operation_id"])

        terminate.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(pending["status"], "running")
        self.assertTrue(pending["progress"]["cancel_requested"])
        self.assertEqual(pending["progress"]["cancel_pending_reason"], "worker_identity_unavailable")

    def test_pending_cancel_is_consumed_by_worker_before_dispatch(self):
        from autoresearch_runtime.worker import run_operation_worker

        (self.candidate / "value.txt").write_text("uncommitted", encoding="utf-8")
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cancel-pending-worker-check",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        self.store.record_operation_spawn(
            operation["operation_id"], 23456,
            worker_identity={
                "kind": "linux_proc", "start_token": "boot:stale",
                "process_group": 23456, "session_id": 23456,
            },
        )
        with patch.object(runtimeops, "_pid_alive", return_value=True), \
             patch.object(runtimeops, "_worker_identity_matches", return_value=False), \
             patch.object(runtimeops, "_terminate_worker_tree") as terminate:
            pending = runtimeops.cancel_operation(operation["operation_id"])

        terminate.assert_not_called()
        self.assertEqual(pending["status"], "queued")
        self.assertEqual(pending["progress"]["cancel_pending_reason"], "worker_identity_mismatch")
        with patch("autoresearch_runtime.worker._dispatch_operation") as dispatch:
            run_operation_worker(str(operation["operation_id"]))

        dispatch.assert_not_called()
        finished = self.store.get_operation(operation["operation_id"])
        self.assertEqual(finished["status"], "failed")
        self.assertEqual(finished["error"]["code"], "operation_cancelled")
        self.assertEqual(finished["progress"]["cancel_pending_reason"], "worker_identity_mismatch")
        self.assertEqual((self.candidate / "value.txt").read_text(encoding="utf-8"), "1")

    def test_cancel_with_mismatched_worker_identity_never_signals_reused_pid(self):
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cancel-mismatched-identity",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        self.store.record_operation_spawn(
            operation["operation_id"], 23456,
            worker_identity={
                "kind": "linux_proc", "start_token": "boot:stale",
                "process_group": 23456, "session_id": 23456,
            },
        )
        self.store.claim_operation(operation["operation_id"], 23456)

        with patch.object(runtimeops, "_pid_alive", return_value=True), \
             patch.object(runtimeops, "_worker_identity_matches", return_value=False), \
             patch.object(runtimeops, "_worker_tree_stopped", return_value=False), \
             patch.object(runtimeops, "_terminate_worker_tree") as terminate, \
             patch.object(runtimeops, "_wait_worker_exit") as wait:
            pending = runtimeops.cancel_operation(operation["operation_id"])

        terminate.assert_not_called()
        wait.assert_not_called()
        self.assertEqual(pending["status"], "running")
        self.assertTrue(pending["progress"]["cancel_requested"])
        self.assertEqual(pending["progress"]["cancel_pending_reason"], "worker_identity_mismatch")

    def test_termination_rechecks_identity_at_the_signal_boundary(self):
        identity = {
            "kind": "linux_proc", "start_token": "boot:captured",
            "process_group": 23456, "session_id": 23456,
        }
        with patch.object(runtimeops.os, "name", "posix"), \
             patch.object(runtimeops, "_pid_alive", return_value=True), \
             patch.object(runtimeops, "_worker_identity_matches", return_value=False), \
             patch.object(runtimeops.os, "killpg", create=True) as kill_group:
            self.assertFalse(runtimeops._terminate_worker_tree(23456, identity))

        kill_group.assert_not_called()

    def test_reaper_keeps_job_handle_until_late_child_exits(self):
        class FakeProcess:
            exited = False

            def wait(self):
                self.exited = True
                return 0

        process = FakeProcess()
        job_handle = object()
        identity = {"kind": "windows_job", "job_name": "Local\\AutoResearch-test"}
        active_processes = iter([1, 0])
        closed = []
        queries = []

        def query(*, job_handle):
            self.assertIs(job_handle, job_handle_ref)
            self.assertEqual(closed, [], "the Job handle must stay open while a child is active")
            queries.append(job_handle)
            return next(active_processes)

        job_handle_ref = job_handle
        terminated = []
        with patch("autoresearch_runtime.runtimeops._windows_job_active_processes", side_effect=query), \
             patch("autoresearch_runtime.runtimeops._terminate_windows_job_handle",
                   side_effect=lambda handle: terminated.append(handle) or True), \
             patch("autoresearch_runtime.runtimeops.close_worker_identity_resource",
                   side_effect=lambda handle: closed.append(handle)):
            common_commands._reap_worker(process, job_handle, identity)

        self.assertTrue(process.exited)
        self.assertEqual(len(queries), 2, "the reaper must verify that the orphan group is empty")
        self.assertEqual(terminated, [job_handle], "a worker exit must stop any remaining Job members")
        self.assertEqual(closed, [job_handle])

    def test_reaper_keeps_job_handle_when_orphan_termination_is_uncertain(self):
        class FakeProcess:
            def wait(self):
                return 1

        job_handle = object()
        with patch("autoresearch_runtime.runtimeops._windows_job_active_processes", return_value=None), \
             patch("autoresearch_runtime.runtimeops._terminate_windows_job_handle", return_value=False), \
             patch("autoresearch_runtime.runtimeops.close_worker_identity_resource") as close_job:
            common_commands._reap_worker(
                FakeProcess(), job_handle, {"kind": "windows_job", "job_name": "Local\\AutoResearch-test"},
            )

        close_job.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires Windows Job Object APIs")
    def test_missing_job_query_is_unknown_and_cancel_stays_pending(self):
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "cancel-missing-job",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        identity = {
            "kind": "windows_job", "start_token": "captured-process-token",
            "job_name": f"Local\\AutoResearch-{operation['operation_id']}",
        }
        self.store.record_operation_spawn(operation["operation_id"], 23456, worker_identity=identity)
        self.store.claim_operation(operation["operation_id"], 23456)
        kernel = SimpleNamespace(OpenJobObjectW=__import__("unittest.mock", fromlist=["Mock"]).Mock(return_value=None))

        with patch.object(runtimeops, "_windows_kernel32", return_value=kernel), \
             patch.object(runtimeops.ctypes, "get_last_error", return_value=2), \
             patch.object(runtimeops, "_pid_alive", return_value=False), \
             patch.object(runtimeops, "_terminate_worker_tree") as terminate:
            self.assertIsNone(runtimeops._windows_job_active_processes(identity))
            self.assertFalse(runtimeops._worker_tree_stopped(23456, identity))
            pending = runtimeops.cancel_operation(operation["operation_id"])

        terminate.assert_not_called()
        self.assertEqual(pending["status"], "running")
        self.assertTrue(pending["progress"]["cancel_requested"])
        self.assertEqual(pending["progress"]["cancel_pending_reason"], "worker_tree_unavailable")

    def test_operation_identity_is_persisted_before_launcher_resumes_worker(self):
        from autoresearch_runtime.config import load_config

        identity = {
            "kind": "linux_proc", "start_token": "boot:captured",
            "process_group": 24680, "session_id": 24680,
        }
        observed = []

        def launcher(_operation_id):
            def resume():
                operation = self.store.get_active_operation(self.lab_id)
                observed.append(operation["progress"].get("worker_identity"))

            return {"pid": 24680, "worker_identity": identity, "resume": resume}

        config = load_config(str(self.lab_id), allow_pending=True)
        with approved_offline_profile(config):
            operation = runtimeops.reserve_and_launch_operation(
                config, "recover", "identity-before-resume",
                {"command": "recover", "params": {"confirmed": True}}, launcher,
            )

        self.assertEqual(observed, [identity])
        self.assertEqual(self.store.get_operation(operation["operation_id"])["progress"]["worker_identity"], identity)
        self.assertFalse(operation.get("progress") and "worker_identity" in operation["progress"])

    def test_api_job_view_exposes_cancel_pending_reason_but_not_process_identity(self):
        view = api_v1._job_view({
            "operation_id": "job-1", "lab_id": "lab-1", "kind": "trial", "status": "running", "progress": {
                "cancel_requested": True,
                "cancel_pending_reason": "worker_identity_mismatch",
                "worker_identity": {"start_token": "private-process-token"},
                "worker_job_handle": 4660,
            },
        })

        self.assertEqual(view["progress"]["cancel_pending_reason"], "worker_identity_mismatch")
        serialized = json.dumps(view)
        self.assertNotIn('"worker_identity":', serialized)
        self.assertNotIn("private-process-token", serialized)
        self.assertNotIn("worker_job_handle", serialized)
        self.assertNotIn("4660", serialized)

    @unittest.skipUnless(os.name == "nt", "requires Windows process and Job Object APIs")
    def test_windows_worker_fails_closed_without_its_duplicated_job_handle(self):
        from autoresearch_runtime import worker as worker_module

        operation = self.store.enqueue_operation(
            self.lab_id, "recover", "missing-worker-job-handle",
            {"command": "recover", "params": {}},
        )
        identity = {
            "kind": "windows_job", "start_token": "test-token",
            "job_name": f"Local\\AutoResearch-{operation['operation_id']}",
        }
        self.store.record_operation_spawn(
            operation["operation_id"], os.getpid() + 100000, worker_identity=identity,
        )

        with patch.object(runtimeops, "_worker_identity_matches", return_value=True), \
             patch.object(runtimeops, "_worker_job_handle_matches_current_process", return_value=False), \
             patch.object(worker_module, "_dispatch_operation", return_value={"ok": True}) as dispatch:
            worker_module.run_operation_worker(str(operation["operation_id"]))

        dispatch.assert_not_called()
        recorded = self.store.get_operation(operation["operation_id"])
        self.assertEqual(recorded["status"], "failed")
        self.assertEqual(recorded["error"]["code"], "worker_identity_unavailable")

    @unittest.skipUnless(os.name == "nt", "requires Windows Job Object APIs")
    def test_windows_worker_waits_for_job_descendant_before_succeeding(self):
        from autoresearch_runtime import worker as worker_module
        from autoresearch_runtime.config import load_config

        config = load_config(str(self.lab_id), allow_pending=True)
        identity = {
            "kind": "windows_job", "start_token": "test-token",
        }
        with approved_offline_profile(config):
            operation = runtimeops.reserve_operation(
                config, "recover", "worker-orphan-before-terminal",
                {"command": "recover", "params": {"confirmed": True}},
            )
            identity["job_name"] = f"Local\\AutoResearch-{operation['operation_id']}"
            self.store.record_operation_spawn(
                operation["operation_id"], os.getpid(), worker_identity=identity, worker_job_handle=4660,
            )

            samples = [2] * 25 + [1]
            with patch.object(runtimeops, "_windows_job_active_processes",
                               side_effect=lambda **_kwargs: samples.pop(0) if samples else 1), \
                 patch.object(worker_module, "_dispatch_operation", return_value={"completed": True}) as dispatch:
                worker_module.run_operation_worker(str(operation["operation_id"]))

        dispatch.assert_called_once()
        recorded = self.store.get_operation(operation["operation_id"])
        self.assertEqual(recorded["status"], "succeeded")

    @unittest.skipUnless(os.name == "nt", "requires Windows process and Job Object APIs")
    def test_windows_job_handle_survives_launcher_exit_and_new_cli_cancels_worker(self):
        run_id = self.store.create_run(self.lab_id, "trial", "trial-win-cancel", 1, {})
        self.store.save_manifest(self.lab_id, run_id, {
            "id": "trial-win-cancel", "status": "running", "runs": [],
        })
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "windows-job-cancel",
            {"command": "trial", "params": {"run_id": str(run_id), "hypothesis": {"title": "safe"}}},
        )
        scripts = str(Path(common_commands.__file__).resolve().parents[2])
        env = os.environ.copy()
        env["AUTORESEARCH_HOME"] = str(self.home)
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (scripts, env.get("PYTHONPATH", "")) if item
        )
        launcher_code = """
import subprocess, sys
from autoresearch_runtime.runtimeops import capture_worker_identity, close_worker_identity_resource, duplicate_job_handle_to_process
from autoresearch_runtime.sqlite_store import SQLiteStore
worker = None
job = None
retained = False
try:
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=0x08000000 | 0x00000200)
    identity, job = capture_worker_identity(worker, sys.argv[1])
    remote = duplicate_job_handle_to_process(job, worker.pid)
    if not identity or not remote:
        raise RuntimeError('could not bind worker Job handle')
    SQLiteStore().record_operation_spawn(sys.argv[1], worker.pid,
        worker_identity=identity, worker_job_handle=remote)
    retained = True
finally:
    if not retained and worker is not None:
        if worker.poll() is None:
            worker.kill()
        worker.wait()
    if job:
        close_worker_identity_resource(job)
"""
        launched = subprocess.run(
            [sys.executable, "-c", launcher_code, str(operation["operation_id"])],
            cwd=scripts, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
            **support.hidden_process_kwargs(),
        )
        self.assertEqual(launched.returncode, 0, launched.stderr)
        spawned = self.store.get_operation(operation["operation_id"])
        identity = spawned["progress"]["worker_identity"]
        worker_pid = spawned["pid"]
        self.assertGreater(runtimeops._windows_job_active_processes(identity) or 0, 0)

        # The launcher has exited; a separate CLI process must still open and cancel its named Job.
        args = ["--lab", str(self.lab_id), "cancel", "trial-win-cancel"]
        code = f"from autoresearch_runtime.cli import main; raise SystemExit(main({args!r}))"
        cancelled = subprocess.run(
            [sys.executable, "-c", code], cwd=scripts, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10, **support.hidden_process_kwargs(),
        )
        self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
        payload = json.loads(cancelled.stdout)
        self.assertTrue(payload["ok"], cancelled.stdout)
        self.assertEqual(payload["data"]["status"], "failed")
        self.assertEqual(payload["data"]["error_code"], "operation_cancelled")
        self.assertFalse(runtimeops._pid_alive(worker_pid))

    @unittest.skipUnless(os.name == "nt", "requires Windows Job Object APIs")
    def test_windows_last_worker_job_handle_kills_orphan_child(self):
        child_pid_file = self.root / "orphan-child.pid"
        parent_code = (
            "import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='ascii')"
        )
        options = support.hidden_process_kwargs()
        options["creationflags"] |= 0x00000200 | 0x00000004
        worker = subprocess.Popen(
            [sys.executable, "-c", parent_code, str(child_pid_file)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **options,
        )
        identity, launcher_job_handle = runtimeops.capture_worker_identity(worker, str(uuid.uuid4()))
        worker_job_handle = None
        try:
            worker_job_handle = runtimeops.duplicate_job_handle_to_process(launcher_job_handle, worker.pid)
            self.assertIsInstance(worker_job_handle, int)
            runtimeops.close_worker_identity_resource(launcher_job_handle)
            launcher_job_handle = None
            self.assertTrue(runtimeops.resume_suspended_worker(worker))
            self.assertEqual(worker.wait(timeout=5), 0)

            deadline = time.monotonic() + 5
            while not child_pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(child_pid_file.exists(), "the helper worker must create its child")
            child_pid = int(child_pid_file.read_text(encoding="ascii"))
            deadline = time.monotonic() + 5
            while runtimeops._pid_alive(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(runtimeops._pid_alive(child_pid), "closing the last Job handle must kill an orphan")
        finally:
            if launcher_job_handle:
                runtimeops.close_worker_identity_resource(launcher_job_handle)
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)

    def test_trial_validation_rejects_an_empty_candidate_before_enqueue(self):
        from autoresearch_runtime.commands.common import normalize_trial_hypothesis
        from autoresearch_runtime.config import load_config
        from autoresearch_runtime.trial import validate_trial_request

        baseline_id = self.store.create_run(self.lab_id, "baseline", "baseline", 1, {})
        self.store.save_manifest(self.lab_id, baseline_id, {"id": "baseline", "status": "accepted", "runs": []})
        state = self.store.load_state(self.lab_id)
        state.update({"baseline_id": "baseline", "best_id": "baseline", "baseline_verified": True})
        self.store.save_projection(self.lab_id, "state", state)
        hypothesis = normalize_trial_hypothesis({
            "title": "No change", "observation": "Candidate is unchanged",
            "expected_effect": "Score improves",
        })

        with self.assertRaises(LabError) as rejected:
            validate_trial_request(load_config(str(self.lab_id)), hypothesis)
        self.assertIn("No candidate change", str(rejected.exception))
        with self.store._reader() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM operations WHERE lab_id=?", (str(self.lab_id),)).fetchone()[0], 0)

    def test_operation_contract_passes_safe_configuration_without_evaluator_args(self):
        from autoresearch_runtime.config import load_config

        config = load_config(str(self.lab_id), allow_pending=True)
        profile_config = config
        config = replace(profile_config, evaluator=("python", "local-evaluator.py", "--api-key", "secret-value"))
        # The operation context is derived from safe config fields and must never copy argv values.
        with approved_offline_profile(profile_config):
            operation = runtimeops.reserve_operation(
                config, "recover", "safe-context",
                {"command": "recover", "params": {"confirmed": True}},
            )
        payload = self.store.get_operation(operation["operation_id"])["payload"]
        self.assertIn("operation_context", payload["params"])
        self.assertNotIn("secret-value", json.dumps(payload))
        self.assertNotIn("evaluator", payload["params"]["operation_context"])

    def test_running_cancel_recovers_lab_and_preserves_completed_repeat(self):
        run_id = self.store.create_run(self.lab_id, "trial", "trial-0001", 1, {})
        result = {"metrics": {"score": 0.8}, "usage": {"requests": 2, "cost_usd": 0.25}, "status": "ok"}
        self.store.append_result(self.lab_id, run_id, 1, result)
        self.store.ensure_repeat_started(self.lab_id, run_id, 2, started_at="2026-10-01T00:00:00Z")
        self.store.save_manifest(self.lab_id, run_id, {"id": "trial-0001", "status": "running", "runs": [result]})
        (self.candidate / "value.txt").write_text("uncommitted", encoding="utf-8")
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "running-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )
        identity = {
            "kind": "linux_proc", "start_token": "boot:captured",
            "process_group": 23456, "session_id": 23456,
        }
        self.store.record_operation_spawn(operation["operation_id"], 23456, worker_identity=identity)
        self.store.claim_operation(operation["operation_id"], 23456)
        cancel = getattr(runtimeops, "cancel_operation", None)
        self.assertTrue(callable(cancel), "runtimeops must implement worker cancellation and recovery")

        with patch.object(runtimeops, "_pid_alive", return_value=True), \
             patch.object(runtimeops, "_worker_identity_matches", return_value=True), \
             patch.object(runtimeops, "_terminate_worker_tree") as terminate, \
             patch.object(runtimeops, "_wait_worker_exit", return_value=True):
            cancelled = cancel(operation["operation_id"])

            terminate.assert_called_once_with(23456, identity, job_handle=None)
        self.assertEqual(cancelled["status"], "failed")
        self.assertEqual(cancelled["error"]["code"], "operation_cancelled")
        cancelled_run = self.store.get_run(self.lab_id, run_id)
        self.assertEqual(cancelled_run["status"], "interrupted")
        self.assertEqual(cancelled_run["manifest"]["reason"], "Operation cancelled by request")
        self.assertEqual(self.store.load_result(self.lab_id, run_id, 1)["metrics"], {"score": 0.8})
        self.assertEqual(self.store.load_result(self.lab_id, run_id, 1)["usage"], {"requests": 2, "cost_usd": 0.25})
        self.assertIsNone(self.store.load_result(self.lab_id, run_id, 2))
        with self.store._reader() as conn:
            repeat = conn.execute(
                "SELECT status,finished_at FROM repeats WHERE lab_id=? AND run_id=? AND repeat_index=2",
                (str(self.lab_id), str(run_id)),
            ).fetchone()
        self.assertEqual(repeat["status"], "interrupted")
        self.assertIsNotNone(repeat["finished_at"])
        self.assertEqual((self.candidate / "value.txt").read_text(encoding="utf-8"), "1")

    def test_cancel_closes_a_created_run_before_its_first_manifest(self):
        run_id = self.store.create_run(self.lab_id, "trial", "trial-early", 1, {})
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "early-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "safe"}}},
        )

        cancelled = runtimeops.cancel_run(str(self.lab_id), "trial-early")

        self.assertEqual(cancelled["operation_id"], operation["operation_id"])
        self.assertEqual(cancelled["status"], "failed")
        run = self.store.get_run(self.lab_id, run_id)
        self.assertEqual(run["status"], "interrupted")
        self.assertEqual(run["manifest"]["reason"], "Operation cancelled by request")

    def test_worker_observes_durable_cancel_before_dispatch_and_restores_candidate(self):
        from autoresearch_runtime.worker import run_operation_worker

        (self.candidate / "value.txt").write_text("uncommitted", encoding="utf-8")
        operation = self.store.enqueue_operation(
            self.lab_id, "trial", "pre-dispatch-cancel",
            {"command": "trial", "params": {"hypothesis": {"title": "unused"}}},
        )
        self.store.request_operation_cancel(operation["operation_id"])

        with patch("autoresearch_runtime.worker._dispatch_operation") as dispatch:
            self.assertIsNone(run_operation_worker(str(operation["operation_id"])))

        dispatch.assert_not_called()
        recorded = self.store.get_operation(operation["operation_id"])
        self.assertEqual(recorded["status"], "failed")
        self.assertEqual(recorded["error"]["code"], "operation_cancelled")
        self.assertEqual((self.candidate / "value.txt").read_text(encoding="utf-8"), "1")

    def test_recover_lab_closes_sqlite_run_and_open_repeats_but_keeps_completed_results(self):
        from autoresearch_runtime.config import load_config
        from autoresearch_runtime.trial import recover_lab

        run_id = self.store.create_run(self.lab_id, "trial", "trial-0001", 1, {})
        completed = {"metrics": {"score": 0.8}, "usage": {"requests": 2, "cost_usd": 0.25}, "status": "ok"}
        self.store.append_result(self.lab_id, run_id, 1, completed)
        self.store.ensure_repeat_started(self.lab_id, run_id, 2, started_at="2026-10-01T00:00:00Z")
        self.store.save_manifest(self.lab_id, run_id, {"id": "trial-0001", "status": "running", "runs": [completed]})

        interrupted = recover_lab(load_config(str(self.lab_id), allow_pending=True))

        self.assertEqual(interrupted, ["trial-0001"])
        self.assertEqual(self.store.get_run(self.lab_id, run_id)["status"], "interrupted")
        self.assertEqual(self.store.load_result(self.lab_id, run_id, 1)["metrics"], {"score": 0.8})
        self.assertIsNone(self.store.load_result(self.lab_id, run_id, 2))
        with self.store._reader() as conn:
            rows = conn.execute(
                "SELECT repeat_index,status,finished_at,cost_usd,requests,result_json FROM repeats "
                "WHERE lab_id=? AND run_id=? ORDER BY repeat_index",
                (str(self.lab_id), str(run_id)),
            ).fetchall()
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual((rows[0]["cost_usd"], rows[0]["requests"]), (0.25, 2))
        self.assertEqual(rows[1]["status"], "interrupted")
        self.assertTrue(rows[1]["finished_at"])
        self.assertEqual((rows[1]["cost_usd"], rows[1]["requests"], rows[1]["result_json"]),
                         (None, None, "{}"))

    def test_reconcile_preview_and_apply_fix_run_and_stale_repeat_without_guessing_data(self):
        run_id = self.store.create_run(self.lab_id, "baseline", "baseline", 1, {})
        self.store.ensure_repeat_started(self.lab_id, run_id, 1, started_at="2026-10-01T00:00:00Z")
        manifest = {"id": "baseline", "status": "interrupted", "runs": []}
        self.store.save_manifest(self.lab_id, run_id, manifest)
        with self.store._writer() as conn:
            conn.execute("UPDATE runs SET status='running' WHERE lab_id=? AND run_id=?", (str(self.lab_id), str(run_id)))
        with self.store._reader() as conn:
            before_metrics = conn.execute("SELECT count(*) FROM metric_values WHERE lab_id=?", (str(self.lab_id),)).fetchone()[0]
            before_attempts = conn.execute("SELECT count(*) FROM attempts WHERE lab_id=?", (str(self.lab_id),)).fetchone()[0]

        preview = reconcile_storage(home=self.home, lab=str(self.lab_id), apply=False)
        self.assertFalse(preview["applied"])
        self.assertTrue(any(item.get("status") == "repairable" for item in preview["items"]))
        findings = preview["items"][0]["findings"]
        self.assertEqual({item["kind"] for item in findings}, {"run_status", "repeat_status"})
        with self.store._reader() as conn:
            self.assertEqual(conn.execute("SELECT status FROM runs WHERE lab_id=? AND run_id=?", (str(self.lab_id), str(run_id))).fetchone()[0], "running")
            repeat = conn.execute("SELECT status,finished_at,cost_usd,requests FROM repeats WHERE lab_id=? AND run_id=?", (str(self.lab_id), str(run_id))).fetchone()
            self.assertEqual(tuple(repeat), ("running", None, None, None))

        applied = reconcile_storage(home=self.home, lab=str(self.lab_id), apply=True)
        self.assertTrue(applied["applied"])
        with self.store._reader() as conn:
            self.assertEqual(conn.execute("SELECT status FROM runs WHERE lab_id=? AND run_id=?", (str(self.lab_id), str(run_id))).fetchone()[0], "interrupted")
            repeat = conn.execute("SELECT status,finished_at,cost_usd,requests,result_json FROM repeats WHERE lab_id=? AND run_id=?", (str(self.lab_id), str(run_id))).fetchone()
            self.assertEqual(repeat[0], "interrupted")
            self.assertTrue(repeat[1])
            self.assertIsNone(repeat[2])
            self.assertIsNone(repeat[3])
            self.assertEqual(repeat[4], "{}")
            self.assertEqual(conn.execute("SELECT count(*) FROM metric_values WHERE lab_id=?", (str(self.lab_id),)).fetchone()[0], before_metrics)
            self.assertEqual(conn.execute("SELECT count(*) FROM attempts WHERE lab_id=?", (str(self.lab_id),)).fetchone()[0], before_attempts)

    def test_storage_reconcile_command_is_wired_to_preview(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = cli.main(["storage", "reconcile", "--lab", str(self.lab_id)])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["data"]["applied"] is False)
        self.assertEqual(payload["data"]["items"][0]["status"], "clean")

    def test_agent_guide_emits_one_compact_canonical_guide(self):
        guide = {"state": {"stage": "ITERATING"}, "next_steps": [{"command": "trial"}]}
        output = io.StringIO()
        with patch.object(cli, "run_command", return_value=guide), \
             patch("autoresearch_runtime.commands.common.resolve_config", return_value=object()), \
             patch("autoresearch_runtime.state.load_state", return_value={}), \
             patch("autoresearch_runtime.cockpit.build_cockpit", return_value=guide), \
             patch("sys.stdout", output):
            code = cli.main(["--agent", "guide", "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["data"], guide)
        self.assertNotIn("cockpit", payload)


if __name__ == "__main__":
    unittest.main()
