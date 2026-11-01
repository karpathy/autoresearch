try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from autoresearch_runtime.envelope import exit_code_for, fail
from autoresearch_runtime.errors import LabError
from autoresearch_runtime.locking import lab_lock

_SCRIPTS = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
_LAB = _SCRIPTS / "lab.py"


def _run_lab(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged["PYTHONDONTWRITEBYTECODE"] = "1"
    merged["PYTHONPATH"] = str(_SCRIPTS)
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, str(_LAB), *args],
        cwd=_SCRIPTS,
        env=merged,
        capture_output=True,
        text=True,
        encoding="utf-8",
        **support.hidden_process_kwargs(),
    )


class CliContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.root / "home")
        self.addCleanup(self._restore_home)

    def _restore_home(self):
        if self.previous_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.previous_home

    def test_envelope_shape_on_doctor(self):
        result = _run_lab(["doctor"])
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        for key in ("ok", "command", "data", "warnings", "next", "error"):
            self.assertIn(key, payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["command"], "doctor")
        self.assertIsInstance(payload["data"], dict)

    def test_e_locked_exit_code(self):
        self.assertEqual(exit_code_for("E_LOCKED"), 5)
        code = fail("trial", LabError("busy", code="E_LOCKED"))
        self.assertEqual(code, 5)

    def test_offline_profile_skips_missing_optional_protected_roots(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.durable_workflow_helpers import approved_offline_profile

        with patch("tests.durable_workflow_helpers.contract_fingerprint", return_value="0" * 64):
            with approved_offline_profile(SimpleNamespace()) as profile:
                self.assertTrue(profile.is_file())

    def test_offline_profile_skips_none_optional_protected_roots(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.durable_workflow_helpers import approved_offline_profile

        config = SimpleNamespace(source_root=None, state_dir=None)
        with patch("tests.durable_workflow_helpers.contract_fingerprint", return_value="0" * 64):
            with approved_offline_profile(config) as profile:
                self.assertTrue(profile.is_file())

    def test_lab_lock_blocks_second_holder(self):
        state_dir = self.root / "lock-lab"
        state_dir.mkdir()
        holder = threading.Event()
        errors: list[Exception] = []

        def hold() -> None:
            try:
                with lab_lock(state_dir):
                    holder.set()
                    time.sleep(2)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        self.assertTrue(holder.wait(5))
        with self.assertRaises(LabError) as ctx:
            with lab_lock(state_dir):
                pass
        self.assertEqual(getattr(ctx.exception, "code", None), "E_LOCKED")
        thread.join(timeout=5)

    def test_lab_lock_clears_dead_pid(self):
        state_dir = self.root / "dead-lock-lab"
        state_dir.mkdir()
        lock_file = state_dir / ".lock"
        # Write dead PID 999999 into lock file
        lock_file.write_text(json.dumps({"pid": 999999, "heartbeat": time.time()}), encoding="utf-8")
        with lab_lock(state_dir):
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            self.assertEqual(data["pid"], os.getpid())

    def test_lab_lock_clears_stale_lock(self):
        state_dir = self.root / "stale-lock-lab"
        state_dir.mkdir()
        lock_file = state_dir / ".lock"
        # Write alive PID but heartbeat older than 300s
        lock_file.write_text(json.dumps({"pid": os.getpid(), "heartbeat": time.time() - 400}), encoding="utf-8")
        with lab_lock(state_dir):
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            self.assertEqual(data["pid"], os.getpid())

    def test_guide_command_and_agent_flag(self):
        from autoresearch_runtime.bootstrap import prepare_lab
        from autoresearch_runtime.config import load_config
        from autoresearch_runtime.sqlite_store import SQLiteStore
        from tests import test_skill_workflow as workflow
        from tests.durable_workflow_helpers import approved_offline_profile

        fixture = workflow.SkillWorkflowTest(methodName="runTest")
        workflow.SkillWorkflowTest.setUp(fixture)
        self.addCleanup(fixture.doCleanups)
        prepared = prepare_lab(fixture.program, fixture.spec(), confirmed=True)
        config = load_config(Path(prepared["config"]))
        store = SQLiteStore()
        lab_id = prepared["lab_id"]

        # Run guide on initialized lab
        result = _run_lab(["--config", str(config.path), "guide", "--json"])
        self.assertEqual(result.returncode, 0, result.stderr)
        guide_data = json.loads(result.stdout)
        self.assertTrue(guide_data["ok"])
        self.assertEqual(guide_data["data"]["state"]["stage"], "SETUP_PENDING")

        # Run baseline with --agent flag
        with approved_offline_profile(config):
            result_base = _run_lab(["--config", str(config.path), "--agent", "--confirmed", "baseline"])
        self.assertEqual(result_base.returncode, 0, result_base.stderr)
        base_data = json.loads(result_base.stdout)
        self.assertTrue(base_data["ok"])
        self.assertIn("cockpit", base_data)
        self.assertEqual(base_data["cockpit"]["state"]["stage"], "ITERATING")
        operation = store.get_latest_operation(lab_id)
        self.assertIsNotNone(operation)
        self.assertEqual(operation["status"], "succeeded", operation)

    @unittest.skip("pending port from a legacy file lab to the SQLite flow")
    def test_agent_promote_conflict_reports_worktree_and_apply_error(self):
        from autoresearch_runtime.lab import load_state
        from tests import test_promote as promote_tests

        fixture = promote_tests.PromoteTest(methodName="runTest")
        promote_tests.PromoteTest.setUp(fixture)
        self.addCleanup(fixture.doCleanups)
        config = fixture._ready_lab()
        # Upstream edits the promoted file, so the patch cannot apply to the recorded head.
        (fixture.source / "value.txt").write_bytes(b"upstream")
        fixture._commit_source("upstream edit to the promoted file")
        state = load_state(config)
        state["source_head"] = fixture._git_out("rev-parse", "HEAD").strip()
        (config.state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

        result = _run_lab(["--config", str(fixture.config_path), "--agent", "promote", "--to-branch", "fix-branch"])

        self.assertEqual(result.returncode, 0, result.stderr)
        envelope = json.loads(result.stdout)
        self.assertIn("cockpit", envelope)
        data = envelope["data"]
        self.assertFalse(data["applied"], data)
        self.assertEqual(data["conflicts"], ["value.txt"])
        self.assertTrue(data["apply_error"], data)
        self.assertTrue(Path(data["worktree"]).is_dir(), data)


if __name__ == "__main__":
    unittest.main()
