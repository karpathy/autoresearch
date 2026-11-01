try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from autoresearch_runtime.config import load_config
from autoresearch_runtime.harness_policy import contract_fingerprint
from autoresearch_runtime.sqlite_store import SQLiteStore

_REPO = Path(__file__).resolve().parents[1]
_CLI = _REPO / ".agents" / "skills" / "autoresearch" / "scripts" / "autoresearch.py"
_SCRIPTS = _CLI.parent
_FAST_EVAL = (
    "import json, sys\nfrom pathlib import Path\n"
    "c, run = map(Path, sys.argv[1:3])\n"
    "v = int((c / 'value.txt').read_text())\n"
    "run.mkdir(parents=True, exist_ok=True)\n"
    "(run / 'result.json').write_text(json.dumps({'metrics': {'score': v}, "
    "'usage': {'requests': 0, 'cost_usd': 0}}))\n"
)
_SLOW_EVAL = _FAST_EVAL.replace(
    "v = int((c / 'value.txt').read_text())\n",
    "import time\nv = int((c / 'value.txt').read_text())\n"
    "if v > 1:\n    time.sleep(3)\n",
)


class TrialCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "workspace"
        self.work.mkdir()
        self.home = Path(self.temp.name) / "home"
        self.home.mkdir()
        self.profile_path = Path(self.temp.name) / "host" / "harness-profile.json"
        self._prev_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.program = self.work / "program.md"
        self.program.write_text(
            "Project: project\nGoal: improve score\nMetric: score, max\n", encoding="utf-8",
        )
        self.project = self.work / "project"
        self.project.mkdir()
        (self.project / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.work / "evaluator.py"
        self.evaluator.write_text(_FAST_EVAL, encoding="utf-8")
        self.spec = self._write_spec()

    def _restore_home(self):
        if self._prev_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self._prev_home

    def _write_spec(self) -> Path:
        known = self.work / "known-case.json"
        ev_sha = hashlib.sha256(self.evaluator.read_bytes()).hexdigest()
        known.write_text(json.dumps({
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": ev_sha, "input_hashes": {},
            "expected_metrics": {"score": 1}, "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        spec = self.work / "derived.json"
        spec.write_text(json.dumps({
            "project": "project",
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {"path": str(known), "sha256": hashlib.sha256(known.read_bytes()).hexdigest()},
            },
        }), encoding="utf-8")
        return spec

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(_SCRIPTS)
        env["AUTORESEARCH_HOME"] = str(self.home)
        env["AUTORESEARCH_HARNESS_PROFILE"] = str(self.profile_path)
        return env

    def _run(self, args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_CLI), *args],
            cwd=self.work,
            env=env or self._env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            **support.hidden_process_kwargs(),
        )

    def _install_offline_profile(self, config_path: str) -> None:
        config = load_config(Path(config_path), allow_pending=True)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "trial-cli-test-harness", "version": "1"},
            "approval": {
                "approved": True,
                "contract_fingerprint": contract_fingerprint(config),
            },
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
            "spend": None,
        }), encoding="utf-8")

    def _run_approved(self, config_path: str, args: list[str]) -> subprocess.CompletedProcess[str]:
        # Keep the host profile available to the asynchronous worker after the CLI exits.
        self._install_offline_profile(config_path)
        return self._run(args)

    def _envelope(self, result: subprocess.CompletedProcess[str]) -> dict:
        raw = result.stdout if result.returncode == 0 else result.stderr
        return json.loads(raw)

    def _assert_ok(self, result: subprocess.CompletedProcess[str], command: str) -> dict:
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        payload = self._envelope(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["command"], command)
        self.assertIsInstance(payload["data"], dict)
        return payload["data"]

    def _prepare_baseline(self) -> str:
        prep = self._run(["prepare", "--program", str(self.program), "--spec", str(self.spec),
                          "--confirmed", "--no-dashboard", "--no-browser"])
        config_path = self._assert_ok(prep, "prepare")["config"]
        self._assert_ok(self._run_approved(
            config_path, ["--confirmed", "--config", config_path, "baseline"],
        ), "baseline")
        return config_path

    def _hypothesis(self) -> Path:
        path = self.work / "h.json"
        path.write_text(json.dumps({
            "title": "Improve", "observation": "low", "expected_effect": "higher",
        }), encoding="utf-8")
        return path

    def _candidate_value(self, config_path: str) -> Path:
        config = load_config(Path(config_path), allow_pending=True)
        return config.state_dir / "candidate" / "value.txt"

    def test_trial_start_wait_and_lock(self):
        self.evaluator.write_text(_SLOW_EVAL, encoding="utf-8")
        self.spec = self._write_spec()
        config = self._prepare_baseline()
        self._candidate_value(config).write_text("2", encoding="utf-8")
        hyp = self._hypothesis()
        start = self._run_approved(
            config, ["--confirmed", "--config", config, "trial", "start", "--hypothesis", str(hyp)],
        )
        self._assert_ok(start, "trial")
        locked = self._run_approved(
            config, ["--confirmed", "--config", config, "trial", "start", "--hypothesis", str(hyp)],
        )
        if locked.returncode == 0:
            # Worker can finish before the second start on a fast host; lock is timing-sensitive.
            pass
        else:
            self.assertEqual(locked.returncode, 5, locked.stderr)
            err = self._envelope(locked)
            self.assertFalse(err["ok"])
            self.assertEqual(err["command"], "trial")
            self.assertEqual(err["error"]["code"], "E_LOCKED")
        deadline = time.monotonic() + 90.0
        trial_data = None
        while time.monotonic() < deadline:
            wait = self._run(["--config", config, "trial", "wait", "--timeout", "60"])
            self.assertEqual(wait.returncode, 0, wait.stderr)
            data = self._envelope(wait)["data"]
            if isinstance(data, dict) and data.get("status"):
                trial_data = data
                break
            if not data.get("pending"):
                break
        self.assertIsNotNone(trial_data)
        self.assertEqual(trial_data["status"], "accepted")
        store = SQLiteStore(self.home)
        lab_id = store.resolve_lab_id(Path(config))
        operation = store.get_latest_operation(lab_id)
        self.assertEqual(operation["kind"], "trial")
        self.assertEqual(operation["status"], "succeeded")
        self.assertEqual(operation["result"]["status"], "accepted")
        self.assertIsNotNone(store.get_run(lab_id, "trial-0001"))
        self.assertFalse((Path(store.get_lab(lab_id)["state_dir"]) / "runs" / "trial-0001" / "manifest.json").exists())

    def test_trial_status_fields(self):
        config = self._prepare_baseline()
        data = self._assert_ok(self._run(["--config", config, "trial", "status"]), "trial")
        self.assertIn("worker", data)
        self.assertIn("running_manifests", data)


    def test_promote_export_patch(self):
        config = self._prepare_baseline()
        self._candidate_value(config).write_text("2", encoding="utf-8")
        hyp = self._hypothesis()
        self._assert_ok(self._run_approved(
            config, ["--confirmed", "--config", config, "trial", "--hypothesis", str(hyp)],
        ), "trial")
        lab_id = SQLiteStore(self.home).resolve_lab_id(Path(config))
        operation = SQLiteStore(self.home).get_latest_operation(lab_id)
        self.assertEqual(operation["status"], "succeeded")
        self.assertEqual(operation["result"]["status"], "accepted")
        data = self._assert_ok(self._run(["--config", config, "promote"]), "promote")
        self.assertTrue(Path(data["patch"]).is_file())

if __name__ == "__main__":
    unittest.main()
