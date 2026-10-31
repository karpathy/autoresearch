"""Public CLI flow for a host-attested offline harness profile."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import


_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / ".agents" / "skills" / "autoresearch" / "scripts"
_PREPARE = _SCRIPTS / "prepare.py"
_LAB = _SCRIPTS / "lab.py"


class HarnessPublicFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "measured-project"
        self.project.mkdir()
        self.home = self.root / "lab-home"
        self.host = self.root / "host"
        self.host.mkdir()
        self.marker = self.root / "evaluator-runs.txt"
        self.profile_env_marker = self.root / "profile-env.txt"
        self.program = self.root / "program.md"
        self.program.write_text(
            "Project: measured-project\nGoal: improve the offline score\nMetric: score, max\n",
            encoding="utf-8",
        )
        (self.project / "score.txt").write_text("1\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            f"Path({str(self.profile_env_marker)!r}).write_text('present' if 'AUTORESEARCH_HARNESS_PROFILE' in os.environ else 'absent')\n"
            "score = float((candidate / 'score.txt').read_text(encoding='utf-8'))\n"
            "(run_dir / 'result.json').write_text(json.dumps({"
            "'metrics': {'score': score}, 'usage': {'requests': 0, 'cost_usd': 0}}))\n"
            f"with Path({str(self.marker)!r}).open('a', encoding='utf-8') as marker:\n"
            "    marker.write('evaluated\\n')\n",
            encoding="utf-8",
        )
        self.known_case = self.root / "known-case.json"
        self.known_case.write_text(json.dumps({
            "schema": 1,
            "id": "offline-score-v1",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        self.spec = self.root / "derived-spec.json"
        self.spec.write_text(json.dumps({
            "project": str(self.project),
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 1,
            "timeout_seconds": 30,
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(self.known_case),
                    "sha256": hashlib.sha256(self.known_case.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")

        self._old_profile = os.environ.get("AUTORESEARCH_HARNESS_PROFILE")
        os.environ.pop("AUTORESEARCH_HARNESS_PROFILE", None)
        self.addCleanup(self._restore_profile_env)
        self.env_patch = patch.dict(os.environ, {
            "AUTORESEARCH_HOME": str(self.home),
            "AUTORESEARCH_NO_DASHBOARD": "1",
            "AUTORESEARCH_NO_BROWSER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(_SCRIPTS),
        })
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _restore_profile_env(self):
        if self._old_profile is None:
            os.environ.pop("AUTORESEARCH_HARNESS_PROFILE", None)
        else:
            os.environ["AUTORESEARCH_HARNESS_PROFILE"] = self._old_profile

    def _run(self, script: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=_SCRIPTS,
            capture_output=True,
            text=True,
            encoding="utf-8",
            **support.hidden_process_kwargs(),
        )

    def _json(self, result: subprocess.CompletedProcess[str]) -> dict:
        output = result.stdout.strip() or result.stderr.strip()
        try:
            return json.loads(output)
        except ValueError as error:
            self.fail(f"CLI did not return JSON (exit {result.returncode}): {output!r}; {error}")

    def _run_lab(self, config: str, *args: str) -> subprocess.CompletedProcess[str]:
        return self._run(_LAB, ["--config", config, "--headless", *args])

    def test_offline_profile_binds_to_public_guide_and_preserves_cli_results(self):
        prepared_result = self._run(_PREPARE, [
            "--program", str(self.program), "--spec", str(self.spec), "--confirmed",
            "--no-dashboard", "--no-browser",
        ])
        prepared = self._json(prepared_result)
        self.assertEqual(prepared_result.returncode, 0, prepared)
        self.assertTrue(prepared["ok"], prepared)
        self.assertEqual(prepared["command"], "prepare")
        lab = prepared["data"]
        self.assertTrue(lab["created"])
        self.assertTrue(lab["lab_id"])
        config = lab["config"]

        guide_result = self._run_lab(config, "guide", "--json")
        guide = self._json(guide_result)
        self.assertEqual(guide_result.returncode, 0, guide)
        self.assertTrue(guide["ok"], guide)
        fingerprint = guide["data"]["contract_fingerprint"]
        self.assertRegex(fingerprint, r"^[a-f0-9]{64}$")

        from autoresearch_runtime.sqlite_store import SQLiteStore

        store = SQLiteStore(self.home)
        lab_id = lab["lab_id"]
        self.assertIsNone(store.get_latest_operation(lab_id))

        # Missing caller confirmation and missing host profile both stop before enqueue/evaluation.
        missing_confirmation = self._run_lab(config, "baseline")
        missing_confirmation_data = self._json(missing_confirmation)
        self.assertNotEqual(missing_confirmation.returncode, 0)
        self.assertFalse(missing_confirmation_data["ok"])
        self.assertIn("confirmed=true", missing_confirmation_data["error"]["message"])
        self.assertIsNone(store.get_latest_operation(lab_id))
        self.assertFalse(self.marker.exists())

        missing_profile = self._run_lab(config, "--confirmed", "baseline")
        missing_profile_data = self._json(missing_profile)
        self.assertNotEqual(missing_profile.returncode, 0)
        self.assertFalse(missing_profile_data["ok"])
        self.assertEqual(missing_profile_data["error"]["code"], "E_HARNESS_PROFILE_REQUIRED")
        self.assertIsNone(store.get_latest_operation(lab_id))
        self.assertFalse(self.marker.exists())

        profile_path = self.host / "harness-profile.json"
        self.assertFalse(profile_path.is_relative_to(self.project))
        self.assertFalse(profile_path.is_relative_to(self.home))
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "offline-test-harness", "version": "1"},
            "approval": {"approved": True, "contract_fingerprint": fingerprint},
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
            "spend": None,
        }), encoding="utf-8")

        with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(profile_path)}):
            baseline_result = self._run_lab(config, "--confirmed", "baseline")
        baseline = self._json(baseline_result)
        self.assertEqual(baseline_result.returncode, 0, {
            "response": baseline,
            "operation": store.get_latest_operation(lab_id),
        })
        self.assertTrue(baseline["ok"], baseline)
        self.assertEqual(baseline["command"], "baseline")
        baseline_manifest = baseline["data"]
        self.assertEqual(baseline_manifest["id"], "baseline")
        self.assertEqual(baseline_manifest["status"], "accepted")
        self.assertEqual(len(baseline_manifest["runs"]), 1)
        self.assertTrue(self.marker.is_file())
        self.assertEqual(self.profile_env_marker.read_text(encoding="utf-8"), "absent")

        candidate = Path(lab["candidate"])
        (candidate / "score.txt").write_text("2\n", encoding="utf-8")
        with patch.dict(os.environ, {"AUTORESEARCH_HARNESS_PROFILE": str(profile_path)}):
            trial_result = self._run_lab(config, "--confirmed", "trial",
                                         "--title", "Raise the offline score",
                                         "--observation", "The candidate score can increase",
                                         "--expected-effect", "Score increases from one to two")
        trial = self._json(trial_result)
        self.assertEqual(trial_result.returncode, 0, {
            "response": trial,
            "operation": store.get_latest_operation(lab_id),
        })
        self.assertTrue(trial["ok"], trial)
        self.assertEqual(trial["command"], "trial")
        trial_manifest = trial["data"]
        self.assertEqual(trial_manifest["id"], "trial-0001")
        self.assertEqual(trial_manifest["title"], "Raise the offline score")
        self.assertEqual(trial_manifest["status"], "accepted")
        self.assertEqual(len(trial_manifest["runs"]), 1)
        self.assertEqual(self.profile_env_marker.read_text(encoding="utf-8"), "absent")

        history_guide_result = self._run_lab(config, "guide", "--json", "--run", "trial-0001")
        history_guide = self._json(history_guide_result)
        self.assertEqual(history_guide_result.returncode, 0, history_guide)
        self.assertTrue(history_guide["ok"], history_guide)
        self.assertEqual(history_guide["data"]["run_detail"]["id"], "trial-0001")
        self.assertIn("trial-0001", {item["id"] for item in history_guide["data"]["evidence"]})
        self.assertEqual(history_guide["data"]["contract_fingerprint"], fingerprint)

        export_path = self.host / "lab-export.json"
        export_result = self._run(_LAB, ["--headless", "storage", "export", "--lab", lab_id,
                                         "--output", str(export_path)])
        exported = self._json(export_result)
        self.assertEqual(export_result.returncode, 0, exported)
        self.assertTrue(exported["ok"], exported)
        self.assertEqual(exported["data"]["run_count"], 2)
        snapshot = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["format"], "autoresearch-sqlite-export-v1")
        self.assertIsInstance(snapshot["history"], list)
        exported_aliases = {item["alias"] for item in snapshot["runs"]}
        self.assertTrue({"baseline", "trial-0001"}.issubset(exported_aliases))

        global_export_path = self.host / "lab-export-global.json"
        global_export_result = self._run(_LAB, ["--headless", "--lab", lab_id, "storage", "export",
                                                "--output", str(global_export_path)])
        global_exported = self._json(global_export_result)
        self.assertEqual(global_export_result.returncode, 0, global_exported)
        self.assertEqual(global_exported["data"]["run_count"], 2)

        conflict_path = self.host / "should-not-exist.json"
        conflicting_export = self._run(_LAB, [
            "--headless", "--config", config, "--lab", lab_id,
            "storage", "export", "--output", str(conflict_path),
        ])
        conflicting_payload = self._json(conflicting_export)
        self.assertEqual(conflicting_export.returncode, 2, conflicting_payload)
        self.assertEqual(conflicting_payload["error"]["code"], "E_USAGE_SELECTOR")
        self.assertFalse(conflict_path.exists())


if __name__ == "__main__":
    unittest.main()
