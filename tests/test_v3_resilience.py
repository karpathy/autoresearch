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
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / ".agents" / "skills" / "autoresearch" / "scripts"
PYTHON = sys.executable


class TestV3Resilience(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(SCRIPTS_DIR)
        self.env["AUTORESEARCH_HOME"] = str(self.root / ".labs")
        self.cli = [PYTHON, str(SCRIPTS_DIR / "autoresearch.py")]

    def tearDown(self):
        self.temp.cleanup()

    def run_cmd(self, args: list[str], cwd: Path | None = None) -> tuple[int, dict, str]:
        res = subprocess.run(
            args,
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=self.env,
            **support.hidden_process_kwargs(),
        )
        try:
            data = json.loads(res.stdout) if res.stdout else {}
        except Exception:
            data = {"raw_stdout": res.stdout}
        return res.returncode, data, res.stderr

    def _create_project(self) -> tuple[Path, Path, Path]:
        project = self.root / "proj"
        project.mkdir()
        subprocess.run(["git", "init"], cwd=project, capture_output=True, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=project, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=project, check=True, **support.hidden_process_kwargs())

        (project / "main.py").write_text("SCORE = 5\n", encoding="utf-8")
        evaluator = project / "eval.py"
        evaluator_called = self.root / "evaluator-called.txt"
        evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            f"marker = Path({str(evaluator_called)!r})\n"
            "count = int(marker.read_text()) if marker.exists() else 0\n"
            "marker.write_text(str(count + 1), encoding='utf-8')\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "sys.path.insert(0, str(candidate))\n"
            "import main\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({"
            "'metrics': {'score': float(main.SCORE)}, 'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "add", "."], cwd=project, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "commit", "-m", "init"], cwd=project, check=True, **support.hidden_process_kwargs())
        return project, evaluator, evaluator_called

    def test_auto_prepare_fails_closed_without_side_effects(self):
        proj, eval_file, evaluator_called = self._create_project()

        # --auto no longer runs an evaluator during preparation.
        eval_cmd = f'"{PYTHON}" "{eval_file}" {{candidate}} {{run_dir}}'
        rc, out, err = self.run_cmd(self.cli + [
            "prepare",
            "--project", str(proj),
            "--auto",
            "--eval", eval_cmd,
            "--metric", "score:max",
            "--brief-text", "Goal: optimize score\n",
        ])
        self.assertEqual(rc, 2, f"Auto-prepare unexpectedly succeeded: {out}\nSTDERR: {err}")
        failure = json.loads(err)
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error"]["code"], "E_AUTO_MIGRATION_REQUIRED")
        self.assertFalse(evaluator_called.exists())
        self.assertFalse((self.root / ".labs" / "autoresearch.sqlite3").exists())
        self.assertFalse((self.root / ".labs" / "projects").exists())

    def test_confirmed_prepare_step_trial_and_promote(self):
        from autoresearch_runtime.config import load_config
        from autoresearch_runtime.harness_policy import contract_fingerprint
        from unittest.mock import patch

        project, evaluator, evaluator_called = self._create_project()
        program = self.root / "program.md"
        program.write_text("Project: proj\nGoal: optimize score\nMetric: score, max\n", encoding="utf-8")

        known_case = self.root / "known-case.json"
        known_case.write_text(json.dumps({
            "schema": 1,
            "id": "v3-local-score",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 5},
            "observed_metrics": {"score": 5},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps({
            "project": str(project),
            "evaluator": ["{python}", str(evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(known_case),
                    "sha256": hashlib.sha256(known_case.read_bytes()).hexdigest(),
                },
            },
            "exclude_globs": ["eval.py"],
        }), encoding="utf-8")

        rc, out, err = self.run_cmd(self.cli + [
            "prepare", "--project", str(project), "--program", str(program),
            "--spec", str(spec_path), "--confirmed",
        ])
        self.assertEqual(rc, 0, f"Confirmed prepare failed: {out}\nSTDERR: {err}")
        config_path = out["data"]["config"]
        with patch.dict(os.environ, {"AUTORESEARCH_HOME": self.env["AUTORESEARCH_HOME"]}):
            config = load_config(Path(config_path), allow_pending=True)
            fingerprint = contract_fingerprint(config)
        profile_path = self.root / "host" / "harness-profile.json"
        profile_path.parent.mkdir()
        profile_path.write_text(json.dumps({
            "schema_version": 1,
            "harness": {"id": "v3-test-harness", "version": "1"},
            "approval": {
                "approved": True,
                "contract_fingerprint": fingerprint,
            },
            "sandbox": {"enabled": True, "scope": "workspace"},
            "network": {"mode": "disabled"},
            "spend": None,
        }), encoding="utf-8")
        self.env["AUTORESEARCH_HARNESS_PROFILE"] = str(profile_path)
        state_dir = Path(out["data"]["candidate"]).parent

        rc, out, err = self.run_cmd(self.cli + ["--config", config_path, "step"])
        self.assertEqual(rc, 0, f"Initial step failed: {out}\nSTDERR: {err}")
        self.assertIn(out["data"]["stage"], ("SETUP_PENDING", "BASELINE_NEEDED"))
        self.assertEqual(out["data"]["recommended_action"]["action"], "RUN_BASELINE")

        rc, out, err = self.run_cmd(self.cli + [
            "--confirmed", "--config", config_path, "step", "--run",
        ])
        self.assertEqual(rc, 0, f"Step baseline failed: {out}\nSTDERR: {err}")
        self.assertEqual(out["data"]["action_taken"], "baseline")

        (state_dir / "candidate" / "main.py").write_text("SCORE = 10\n", encoding="utf-8")
        rc, out, err = self.run_cmd(self.cli + ["--config", config_path, "step"])
        self.assertEqual(rc, 0, f"Trial recommendation failed: {out}\nSTDERR: {err}")
        self.assertEqual(out["data"]["stage"], "ITERATING")
        self.assertEqual(out["data"]["recommended_action"]["action"], "RUN_TRIAL")

        hypothesis_path = self.root / "hypothesis.json"
        hypothesis_path.write_text(json.dumps({
            "title": "Raise the candidate score from 5 to 10",
            "observation": "The edited main.py sets SCORE to 10 while the baseline scored 5.",
            "expected_effect": "The evaluator should report a score of 10.",
        }), encoding="utf-8")
        rc, out, err = self.run_cmd(self.cli + [
            "--confirmed", "--config", config_path, "trial", "--hypothesis", str(hypothesis_path),
        ])
        self.assertEqual(rc, 0, f"Trial failed: {out}\nSTDERR: {err}")
        self.assertEqual(out["data"]["status"], "accepted")
        self.assertEqual(out["data"]["title"], "Raise the candidate score from 5 to 10")

        rc, out, err = self.run_cmd(self.cli + ["--config", config_path, "promote", "--in-place"])
        self.assertEqual(rc, 0, f"Promote failed: {out}\nSTDERR: {err}")
        self.assertTrue(out["data"]["applied"])
        self.assertIn("SCORE = 10", (project / "main.py").read_text(encoding="utf-8"))
        self.assertTrue(evaluator_called.exists())


if __name__ == "__main__":
    unittest.main()
