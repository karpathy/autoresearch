"""The offline toy example runs through the public SQLite flow and accepts its improving trial."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import


_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "examples" / "toy" / "run_demo.py"


class ToyExampleTest(unittest.TestCase):
    def test_demo_accepts_improving_trial_without_user_profile_path(self):
        result = subprocess.run(
            [sys.executable, "-B", str(_DEMO)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            **support.hidden_process_kwargs(),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        baseline = re.search(r"^baseline\s+mean_error=([0-9.]+)", result.stdout, re.MULTILINE)
        trial = re.search(r"^trial\s+mean_error=([0-9.]+)", result.stdout, re.MULTILINE)
        decision = re.search(r"^decision\s+(\w+)", result.stdout, re.MULTILINE)
        self.assertIsNotNone(baseline, result.stdout)
        self.assertIsNotNone(trial, result.stdout)
        self.assertIsNotNone(decision, result.stdout)
        self.assertEqual(decision.group(1), "accepted", result.stdout)
        self.assertLess(float(trial.group(1)), float(baseline.group(1)), result.stdout)
        self.assertNotIn(str(Path.home()), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()


class FailedBaselineCliRetryTest(unittest.TestCase):
    def test_failed_baseline_is_retried_through_cli_after_recover_and_keeps_both_attempts(self):
        spec = importlib.util.spec_from_file_location("toy_run_demo", _DEMO)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            host = base / "host"
            host.mkdir()
            project = base / "toy"
            shutil.copytree(demo.TOY, project)
            flag = base / "fail-once.flag"
            evaluator = project / "evaluator.py"
            evaluator.write_text(
                "from pathlib import Path\n"
                f"flag = Path({str(flag)!r})\n"
                "if not flag.exists():\n"
                "    flag.write_text('done')\n"
                "    raise SystemExit(1)\n" + evaluator.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            demo._pin_known_case(project)
            lab = demo.Lab(project, base / "lab-home", host, base)
            prepared = lab.run(demo.PREPARE, [
                "--program", str(project / "brief.md"), "--spec", str(project / "spec.json"),
                "--confirmed", "--no-dashboard", "--no-browser",
            ])
            config = str(prepared["config"])
            fingerprint = lab.lab(config, "guide", "--json")["contract_fingerprint"]
            profile = host / "harness-profile.json"
            profile.write_text(json.dumps({
                "schema_version": 1, "harness": {"id": "toy-offline-test", "version": "1"},
                "approval": {"approved": True, "contract_fingerprint": fingerprint},
                "sandbox": {"enabled": True, "scope": "workspace"}, "network": {"mode": "disabled"},
            }), encoding="utf-8")
            lab.env["AUTORESEARCH_HARNESS_PROFILE"] = str(profile)

            with self.assertRaisesRegex(demo.DemoError, "outcome"):
                lab.lab(config, "--confirmed", "baseline")
            # A failed operation is held for reconciliation until `recover` is run.
            with self.assertRaisesRegex(demo.DemoError, "recover"):
                lab.lab(config, "--confirmed", "baseline")
            lab.lab(config, "--confirmed", "recover")
            baseline = lab.lab(config, "--confirmed", "baseline")

            self.assertEqual(baseline["id"], "baseline-0002")
            self.assertEqual(baseline["status"], "accepted")
            status = lab.lab(config, "status")
            self.assertEqual(status["baseline_id"], "baseline-0002")
            self.assertEqual([(item["id"], item["status"]) for item in status["runs"]],
                             [("baseline", "crash"), ("baseline-0002", "accepted")])
            self.assertGreater(status["spent"]["seconds"], baseline["runs"][0]["duration_seconds"])
