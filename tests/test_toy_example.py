"""The offline toy example runs through the public SQLite flow and accepts its improving trial."""

from __future__ import annotations

import re
import subprocess
import sys
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
