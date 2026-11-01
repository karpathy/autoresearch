"""The harness smoke driver exits 2 before any workspace or agent starts: for a missing agent binary,
and for a codex run without --model (codex has no default model)."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "tests" / "harness_smoke" / "run.py"


def run_driver(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(
        [sys.executable, "-B", str(DRIVER), *args],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
    )


class MissingBinaryExitsTwo(unittest.TestCase):
    def test_missing_binary_exits_2(self):
        missing = str(REPO / ".tests" / "no-such-agent-binary")
        proc = run_driver("codex", "--model", "test-model", "--binary", missing)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("agent binary not found", proc.stderr)


class CodexNeedsModel(unittest.TestCase):
    def test_codex_without_model_exits_2_before_launch(self):
        missing = str(REPO / ".tests" / "no-such-agent-binary")
        proc = run_driver("codex", "--binary", missing)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("--model", proc.stderr)
        # The model check runs first: the binary lookup never happens, so nothing is launched.
        self.assertNotIn("agent binary not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
