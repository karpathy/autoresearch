"""The harness smoke driver exits 2 for a missing agent binary, before any workspace or agent starts."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "tests" / "harness_smoke" / "run.py"


class MissingBinaryExitsTwo(unittest.TestCase):
    def test_missing_binary_exits_2(self):
        env = dict(os.environ, PYTHONPATH=str(REPO))
        missing = str(REPO / ".tests" / "no-such-agent-binary")
        proc = subprocess.run(
            [sys.executable, "-B", str(DRIVER), "codex", "--binary", missing],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)


if __name__ == "__main__":
    unittest.main()
