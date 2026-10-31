try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from autoresearch_runtime.errors import LabError
from autoresearch_runtime.locking import lab_lock


class AdvisoryLockTest(unittest.TestCase):
    def test_live_owner_cannot_be_replaced_by_stale_heartbeat(self):
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            ready = state_dir / "ready"
            script = (
                "import sys, time; from pathlib import Path; "
                "from autoresearch_runtime.locking import lab_lock; "
                "state, ready = Path(sys.argv[1]), Path(sys.argv[2]); "
                "ctx = lab_lock(state); ctx.__enter__(); "
                "ready.write_text('ready', encoding='utf-8'); time.sleep(60)"
            )
            env = os.environ.copy()
            scripts = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "autoresearch" / "scripts"
            env["PYTHONPATH"] = str(scripts)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(state_dir), str(ready)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **support.hidden_process_kwargs(),
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail(f"lock holder exited early: {stdout}\n{stderr}")
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), "child did not acquire the lab lock")

                (state_dir / ".lock").write_text(
                    json.dumps({"pid": process.pid, "heartbeat": time.time() - 600}),
                    encoding="utf-8",
                )
                with self.assertRaises(LabError) as ctx:
                    with lab_lock(state_dir):
                        pass
                self.assertEqual(ctx.exception.code, "E_LOCKED")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=10)

            # The OS releases the advisory lock when the former owner exits.
            with lab_lock(state_dir):
                self.assertTrue((state_dir / ".lock.guard").is_file())

    def test_advisory_lock_inode_persists_across_acquisitions(self):
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            guard = state_dir / ".lock.guard"
            with lab_lock(state_dir):
                self.assertTrue(guard.is_file(), "the advisory lock must use a persistent guard file")
                first_inode = guard.stat().st_ino
            with lab_lock(state_dir):
                second_inode = guard.stat().st_ino

            self.assertTrue(guard.is_file())
            self.assertEqual(first_inode, second_inode)

    def test_exit_preserves_metadata_owned_by_another_token(self):
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            metadata = state_dir / ".lock"
            replacement = {"pid": 123, "heartbeat": 456.0, "owner_token": "another-owner"}
            with lab_lock(state_dir):
                metadata.write_text(json.dumps(replacement), encoding="utf-8")

            self.assertTrue(metadata.is_file(), "cleanup must leave metadata owned by another token")
            self.assertEqual(json.loads(metadata.read_text(encoding="utf-8")), replacement)


if __name__ == "__main__":
    unittest.main()
