from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.sqlite_store import SQLiteStore


class StorageOpsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "program.md").write_text("goal\n", encoding="utf-8")
        self.state_dir = self.root / "lab"
        self.candidate = support.git_project(self.state_dir / "candidate", {"value.txt": "1"})
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.candidate, text=True,
            **support.hidden_process_kwargs(),
        ).strip()
        pin = {"program": {"path": str(self.project / "program.md"), "sha256": "a" * 64}}
        config = {
            "confirmed": True,
            "source_root": str(self.project),
            "state_dir": str(self.state_dir),
            "config_path": str(self.state_dir / "lab.json"),
            "goal": "test storage operations",
            "repeats": 1,
            "timeout_seconds": 30,
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "evaluator": ["python", "local-evaluator.py"],
            "budget": {"max_trials": 1, "max_cost_usd": 0},
            "pins": pin,
        }
        state = {"best_commit": commit, "current_epoch": 1, "source": [], "inputs": []}
        self.store = SQLiteStore(self.home)
        self.lab_id = self.store.create_lab(config, state, self.candidate)

    def _storage_ops(self):
        spec = importlib.util.find_spec("autoresearch_runtime.storage_ops")
        self.assertIsNotNone(spec, "storage_ops module has not been extracted yet")
        return importlib.import_module("autoresearch_runtime.storage_ops")

    def test_export_preserves_an_existing_migration_binding(self):
        storage_ops = self._storage_ops()
        binding = {"verified": True, "source_config_path": "legacy/lab.json", "custom": [1, 2]}
        self.store.set_document(self.lab_id, "migration_binding", binding)
        output = self.root / "exports" / "lab.json"

        result = storage_ops.export_lab_json(home=self.home, lab=str(self.lab_id), output=output)

        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["lab_id"], str(self.lab_id))
        self.assertEqual(payload["migration_binding"], binding)
        self.assertEqual(base64.b64decode(payload["config_bytes_base64"]), self.store.load_config_bytes(self.lab_id))

    def test_reconcile_repairs_status_without_running_recover_or_restoring_candidate(self):
        storage_ops = self._storage_ops()
        run_id = self.store.create_run(self.lab_id, "baseline", "baseline", 1, {})
        self.store.ensure_repeat_started(self.lab_id, run_id, 1, started_at="2026-10-01T00:00:00Z")
        self.store.save_manifest(self.lab_id, run_id, {"id": "baseline", "status": "interrupted", "runs": []})
        with self.store._writer() as conn:
            conn.execute(
                "UPDATE runs SET status='running' WHERE lab_id=? AND run_id=?",
                (str(self.lab_id), str(run_id)),
            )

        candidate_file = self.candidate / "value.txt"
        candidate_file.write_text("uncommitted candidate edit", encoding="utf-8")
        with patch("autoresearch_runtime.trial.recover_lab") as recover:
            result = storage_ops.reconcile_storage(home=self.home, lab=str(self.lab_id), apply=True)

        recover.assert_not_called()
        self.assertEqual(candidate_file.read_text(encoding="utf-8"), "uncommitted candidate edit")
        self.assertEqual(result["repaired"], 2)
        self.assertEqual(self.store.get_run(self.lab_id, run_id)["status"], "interrupted")
        with self.store._reader() as conn:
            repeat = conn.execute(
                "SELECT status,finished_at,cost_usd,requests,result_json FROM repeats WHERE lab_id=? AND run_id=?",
                (str(self.lab_id), str(run_id)),
            ).fetchone()
        self.assertEqual(repeat[0], "interrupted")
        self.assertTrue(repeat[1])
        self.assertEqual(tuple(repeat[2:]), (None, None, "{}"))


if __name__ == "__main__":
    unittest.main()
