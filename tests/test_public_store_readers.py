try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_runtime import doctor
from autoresearch_runtime.cockpit import _recent_manifests
from autoresearch_runtime.sqlite_store import SQLiteStore


def _create_lab(store: SQLiteStore, root: Path):
    project = root / "project"
    candidate = root / "candidate"
    state_dir = root / "state"
    project.mkdir(parents=True)
    (candidate / ".git").mkdir(parents=True)
    state_dir.mkdir()
    metrics = {"primary": "score", "direction": "max", "guards": []}
    evaluator = ["python", "eval.py"]
    limits = {"max_trials": 5}
    pins = {"evaluator": {"sha256": "a" * 64}}
    config = {
        "id": "test-lab",
        "root_ref": "root-test",
        "project": str(project),
        "state_dir": str(state_dir),
        "config_path": str(state_dir / "lab.json"),
        "metrics": metrics,
        "evaluator": evaluator,
        "limits": limits,
        "pins": pins,
        "_passport": {
            "confirmed": True,
            "goal": "Exercise public SQLite readers",
            "project": str(project),
            "metrics": metrics,
            "evaluator": evaluator,
            "limits": limits,
            "pins": pins,
        },
    }
    lab_id = store.create_lab(
        config,
        {"best_commit": "b" * 40, "current_epoch": 1, "setup_pending": True},
        candidate,
    )
    return lab_id, state_dir


def _set_run_tie(store: SQLiteStore, lab_id, run_ids) -> None:
    placeholders = ",".join("?" for _ in run_ids)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            f"UPDATE runs SET created_at=? WHERE lab_id=? AND run_id IN ({placeholders})",
            ("2026-01-01T00:00:00.000Z", str(lab_id), *(str(run_id) for run_id in run_ids)),
        )


class PublicStoreReadersTest(unittest.TestCase):
    def test_cockpit_calls_public_recent_reader_with_one_join_for_saved_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "home"
            store = SQLiteStore(home)
            lab_id, state_dir = _create_lab(store, root)
            runs = []
            for alias in ("saved-a", "saved-b", "saved-c"):
                run_id = store.create_run(lab_id, "trial", alias, 1, {})
                store.save_manifest(lab_id, run_id, {"id": alias, "epoch": 1, "runs": []})
                runs.append((run_id, alias))
            _set_run_tie(store, lab_id, [run_id for run_id, _ in runs])
            config = SimpleNamespace(storage_mode="sqlite", lab_id=lab_id, state_dir=state_dir)

            public_calls = []
            original_reader = getattr(SQLiteStore, "list_recent_manifests", None)

            def traced_reader(self, requested_lab_id, epoch_id, *, limit=100):
                public_calls.append((requested_lab_id, epoch_id, limit))
                if original_reader is None:
                    return []
                return original_reader(self, requested_lab_id, epoch_id, limit=limit)

            statements = []
            original_connect = SQLiteStore._connect

            def traced_connect(self, write=False):
                conn = original_connect(self, write=write)
                conn.set_trace_callback(statements.append)
                return conn

            with patch.dict(os.environ, {"AUTORESEARCH_HOME": str(home)}):
                with patch.object(SQLiteStore, "list_recent_manifests", traced_reader, create=True):
                    with patch.object(SQLiteStore, "_connect", traced_connect):
                        manifests = _recent_manifests(config, 1, limit=2)

            self.assertEqual(len(public_calls), 1)
            self.assertEqual(public_calls[0][1:], (1, 2))
            expected = [alias for _, alias in sorted(runs, key=lambda item: str(item[0]), reverse=True)[:2]]
            self.assertEqual([item["id"] for item in manifests], expected)
            selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
            self.assertEqual(len(selects), 1)
            self.assertIn("LEFT JOIN run_documents", selects[0])

    def test_public_recent_reader_rehydrates_only_missing_manifest_documents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = SQLiteStore(root / "home")
            lab_id, _ = _create_lab(store, root)
            saved_id = store.create_run(lab_id, "trial", "saved", 1, {})
            missing_id = store.create_run(lab_id, "trial", "missing", 1, {})
            store.save_manifest(lab_id, saved_id, {"id": "saved", "epoch": 1, "runs": []})
            _set_run_tie(store, lab_id, [saved_id, missing_id])

            reader = getattr(store, "list_recent_manifests", None)
            self.assertTrue(callable(reader), "SQLiteStore must expose its recent manifest reader")
            if not callable(reader):
                return
            with patch.object(store, "load_manifest", wraps=store.load_manifest) as load_manifest:
                manifests = reader(lab_id, 1, limit=10)

            expected = [
                alias for _, alias in sorted(((saved_id, "saved"), (missing_id, "missing")),
                                             key=lambda item: str(item[0]), reverse=True)
            ]
            self.assertEqual([item["id"] for item in manifests], expected)
            self.assertEqual(load_manifest.call_count, 1)
            self.assertEqual(str(load_manifest.call_args.args[1]), str(missing_id))

    def test_doctor_uses_public_active_reader_and_reports_manifest_failure_without_repair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            home = root / "home"
            store = SQLiteStore(home)
            lab_id, _ = _create_lab(store, root)
            operation = store.enqueue_operation(lab_id, "trial", "request-1", {"title": "queued"})

            public_calls = []
            original_reader = getattr(SQLiteStore, "list_active_operations", None)

            def traced_reader(self):
                public_calls.append(True)
                return original_reader(self) if original_reader is not None else []

            with patch.object(doctor, "resolve_home", return_value=home):
                with patch.object(doctor, "check_skill", return_value={
                    "ok": False, "modified": False, "error": "invalid skill manifest: unexpected value",
                }):
                    with patch.object(SQLiteStore, "list_active_operations", traced_reader, create=True):
                        result = doctor.run_doctor()

            self.assertEqual(public_calls, [True])
            manifest_problems = [problem for problem in result["problems"] if "invalid skill manifest" in problem.lower()]
            self.assertEqual(len(manifest_problems), 1, result["problems"])
            self.assertIn("install.py", manifest_problems[0])
            self.assertIn(f"operation:{operation['operation_id']}", result["stale_running_manifests"])


if __name__ == "__main__":
    unittest.main()
