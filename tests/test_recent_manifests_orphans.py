"""Regression tests for SQLite recent-run guidance with missing manifests."""

from __future__ import annotations

try:
    import support  # noqa: F401 - discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import os
import sqlite3
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from autoresearch_runtime.cockpit import _recent_manifests
from autoresearch_runtime.config import load_config
from autoresearch_runtime.sqlite_store import SQLiteStore


class RecentManifestsOrphansTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.home = root / "home"
        self.home.mkdir()
        self.home_patch = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(self.home)})
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)

        state_dir = self.home / "state"
        state_dir.mkdir()
        candidate = support.git_project(state_dir / "candidate", {"v.txt": "1"})
        self.commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=candidate, text=True,
            **support.hidden_process_kwargs(),
        ).strip()
        config_data = {
            "confirmed": True,
            "source_root": str(candidate),
            "state_dir": str(state_dir),
            "config_path": str(state_dir / "lab.json"),
            "goal": "recent-manifests test",
            "repeats": 1,
            "timeout_seconds": 15,
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "evaluator": ["python", "eval.py"],
            "budget": {"max_trials": 20, "max_cost_usd": 0},
            "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
            "_passport": {
                "confirmed": True,
                "goal": "recent-manifests test",
                "constraints": [],
                "limits": {"max_trials": 20},
                "pins": {"program": {"path": str(candidate / "program.md"), "sha256": "a" * 64}},
            },
        }
        self.store = SQLiteStore(self.home)
        self.lab_id = self.store.create_lab(
            config_data,
            {"best_commit": self.commit, "current_epoch": 1, "inputs": [], "source": []},
            candidate,
        )
        (state_dir / "lab.json").write_bytes(self.store.load_config_bytes(self.lab_id))
        self.config = load_config(str(self.lab_id))

    def _add_run(self, alias, created_at, *, manifest=None):
        run_id = self.store.create_run(self.lab_id, "trial", alias, 1, {})
        with self.store._writer() as conn:
            conn.execute(
                "UPDATE runs SET created_at=?,updated_at=? WHERE lab_id=? AND run_id=?",
                (created_at, created_at, str(self.lab_id), str(run_id)),
            )
        if manifest is not None:
            self.store.save_manifest(self.lab_id, run_id, manifest)
        return str(run_id)

    def _manifest(self, alias, marker):
        return {"id": alias, "epoch": 1, "status": "accepted", "marker": marker}

    def _trace_recent(self, limit=100):
        statements = []
        original_connect = SQLiteStore._connect

        def traced_connect(store, write=False):
            conn = original_connect(store, write=write)
            conn.set_trace_callback(statements.append)
            return conn

        with patch.object(SQLiteStore, "_connect", new=traced_connect):
            manifests = _recent_manifests(self.config, 1, limit=limit)
        return manifests, statements

    def test_empty_store_returns_no_manifests_with_one_query(self):
        manifests, statements = self._trace_recent()

        self.assertEqual(manifests, [])
        self.assertEqual(len(statements), 1)

    def test_normal_manifests_keep_payload_order_and_one_query(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        expected = []
        for offset, alias in ((1, "trial-old"), (3, "trial-new"), (2, "trial-middle")):
            payload = self._manifest(alias, alias)
            run_id = self._add_run(alias, (base + timedelta(seconds=offset)).isoformat(), manifest=payload)
            expected.append((offset, run_id, payload))

        manifests, statements = self._trace_recent()

        self.assertEqual(manifests, [item[2] for item in sorted(expected, reverse=True)])
        self.assertEqual(len(statements), 1)

    def test_mixed_manifests_and_orphan_run_keep_database_order(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        newest = self._manifest("trial-newest", "newest")
        oldest = self._manifest("trial-oldest", "oldest")
        self._add_run("trial-newest", (base + timedelta(seconds=3)).isoformat(), manifest=newest)
        self._add_run("trial-orphan", (base + timedelta(seconds=2)).isoformat())
        self._add_run("trial-oldest", (base + timedelta(seconds=1)).isoformat(), manifest=oldest)

        manifests, _ = self._trace_recent()

        self.assertEqual(
            manifests,
            [
                newest,
                {
                    "id": "trial-orphan",
                    "epoch": 1,
                    "status": "created",
                    "candidate_commit": self.commit,
                },
                oldest,
            ],
        )

    def test_newest_orphan_is_included_when_limit_is_one(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self._add_run("trial-old", base.isoformat(), manifest=self._manifest("trial-old", "old"))
        self._add_run("trial-new-orphan", (base + timedelta(seconds=1)).isoformat())

        manifests, _ = self._trace_recent(limit=1)

        self.assertEqual(
            manifests,
            [{"id": "trial-new-orphan", "epoch": 1, "status": "created", "candidate_commit": self.commit}],
        )

    def test_tied_timestamps_use_run_id_descending(self):
        timestamp = "2025-01-01T00:00:00+00:00"
        runs = []
        for index in range(3):
            alias = f"trial-tie-{index}"
            payload = self._manifest(alias, alias)
            run_id = self._add_run(alias, timestamp, manifest=payload)
            runs.append((run_id, payload))

        manifests, _ = self._trace_recent()

        self.assertEqual(manifests, [payload for _, payload in sorted(runs, key=lambda item: item[0], reverse=True)])

    def test_nonpositive_limit_keeps_existing_one_row_minimum(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self._add_run("trial-old", base.isoformat(), manifest=self._manifest("trial-old", "old"))
        self._add_run("trial-new-orphan", (base + timedelta(seconds=1)).isoformat())

        manifests, _ = self._trace_recent(limit=0)

        self.assertEqual([manifest["id"] for manifest in manifests], ["trial-new-orphan"])

    def test_limit_is_capped_at_100_and_older_orphan_is_not_appended(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for index in range(100):
            alias = f"trial-{index:03d}"
            payload = self._manifest(alias, alias)
            self._add_run(alias, (base + timedelta(seconds=index + 1)).isoformat(), manifest=payload)
        self._add_run("trial-old-orphan", "2000-01-01T00:00:00+00:00")
        expected = [item["alias"] for item in self.store.list_runs(self.lab_id, limit=100, epoch_id=1)["items"]]

        manifests, statements = self._trace_recent(limit=101)

        self.assertEqual(len(manifests), 100)
        self.assertEqual([manifest["id"] for manifest in manifests], expected)
        self.assertNotIn("trial-old-orphan", [manifest["id"] for manifest in manifests])
        self.assertEqual(len(statements), 1)

    def test_sql_exception_still_uses_list_runs_fallback(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        old = self._manifest("trial-old", "old")
        new = self._manifest("trial-new", "new")
        self._add_run("trial-old", base.isoformat(), manifest=old)
        self._add_run("trial-new", (base + timedelta(seconds=1)).isoformat(), manifest=new)

        original_connect = SQLiteStore._connect
        should_fail = True

        def fail_first_connect(store, write=False):
            nonlocal should_fail
            if should_fail:
                should_fail = False
                raise sqlite3.OperationalError("injected read failure")
            return original_connect(store, write=write)

        with patch.object(SQLiteStore, "_connect", new=fail_first_connect):
            manifests = _recent_manifests(self.config, 1)

        self.assertEqual(manifests, [new, old])


if __name__ == "__main__":
    unittest.main()
