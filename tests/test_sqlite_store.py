from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import contextlib
import multiprocessing
import sqlite3
import tempfile
import unittest
import uuid
import hashlib
import json
from pathlib import Path

from autoresearch_runtime.sqlite_store import (
    AmbiguousAliasError,
    IdempotencyConflictError,
    SQLiteStore,
    StoreCorruptError,
    StoreIntegrityError,
    StoreNotInitializedError,
    StoreVersionError,
)


def _create_in_process(home: str, candidate: str, project: str, state_dir: str, root_ref: str) -> str:
    store = SQLiteStore(home)
    config = _config(Path(project), Path(state_dir), root_ref=root_ref)
    lab_id = store.create_lab(config, _state(), candidate)
    return str(lab_id)


def _config(project: Path, state_dir: Path, *, root_ref: str = "root-A", alias: str = "legacy-project"):
    return {
        "id": alias,
        "root_ref": root_ref,
        "project": str(project),
        "state_dir": str(state_dir),
        "config_path": str(state_dir / "lab.json"),
        "metrics": {"primary": "score", "direction": "max", "guards": ["latency"]},
        "evaluator": ["python", "eval.py"],
        "limits": {"max_trials": 5},
        "pins": {"evaluator": {"sha256": "a" * 64}},
        "_passport": {
            "confirmed": True,
            "goal": "Improve the score",
            "project": str(project),
            "metrics": {"primary": "score", "direction": "max", "guards": ["latency"]},
            "evaluator": ["python", "eval.py"],
            "limits": {"max_trials": 5},
            "pins": {"evaluator": {"sha256": "a" * 64}},
        },
    }


def _state():
    return {"best_commit": "b" * 40, "current_epoch": 1, "setup_pending": True}


class SQLiteStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / ".autoresearch"
        self.project = self.root / "project"
        self.project.mkdir()
        self.candidate = self.root / "candidate"
        (self.candidate / ".git").mkdir(parents=True)
        self.store = SQLiteStore(self.home)

    def create_lab(self, *, root_ref="root-A", alias="legacy-project", state_name="lab-a", config=None,
                   source_config_bytes=b'{"source":true}'):
        state_dir = self.root / "labs" / state_name
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        config = config or _config(self.project, state_dir, root_ref=root_ref, alias=alias)
        return self.store.create_lab(
            config,
            _state(),
            self.candidate,
            source_config_bytes=source_config_bytes,
        )

    def test_absent_store_reads_do_not_create_home_or_database(self):
        self.assertFalse(self.home.exists())
        self.assertEqual(self.store.list_labs(), {
            "items": [], "next_cursor": None, "total": 0, "limit": 20, "offset": 0,
        })
        self.assertEqual(self.store.get_revision(), 0)
        with self.assertRaises(StoreNotInitializedError):
            self.store.load_state(uuid.uuid4())
        self.assertFalse(self.home.exists())

    def test_schema_is_strict_and_foreign_key_failure_rolls_back(self):
        lab_id = self.create_lab()
        with self.store._reader() as conn:
            strict_tables = {
                row["name"] for row in conn.execute("SELECT name,strict FROM pragma_table_list WHERE schema='main'")
                if row["name"] in {"labs", "runs", "repeats", "operations"} and row["strict"] == 1
            }
            self.assertEqual(strict_tables, {"labs", "runs", "repeats", "operations"})
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        with self.assertRaises(StoreIntegrityError):
            with self.store._writer() as conn:
                conn.execute(
                    "INSERT INTO metric_values(lab_id,run_id,epoch_id,repeat_index,metric_name,value) "
                    "VALUES(?,?,?,?,?,?)",
                    (str(lab_id), str(uuid.uuid4()), 1, 1, "score", 1.0),
                )
        with self.store._reader() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM metric_values").fetchone()[0], 0)

    def test_version_and_corruption_errors_leave_database_bytes_unchanged(self):
        self.create_lab()
        with contextlib.closing(sqlite3.connect(self.store.db_path)) as conn, conn:
            conn.execute("PRAGMA user_version=999")
        before = self.store.db_path.read_bytes()
        with self.assertRaises(StoreVersionError):
            self.store.list_labs()
        self.assertEqual(self.store.db_path.read_bytes(), before)

        self.store.db_path.write_bytes(b"not a SQLite database")
        before = self.store.db_path.read_bytes()
        with self.assertRaises(StoreCorruptError):
            self.store.list_labs()
        self.assertEqual(self.store.db_path.read_bytes(), before)

    def test_scoped_aliases_support_multiple_projects_and_bare_collision_is_explicit(self):
        first = self.create_lab(root_ref="root-A", state_name="lab-a")
        second = self.create_lab(root_ref="root-B", state_name="lab-b")
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.resolve_id("lab", "legacy-project", "root-A"), first)
        self.assertEqual(self.store.resolve_id("lab", "legacy-project", "root-B"), second)
        with self.assertRaises(AmbiguousAliasError):
            self.store.resolve_lab_id("legacy-project")

    def test_operation_idempotency_returns_same_operation_and_rejects_changed_payload(self):
        lab_id = self.create_lab()
        first = self.store.enqueue_operation(lab_id, "trial", "request-1", {"title": "test"})
        replay = self.store.enqueue_operation(lab_id, "trial", "request-1", {"title": "test"})
        self.assertEqual(first["operation_id"], replay["operation_id"])
        self.assertTrue(replay["replayed"])
        with self.assertRaises(IdempotencyConflictError):
            self.store.enqueue_operation(lab_id, "trial", "request-1", {"title": "changed"})
        self.store.record_operation_spawn(first["operation_id"], 12345)
        self.assertEqual(self.store.get_operation(first["operation_id"])["pid"], 12345)
        self.store.transition_operation(first["operation_id"], "queued", "recovery_required")
        self.assertEqual(self.store.get_operation(first["operation_id"])["status"], "recovery_required")

    def test_repeat_placeholder_is_not_a_completed_result_and_cases_are_repeat_scoped(self):
        lab_id = self.create_lab()
        run_id = self.store.create_run(lab_id, "baseline", "baseline", 1, {})
        self.store.ensure_repeat_started(lab_id, run_id, 1)
        self.assertEqual(self.store.load_results(lab_id, run_id), [])
        self.assertIsNone(self.store.load_result(lab_id, run_id, 1))
        self.store.append_checkpoint(lab_id, run_id, "same-case", {"repeat_index": 1, "status": "ok"})
        self.store.append_checkpoint(lab_id, run_id, "same-case", {"repeat_index": 2, "status": "ok"})
        self.assertEqual(len(self.store.load_checkpoints(lab_id, run_id, repeat_index=1)), 1)
        self.assertEqual(len(self.store.load_checkpoints(lab_id, run_id, repeat_index=2)), 1)
        self.store.append_result(lab_id, run_id, 1, {"metrics": {"score": 1}, "status": "accepted"})
        self.store.append_result(lab_id, run_id, 2, {"metrics": {"score": 2}, "status": "accepted"})
        self.assertEqual([item["repeat_index"] for item in self.store.load_results(lab_id, run_id)], [1, 2])
        self.assertEqual(self.store.load_result(lab_id, run_id, 1)["metrics"], {"score": 1})

    def test_implicit_success_status_keeps_dashboard_metric_summary_visible(self):
        from autoresearch_runtime.cockpit import _metric_summary

        lab_id = self.create_lab()
        run_id = self.store.create_run(lab_id, "baseline", "baseline", 1, {})
        self.store.append_result(lab_id, run_id, 1, {"metrics": {"score": 1}})
        self.store.append_result(lab_id, run_id, 2, {"metrics": {"score": 1}})
        manifest = self.store.load_manifest(lab_id, run_id)
        self.assertEqual([item["status"] for item in manifest["runs"]], ["ok", "ok"])
        self.assertEqual(_metric_summary(manifest, "score", "max"), {
            "mean": 1.0, "decision_worst": 1.0, "min": 1.0, "max": 1.0,
            "completed_repeats": 2,
        })

    def _epoch_two_payload(self, lab_id, *, epoch_id=2, source_hash_override=None):
        config = self.store.load_config(lab_id)
        config["metrics"] = {
            "primary": "score", "direction": "max", "unit": "points",
            "target_threshold": 4,
            "guards": [{"name": "latency", "direction": "min", "unit": "ms", "max_threshold": 5}],
        }
        raw = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        prior = self.store.load_state(lab_id)
        prior.update({
            "current_epoch": epoch_id,
            "config_sha256": source_hash_override or digest,
            "epochs": [
                {"id": 1, "superseded_at": "2026-01-01T00:00:00Z", "supersede_reason": "contract changed"},
                {"id": epoch_id, "started_at": "2026-01-02T00:00:00Z", "previous_epoch_id": 1,
                 "superseded_at": None, "config_sha256": digest,
                 "metrics": {"primary": "score", "direction": "max"}},
            ],
        })
        return config, raw, prior, digest

    def _create_with_exact_config_source(self):
        state_dir = self.root / "labs" / "epoched"
        state_dir.parent.mkdir(parents=True, exist_ok=True)
        config = _config(self.project, state_dir)
        stored_config = {key: value for key, value in config.items() if key != "_passport"}
        raw = (json.dumps(stored_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        return self.create_lab(state_name="epoched", config=config, source_config_bytes=raw)

    def test_explicit_config_state_save_appends_immutable_epoch_and_full_metric_definitions(self):
        lab_id = self._create_with_exact_config_source()
        config, raw, state, digest = self._epoch_two_payload(lab_id)
        with self.store._reader() as conn:
            old_epoch = tuple(conn.execute(
                "SELECT status,created_at,superseded_at,reason,contract_hash,payload_json "
                "FROM epochs WHERE lab_id=? AND epoch_id=1", (str(lab_id),),
            ).fetchone())
            old_pins = [tuple(row) for row in conn.execute(
                "SELECT name,sha256,source_path,payload_json,created_at FROM pins WHERE lab_id=? ORDER BY name",
                (str(lab_id),),
            )]
            old_definitions = [tuple(row) for row in conn.execute(
                "SELECT metric_name,role,direction,unit,payload_json FROM metric_definitions "
                "WHERE lab_id=? AND epoch_id=1 ORDER BY metric_name", (str(lab_id),),
            )]
        before_revision = self.store.get_revision()

        revision = self.store.save_config_and_state(lab_id, config, state, raw_config_bytes=raw)

        self.assertEqual(revision, before_revision + 1)
        self.assertEqual(self.store.load_config_bytes(lab_id), raw)
        self.assertEqual(self.store.load_state(lab_id), state)
        with self.store._reader() as conn:
            new_epoch = conn.execute(
                "SELECT status,contract_hash,payload_json FROM epochs WHERE lab_id=? AND epoch_id=2",
                (str(lab_id),),
            ).fetchone()
            self.assertEqual(tuple(conn.execute(
                "SELECT status,created_at,superseded_at,reason,contract_hash,payload_json "
                "FROM epochs WHERE lab_id=? AND epoch_id=1", (str(lab_id),),
            ).fetchone()), old_epoch)
            self.assertEqual([tuple(row) for row in conn.execute(
                "SELECT name,sha256,source_path,payload_json,created_at FROM pins WHERE lab_id=? ORDER BY name",
                (str(lab_id),),
            )], old_pins)
            self.assertEqual([tuple(row) for row in conn.execute(
                "SELECT metric_name,role,direction,unit,payload_json FROM metric_definitions "
                "WHERE lab_id=? AND epoch_id=1 ORDER BY metric_name", (str(lab_id),),
            )], old_definitions)
            definitions = [dict(row) for row in conn.execute(
                "SELECT metric_name,role,direction,unit,payload_json FROM metric_definitions "
                "WHERE lab_id=? AND epoch_id=2 ORDER BY metric_name", (str(lab_id),),
            )]
        snapshot = json.loads(new_epoch["payload_json"])
        self.assertEqual(new_epoch["contract_hash"], digest)
        self.assertEqual(snapshot["goal"], "Improve the score")
        self.assertEqual(snapshot["metrics"]["target_threshold"], 4)
        self.assertEqual(snapshot["pins"], self.store.load_pins(lab_id))
        self.assertEqual({item["metric_name"]: item["role"] for item in definitions}, {
            "latency": "guard", "score": "primary",
        })
        latency = next(item for item in definitions if item["metric_name"] == "latency")
        self.assertEqual(json.loads(latency["payload_json"])["max_threshold"], 5)
        run_id = self.store.create_run(lab_id, "baseline", "baseline-epoch-2", 2, {})
        self.store.append_result(lab_id, run_id, 1, {"metrics": {"score": 5, "latency": 4}})
        with self.store._reader() as conn:
            self.assertEqual(conn.execute(
                "SELECT count(*) FROM metric_values WHERE lab_id=? AND run_id=?", (str(lab_id), str(run_id)),
            ).fetchone()[0], 2)

    def test_only_next_contiguous_epoch_can_be_added_and_failure_rolls_back_config_state_and_revision(self):
        lab_id = self._create_with_exact_config_source()
        config, raw, state, _ = self._epoch_two_payload(lab_id, epoch_id=3)
        before_config = self.store.load_config(lab_id)
        before_bytes = self.store.load_config_bytes(lab_id)
        before_state = self.store.load_state(lab_id)
        before_revision = self.store.get_revision()
        with self.store._reader() as conn:
            before_epochs = [tuple(row) for row in conn.execute(
                "SELECT epoch_id,status,created_at,superseded_at,reason,contract_hash,payload_json "
                "FROM epochs WHERE lab_id=? ORDER BY epoch_id", (str(lab_id),),
            )]

        with self.assertRaises(StoreIntegrityError):
            self.store.save_config_and_state(lab_id, config, state, raw_config_bytes=raw)

        self.assertEqual(self.store.load_config(lab_id), before_config)
        self.assertEqual(self.store.load_config_bytes(lab_id), before_bytes)
        self.assertEqual(self.store.load_state(lab_id), before_state)
        self.assertEqual(self.store.get_revision(), before_revision)
        with self.store._reader() as conn:
            self.assertEqual([tuple(row) for row in conn.execute(
                "SELECT epoch_id,status,created_at,superseded_at,reason,contract_hash,payload_json "
                "FROM epochs WHERE lab_id=? ORDER BY epoch_id", (str(lab_id),),
            )], before_epochs)

    def test_config_hash_mismatch_and_generic_state_projection_never_create_epochs(self):
        lab_id = self._create_with_exact_config_source()
        config, raw, state, digest = self._epoch_two_payload(lab_id)
        before_config = self.store.load_config(lab_id)
        before_bytes = self.store.load_config_bytes(lab_id)
        before_revision = self.store.get_revision()
        state["config_sha256"] = "0" * 64
        with self.assertRaises(StoreIntegrityError):
            self.store.save_config_and_state(lab_id, config, state, raw_config_bytes=raw)
        self.assertEqual(self.store.load_config(lab_id), before_config)
        self.assertEqual(self.store.load_state(lab_id)["current_epoch"], 1)
        self.assertEqual(self.store.load_config_bytes(lab_id), before_bytes)
        self.assertEqual(self.store.get_revision(), before_revision)

        # State projections are ordinary mutable documents and cannot synthesize relational contract rows.
        state["config_sha256"] = digest
        state["current_epoch"] = 2
        self.store.save_state(lab_id, state)
        with self.store._reader() as conn:
            self.assertEqual([row[0] for row in conn.execute(
                "SELECT epoch_id FROM epochs WHERE lab_id=? ORDER BY epoch_id", (str(lab_id),),
            )], [1])

    def test_large_config_and_result_are_externalized_and_hydrated_on_demand(self):
        large_text = "z" * (256 * 1024 + 128)
        state_dir = self.root / "labs" / "large"
        config = _config(self.project, state_dir)
        config["large_setting"] = large_text
        raw_config = (b'{"source":"' + b"q" * (256 * 1024 + 64) + b'"}')
        lab_id = self.create_lab(state_name="large", config=config, source_config_bytes=raw_config)
        revision = self.store.get_revision()
        loaded_config = self.store.load_config(lab_id)
        self.assertEqual(loaded_config["large_setting"], large_text)
        self.assertEqual(self.store.load_config_bytes(lab_id), raw_config)
        with self.store._reader() as conn:
            rows = conn.execute(
                "SELECT payload_artifact_id,source_artifact_id,source_bytes FROM projections "
                "WHERE lab_id=? AND view='config'", (str(lab_id),)
            ).fetchone()
        self.assertIsNotNone(rows["payload_artifact_id"])
        self.assertIsNotNone(rows["source_artifact_id"])
        self.assertIsNone(rows["source_bytes"])

        run_id = self.store.create_run(lab_id, "trial", "trial-0001", 1, {})
        result = {"metrics": {"score": 3}, "details": large_text}
        self.store.append_result(lab_id, run_id, 1, result)
        self.store.append_attempt(
            lab_id, run_id, "attempt-1", "finished",
            {"repeat_index": 1, "case_id": "case-1", "cost_usd": None, "requests": None},
        )
        self.assertEqual(self.store.load_result(lab_id, run_id, 1), result)
        self.assertTrue(self.store.load_results(lab_id, run_id)[0]["details_available"])
        self.assertGreater(self.store.get_revision(), revision)
        artifact_rows = self.store.list_artifacts(lab_id, run_id)
        self.assertTrue(artifact_rows)
        self.assertTrue(all(item["run_id"] == run_id for item in artifact_rows))
        self.assertTrue(all("content" not in item for item in artifact_rows))
        with self.store._reader() as conn:
            row = conn.execute(
                "SELECT result_json,result_artifact_id FROM repeats WHERE lab_id=? AND run_id=?",
                (str(lab_id), str(run_id)),
            ).fetchone()
        self.assertEqual(row["result_json"], "{}")
        self.assertIsNotNone(row["result_artifact_id"])
        self.assertEqual(self.store.usage_totals(lab_id, run_id), {
            "requests": 1, "unknown_requests": 0, "cost_usd": 0.0, "unknown_costs": 1,
        })

    def test_usage_counts_started_attempts_once_and_prefers_lifecycle_over_meter_fallback(self):
        lab_id = self.create_lab()
        run_id = self.store.create_run(lab_id, "trial", "trial-usage", 1, {})
        started = {"repeat_index": 1, "case_id": "case-1"}
        finished = {"repeat_index": 1, "case_id": "case-1", "cost_usd": 0.25}
        self.store.append_attempt(lab_id, run_id, "http-1", "started", started)
        self.assertEqual(self.store.usage_totals(lab_id, run_id), {
            "requests": 1, "unknown_requests": 1, "cost_usd": 0.0, "unknown_costs": 1,
        })
        self.store.append_attempt(lab_id, run_id, "http-1", "finished", finished)
        self.store.append_attempt(lab_id, run_id, "http-1", "finished", finished)
        self.store.append_usage(lab_id, run_id, 1, {"requests": 9, "cost_usd": 99.0})
        self.assertEqual(self.store.usage_totals(lab_id, run_id, repeat_index=1), {
            "requests": 1, "unknown_requests": 0, "cost_usd": 0.25, "unknown_costs": 0,
        })

        self.store.append_usage(lab_id, run_id, 2, {"requests": 3, "cost_usd": 0.5})
        self.store.append_usage(lab_id, run_id, 2, {"requests": 3, "cost_usd": 0.5})
        totals = self.store.usage_totals(lab_id, run_id)
        self.assertEqual(totals, {
            "requests": 7, "unknown_requests": 0, "cost_usd": 1.25, "unknown_costs": 0,
        })

    def test_first_schema_initialization_is_serialized_across_processes(self):
        context = multiprocessing.get_context("spawn")
        jobs = [
            (str(self.home), str(self.candidate), str(self.project), str(self.root / "labs" / f"p{i}"), f"root-{i}")
            for i in range(2)
        ]
        with context.Pool(2) as pool:
            lab_ids = pool.starmap(_create_in_process, jobs)
        self.assertEqual(len(set(lab_ids)), 2)
        self.assertEqual(SQLiteStore(self.home).list_labs()["total"], 2)


if __name__ == "__main__":
    unittest.main()
