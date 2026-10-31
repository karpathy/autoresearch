from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime import api_v1
from autoresearch_runtime.sqlite_store import EntityNotFoundError, SQLiteStore


def _config(project: Path, state_dir: Path) -> dict:
    pins = {"program": {"path": str(project / "program.md"), "sha256": "a" * 64}}
    return {
        "confirmed": True,
        "source_root": str(project),
        "state_dir": str(state_dir),
        "config_path": str(state_dir / "lab.json"),
        "goal": "test dashboard lifecycle",
        "repeats": 2,
        "timeout_seconds": 30,
        "metrics": {"primary": "score", "direction": "max",
                     "guards": [{"name": "safety", "direction": "min", "unit": "ratio"}]},
        "evaluator": ["python", str(project / "evaluator.py")],
        "budget": {"max_trials": 3, "max_cost_usd": 0},
        "pins": pins,
        "_passport": {"confirmed": True, "goal": "test dashboard lifecycle",
                       "constraints": [], "limits": {"max_trials": 3}, "pins": pins},
    }


class _Handler:
    def __init__(self, method: str, path: str, payload: dict | None = None):
        raw = json.dumps(payload or {}).encode("utf-8")
        self.command = method
        self.path = path
        self.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        self.rfile = io.BytesIO(raw)
        self.response = None

    def _send(self, status, body, content_type, extra=None):
        self.response = (status, body, content_type, extra or {})


class DashboardLifecycleApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.store = SQLiteStore(self.home)
        self.old_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        self.addCleanup(self._restore_home)
        self.labs = []
        self.paths = []
        for name in ("first", "second"):
            project = self.root / f"project-{name}"
            project.mkdir()
            (project / "program.md").write_text("goal\n", encoding="utf-8")
            (project / "evaluator.py").write_text("# evaluator\n", encoding="utf-8")
            (project / "input.csv").write_text("value\n1\n", encoding="utf-8")
            state_dir = self.root / f"state-{name}"
            state_dir.mkdir()
            candidate = support.git_project(state_dir / "candidate", {"value.txt": "1"})
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=candidate, text=True,
                **support.hidden_process_kwargs(),
            ).strip()
            config = _config(project, state_dir)
            state = {"best_commit": commit, "best_id": "baseline", "baseline_id": "baseline",
                     "baseline_verified": True, "current_epoch": 1, "inputs": [], "source": []}
            lab_id = self.store.create_lab(config, state, candidate)
            self.labs.append(lab_id)
            self.paths.append((project, state_dir, candidate))

    def _restore_home(self):
        if self.old_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.old_home

    def _api(self, method: str, path: str, payload: dict | None = None):
        handler = _Handler(method, path, payload)
        with patch.object(api_v1, "authenticate"), \
             patch.object(api_v1, "resolve_home", return_value=self.home):
            handled = api_v1.dispatch(handler)
        self.assertTrue(handled)
        status, raw, _content_type, headers = handler.response
        return status, json.loads(raw) if raw else None, headers

    def _record_run(self, alias: str, *, parent_id=None, scores=(0.5, 0.7), safety=(0.9, 0.8)):
        lab_id = self.labs[0]
        run_id = self.store.create_run(
            lab_id, "baseline" if alias == "baseline" else "trial", alias, 1,
            {"parent_run_id": str(parent_id)} if parent_id else {},
        )
        state = self.store.load_state(lab_id)
        manifest = {"id": alias, "status": "accepted", "parent": str(parent_id) if parent_id else None}
        results = [{"status": "ok", "metrics": {"score": score, "safety": guard}}
                   for score, guard in zip(scores, safety)]
        self.store.finalize_run(lab_id, run_id, state, manifest, results)
        return run_id

    def test_dashboard_list_defaults_active_and_returns_bounded_summary(self):
        baseline = self._record_run("baseline")
        trial = self._record_run("trial-0001", parent_id=baseline,
                                 scores=(0.7, 0.9), safety=(0.85, 0.95))
        state = self.store.load_state(self.labs[0])
        state.update({"baseline_id": "baseline", "best_id": "trial-0001"})
        self.store.save_state(self.labs[0], state)

        status, payload, _ = self._api("POST", f"/api/v1/labs/{self.labs[1]}/lifecycle", {"action": "archive"})
        self.assertEqual(status, 200, payload)

        # Existing non-dashboard callers retain the all-labs behavior unless filtered.
        self.assertEqual(self.store.list_labs()["total"], 2)
        self.assertEqual(self.store.list_labs(filters={"lifecycle": "archived"})["total"], 1)
        status, payload, _ = self._api("GET", "/api/v1/labs")
        self.assertEqual(status, 200, payload)
        page = payload["data"]
        self.assertEqual(page["total"], 1)
        self.assertEqual(page["items"][0]["lab_id"], str(self.labs[0]))
        lab = page["items"][0]
        self.assertEqual(lab["lifecycle"], "active")
        summary = lab["dashboard_summary"]
        self.assertEqual(summary["current_epoch"], 1)
        self.assertEqual(summary["metric_definitions"]["primary"]["name"], "score")
        self.assertEqual(summary["metric_definitions"]["guards"][0]["name"], "safety")
        self.assertEqual(summary["baseline"]["run_id"], str(baseline))
        self.assertAlmostEqual(summary["baseline"]["metric_means"]["score"], 0.6)
        self.assertEqual(summary["best"]["run_id"], str(trial))
        self.assertAlmostEqual(summary["best"]["metric_means"]["score"], 0.8)
        self.assertEqual(summary["latest"]["run_id"], str(trial))
        self.assertEqual(len(summary["trend"]), 2)
        self.assertEqual(summary["trend"][-1]["parent_run_id"], str(baseline))
        self.assertAlmostEqual(summary["trend"][-1]["metric_means"]["safety"], 0.9)
        self.assertEqual(summary["trend"][-1]["time"], summary["trend"][-1]["created_at"])

        status, archived, _ = self._api("GET", "/api/v1/labs?lifecycle=archived")
        self.assertEqual(status, 200, archived)
        self.assertEqual([item["lab_id"] for item in archived["data"]["items"]], [str(self.labs[1])])
        status, invalid, _ = self._api("GET", "/api/v1/labs?lifecycle=all")
        self.assertEqual(status, 400)
        self.assertEqual(invalid["error"]["code"], "invalid_request")

    def test_dashboard_trend_contains_at_most_five_current_epoch_points(self):
        for index in range(7):
            run_id = self.store.create_run(self.labs[0], "trial", f"trial-{index:04d}", 1, {})
            self.store.save_manifest(self.labs[0], run_id,
                                      {"id": f"trial-{index:04d}", "status": "accepted", "runs": []})

        status, payload, _ = self._api("GET", "/api/v1/labs?limit=2")
        self.assertEqual(status, 200, payload)
        lab = next(item for item in payload["data"]["items"] if item["lab_id"] == str(self.labs[0]))
        summary = lab["dashboard_summary"]
        self.assertEqual(len(summary["trend"]), 5)
        self.assertEqual(summary["latest_id"], summary["trend"][-1]["run_id"])

    def test_active_run_progress_is_included_without_per_lab_fetches(self):
        run_id = self.store.create_run(self.labs[0], "trial", "trial-live", 1, {})
        self.store.save_manifest(self.labs[0], run_id, {"id": "trial-live", "status": "running", "runs": []})
        operation = self.store.enqueue_operation(self.labs[0], "trial", "live", {"command": "trial"})
        self.store.claim_operation(operation["operation_id"], 4321)
        self.store.heartbeat_operation(operation["operation_id"], 4321,
                                       {"phase": "evaluate", "completed": 2, "total": 5})

        status, payload, _ = self._api("GET", "/api/v1/labs?limit=2")
        self.assertEqual(status, 200, payload)
        lab = next(item for item in payload["data"]["items"] if item["lab_id"] == str(self.labs[0]))
        active = lab["dashboard_summary"]["active_run"]
        self.assertEqual(active["run_id"], str(run_id))
        self.assertEqual(active["status"], "running")
        self.assertEqual(active["progress"], {"phase": "evaluate", "completed": 2, "total": 5})

    def test_paginated_run_list_includes_batched_primary_and_guard_metrics(self):
        self._record_run("baseline", scores=(0.5, 0.7), safety=(0.9, 0.8))
        latest_run = self._record_run("trial-0001", scores=(0.7, 0.9), safety=(0.85, 0.95))

        # Dashboard data must come from the paginated list itself, without one detail fetch per run.
        with patch.object(SQLiteStore, "get_run", side_effect=AssertionError("unexpected run detail fetch")), \
             patch.object(SQLiteStore, "load_results", side_effect=AssertionError("unexpected repeat fetch")):
            status, payload, _ = self._api(
                "GET", f"/api/v1/labs/{self.labs[0]}/runs?limit=1",
            )

        self.assertEqual(status, 200, payload)
        page = payload["data"]
        self.assertEqual(page["total"], 2)
        self.assertEqual(page["limit"], 1)
        self.assertIsNotNone(page["next_cursor"])
        item = page["items"][0]
        self.assertEqual(item["run_id"], str(latest_run))
        self.assertEqual(item["metrics"]["score"]["decision_worst"], 0.7)
        self.assertAlmostEqual(item["metrics"]["score"]["mean"], 0.8)
        self.assertEqual(item["metrics"]["score"]["completed_repeats"], 2)
        self.assertEqual(item["metrics"]["safety"]["decision_worst"], 0.95)
        self.assertAlmostEqual(item["metrics"]["safety"]["mean"], 0.9)
        self.assertEqual(item["metrics"]["safety"]["completed_repeats"], 2)
        self.assertNotIn("metrics", self.store.list_runs(self.labs[0], limit=1)["items"][0])

    def test_dashboard_summary_and_run_list_return_structured_fields(self):
        lab_id = self.labs[0]
        config = self.store.load_config(lab_id)
        config["acceptance"] = {"statistic": "median"}
        self.store.save_config(lab_id, config)

        run_id = self.store.create_run(
            lab_id, "trial", "trial-structured", 1, {},
        )
        state = self.store.load_state(lab_id)
        hyp = {
            "title": "Hypothesis on caching",
            "observation": "Cache misses cause latency",
            "expected_effect": "Reduce latency by 20%",
        }
        manifest = {
            "id": "trial-structured",
            "status": "accepted",
            "title": "Run Title Fallback",
            "reason": "Cache hit optimization",
            "hypothesis": hyp,
            "regressed_case_ids": ["case-latency-1", "case-latency-2"],
        }
        results = [
            {"status": "ok", "metrics": {"score": 0.6, "safety": 0.9}},
            {"status": "ok", "metrics": {"score": 0.9, "safety": 0.95}},
            {"status": "ok", "metrics": {"score": 0.7, "safety": 0.92}},
        ]
        self.store.finalize_run(lab_id, run_id, state, manifest, results)

        # 1. Verify /api/v1/labs returns statistic, goal_status, confirmed_cost_usd, and signals
        status, labs_payload, _ = self._api("GET", "/api/v1/labs")
        self.assertEqual(status, 200)
        lab_item = next(item for item in labs_payload["data"]["items"] if item["lab_id"] == str(lab_id))
        summary = lab_item["dashboard_summary"]
        self.assertEqual(summary["statistic"], "median")
        self.assertIn("goal_status", summary)
        self.assertIn("confirmed_cost_usd", summary)
        self.assertEqual(summary["confirmed_cost_usd"], 0.0)
        self.assertIsInstance(summary["signals"], list)

        # 2. Verify /api/v1/labs/{id}/runs returns title, reason, hypothesis, regressed_case_ids, and median
        status, runs_payload, _ = self._api("GET", f"/api/v1/labs/{lab_id}/runs?limit=10")
        self.assertEqual(status, 200)
        run_item = next(item for item in runs_payload["data"]["items"] if item["run_id"] == str(run_id))
        self.assertEqual(run_item["title"], "Hypothesis on caching")
        self.assertEqual(run_item["reason"], "Cache hit optimization")
        self.assertEqual(run_item["hypothesis"], hyp)
        self.assertEqual(run_item["regressed_case_ids"], ["case-latency-1", "case-latency-2"])
        self.assertAlmostEqual(run_item["metrics"]["score"]["median"], 0.7)


    def test_archive_restore_preserves_path_history_and_bumps_revisions(self):
        self._record_run("baseline")
        _project, state_dir, _candidate = self.paths[0]
        path_before = self.store.get_lab(self.labs[0])["state_dir"]
        revision_before = self.store.get_revision()

        status, archived, _ = self._api(
            "POST", f"/api/v1/labs/{self.labs[0]}/lifecycle", {"action": "archive"},
        )
        self.assertEqual(status, 200, archived)
        self.assertEqual(archived["data"]["lifecycle"], "archived")
        self.assertEqual(self.store.get_revision(), revision_before + 1)
        self.assertEqual(self.store.get_lab(self.labs[0])["state_dir"], path_before)
        self.assertEqual(self.store.list_runs(self.labs[0])["total"], 1)
        self.assertTrue(state_dir.is_dir())

        status, restored, _ = self._api(
            "POST", f"/api/v1/labs/{self.labs[0]}/lifecycle", {"action": "restore"},
        )
        self.assertEqual(status, 200, restored)
        self.assertEqual(restored["data"]["lifecycle"], "active")
        self.assertEqual(self.store.get_revision(), revision_before + 2)
        self.assertEqual(self.store.list_runs(self.labs[0])["total"], 1)

    def test_delete_requires_exact_confirmation_and_preserves_shared_files(self):
        first_project, _first_state, first_candidate = self.paths[0]
        second_project, _second_state, _second_candidate = self.paths[1]
        run_id = self._record_run("baseline")
        self.store.append_history(self.labs[0], {"id": "baseline", "kind": "record"})
        shared_bytes = b"shared evidence" * 30000
        owned_bytes = b"owned evidence" * 30000
        shared_first = self.store.register_artifact(self.labs[0], "shared.bin", shared_bytes)
        shared_second = self.store.register_artifact(self.labs[1], "shared.bin", shared_bytes)
        owned = self.store.register_artifact(self.labs[0], "owned.bin", owned_bytes, run_id=run_id)
        shared_path = self.home / shared_first["object_path"]
        owned_path = self.home / owned["object_path"]
        self.assertTrue(shared_path.is_file())
        self.assertTrue(owned_path.is_file())

        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            root_ref = str(uuid.uuid4())
            conn.execute("INSERT INTO roots(root_ref,path,created_at,payload_json) VALUES(?,?,?,?)",
                         (root_ref, str(self.root), "now", "{}"))
            conn.execute("INSERT INTO catalog_metadata(root_ref,project_ref,lab_id,updated_at) VALUES(?,?,?,?)",
                         (root_ref, "project-first", str(self.labs[0]), "now"))
            conn.execute("INSERT INTO active_labs(root_ref,project_ref,lab_id,updated_at) VALUES(?,?,?,?)",
                         (root_ref, "project-first", str(self.labs[0]), "now"))
            conn.commit()
        finally:
            conn.close()

        path = f"/api/v1/labs/{self.labs[0]}/lifecycle"
        status, missing, _ = self._api("POST", path, {"action": "delete"})
        self.assertEqual(status, 400)
        self.assertEqual(missing["error"]["code"], "invalid_request")
        status, wrong, _ = self._api("POST", path,
                                     {"action": "delete", "confirm_lab_id": str(self.labs[1])})
        self.assertEqual(status, 400)
        self.assertEqual(wrong["error"]["code"], "invalid_request")
        self.assertTrue(self.store.get_lab(self.labs[0]))

        revision = self.store.get_revision()
        status, deleted, _ = self._api(
            "POST", path, {"action": "delete", "confirm_lab_id": str(self.labs[0])},
        )
        self.assertEqual(status, 200, deleted)
        self.assertEqual(deleted["data"]["deleted"], True)
        self.assertEqual(self.store.get_revision(), revision + 1)
        with self.assertRaises(EntityNotFoundError):
            self.store.get_lab(self.labs[0])
        self.assertTrue(self.store.get_lab(self.labs[1]))
        self.assertEqual(self.store.load_artifact(shared_second["artifact_id"]), shared_bytes)
        self.assertTrue(shared_path.is_file(), "shared evidence objects must remain available")
        self.assertFalse(owned_path.exists(), "unreferenced evidence objects must be removed")
        self.assertTrue(first_project.joinpath("program.md").is_file())
        self.assertTrue(first_project.joinpath("evaluator.py").is_file())
        self.assertTrue(first_project.joinpath("input.csv").is_file())
        self.assertTrue(first_candidate.joinpath("value.txt").is_file())

        conn = sqlite3.connect(self.store.db_path)
        try:
            self.assertEqual(conn.execute("SELECT count(*) FROM runs WHERE lab_id=?", (str(self.labs[0]),)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM repeats WHERE lab_id=?", (str(self.labs[0]),)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM metric_values WHERE lab_id=?", (str(self.labs[0]),)).fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM history WHERE lab_id=?", (str(self.labs[0]),)).fetchone()[0], 0)
            metadata_lab = conn.execute("SELECT lab_id FROM catalog_metadata WHERE project_ref='project-first'").fetchone()[0]
            active_count = conn.execute("SELECT count(*) FROM active_labs WHERE project_ref='project-first'").fetchone()[0]
            roots_count = conn.execute("SELECT count(*) FROM roots").fetchone()[0]
        finally:
            conn.close()
        self.assertIsNone(metadata_lab)
        self.assertEqual(active_count, 0)
        self.assertEqual(roots_count, 1)

    def test_delete_gc_serializes_concurrent_shared_artifact_registration(self):
        content = b"concurrent shared evidence" * 30000
        artifact = self.store.register_artifact(self.labs[0], "owned.bin", content)
        object_path = (self.home / artifact["object_path"]).resolve()
        registering_store = SQLiteStore(self.home)
        at_unlink = threading.Event()
        resume_gc = threading.Event()
        writer_attempted = threading.Event()
        writer_acquired = threading.Event()
        register_done = threading.Event()
        failures = []
        registered = []
        original_unlink = Path.unlink
        original_writer = registering_store._writer

        def pause_before_object_unlink(path, *args, **kwargs):
            if path.resolve() == object_path:
                at_unlink.set()
                if not resume_gc.wait(5):
                    raise TimeoutError("test did not release evidence GC")
            return original_unlink(path, *args, **kwargs)

        def delete_lab():
            try:
                self.store.set_lab_lifecycle(
                    self.labs[0], "delete", confirm_lab_id=str(self.labs[0]),
                )
            except BaseException as exc:
                failures.append(exc)

        def register_shared_artifact():
            try:
                registered.append(registering_store.register_artifact(
                    self.labs[1], "shared.bin", content,
                ))
            except BaseException as exc:
                failures.append(exc)
            finally:
                register_done.set()

        @contextmanager
        def observe_writer_attempt():
            writer_attempted.set()
            with original_writer() as conn:
                writer_acquired.set()
                yield conn

        with patch.object(Path, "unlink", new=pause_before_object_unlink), \
             patch.object(registering_store, "_writer", new=observe_writer_attempt):
            deleting = threading.Thread(target=delete_lab, daemon=True)
            deleting.start()
            at_unlink_seen = at_unlink.wait(5)
            if at_unlink_seen:
                registering = threading.Thread(target=register_shared_artifact, daemon=True)
                registering.start()
                writer_attempted_seen = writer_attempted.wait(5)
                # Registration must reach the shared writer and remain blocked until GC releases it.
                writer_acquired_while_locked = writer_acquired.wait(0.5)
                register_finished_while_locked = register_done.is_set()
            else:
                registering = None
                writer_attempted_seen = False
                writer_acquired_while_locked = False
                register_finished_while_locked = False
            resume_gc.set()
            deleting.join(5)
            if registering is not None:
                registering.join(5)

        self.assertTrue(at_unlink_seen, "delete did not reach final object GC")
        self.assertTrue(writer_attempted_seen, "registration did not attempt the shared writer lock")
        self.assertFalse(writer_acquired_while_locked,
                         "registration acquired the writer lock while GC held it")
        self.assertFalse(register_finished_while_locked,
                         "registration completed while GC held the shared writer lock")
        self.assertFalse(deleting.is_alive(), "delete thread did not finish")
        self.assertIsNotNone(registering)
        self.assertFalse(registering.is_alive(), "registration thread did not finish")
        self.assertTrue(writer_acquired.is_set(), "registration did not acquire the writer lock after GC")
        self.assertEqual(failures, [])
        self.assertEqual(len(registered), 1)
        self.assertTrue(object_path.is_file(), "a registered object must survive concurrent GC")
        self.assertEqual(registering_store.load_artifact(registered[0]["artifact_id"]), content)

    def test_lifecycle_changes_are_refused_for_active_and_recovery_required_operations(self):
        for index, status_name in enumerate(("queued", "recovery_required")):
            lab_id = self.labs[index]
            operation = self.store.enqueue_operation(lab_id, "trial", f"block-{index}", {"command": "trial"})
            if status_name == "recovery_required":
                self.store.transition_operation(operation["operation_id"], "queued", "recovery_required")
            status, payload, _ = self._api("POST", f"/api/v1/labs/{lab_id}/lifecycle", {"action": "archive"})
            self.assertEqual(status, 409, payload)
            self.assertTrue(self.store.get_lab(lab_id))

    def test_existing_v1_database_is_migrated_in_place_before_dashboard_reads(self):
        lab_id = self.labs[0]
        self._record_run("baseline")
        self.store.append_history(lab_id, {"id": "baseline", "kind": "record"})
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("ALTER TABLE labs DROP COLUMN lifecycle")
            conn.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
            conn.execute("PRAGMA user_version=1")
            conn.commit()
        finally:
            conn.close()

        status, payload, _ = self._api("GET", "/api/v1/labs")
        self.assertEqual(status, 200, payload)
        conn = sqlite3.connect(self.store.db_path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 2)
        finally:
            conn.close()
        self.assertEqual(self.store.get_lab(lab_id)["lifecycle"], "active")
        self.assertEqual(self.store.load_config(lab_id)["goal"], "test dashboard lifecycle")
        self.assertEqual(self.store.list_runs(lab_id)["total"], 1)

        # Verify _run_summary without config uses module-level _metric_summary without UnboundLocalError
        raw_run = {
            "run_id": "run-no-cfg",
            "alias": "trial-no-cfg",
            "kind": "trial",
            "status": "accepted",
            "repeats": [{"metrics": {"score": 0.75}}, {"metrics": {"score": 0.85}}],
            "manifest": {"id": "trial-no-cfg", "status": "accepted"},
        }
        summary_no_cfg = api_v1._run_summary(raw_run, config=None)
        self.assertEqual(summary_no_cfg["metrics"], {"score": [0.75, 0.85]})
        self.assertEqual(summary_no_cfg["statistic"], "worst")


if __name__ == "__main__":
    unittest.main()
