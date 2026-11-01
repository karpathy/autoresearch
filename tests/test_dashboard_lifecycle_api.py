from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime import api_v1
from autoresearch_runtime.sqlite_store import SQLiteStore


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
    def __init__(self, method: str, path: str):
        self.command = method
        self.path = path
        self.headers = {"Content-Length": "0"}
        self.rfile = io.BytesIO(b"")
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

    def _restore_home(self):
        if self.old_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.old_home

    def _api(self, method: str, path: str):
        handler = _Handler(method, path)
        with patch.object(api_v1, "authenticate"), \
             patch.object(api_v1, "resolve_home", return_value=self.home):
            handled = api_v1.dispatch(handler)
        self.assertTrue(handled)
        status, raw, _content_type, headers = handler.response
        return status, json.loads(raw) if raw else None, headers

    def _archive(self, lab_id) -> None:
        """Seed an archived lab in the database; the dashboard only lists labs, so nothing else writes this."""
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute("UPDATE labs SET lifecycle='archived' WHERE lab_id=?", (str(lab_id),))
            conn.commit()
        finally:
            conn.close()

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

        self._archive(self.labs[1])

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
