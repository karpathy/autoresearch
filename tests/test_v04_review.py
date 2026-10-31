from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from unittest import mock

from autoresearch_runtime.bootstrap import prepare_lab
from autoresearch_runtime.config import load_config
from autoresearch_runtime.cockpit import build_cockpit
from autoresearch_runtime.commands.guide import run as run_cli_guide
from autoresearch_runtime.dash_http import create_server
from autoresearch_runtime.lab import LabError, load_state
from autoresearch_runtime.resume import resume_run
from autoresearch_runtime.sqlite_store import SQLiteStore, StoreCorruptError
from tests.durable_workflow_helpers import run_baseline as run_durable_baseline, run_trial as run_durable_trial


class V04IndependentReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.old_home = os.environ.get("AUTORESEARCH_HOME")
        self.old_counter = os.environ.get("AUTORESEARCH_REVIEW_EVAL_COUNTER")
        os.environ["AUTORESEARCH_HOME"] = str(self.home)
        os.environ["AUTORESEARCH_REVIEW_EVAL_COUNTER"] = str(self.root / "eval-count.txt")
        self.addCleanup(self._restore_env)

        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "score.txt").write_text("1", encoding="utf-8")
        self.program = self.root / "program.md"
        self.program.write_text("Project: project\nGoal: improve score\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "counter = Path(os.environ['AUTORESEARCH_REVIEW_EVAL_COUNTER'])\n"
            "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "score = int((candidate / 'score.txt').read_text())\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': score}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        self.known_case = self.root / "known-case.json"
        self.known_case.write_text(json.dumps({
            "schema": 1,
            "id": "review-local-case",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        self.spec = self.root / "derived.json"
        self.spec.write_text(json.dumps({
            "project": str(self.project),
            "goal": "Improve the local score",
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(self.known_case),
                    "sha256": hashlib.sha256(self.known_case.read_bytes()).hexdigest(),
                },
            },
            "budget": {"max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")
        prepared = prepare_lab(self.program, self.spec, confirmed=True)
        self.config = load_config(Path(prepared["config"]))

    def _restore_env(self) -> None:
        for key, value in (("AUTORESEARCH_HOME", self.old_home),
                           ("AUTORESEARCH_REVIEW_EVAL_COUNTER", self.old_counter)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _interrupt_accepted_trial(self) -> tuple[SQLiteStore, dict, str]:
        config = self.config
        self.assertEqual(run_durable_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        store = SQLiteStore(self.home)
        original_finalize = SQLiteStore.finalize_run

        def fail_after_candidate_commit(instance, lab_id, run_id, state, manifest, results=None):
            if str(manifest.get("id", "")).startswith("trial-") and manifest.get("status") == "accepted":
                raise RuntimeError("injected crash after candidate commit")
            return original_finalize(instance, lab_id, run_id, state, manifest, results)

        with mock.patch.object(SQLiteStore, "finalize_run", fail_after_candidate_commit):
            with self.assertRaisesRegex(RuntimeError, "after candidate commit"):
                run_durable_trial(config, {
                    "title": "Raise the local score",
                    "observation": "The candidate contains score 2",
                    "expected_effect": "The score improves from 1 to 2",
                })
        manifest = store.load_manifest(config.lab_id, "trial-0001")
        self.assertEqual(manifest["status"], "finalizing")
        return store, manifest, self.config.state_dir.name

    def test_corrupt_sqlite_does_not_fall_back_to_present_json_shadows(self):
        config = self.config
        config.path.parent.mkdir(parents=True, exist_ok=True)
        config.path.write_bytes(config.config_bytes)
        (config.state_dir / "state.json").write_text(
            json.dumps({"best_id": "legacy-shadow", "baseline_verified": True}), encoding="utf-8",
        )
        store = SQLiteStore(self.home)
        store.db_path.write_bytes(b"not a SQLite database")
        before = store.db_path.read_bytes()

        with self.assertRaises(StoreCorruptError):
            load_config(config.path)
        with self.assertRaises(StoreCorruptError):
            load_state(config)
        self.assertEqual(store.db_path.read_bytes(), before)

    def test_untrusted_migration_binding_blocks_trial_without_importer(self):
        config = self.config
        store = SQLiteStore(self.home)
        state = load_state(config)
        state["baseline_id"] = "legacy-baseline"
        state["baseline_verified"] = True
        store.save_state(config.lab_id, state)
        binding = {
            "complete": True,
            "baseline_trust": {
                "source_baseline_id": "legacy-baseline",
                "source_baseline_verified": True,
                "usable_for_comparison": False,
                "warnings": ["historical result evidence is unavailable"],
            },
        }
        store.set_document(config.lab_id, "migration_binding", binding)
        self.assertEqual(store.get_document(config.lab_id, "migration_binding"), binding)

        counter = Path(os.environ["AUTORESEARCH_REVIEW_EVAL_COUNTER"])
        with self.assertRaisesRegex(LabError, "not verified for comparison"):
            run_durable_trial(config, {
                "title": "Must stay blocked",
                "observation": "A saved historical decision cannot repair missing evidence",
                "expected_effect": "No trial is compared against unverified history",
            })
        self.assertFalse(counter.exists(), "an untrusted migration binding must block before evaluator calls")

    def test_finalizing_resume_rechecks_epoch_parent_and_complete_candidate_snapshot(self):
        store, manifest, _ = self._interrupt_accepted_trial()
        config = self.config
        original_state = load_state(config)

        changed_state = dict(original_state)
        changed_state["current_epoch"] = int(original_state.get("current_epoch", 1)) + 1
        store.save_state(config.lab_id, changed_state)
        with self.assertRaisesRegex(LabError, "Resume epoch changed"):
            resume_run(config, "trial-0001")
        store.save_state(config.lab_id, original_state)

        manifest["parent"] = "trial-unknown-parent"
        store.save_manifest(config.lab_id, "trial-0001", manifest)
        with self.assertRaisesRegex(LabError, "parent or verified baseline changed"):
            resume_run(config, "trial-0001")
        manifest["parent"] = "baseline"
        store.save_manifest(config.lab_id, "trial-0001", manifest)

        unexpected = config.state_dir / "candidate" / "untracked-review-file.txt"
        unexpected.write_text("untracked candidate content", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "Candidate changed during evaluation"):
            resume_run(config, "trial-0001")
        unexpected.unlink()

        calls_before = (self.root / "eval-count.txt").read_text(encoding="utf-8")
        recovered = resume_run(config, str(store.resolve_run_id(config.lab_id, "trial-0001")))
        self.assertEqual(recovered["status"], "accepted")
        self.assertEqual((self.root / "eval-count.txt").read_text(encoding="utf-8"), calls_before)

    def test_sqlite_guide_handles_10000_runs_and_api_alias_or_uuid_refs(self):
        config = self.config
        store = SQLiteStore(self.home)
        state = load_state(config)
        baseline_id = store.create_run(config.lab_id, "baseline", "baseline", 1, {})
        store.save_manifest(config.lab_id, baseline_id, {
            "id": "baseline", "status": "accepted", "epoch": 1, "runs": [],
            "config_sha256": state["config_sha256"],
            "input_hashes": {item["path"]: item["sha256"] for item in state.get("inputs", [])},
            "evaluator": list(config.evaluator),
        })
        store.append_result(config.lab_id, baseline_id, 1, {
            "status": "ok", "metrics": {"score": 1}, "duration_seconds": 0.0,
        })
        state.update({"setup_pending": False, "baseline_id": "baseline", "baseline_verified": True,
                      "best_id": "baseline"})
        store.save_state(config.lab_id, state)

        aliases: list[tuple[str, str, str, str]] = []
        run_rows: list[tuple[str, str, int, str, str, str, str, str, str, str, str]] = []
        document_rows: list[tuple[str, str, str, str, int, str]] = []
        first_time = datetime(2027, 1, 1, tzinfo=timezone.utc)
        for index in range(1, 10_000):
            alias = f"trial-{index:05d}"
            run_id = str(uuid.uuid4())
            created_at = (first_time + timedelta(seconds=index)).isoformat()
            manifest = {
                "id": alias, "status": "rejected", "epoch": 1, "parent": "baseline",
                "runs": [{"repeat_index": 1, "status": "ok", "metrics": {"score": 0}}],
                "config_sha256": state["config_sha256"],
                "input_hashes": {item["path"]: item["sha256"] for item in state.get("inputs", [])},
                "evaluator": list(config.evaluator),
            }
            run_rows.append((str(config.lab_id), run_id, 1, "trial", alias, "rejected", str(baseline_id),
                             state["best_commit"], created_at, created_at, "{}"))
            aliases.append(("run", str(config.lab_id), alias, run_id))
            document_rows.append((str(config.lab_id), run_id, "manifest",
                                  json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                                  0, created_at))
        with store._writer() as conn:
            conn.executemany(
                "INSERT INTO runs(lab_id,run_id,epoch_id,kind,alias,status,parent_run_id,candidate_commit,"
                "created_at,updated_at,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)", run_rows,
            )
            conn.executemany(
                "INSERT INTO aliases(kind,scope_id,alias,run_id) VALUES(?,?,?,?)", aliases,
            )
            conn.executemany(
                "INSERT INTO run_documents(lab_id,run_id,document_kind,payload_json,revision,updated_at) "
                "VALUES(?,?,? ,?,?,?)", document_rows,
            )
        latest_alias = "trial-09999"
        latest_id = str(store.resolve_run_id(config.lab_id, latest_alias))

        samples: list[float] = []
        for _ in range(3):
            build_cockpit(config, load_state(config), run_id=latest_alias)
        for _ in range(40):
            started = time.perf_counter()
            guide = build_cockpit(config, load_state(config), run_id=latest_alias)
            samples.append((time.perf_counter() - started) * 1000)
        p95 = sorted(samples)[int(0.95 * len(samples)) - 1]
        print(f"10k-run guide: p50={sorted(samples)[len(samples)//2]:.1f}ms p95={p95:.1f}ms max={max(samples):.1f}ms")
        self.assertEqual(store.list_runs(config.lab_id, limit=1)["total"], 10_000)
        self.assertLessEqual(p95, 200.0)
        self.assertEqual(guide["run_detail"]["manifest"], f"/api/v1/labs/{config.lab_id}/runs/{latest_id}")
        self.assertLessEqual(len(guide["next_steps"]), 1)
        self.assertLessEqual(len(guide["signals"]), 3)
        self.assertLessEqual(len(guide["evidence"]), 5)
        cli_guide = run_cli_guide(SimpleNamespace(
            config=None, lab=str(config.lab_id), project=None, json=True, run_id=latest_alias,
        ), None)
        self.assertEqual(cli_guide["run_detail"]["manifest"],
                         f"/api/v1/labs/{config.lab_id}/runs/{latest_id}")
        self.assertTrue(Path(cli_guide["next_steps"][0]["argv"][0]).is_absolute())

        server = create_server(config.state_dir, port=0, stop_token="review-local-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        origin = f"http://127.0.0.1:{server.server_port}"
        for run_ref in (latest_alias, latest_id):
            request = Request(
                f"{origin}/api/v1/labs/{config.lab_id}/guide?run={run_ref}",
                headers={"Authorization": "Bearer review-local-token"},
            )
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read())
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["data"]["run_detail"]["manifest"],
                             f"/api/v1/labs/{config.lab_id}/runs/{latest_id}")
        request = Request(
            f"{origin}/api/v1/labs/{config.lab_id}/runs?limit=1000",
            headers={"Authorization": "Bearer review-local-token"},
        )
        with urlopen(request, timeout=5) as response:
            page = json.loads(response.read())["data"]
        self.assertEqual(page["limit"], 100)
        self.assertLessEqual(len(page["items"]), 100)


if __name__ == "__main__":
    unittest.main()
