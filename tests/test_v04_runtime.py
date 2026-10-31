from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from autoresearch_runtime.bootstrap import prepare_lab
from autoresearch_runtime.lab import LabError, load_config, load_state
from autoresearch_runtime.resume import resume_run
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.case_checkpoint import summarize_usage
from autoresearch_runtime.runtimeops import reserve_and_launch_operation
from autoresearch_runtime.worker import run_operation_worker
from autoresearch_runtime.measurement import open_contract_epoch
from autoresearch_runtime.paths import add_root, all_roots
from autoresearch_runtime.cockpit import build_cockpit
from autoresearch_runtime.archive import archive_idle_labs, inspect_archive_candidates
from autoresearch_runtime.control_import import (
    import_control_report, load_control_manifests, promote_control_to_baseline,
)
from autoresearch_runtime.ingest import ingest_reports
from tests.durable_workflow_helpers import (
    approved_offline_profile, run_baseline, run_durable_operation, run_probe, run_replay, run_trial,
)


class SQLiteRuntimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_home = os.environ.get("AUTORESEARCH_HOME")
        os.environ["AUTORESEARCH_HOME"] = str(self.root / ".labs")
        self.addCleanup(self._restore_home)
        self.counter = self.root / "evaluation-count.txt"
        self.old_counter = os.environ.get("AUTORESEARCH_TEST_EVAL_COUNTER")
        os.environ["AUTORESEARCH_TEST_EVAL_COUNTER"] = str(self.counter)
        self.addCleanup(self._restore_counter)

        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "score.txt").write_text("1", encoding="utf-8")
        self.program = self.root / "program.md"
        self.program.write_text("Project: project\nGoal: improve score\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "import os\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "counter = Path(os.environ['AUTORESEARCH_TEST_EVAL_COUNTER'])\n"
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
            "id": "runtime-v04-score",
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

    def _restore_home(self):
        if self.old_home is None:
            os.environ.pop("AUTORESEARCH_HOME", None)
        else:
            os.environ["AUTORESEARCH_HOME"] = self.old_home

    def _restore_counter(self):
        if self.old_counter is None:
            os.environ.pop("AUTORESEARCH_TEST_EVAL_COUNTER", None)
        else:
            os.environ["AUTORESEARCH_TEST_EVAL_COUNTER"] = self.old_counter

    def _prepare(self):
        prepared = prepare_lab(self.program, self.spec, confirmed=True)
        return prepared, load_config(Path(prepared["config"]))

    def test_confirmed_prepare_runs_from_sqlite_without_json_state_shadows(self):
        prepared, config = self._prepare()

        home = self.root / ".labs"
        self.assertTrue((home / "autoresearch.sqlite3").is_file())
        self.assertFalse(Path(prepared["config"]).exists())
        self.assertFalse((config.state_dir / "state.json").exists())
        self.assertFalse((config.state_dir / "history.json").exists())
        self.assertEqual(hashlib.sha256(config.config_bytes).hexdigest(), load_state(config)["config_sha256"])
        self.assertEqual(load_state(config)["setup_pending"], True)

        self.assertEqual(run_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "score.txt").write_text("0", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Lower score",
            "observation": "Check rejection path",
            "expected_effect": "A regression is rejected",
        })
        self.assertEqual(trial["status"], "rejected")

        guide = build_cockpit(config, load_state(config))
        self.assertEqual(guide["state"]["baseline"]["score"], 1)
        self.assertEqual(guide["state"]["best"]["score"], 1)
        rejected_evidence = next(item for item in guide["evidence"] if item["id"] == trial["id"])
        self.assertEqual(rejected_evidence["primary"], 0)
        self.assertEqual(rejected_evidence["completed_repeats"], 1)

        self.assertEqual(load_state(config)["best_id"], "baseline")
        self.assertEqual((config.state_dir / "candidate" / "score.txt").read_text(encoding="utf-8"), "1")
        self.assertFalse((home / "catalog.json").exists())
        self.assertFalse((home / ".active").exists())
        for name in ("lab.json", "state.json", "history.json", "worker.json", "dashboard.html"):
            self.assertFalse((config.state_dir / name).exists(), name)
        for name in ("manifest.json", "result.json", "usage.json", "attempts.jsonl", "cases.jsonl", "sent-requests.jsonl"):
            self.assertEqual(list((config.state_dir / "runs").rglob(name)), [], name)

        again = prepare_lab(self.program, self.spec, confirmed=True)
        self.assertEqual(again["lab_id"], prepared["lab_id"])
        self.assertFalse(again["created"])

    def test_sqlite_manifest_updates_do_not_rewrite_legacy_catalog(self):
        _prepared, config = self._prepare()
        catalog_path = self.root / ".labs" / "catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({
            "schema": 1,
            "projects": [{"id": "legacy-project", "labs": [{
                "id": "legacy-lab", "path": str(config.state_dir), "status": "manual",
            }]}],
            "archived": [],
        }, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        catalog_path.write_bytes(original)

        self.assertEqual(run_baseline(config)["status"], "accepted")
        self.assertEqual(catalog_path.read_bytes(), original)

    def test_trial_finalizing_resumes_after_git_commit_without_evaluator_retry(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        candidate = config.state_dir / "candidate" / "score.txt"
        candidate.write_text("2", encoding="utf-8")
        store = SQLiteStore()
        crash_hook_dir = self.root / "worker-crash-hook"
        crash_hook_dir.mkdir()
        crash_marker = self.root / "worker-crash-marker.txt"
        hook_log = self.root / "worker-hook-loads.txt"
        (crash_hook_dir / "sitecustomize.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "with Path(os.environ['AUTORESEARCH_TEST_WORKER_HOOK_LOG']).open('a', encoding='utf-8') as _log:\n"
            "    _log.write(f'{os.getpid()} {sys.argv!r}\\n')\n"
            "if sys.argv and sys.argv[0] == '-m' and '--operation-id' in sys.argv:\n"
            "    from autoresearch_runtime.sqlite_store import SQLiteStore\n"
            "    _original_finalize_run = SQLiteStore.finalize_run\n"
            "    def _fail_after_git_commit(instance, lab_id, run_id, state, manifest, results=None):\n"
            "        if str(manifest.get('id', '')).startswith('trial-') and manifest.get('status') == 'accepted':\n"
            "            Path(os.environ['AUTORESEARCH_TEST_WORKER_CRASH_MARKER']).write_text('injected', encoding='utf-8')\n"
            "            raise RuntimeError('injected crash after candidate commit')\n"
            "        return _original_finalize_run(instance, lab_id, run_id, state, manifest, results)\n"
            "    SQLiteStore.finalize_run = _fail_after_git_commit\n",
            encoding="utf-8",
        )
        from autoresearch_runtime.commands.common import enqueue_and_start_operation, wait_for_operation
        import autoresearch_runtime.runtimeops as runtimeops
        with approved_offline_profile(config), mock.patch.dict(os.environ, {
            "PYTHONPATH": str(crash_hook_dir),
            "AUTORESEARCH_TEST_WORKER_CRASH_MARKER": str(crash_marker),
            "AUTORESEARCH_TEST_WORKER_HOOK_LOG": str(hook_log),
        }):
            operation = enqueue_and_start_operation(config.lab_id, "trial", {
                "hypothesis": {
                    "title": "Raise score",
                    "observation": "A candidate improves the measured score",
                    "expected_effect": "The best score should increase",
                },
                "confirmed": True,
            }, "test-trial-finalizing-worker-crash")
            worker_error = None
            try:
                wait_for_operation(operation["operation_id"])
            except LabError as error:
                worker_error = error
        self.assertTrue(hook_log.is_file(), "child Python did not load the temporary crash hook")
        self.assertRegex(hook_log.read_text(encoding="utf-8"), r"\['-m', '--operation-id'")
        self.assertIsNotNone(worker_error, "the injected worker crash must leave an uncertain operation")
        self.assertEqual(worker_error.code, "E_WORKER_DIED")
        self.assertTrue(crash_marker.is_file(), "the child worker must hit the injected finalization crash")
        crashed_operation = store.get_operation(operation["operation_id"])
        self.assertEqual(crashed_operation["status"], "recovery_required")
        self.assertEqual(crashed_operation["error"]["code"], "operation_outcome_uncertain")
        worker_pid = int(crashed_operation.get("pid") or 0)
        deadline = time.monotonic() + 10
        while worker_pid and runtimeops._pid_alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(runtimeops._pid_alive(worker_pid), "the crashed worker must exit before explicit recovery")

        manifest = store.load_manifest(config.lab_id, "trial-0001")
        self.assertEqual(manifest["status"], "finalizing")
        self.assertEqual(manifest["finalization"]["decision"], "accepted")
        self.assertEqual(manifest["parent"], "baseline")
        evaluator_calls_before_resume = self.counter.read_text()

        original_input = Path(load_state(config)["inputs"][0]["path"])
        original_bytes = original_input.read_bytes()
        original_input.write_bytes(original_bytes + b" ")
        with self.assertRaisesRegex(LabError, "Input changed"):
            resume_run(config, "trial-0001")
        original_input.write_bytes(original_bytes)

        import autoresearch_runtime.integrity as integrity
        original_stamp = integrity.skill_stamp
        with mock.patch.object(integrity, "skill_stamp", lambda: ("changed", "0" * 64)):
            with self.assertRaisesRegex(LabError, "runtime build differs"):
                resume_run(config, "trial-0001")
        self.assertIsNotNone(original_stamp)

        recovery = run_durable_operation(config, "recover", {})
        self.assertEqual(recovery["interrupted"], [])
        self.assertEqual(store.get_latest_operation(config.lab_id)["status"], "succeeded")
        self.assertEqual(store.load_manifest(config.lab_id, "trial-0001")["status"], "finalizing")

        operation = store.enqueue_operation(config.lab_id, "resume", "live-resume-test", {
            "command": "resume", "params": {"run_id": "trial-0001"},
        })
        self.assertTrue(store.claim_operation(operation["operation_id"], os.getpid()))
        with self.assertRaisesRegex(LabError, "live background worker"):
            resume_run(config, "trial-0001")
        store.transition_operation(operation["operation_id"], "running", "failed")

        candidate.write_text("999", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "Candidate changed"):
            resume_run(config, "trial-0001")
        candidate.write_text("2", encoding="utf-8")

        original_pin = store.load_pins(config.lab_id)[0]
        changed_payload = dict(original_pin["payload"])
        changed_payload["test_mutation"] = True
        changed_digest = hashlib.sha256(json.dumps(
            changed_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        store.register_pin(config.lab_id, original_pin["name"], changed_payload,
                           source_path=original_pin["source_path"], sha256=changed_digest)
        with self.assertRaisesRegex(LabError, "Passport pins changed"):
            resume_run(config, "trial-0001")
        store.register_pin(config.lab_id, original_pin["name"], original_pin["payload"],
                           source_path=original_pin["source_path"], sha256=original_pin["sha256"])

        store.append_attempt(config.lab_id, "trial-0001", "uncertain-attempt", "started", {
            "repeat_index": 1, "case_id": "case-1",
        })
        with self.assertRaisesRegex(LabError, "Uncertain unfinished HTTP attempt"):
            resume_run(config, "trial-0001")
        store.append_attempt(config.lab_id, "trial-0001", "uncertain-attempt", "finished", {
            "repeat_index": 1, "case_id": "case-1", "http_status": 429, "cost_usd": None,
        })

        native_run_id = str(store.resolve_run_id(config.lab_id, "trial-0001"))
        recovered = run_durable_operation(config, "resume", {"run_id": native_run_id})
        self.assertEqual(recovered["status"], "accepted")
        self.assertEqual(load_state(config)["best_id"], "trial-0001")
        self.assertEqual(candidate.read_text(encoding="utf-8"), "2")
        self.assertEqual(self.counter.read_text(), evaluator_calls_before_resume)

    def test_sqlite_case_resume_skips_completed_stub_transport_call(self):
        calls = self.root / "case-calls.jsonl"
        self.evaluator.write_text(
            "import io, json, os, sys\n"
            "from pathlib import Path\n"
            "import urllib.request, sitecustomize\n"
            "from autoresearch_runtime.case_checkpoint import append_case, case_scope, completed_ids\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "score = int((candidate / 'score.txt').read_text())\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "resume_id = os.environ.get('AUTORESEARCH_RESUME_ID', '')\n"
            "if score == 2:\n"
            "    class Response:\n"
            "        status = 200\n"
            "        code = 200\n"
            "        def __init__(self, body): self.fp = io.BytesIO(body); self.length = len(body)\n"
            "        def read(self): return self.fp.read()\n"
            "    for case_id in ('case-1', 'case-2'):\n"
            "        if case_id in completed_ids(run_dir / 'cases.jsonl'): continue\n"
            "        def stub(request, *args, **kwargs): return Response(('reply:' + case_id).encode())\n"
            "        urllib.request.urlopen = stub\n"
            "        sitecustomize._install_urllib(sitecustomize._STATE)\n"
            "        with case_scope(case_id):\n"
            "            request = urllib.request.Request('https://stub.invalid/' + case_id)\n"
            "            reply = urllib.request.urlopen(request, timeout=2).read().decode()\n"
            "        append_case(run_dir / 'cases.jsonl', {'id': case_id, 'status': 'ok', 'reply': reply})\n"
            f"        with Path({str(calls)!r}).open('a', encoding='utf-8') as log: log.write(case_id + ' resume=' + str(int(bool(resume_id))) + '\\n')\n"
            "        if not resume_id: raise SystemExit(9)\n"
            "result = {'metrics': {'score': score}, 'usage': {'requests': 0, 'cost_usd': 0}}\n"
            "(run_dir / 'result.json').write_text(json.dumps(result), encoding='utf-8')\n",
            encoding="utf-8",
        )
        evaluator_hash = hashlib.sha256(self.evaluator.read_bytes()).hexdigest()
        fixture = self.root / "offline-resume-fixture.json"
        fixture.write_text(json.dumps({
            "schema": 1, "offline": True, "case_resume": True,
            "evaluator_sha256": evaluator_hash,
        }), encoding="utf-8")
        fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
        known = json.loads(self.known_case.read_text(encoding="utf-8"))
        known["evaluator_sha256"] = evaluator_hash
        known["input_hashes"] = {str(fixture): fixture_hash}
        self.known_case.write_text(json.dumps(known), encoding="utf-8")
        spec = json.loads(self.spec.read_text(encoding="utf-8"))
        spec["repeats"] = 1
        spec["baseline_validation"]["known_case"]["sha256"] = hashlib.sha256(
            self.known_case.read_bytes()
        ).hexdigest()
        spec["evaluator_capabilities"] = {
            "case_resume": True, "offline_verified": True,
            "resume_fixture": {
                "path": str(fixture),
                "sha256": fixture_hash,
            },
        }
        self.spec.write_text(json.dumps(spec), encoding="utf-8")
        prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Resume a pending case", "observation": "The second case did not run",
            "expected_effect": "The resumed evaluation should finish without repeating case one",
        })
        self.assertEqual(trial["status"], "crash")
        store = SQLiteStore()
        self.assertEqual([item["id"] for item in store.load_checkpoints(
            config.lab_id, "trial-0001", repeat_index=1,
        )], ["case-1"])
        repeat_dir = config.state_dir / "runs" / "trial-0001" / "repeat_01"
        self.assertFalse((repeat_dir / "cases.jsonl").exists())
        self.assertFalse((repeat_dir / "attempts.jsonl").exists())
        self.assertFalse((repeat_dir / "sent-requests.jsonl").exists())

        resumed = resume_run(config, "trial-0001")
        self.assertEqual(resumed["status"], "accepted", resumed)
        self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), [
            "case-1 resume=0", "case-2 resume=1",
        ])
        self.assertEqual([item["id"] for item in store.load_checkpoints(
            config.lab_id, "trial-0001", repeat_index=1,
        )], ["case-1", "case-2"])
        attempts = store.load_attempts(config.lab_id, "trial-0001", repeat_index=1)
        self.assertEqual(sum(item.get("event") == "started" for item in attempts), 2)
        self.assertEqual(len(store.load_wire_events(config.lab_id, "trial-0001", repeat_index=1)), 2)

    def test_rejected_finalizing_resume_blocks_foreign_worktree_changes(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        candidate = config.state_dir / "candidate" / "score.txt"
        candidate.write_text("0", encoding="utf-8")

        with mock.patch("autoresearch_runtime.trial._restore_candidate",
                        side_effect=RuntimeError("injected crash before restore")):
            with self.assertRaisesRegex(RuntimeError, "crash before restore"):
                run_trial(config, {
                    "title": "Reject regression",
                    "observation": "This candidate scores below baseline",
                    "expected_effect": "The baseline remains best",
                })

        manifest = SQLiteStore().load_manifest(config.lab_id, "trial-0001")
        self.assertEqual(manifest["status"], "finalizing")
        self.assertEqual(manifest["finalization"]["decision"], "rejected")
        self.assertRegex(manifest["finalization"]["pre_restore_tree"], r"^[0-9a-f]{40}$")

        (config.state_dir / "candidate" / "foreign.tmp").write_text("unrelated", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "worktree changed after the finalizing decision"):
            resume_run(config, "trial-0001")

    def test_run_history_paginates_without_losing_old_best_or_reusing_trial_alias(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        store = SQLiteStore()
        baseline_id = str(store.resolve_run_id(config.lab_id, "baseline"))
        state = load_state(config)
        for index in range(1, 102):
            alias = f"trial-{index:04d}"
            run_id = store.create_run(config.lab_id, "trial", alias, 1, {
                "parent_run_id": baseline_id, "candidate_commit": state["best_commit"],
            })
            store.save_manifest(config.lab_id, run_id, {
                "id": alias, "status": "rejected", "epoch": 1, "runs": [],
                "parent": "baseline", "base_commit": state["best_commit"],
            })
        self.assertEqual(len([item for item in __import__("autoresearch_runtime.state", fromlist=["_runs"])._runs(config)
                              if item["id"].startswith("trial-")]), 101)
        (config.state_dir / "candidate" / "score.txt").write_text("0", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Keep baseline",
            "observation": "The new candidate regresses",
            "expected_effect": "The best should remain baseline",
        })
        self.assertEqual(trial["id"], "trial-0102")
        self.assertEqual(trial["parent"], "baseline")
        self.assertEqual(load_state(config)["best_id"], "baseline")

    def test_usage_and_wire_events_do_not_duplicate_lifecycle_requests(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        store = SQLiteStore()
        store.ensure_repeat_started(config.lab_id, "baseline", 1)
        store.append_attempt(config.lab_id, "baseline", "req-1", "started", {
            "repeat_index": 1, "case_id": "case-1",
        })
        store.append_attempt(config.lab_id, "baseline", "req-1", "finished", {
            "repeat_index": 1, "case_id": "case-1", "http_status": 200, "cost_usd": 0.25,
        })
        store.append_attempt(config.lab_id, "baseline", "legacy-sdk-1", "attempt", {
            "repeat_index": 1, "cost_usd": None,
        })
        store.append_usage(config.lab_id, "baseline", 1, {"requests": 1, "cost_usd": None})
        store.append_usage(config.lab_id, "baseline", 1, {"requests": 1, "cost_usd": None})

        attempts = config.state_dir / "runs" / "baseline" / "repeat_01" / "attempts.jsonl"
        wire = config.state_dir / "runs" / "baseline" / "repeat_01" / "sent-requests.jsonl"
        from autoresearch_runtime.wire_record import record_sent_request
        with mock.patch.dict(os.environ, {
            "AUTORESEARCH_WIRE_LOG": str(wire), "AUTORESEARCH_WIRE_LOG_BODY": "1",
        }):
            record_sent_request(
                url="https://example.test", method="POST", body='{"input":"x"}',
                headers={"Content-Type": "application/json"}, status=200,
                response_body='{"ok":true}', attempt=1,
            )
            record_sent_request(
                url="https://example.test", method="POST", body='{"input":"y"}',
                headers={"Content-Type": "application/json"}, status=200,
                response_body='{"ok":true}', attempt=1,
            )
        first_sidecar = wire.read_bytes()
        from autoresearch_runtime.evaluator import _import_sqlite_wire_log
        _import_sqlite_wire_log(config, wire.parent, (store, "baseline", 1))
        wire_rows = store.load_wire_events(config.lab_id, "baseline", repeat_index=1)
        self.assertEqual(
            {row["body"]: row["wire_sequence"] for row in wire_rows},
            {'{"input":"x"}': 1, '{"input":"y"}': 2},
        )
        from autoresearch_runtime.baseline import _request_identity_digest
        self.assertEqual(
            _request_identity_digest(config, "baseline", 1),
            hashlib.sha256(b'{"input":"x"}{"input":"y"}').hexdigest(),
        )

        # Reimporting the same recovery sidecar must preserve IDs and sequence values.
        wire.write_bytes(first_sidecar)
        _import_sqlite_wire_log(config, wire.parent, (store, "baseline", 1))
        wire_rows = store.load_wire_events(config.lab_id, "baseline", repeat_index=1)
        self.assertEqual(
            {row["body"]: row["wire_sequence"] for row in wire_rows},
            {'{"input":"x"}': 1, '{"input":"y"}': 2},
        )

        with mock.patch.dict(os.environ, {
            "AUTORESEARCH_WIRE_LOG": str(wire), "AUTORESEARCH_WIRE_LOG_BODY": "1",
        }):
            record_sent_request(
                url="https://example.test", method="POST", body='{"input":"z"}',
                headers={"Content-Type": "application/json"}, status=200,
                response_body='{"ok":true}', attempt=1,
            )
        _import_sqlite_wire_log(config, wire.parent, (store, "baseline", 1))
        wire_rows = store.load_wire_events(config.lab_id, "baseline", repeat_index=1)
        self.assertEqual(
            {row["body"]: row["wire_sequence"] for row in wire_rows},
            {'{"input":"x"}': 1, '{"input":"y"}': 2, '{"input":"z"}': 3},
        )
        self.assertEqual(
            _request_identity_digest(config, "baseline", 1),
            hashlib.sha256(b'{"input":"x"}{"input":"y"}{"input":"z"}').hexdigest(),
        )

        # A new evaluator phase restarts the sidecar line counter. An identical
        # request still represents a distinct lifecycle event and must be kept.
        with mock.patch.dict(os.environ, {
            "AUTORESEARCH_WIRE_LOG": str(wire), "AUTORESEARCH_WIRE_LOG_BODY": "1",
        }):
            record_sent_request(
                url="https://example.test", method="POST", body='{"input":"x"}',
                headers={"Content-Type": "application/json"}, status=200,
                response_body='{"ok":true}', attempt=1,
            )
        _import_sqlite_wire_log(config, wire.parent, (store, "baseline", 1))
        wire_rows = store.load_wire_events(config.lab_id, "baseline", repeat_index=1)
        self.assertEqual(
            sorted((row["wire_sequence"], row["body"]) for row in wire_rows),
            [
                (1, '{"input":"x"}'),
                (2, '{"input":"y"}'),
                (3, '{"input":"z"}'),
                (4, '{"input":"x"}'),
            ],
        )
        self.assertEqual(
            _request_identity_digest(config, "baseline", 1),
            hashlib.sha256(b'{"input":"x"}{"input":"y"}{"input":"z"}{"input":"x"}').hexdigest(),
        )
        stats = config.state_dir / "runs" / "baseline" / "repeat_01" / "net-stats.json"
        usage = config.state_dir / "runs" / "baseline" / "repeat_01" / "usage.json"
        store.set_document(config.lab_id, stats, {"rate_limited_responses": 1})
        from autoresearch_runtime.evaluator import _apply_net_stats
        _apply_net_stats(stats.parent)
        self.assertIsNone(store.get_document(config.lab_id, usage, None))
        totals = summarize_usage(attempts)
        self.assertEqual(totals, {
            "requests": 2, "unknown_requests": 1, "cost_usd": None,
            "confirmed_cost_usd": 0.25,
        })
        self.assertEqual(len(store.load_wire_events(config.lab_id, "baseline", repeat_index=1)), 4)
        usage_events = [item for item in store.load_attempts(config.lab_id, "baseline", repeat_index=1)
                        if item.get("event") == "usage"]
        self.assertEqual(len(usage_events), 2)
        self.assertFalse(wire.exists())

    def test_evaluator_clears_inherited_unbounded_retry_flag(self):
        _prepared, config = self._prepare()
        self.assertTrue(config.network_retry)
        from autoresearch_runtime.evaluator import _evaluate_once

        observed = {}

        def fake_process(_command, **kwargs):
            observed.update(kwargs["env"])
            return mock.Mock(returncode=1, stdout="", stderr="")

        with mock.patch.dict(os.environ, {"AUTORESEARCH_HTTP_RETRY": "1"}):
            with mock.patch("autoresearch_runtime.evaluator.subprocess.run", side_effect=fake_process):
                status, _result, _reason = _evaluate_once(
                    config, self.root / "provider-retry-test", 1.0, candidate=self.project,
                    extra_env={"AUTORESEARCH_HTTP_RETRY": "1"},
                )

        self.assertEqual(status, "crash")
        self.assertFalse("AUTORESEARCH_HTTP_RETRY" in observed, "retry flag leaked into evaluator environment")

    def test_worker_waits_for_launcher_lock_then_claims_once(self):
        _prepared, config = self._prepare()
        store = SQLiteStore()
        worker_done = __import__("threading").Event()
        worker_errors = []

        def launcher(operation_id):
            import threading

            def run():
                try:
                    run_operation_worker(operation_id)
                except Exception as error:
                    worker_errors.append(error)
                finally:
                    worker_done.set()

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(timeout=0.1)
            return os.getpid()

        with approved_offline_profile(config):
            operation = reserve_and_launch_operation(
                config,
                "recover",
                "worker-lock-race",
                {"command": "recover", "params": {"confirmed": True}},
                launcher,
            )
            self.assertTrue(worker_done.wait(timeout=5))
            self.assertEqual(worker_errors, [])
            self.assertEqual(store.get_operation(operation["operation_id"])["status"], "succeeded")

    def test_epoch_open_reads_and_writes_virtual_config_in_sqlite(self):
        _prepared, config = self._prepare()
        result = open_contract_epoch(config.path, {"repeats": 1}, reason="Update measurement repeats")
        updated = load_config(str(config.lab_id), allow_pending=True)
        self.assertEqual(result["epoch_id"], 2)
        self.assertEqual(updated.repeats, 1)
        self.assertEqual(load_state(updated)["current_epoch"], 2)
        self.assertFalse(config.path.exists())
        self.assertFalse((config.state_dir / "lab.json").exists())

    def test_sqlite_root_registration_keeps_same_basename_roots_distinct(self):
        self._prepare()
        left = self.root / "left-project" / ".labs"
        right = self.root / "right-project" / ".labs"
        left.mkdir(parents=True)
        right.mkdir(parents=True)

        left_record = add_root(left)
        right_record = add_root(right)
        self.assertNotEqual(left_record["id"], right_record["id"])
        self.assertIn(left.resolve(), all_roots())
        self.assertIn(right.resolve(), all_roots())
        with self.assertRaisesRegex(ValueError, "already registered"):
            add_root(self.root / "third-project" / ".labs", root_id=left_record["id"])

    def test_sqlite_labs_are_explicitly_blocked_from_legacy_archive_moves(self):
        _prepared, config = self._prepare()
        plan = inspect_archive_candidates(self.root / ".labs")
        self.assertIn("SQLite-backed labs", plan["blocked"])
        with self.assertRaises(LabError) as raised:
            archive_idle_labs(self.root / ".labs", apply=True)
        self.assertEqual(raised.exception.code, "E_LEGACY_READ_ONLY")
        self.assertTrue(config.state_dir.exists())

    def test_unverified_migrated_baseline_cannot_be_used_for_trial(self):
        _prepared, config = self._prepare()
        store = SQLiteStore()
        state = load_state(config)
        state["baseline_id"] = "legacy-baseline"
        state["baseline_verified"] = True
        store.save_state(config.lab_id, state)
        store.set_document(config.lab_id, "migration_binding", {
            "complete": True,
            "baseline_trust": {"usable_for_comparison": False, "warnings": ["missing source hash"]},
        })
        with self.assertRaisesRegex(LabError, "not verified for comparison"):
            run_trial(config, {
                "title": "Must be blocked", "observation": "Imported baseline trust is unknown",
                "expected_effect": "No comparison should run",
            })

    def test_new_native_baseline_releases_only_its_own_migration_trust_gate(self):
        _prepared, config = self._prepare()
        store = SQLiteStore()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        binding = {
            "complete": True,
            "baseline_trust": {
                "source_baseline_id": "baseline",
                "source_baseline_verified": True,
                "usable_for_comparison": False,
                "warnings": ["source result hash is missing"],
            },
        }
        store.set_document(config.lab_id, "migration_binding", binding)

        open_contract_epoch(
            config.path,
            {"repeats": config.repeats},
            reason="verify a native baseline",
        )
        config = load_config(str(config.lab_id), allow_pending=True)
        baseline = run_baseline(config)
        self.assertEqual(baseline["status"], "accepted")
        self.assertEqual(baseline["id"], "baseline-0002")
        self.assertEqual(store.get_document(config.lab_id, "migration_binding"), binding)
        self.assertEqual(load_state(config)["baseline_id"], "baseline-0002")

        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Use verified native baseline",
            "observation": "The imported comparison was quarantined",
            "expected_effect": "A later verified native baseline can support a trial",
        })
        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(trial["parent"], "baseline-0002")
        self.assertEqual(store.get_document(config.lab_id, "migration_binding"), binding)

    def test_sqlite_baseline_supersede_persists_epoch_before_creating_run(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")

        baseline = run_baseline(config, supersede=True, reason="verify a fresh measurement epoch")
        self.assertEqual(baseline["status"], "accepted")
        self.assertEqual(baseline["id"], "baseline-0002")
        self.assertEqual(baseline["epoch"], 2)
        self.assertEqual(load_state(config)["current_epoch"], 2)

        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Use superseding baseline",
            "observation": "A new epoch has an independently measured baseline",
            "expected_effect": "Trial should use the native baseline from epoch two",
        })
        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(trial["parent"], "baseline-0002")

    def test_imported_controls_are_sqlite_probes_and_never_adopted_as_baseline(self):
        _prepared, config = self._prepare()
        before = load_state(config)
        manifest = import_control_report(
            config.state_dir, self.root / "control-report.json", result_type="provider_check",
            label="Provider check", entries=[{"metrics": {"score": 4}}, {"metrics": {"score": 5}}],
        )
        self.assertEqual(manifest["id"], "control-0001")
        self.assertEqual(manifest["status"], "non_adoptable")
        self.assertFalse(manifest["adoptable"])
        self.assertFalse((config.state_dir / "controls").exists())
        stored = load_control_manifests(config.state_dir)
        self.assertEqual(len(stored), 1)
        self.assertEqual([item["metrics"]["score"] for item in stored[0]["runs"]], [4, 5])
        run = SQLiteStore().get_run(config.lab_id, "control-0001")
        self.assertEqual(run["kind"], "probe")
        self.assertEqual(load_state(config)["baseline_id"], before["baseline_id"])
        with self.assertRaisesRegex(LabError, "non-adoptable"):
            promote_control_to_baseline(config.state_dir, "control-0001")

    def test_probe_and_ingest_store_measurements_under_runs_without_json_shadows(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        state_before = load_state(config)
        probe = run_probe(config, "best", {}, "runtime", "candidate check", 1)
        self.assertEqual(probe["status"], "complete")
        self.assertFalse(probe["adoptable"])
        store = SQLiteStore()
        probe_manifest = store.load_manifest(config.lab_id, "probe-0001")
        self.assertEqual(probe_manifest["runs"][0]["metrics"]["score"], 1)
        self.assertEqual(load_state(config)["best_id"], state_before["best_id"])
        self.assertFalse((config.state_dir / "probes" / "probe-0001" / "summary.json").exists())
        self.assertEqual(list((config.state_dir / "runs" / "probe-0001").rglob("result.json")), [])

        source = self.root / "imported.json"
        source.write_text(json.dumps({"metrics": {"score": 7}, "commit": "external"}), encoding="utf-8")
        imported = ingest_reports(config.state_dir, source, result_type="experiment")
        self.assertEqual(len(imported), 1)
        imported_manifest = store.load_manifest(config.lab_id, "probe-0002")
        self.assertEqual(imported_manifest["status"], "non_adoptable")
        self.assertEqual(imported_manifest["runs"][0]["metrics"]["score"], 7)
        self.assertFalse(imported[0].exists())
        self.assertEqual(load_state(config)["best_id"], state_before["best_id"])

    def test_replay_stores_results_and_proof_in_sqlite(self):
        _prepared, config = self._prepare()
        self.assertEqual(run_baseline(config)["status"], "accepted")
        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        self.assertEqual(run_trial(config, {
            "title": "Increase score", "observation": "Candidate score rises",
            "expected_effect": "Replay confirms the accepted measurement",
        })["status"], "accepted")

        proof = run_replay(config, "trial-0001")
        self.assertEqual(proof["status"], "complete")
        self.assertFalse(proof["adoptable"])
        store = SQLiteStore()
        replay = store.get_run(config.lab_id, "replay-0001")
        self.assertEqual(replay["kind"], "replay")
        self.assertEqual(replay["manifest"]["runs"][0]["metrics"]["score"], 2)
        replay_dir = config.state_dir / "runs" / "replay-0001"
        self.assertFalse((replay_dir / "manifest.json").exists())
        for name in ("result.json", "usage.json", "attempts.jsonl", "cases.jsonl", "sent-requests.jsonl"):
            self.assertEqual(list(replay_dir.rglob(name)), [], name)
        self.assertFalse((config.state_dir / "runs" / "trial-0001" / "rechecks").exists())


if __name__ == "__main__":
    unittest.main()
