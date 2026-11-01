from __future__ import annotations

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from autoresearch_runtime.errors import LabError
from autoresearch_runtime.state import _save_manifest
from autoresearch_runtime.store import _read_json, _write_json
from autoresearch_runtime.evaluator import _spent
from autoresearch_runtime.resume import resume_run
from autoresearch_runtime.state import _runs, load_state
from durable_workflow_helpers import run_trial
from durable_workflow_helpers import run_baseline
from autoresearch_runtime.evaluator import _validate_result
from autoresearch_runtime.lab import init_lab, load_config


def _rows(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class NativeResumeTest(unittest.TestCase):
    def make_lab(
        self, *, repeats=1, fail_score=2, capabilities=False, cases_url=None,
        crash_before_case_checkpoint=False,
    ):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "source"
        source.mkdir()
        (source / "score.txt").write_text("1", encoding="utf-8")
        evaluator = root / "evaluator.py"
        counter = root / "calls.jsonl"
        case_resume_code = ""
        if cases_url:
            crash_before_checkpoint = (
                "        if not resume_id: raise SystemExit(9)\n"
                if crash_before_case_checkpoint else ""
            )
            case_resume_code = (
                "if value == 2:\n"
                "    from urllib.parse import urlencode\n"
                "    from urllib.request import Request, urlopen\n"
                "    from autoresearch_runtime.case_checkpoint import append_case, case_scope, completed_ids\n"
                "    case_log = run_dir / 'cases.jsonl'\n"
                "    for case_id in ('case-1', 'case-2'):\n"
                "        if case_id in completed_ids(case_log): continue\n"
                "        with case_scope(case_id):\n"
                f"            url = {cases_url!r} + '?' + urlencode({{'case': case_id}})\n"
                "            response = urlopen(Request(url), timeout=2).read().decode('utf-8')\n"
                f"{crash_before_checkpoint}"
                "        append_case(case_log, {'id': case_id, 'status': 'ok', 'response': response})\n"
                "        if not os.environ.get('AUTORESEARCH_RESUME_ID'):\n"
                "            raise SystemExit(9)\n"
            )
        evaluator.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            f"counter = Path({str(counter)!r})\n"
            "value = int((candidate / 'score.txt').read_text(encoding='utf-8'))\n"
            "resume_id = os.environ.get('AUTORESEARCH_RESUME_ID', '')\n"
            "with counter.open('a', encoding='utf-8') as log:\n"
            "    log.write(f'{value} {run_dir.name} resume={int(bool(resume_id))}\\n')\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "details = {'evidence_origin': 'offline_recompute' if value == 2 else 'fresh'}\n"
            f"{case_resume_code}"
            "result = {'metrics': {'score': value}, 'usage': {'requests': 0, 'cost_usd': 0}, 'details': details}\n"
            "(run_dir / 'result.json').write_text(json.dumps(result), encoding='utf-8')\n"
            f"if value == {fail_score!r} and run_dir.name == 'repeat_01' and not resume_id:\n"
            "    raise SystemExit(9)\n",
            encoding="utf-8",
        )
        config_path = root / "lab.json"
        raw = {
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["score.txt"],
            "evaluator": ["{python}", str(evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": repeats,
            "timeout_seconds": 5,
            "budget": {"max_seconds": 60, "max_trials": 4, "max_requests": 0, "max_cost_usd": 0},
        }
        if capabilities:
            fixture = root / "resume-fixture.json"
            report = {
                "schema": 1,
                "offline": True,
                "case_resume": True,
                "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            }
            fixture.write_text(json.dumps(report), encoding="utf-8")
            raw["evaluator_capabilities"] = {
                "case_resume": True,
                "offline_verified": True,
                "resume_fixture": {
                    "path": str(fixture),
                    "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                },
            }
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        config = load_config(config_path)
        init_lab(config)
        return config, counter

    def trial_that_crashes_after_writing_repeat_one(self, config):
        (config.state_dir / "candidate" / "score.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Raise score", "observation": "Score is low", "expected_effect": "Increase score",
            "evidence": ["baseline/repeat_01"], "falsification": "Score does not improve on repeat.",
        })
        self.assertEqual(trial["status"], "crash")
        return trial

    def mark_first_repeat_completed(self, config, trial):
        run_dir = config.state_dir / "runs" / trial["id"]
        result_path = run_dir / "repeat_01" / "result.json"
        result = _read_json(result_path)
        result["duration_seconds"] = 0.1
        result["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["runs"] = [result]
        manifest["status"] = "interrupted"
        _write_json(manifest_path, manifest)
        return result_path

    def test_saved_result_finalizes_without_rerun_and_preserves_provenance(self):
        config, counter = self.make_lab(repeats=1, capabilities=False)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        calls_before = counter.read_text(encoding="utf-8").splitlines()
        attempts = config.state_dir / "runs" / trial["id"] / "repeat_01" / "attempts.jsonl"
        _write_json(attempts, [
            {"event": "started", "attempt_id": "saved-result", "case_id": "case-saved"},
            {
                "event": "finished", "attempt_id": "saved-result", "case_id": "case-saved",
                "http_status": 200, "cost_usd": None,
            },
        ])
        saved_result = config.state_dir / "runs" / trial["id"] / "repeat_01" / "result.json"
        self.assertEqual(_read_json(saved_result)["details"]["evidence_origin"],
                         "offline_recompute")

        resumed = resume_run(config, trial["id"])

        self.assertEqual(resumed["status"], "accepted", resumed)
        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)
        self.assertEqual(resumed["runs"][0]["details"]["evidence_origin"], "offline_recompute")
        self.assertTrue(resumed.get("resume_id"))
        self.assertEqual(resumed["evidence"], ["baseline/repeat_01"])
        self.assertEqual(resumed["falsification"], "Score does not improve on repeat.")
        self.assertEqual(resumed["runs"][0]["usage"]["requests"], 1)
        self.assertEqual(resumed["runs"][0]["usage"]["unknown_requests"], 1)

    def test_invalid_result_can_finalize_from_a_saved_valid_result_without_rerun(self):
        config, counter = self.make_lab(repeats=1, capabilities=False)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        run_dir = config.state_dir / "runs" / trial["id"]
        result_path = run_dir / "repeat_01" / "result.json"
        result = _read_json(result_path)
        result["usage"] = {"requests": 1, "cost_usd": None, "unknown_requests": 1}
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest.update(status="invalid", reason="Invalid result.json: Invalid usage")
        _write_json(manifest_path, manifest)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        resumed = resume_run(config, trial["id"])

        self.assertEqual(resumed["status"], "accepted")
        self.assertEqual(resumed["runs"][0]["usage"], {
            "requests": 1, "cost_usd": None, "unknown_requests": 1,
        })
        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_saved_manifest_origin_is_restored_when_original_file_omitted_it(self):
        config, _counter = self.make_lab(repeats=1, capabilities=False)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        result_path = self.mark_first_repeat_completed(config, trial)
        result = _read_json(result_path)
        result.pop("details", None)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["runs"][0]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        manifest["runs"][0]["details"] = {"evidence_origin": "fresh"}
        _write_json(manifest_path, manifest)

        resumed = resume_run(config, trial["id"])

        self.assertEqual(resumed["runs"][0]["details"]["evidence_origin"], "fresh")

    def test_trial_resume_skips_completed_case_and_calls_only_pending_case(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(parse_qs(urlparse(self.path).query)["case"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"cached-response")

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        url = f"http://127.0.0.1:{server.server_port}/case"
        config, counter = self.make_lab(repeats=1, capabilities=True, cases_url=url)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        first_repeat = config.state_dir / "runs" / trial["id"] / "repeat_01" / "result.json"
        self.assertFalse(first_repeat.exists())
        spent_after_crash = _spent(_runs(config), config.state_dir)
        self.assertEqual(spent_after_crash["requests"], 1)
        self.assertEqual(spent_after_crash["unknown_requests"], 1)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        resumed = resume_run(config, trial["id"])

        calls_after = counter.read_text(encoding="utf-8").splitlines()
        run_log = config.state_dir / "runs" / trial["id"] / "repeat_01" / "run.log"
        self.assertEqual(resumed["status"], "accepted", (
            resumed, run_log.read_text(encoding="utf-8"), counter.read_text(encoding="utf-8"),
        ))
        self.assertEqual(calls_after.count("2 repeat_01 resume=0"), 1)
        self.assertEqual(calls_after.count("2 repeat_01 resume=1"), 1)
        self.assertEqual(len(calls_after), len(calls_before) + 1)
        self.assertEqual(hits, ["case-1", "case-2"])
        cases = config.state_dir / "runs" / trial["id"] / "repeat_01" / "cases.jsonl"
        self.assertEqual([item["id"] for item in _read_json(cases)],
                         ["case-1", "case-2"])
        self.assertEqual(resumed["runs"][0]["metrics"]["score"], 2)
        self.assertEqual(resumed["runs"][0]["usage"]["requests"], 2)
        self.assertEqual(resumed["runs"][0]["usage"]["unknown_requests"], 2)
        spent_after_resume = _spent(_runs(config), config.state_dir)
        self.assertEqual(spent_after_resume["requests"], 2)
        self.assertEqual(spent_after_resume["unknown_requests"], 2)

    def test_redirect_and_informational_responses_require_reconciliation(self):
        for status in (302, 103):
            with self.subTest(status=status):
                config, counter = self.make_lab(repeats=1, capabilities=True)
                run_baseline(config)
                trial = self.trial_that_crashes_after_writing_repeat_one(config)
                repeat_dir = config.state_dir / "runs" / trial["id"] / "repeat_01"
                (repeat_dir / "result.json").unlink()
                attempt_id = f"http-{status}"
                rows = [
                    {"event": "started", "attempt_id": attempt_id, "case_id": "case-redirect"},
                    {"event": "finished", "attempt_id": attempt_id, "case_id": "case-redirect",
                     "http_status": status, "cost_usd": None},
                ]
                _write_json(repeat_dir / "attempts.jsonl", rows)
                calls_before = counter.read_text(encoding="utf-8").splitlines()

                with self.assertRaises(LabError) as caught:
                    resume_run(config, trial["id"])

                self.assertEqual(caught.exception.code, "E_RESUME_RECONCILE")
                self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_http_503_can_resume_and_remains_unknown_cost(self):
        config, counter = self.make_lab(repeats=1, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        repeat_dir = config.state_dir / "runs" / trial["id"] / "repeat_01"
        (repeat_dir / "result.json").unlink()
        rows = [
            {"event": "started", "attempt_id": "http-503", "case_id": "case-retry"},
            {"event": "finished", "attempt_id": "http-503", "case_id": "case-retry",
             "http_status": 503, "cost_usd": None},
        ]
        _write_json(repeat_dir / "attempts.jsonl", rows)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        resumed = resume_run(config, trial["id"])

        self.assertEqual(resumed["status"], "accepted")
        self.assertEqual(len(counter.read_text(encoding="utf-8").splitlines()), len(calls_before) + 1)
        self.assertEqual(resumed["runs"][0]["usage"]["unknown_requests"], 1)

    def test_unknown_finished_http_outcome_requires_case_reconciliation(self):
        config, counter = self.make_lab(repeats=1, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        repeat_dir = config.state_dir / "runs" / trial["id"] / "repeat_01"
        (repeat_dir / "result.json").unlink()
        attempt_id = "unknown-status"
        rows = [
            {"event": "started", "attempt_id": attempt_id, "case_id": "case-unknown"},
            {"event": "finished", "attempt_id": attempt_id, "case_id": "case-unknown"},
        ]
        _write_json(repeat_dir / "attempts.jsonl", rows)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "uncertain|checkpoint|reconcil"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_resume_accumulates_time_from_failed_evaluator_attempts(self):
        config, _counter = self.make_lab(repeats=1, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        repeat_dir = config.state_dir / "runs" / trial["id"] / "repeat_01"
        (repeat_dir / "result.json").unlink()
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["runs"][0]["duration_seconds"] = 5.0
        manifest["status"] = "interrupted"
        _write_json(manifest_path, manifest)

        with patch(
            "autoresearch_runtime.resume._evaluate_once",
            return_value=("crash", {"duration_seconds": 2.0, "usage": {"requests": 0}}, "timed out"),
        ) as evaluate:
            resumed = resume_run(config, trial["id"])

        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(resumed["runs"][0]["duration_seconds"], 7.0)

    def test_successful_http_without_case_checkpoint_requires_reconciliation(self):
        hits = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(parse_qs(urlparse(self.path).query)["case"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"response-already-received")

            def log_message(self, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        url = f"http://127.0.0.1:{server.server_port}/case"
        config, counter = self.make_lab(
            repeats=1, capabilities=True, cases_url=url, crash_before_case_checkpoint=True,
        )
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        first_repeat = config.state_dir / "runs" / trial["id"] / "repeat_01"
        self.assertFalse((first_repeat / "result.json").exists())
        self.assertEqual(hits, ["case-1"])
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaises(LabError) as caught:
            resume_run(config, trial["id"])
        self.assertEqual(caught.exception.code, "E_RESUME_RECONCILE")

        self.assertEqual(hits, ["case-1"])
        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)
        _write_json((first_repeat / "cases.jsonl"), _rows(json.dumps({
            "id": "case-1", "status": "ok", "response": "reconciled offline",
        }) + "\n"))

        resumed = resume_run(config, trial["id"])

        self.assertEqual(resumed["status"], "accepted", resumed)
        self.assertEqual(hits, ["case-1", "case-2"])
        self.assertEqual(resumed["runs"][0]["usage"]["requests"], 2)



    def test_baseline_resume_reuses_saved_repeat_and_repeats_baseline_gates(self):
        config, counter = self.make_lab(repeats=2, fail_score=1, capabilities=True)
        with self.assertRaisesRegex(LabError, "Evaluator exited 9"):
            run_baseline(config)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        resumed = resume_run(config, "baseline")

        self.assertEqual(resumed["status"], "accepted")
        self.assertEqual(len(counter.read_text(encoding="utf-8").splitlines()), len(calls_before) + 1)
        self.assertEqual(load_state(config)["baseline_id"], "baseline")
        self.assertEqual([item["metrics"]["score"] for item in resumed["runs"]], [1, 1])

    def test_missing_repeat_requires_verified_offline_case_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=False)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "offline|resume"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_unfinished_attempt_record_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        attempts = config.state_dir / "runs" / trial["id"] / "repeat_01" / "attempts.jsonl"
        _write_json(attempts, _rows(json.dumps({
            "event": "started", "attempt_id": "uncertain-1", "case_id": "resume-case",
        }) + "\n"))
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "uncertain|attempt"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)


    def test_result_only_finalization_still_validates_attempt_journal(self):
        config, counter = self.make_lab(repeats=1, capabilities=False)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        attempts = config.state_dir / "runs" / trial["id"] / "repeat_01" / "attempts.jsonl"
        _write_json(attempts, _rows(json.dumps({
            "event": "started", "attempt_id": "uncertain-2", "case_id": "case-a",
        }) + "\n"))
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "uncertain|attempt"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)


    def test_completed_manifest_repeat_rejects_tampered_result_file(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        result_path = self.mark_first_repeat_completed(config, trial)
        result = _read_json(result_path)
        result["metrics"]["score"] = 99
        _write_json(result_path, result)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "result|fingerprint|hash"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_completed_manifest_repeat_rejects_tampered_manifest_metrics(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        self.mark_first_repeat_completed(config, trial)
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["runs"][0]["metrics"]["score"] = 99
        _write_json(manifest_path, manifest)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "metrics|result"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_completed_manifest_repeat_without_result_is_not_remeasured(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        result_path = self.mark_first_repeat_completed(config, trial)
        result_path.unlink()
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "completed|missing|result"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)


    def test_config_change_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        changed = _read_json(config.path)
        changed["timeout_seconds"] = changed["timeout_seconds"] + 1
        _write_json(config.path, changed)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "Configuration changed"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_pinned_evaluator_change_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        evaluator_path = Path(config.evaluator[1])
        evaluator_path.write_text(evaluator_path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "Input changed|frozen"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_parent_change_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["parent"] = "trial-9999"
        _write_json(manifest_path, manifest)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "parent|snapshot"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_skill_stamp_change_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        manifest_path = config.state_dir / "runs" / trial["id"] / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["skill_sha"] = "0" * 64
        _write_json(manifest_path, manifest)
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "skill|runtime|build"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_other_running_manifest_blocks_resume(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        _save_manifest(config, {"id": "trial-9999", "status": "running", "runs": []})
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "Another run|running"):
            resume_run(config, trial["id"])

        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)

    def test_unrelated_dirty_candidate_is_not_overwritten(self):
        config, counter = self.make_lab(repeats=2, capabilities=True)
        run_baseline(config)
        trial = self.trial_that_crashes_after_writing_repeat_one(config)
        unrelated = config.state_dir / "candidate" / "unrelated.txt"
        unrelated.write_text("keep this", encoding="utf-8")
        calls_before = counter.read_text(encoding="utf-8").splitlines()

        with self.assertRaisesRegex(LabError, "candidate|changed|snapshot"):
            resume_run(config, trial["id"])

        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep this")
        self.assertEqual(counter.read_text(encoding="utf-8").splitlines(), calls_before)



    def test_result_validation_accepts_legacy_shape_and_validates_optional_evidence(self):
        config, _counter = self.make_lab(repeats=1, capabilities=False)
        run_dir = config.state_dir / "runs" / "validation" / "repeat_01"
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.json"
        _write_json(result_path, {
            "metrics": {"score": 1}, "usage": {"requests": 0, "cost_usd": 0},
        })
        self.assertEqual(_validate_result(config, run_dir)["metrics"]["score"], 1)

        _write_json(result_path, {
            "metrics": {"score": 1},
            "usage": {"requests": 1, "cost_usd": 0, "unknown_requests": 1},
            "details": {"evidence_origin": "offline_recompute", "diagnostics": [{
                "case_id": "case-7", "slice": "decoder", "failure_kind": "timeout",
            }]},
        })
        validated = _validate_result(config, run_dir)
        self.assertEqual(validated["details"]["evidence_origin"], "offline_recompute")
        self.assertEqual(validated["usage"], {
            "requests": 1, "cost_usd": None, "confirmed_cost_usd": 0.0,
            "unknown_requests": 1,
        })

        for invalid in (
            {"metrics": {"score": 1}, "details": {"evidence_origin": "invented"}},
            {"metrics": {"score": 1}, "details": {"diagnostics": [{"case_id": "", "slice": "x", "failure_kind": "y"}]}},
            {"metrics": {"score": 1}, "usage": {"requests": 0, "cost_usd": 0, "unknown_requests": -1}},
        ):
            with self.subTest(invalid=invalid):
                _write_json(result_path, invalid)
                with self.assertRaises(LabError):
                    _validate_result(config, run_dir)

    def test_result_validation_preserves_unknown_cost_when_any_request_is_unpriced(self):
        config, _counter = self.make_lab(repeats=1, capabilities=False)
        run_dir = config.state_dir / "runs" / "unknown-cost" / "repeat_01"
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.json"
        _write_json(result_path, {
            "metrics": {"score": 1},
            "usage": {"requests": 41, "cost_usd": None, "unknown_requests": 41},
        })

        result = _validate_result(config, run_dir)

        self.assertEqual(result["usage"], {
            "requests": 41, "cost_usd": None, "unknown_requests": 41,
        })

        _write_json(result_path, {
            "metrics": {"score": 1},
            "usage": {
                "requests": 2, "cost_usd": None, "confirmed_cost_usd": 0.25,
                "unknown_requests": 1,
            },
        })
        self.assertEqual(_validate_result(config, run_dir)["usage"], {
            "requests": 2, "cost_usd": None, "confirmed_cost_usd": 0.25,
            "unknown_requests": 1,
        })

        _write_json(result_path, {
            "metrics": {"score": 1}, "usage": {"requests": 3, "cost_usd": None},
        })
        with self.assertRaises(LabError):
            _validate_result(config, run_dir)

        _write_json(result_path, {
            "metrics": {"score": 1},
            "usage": {"requests": 3, "cost_usd": None, "unknown_requests": 3},
        })
        self.assertEqual(_validate_result(config, run_dir)["usage"], {
            "requests": 3, "cost_usd": None, "unknown_requests": 3,
        })


if __name__ == "__main__":
    unittest.main()
