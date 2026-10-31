try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.case_checkpoint import (
    append_attempt,
    append_case,
    completed_ids,
    current_case_id,
    case_scope,
    pending_ids,
    summarize_usage,
)


class CaseCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases_path = self.root / "cases.jsonl"
        self.attempts_path = self.root / "attempts.jsonl"

    def test_duplicate_ok_case_ids_deduplicate(self):
        usage = {"requests": 1, "cost_usd": 0.1, "unknown_requests": 0}
        append_case(self.cases_path, {"id": "x", "status": "ok", "usage": usage})
        append_case(self.cases_path, {"id": "x", "status": "ok", "usage": usage})
        self.assertEqual(completed_ids(self.cases_path), {"x"})
        self.assertEqual(len(completed_ids(self.cases_path)), 1)

    def test_pending_ids_skips_done_preserving_order(self):
        self.assertEqual(pending_ids(["a", "b", "c"], {"b"}), ["a", "c"])

    def test_case_scope_is_nested_and_restored(self):
        self.assertIsNone(current_case_id())
        with case_scope("outer"):
            self.assertEqual(current_case_id(), "outer")
            with case_scope("inner"):
                self.assertEqual(current_case_id(), "inner")
            self.assertEqual(current_case_id(), "outer")
        self.assertIsNone(current_case_id())

    def test_append_after_torn_tail_starts_a_new_jsonl_record(self):
        self.cases_path.write_bytes(b'{"id":"unfinished"')

        append_case(self.cases_path, {"id": "finished", "status": "ok"})

        self.assertEqual(completed_ids(self.cases_path), {"finished"})

    def test_started_and_finished_events_count_as_one_attempt(self):
        append_attempt(self.attempts_path, {
            "event": "started", "attempt_id": "req-1", "case_id": "case-1",
        })
        append_attempt(self.attempts_path, {
            "event": "finished", "attempt_id": "req-1", "case_id": "case-1",
            "http_status": 200, "cost_usd": 0.25,
        })
        append_attempt(self.attempts_path, {
            "id": "legacy-case", "attempt": 1, "http_status": 429, "cost_usd": None,
        })

        self.assertEqual(summarize_usage(self.attempts_path), {
            "requests": 2, "unknown_requests": 1, "cost_usd": None,
            "confirmed_cost_usd": 0.25,
        })

    def test_invalid_lifecycle_cost_is_counted_as_unknown(self):
        append_attempt(self.attempts_path, {
            "event": "started", "attempt_id": "bad-cost", "case_id": "case-1",
        })
        append_attempt(self.attempts_path, {
            "event": "finished", "attempt_id": "bad-cost", "case_id": "case-1",
            "http_status": 200, "cost_usd": -1,
        })

        self.assertEqual(summarize_usage(self.attempts_path), {
            "requests": 1, "unknown_requests": 1, "cost_usd": None,
            "confirmed_cost_usd": 0.0,
        })

    def test_attempt_429_unknown_cost_does_not_complete_case(self):
        append_attempt(
            self.attempts_path,
            {"id": "c1", "attempt": 1, "http_status": 429, "cost_usd": None},
        )
        self.assertEqual(completed_ids(self.cases_path), set())
        summary = summarize_usage(self.attempts_path)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["unknown_requests"], 1)
        self.assertIsNone(summary["cost_usd"])
        self.assertEqual(summary["confirmed_cost_usd"], 0.0)

        append_attempt(
            self.attempts_path,
            {"id": "c1", "attempt": 2, "http_status": 200, "cost_usd": 0.5},
        )
        summary = summarize_usage(self.attempts_path)
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["unknown_requests"], 1)
        self.assertIsNone(summary["cost_usd"])
        self.assertEqual(summary["confirmed_cost_usd"], 0.5)
        self.assertEqual(completed_ids(self.cases_path), set())
