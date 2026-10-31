try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.case_repro import compare_repeat_dirs


def _write_sent(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_cases(path: Path, case_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for case_id in case_ids:
            stream.write(json.dumps({"id": case_id, "status": "ok"}) + "\n")


class CaseReproTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.left = self.root / "left"
        self.right = self.root / "right"
        self.left.mkdir()
        self.right.mkdir()

    def test_compare_skips_429_and_counts_mismatches(self):
        case_ids = ["c1", "c2", "c3"]
        _write_cases(self.left / "cases.jsonl", case_ids)
        _write_cases(self.right / "cases.jsonl", case_ids)

        body_a = '{"answer":1}'
        body_b = '{"answer":2}'
        body_c = '{"answer":3}'
        shared_before_c = body_c

        _write_sent(
            self.left / "sent-requests.jsonl",
            [
                {"status": 429, "response_body": shared_before_c},
                {"status": 200, "response_body": body_a},
                {"status": 200, "response_body": body_b},
                {"status": 200, "response_body": body_c},
            ],
        )
        _write_sent(
            self.right / "sent-requests.jsonl",
            [
                {"status": 429, "response_body": shared_before_c},
                {"status": 200, "response_body": body_a},
                {"status": 200, "response_body": body_b + "-diff"},
                {"status": 200, "response_body": body_c + "-diff"},
            ],
        )

        result = compare_repeat_dirs(self.left, self.right)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["mismatched"], 2)
        self.assertEqual(len(result["rows"]), 3)
        by_id = {row["case_id"]: row for row in result["rows"]}
        self.assertTrue(by_id["c1"]["same"])
        self.assertFalse(by_id["c2"]["same"])
        self.assertFalse(by_id["c3"]["same"])

    def test_failed_case_still_consumes_its_response_line(self):
        def write_cases(path: Path) -> None:
            rows = [
                {"id": "c1", "status": "ok"},
                {"id": "bad", "status": "error"},
                {"id": "c2", "status": "ok"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        write_cases(self.left / "cases.jsonl")
        write_cases(self.right / "cases.jsonl")
        _write_sent(
            self.left / "sent-requests.jsonl",
            [
                {"status": 200, "response_body": "A"},
                {"status": 200, "response_body": "FAIL"},
                {"status": 200, "response_body": "C"},
            ],
        )
        _write_sent(
            self.right / "sent-requests.jsonl",
            [
                {"status": 200, "response_body": "A"},
                {"status": 200, "response_body": "FAIL"},
                {"status": 200, "response_body": "C-diff"},
            ],
        )
        result = compare_repeat_dirs(self.left, self.right)
        by_id = {row["case_id"]: row for row in result["rows"]}
        self.assertEqual(result["total"], 2)
        self.assertNotIn("bad", by_id)
        self.assertTrue(by_id["c1"]["same"])
        self.assertFalse(by_id["c2"]["same"])

    def test_compare_uses_safe_response_hashes_without_body_capture(self):
        _write_cases(self.left / "cases.jsonl", ["c1"])
        _write_cases(self.right / "cases.jsonl", ["c1"])
        digest = "a" * 64
        _write_sent(self.left / "sent-requests.jsonl", [{"status": 200, "response_body_sha256": digest}])
        _write_sent(self.right / "sent-requests.jsonl", [{"status": 200, "response_body_sha256": digest}])

        result = compare_repeat_dirs(self.left, self.right)

        self.assertTrue(result["rows"][0]["same"])
        self.assertEqual(result["rows"][0]["left_sha256"], digest)

    def test_compare_errors_when_response_body_was_not_captured(self):
        _write_cases(self.left / "cases.jsonl", ["c1"])
        _write_cases(self.right / "cases.jsonl", ["c1"])
        _write_sent(self.left / "sent-requests.jsonl", [{"status": 200}])
        _write_sent(self.right / "sent-requests.jsonl", [{"status": 200}])

        with self.assertRaisesRegex(ValueError, "(?i)response body.*not captured|capture.*body"):
            compare_repeat_dirs(self.left, self.right)
