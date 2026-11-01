"""Focused CLI tests for explicit confirmation before a new SQLite lab."""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

from autoresearch_runtime.bootstrap import main as bootstrap_main
from autoresearch_runtime.bootstrap import prepare_lab
from autoresearch_runtime.sqlite_store import SQLiteStore


class BootstrapConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / ".labs"
        self.env = patch.dict(os.environ, {"AUTORESEARCH_HOME": str(self.home)})
        self.env.start()
        self.addCleanup(self.env.stop)

        self.program = self.root / "program.md"
        self.program.write_text("Goal: improve score\nProject: project\n", encoding="utf-8")
        self.project = self.root / "project"
        self.project.mkdir()
        (self.project / "value.txt").write_text("1\n", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text("raise SystemExit('must not execute during prepare')\n", encoding="utf-8")
        self.known_case = self.root / "known-case.json"
        self.known_case.write_text(json.dumps({
            "schema": 1,
            "id": "fixture-score-v1",
            "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {},
            "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1},
            "tolerance": 1e-9,
        }), encoding="utf-8")
        self.spec = self.root / "derived.json"
        self.spec.write_text(json.dumps({
            "project": "project",
            "evaluator": ["{python}", "evaluator.py", "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(self.known_case),
                    "sha256": hashlib.sha256(self.known_case.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")

    def invoke(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = bootstrap_main([
                "--program", str(self.program), "--spec", str(self.spec), *extra,
            ])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_new_sqlite_lab_requires_explicit_confirmation_before_writes(self):
        code, _stdout, stderr = self.invoke()

        self.assertNotEqual(code, 0)
        self.assertFalse(self.home.exists())
        self.assertIn("confirmation", stderr.lower())

    def test_prepare_helper_rejects_missing_confirmation_for_new_lab(self):
        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            prepare_lab(self.program, self.spec)

        self.assertFalse(self.home.exists())

    def test_confirmed_cli_prepare_creates_sqlite_lab(self):
        code, stdout, stderr = self.invoke("--confirmed")

        self.assertEqual(code, 0, stderr)
        response = json.loads(stdout)
        self.assertTrue(response["ok"])
        self.assertTrue(response["data"]["created"])
        self.assertTrue(response["data"]["lab_id"])

    def test_prepare_does_not_import_legacy_project_history(self):
        history = self.project / "history"
        history.mkdir()
        (history / "NOTES.md").write_text("Old measurements stay in the source project\n", encoding="utf-8")
        (history / "manifest.json").write_text(json.dumps({
            "title": "Earlier result", "status": "accepted", "metrics": {"score": 99},
        }), encoding="utf-8")

        code, stdout, stderr = self.invoke("--confirmed")

        self.assertEqual(code, 0, stderr)
        lab_id = json.loads(stdout)["data"]["lab_id"]
        store = SQLiteStore(self.home)
        self.assertEqual(store.get_document(lab_id, "history_catalog", {
            "schema": 1, "documents": [], "hypotheses": [],
        }), {"schema": 1, "documents": [], "hypotheses": []})


if __name__ == "__main__":
    unittest.main()
