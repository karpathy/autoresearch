"""Candidate file-set and protected-input rules, checked on draft contracts."""

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.config import validate_config_file
from autoresearch_runtime.errors import LabError

DRAFT = Path(__file__).resolve().parent / "fixtures" / "draft-config"


class CandidateRulesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "toy"
        shutil.copytree(DRAFT, self.root)
        self.path = self.root / "lab.json"
        self.raw = json.loads(self.path.read_text(encoding="utf-8"))

    def load(self, **updates):
        raw = {**self.raw, **updates}
        for key in [key for key, value in raw.items() if value is ...]:
            del raw[key]
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        return validate_config_file(self.path)

    def test_draft_loads(self):
        self.assertEqual(self.load().editable_files, frozenset({"rule.txt"}))

    def test_evaluator_inside_an_unrestricted_candidate_is_rejected(self):
        with self.assertRaisesRegex(LabError, "Evaluator must run outside the editable candidate"):
            self.load(editable_files=...)

    def test_protected_input_cannot_be_editable(self):
        with self.assertRaisesRegex(LabError, "Protected input files cannot be editable"):
            self.load(input_files=[str(self.root / "source" / "rule.txt")])

    def test_editable_files_must_exist_in_the_snapshot(self):
        with self.assertRaisesRegex(LabError, "Editable files must exist in the source snapshot"):
            self.load(editable_files=["missing.txt"])

    def test_missing_source_path_is_rejected(self):
        with self.assertRaisesRegex(LabError, "Missing or linked source path"):
            self.load(source_paths=["rule.txt", "nope.txt"])

    def test_git_internals_cannot_be_snapshotted(self):
        git_dir = self.root / "source" / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(LabError, "Git internals"):
            self.load(source_paths=[".git"], editable_files=...)

    def test_input_files_must_exist(self):
        with self.assertRaisesRegex(LabError, "Input files must exist"):
            self.load(input_files=["absent.json"])


if __name__ == "__main__":
    unittest.main()
