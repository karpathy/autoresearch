try:
    import support  # noqa: F401 - discover -s tests adds tests to sys.path
except ImportError:
    from tests import support  # noqa: F401 - package import

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.config import load_config
from autoresearch_runtime.errors import LabError


REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_LAB = REPO_ROOT / "tests" / "fixtures" / "legacy-toy"


class StrictContractConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lab_dir = Path(self.temp.name) / "toy"
        shutil.copytree(TOY_LAB, self.lab_dir)
        self.config_path = self.lab_dir / "lab.json"
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        cases_path = self.lab_dir / "cases.json"
        self.raw["baseline_validation"] = {
            "mode": "historical",
            "runs": [
                {"name": "repeat-1", "metrics": {"mean_error": 1.0, "coverage": 1.0}},
                {"name": "repeat-2", "metrics": {"mean_error": 1.0, "coverage": 1.0}},
            ],
            "input_hashes": {
                "cases.json": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            },
        }
        self.raw["strict_contract"] = True

    def write_config(self, updates=None, *, snapshot=None):
        raw = copy.deepcopy(self.raw)
        if updates:
            raw.update(updates)
        self.config_path.write_text(json.dumps(raw), encoding="utf-8")
        snapshot_path = self.lab_dir / "program.snapshot.md"
        if snapshot is None:
            snapshot_path.unlink(missing_ok=True)
        else:
            snapshot_path.write_text(snapshot, encoding="utf-8")
        return self.config_path

    def test_strict_contract_requires_a_baseline_validation(self):
        with self.assertRaisesRegex(LabError, "Strict labs require baseline validation"):
            load_config(self.write_config({"baseline_validation": None}))

    def test_strict_contract_requires_snapshot_even_with_no_declared_budget(self):
        with self.assertRaisesRegex(LabError, "Strict lab is missing program.snapshot.md"):
            load_config(self.write_config({"budget_declared": [], "budget_sources": {}}))

    def test_strict_contract_requires_quotes_for_active_time_and_trial_budgets(self):
        cases = (
            ("max_seconds", 60, {}),
            ("max_seconds", 60, {"max_seconds": "not in snapshot"}),
            ("max_trials", 3, {}),
            ("max_trials", 3, {"max_trials": "not in snapshot"}),
        )
        for key, value, sources in cases:
            with self.subTest(key=key, sources=sources):
                raw_budget = dict(self.raw["budget"], **{key: value})
                with self.assertRaisesRegex(LabError, f"Budget source for {key} is not an exact quote"):
                    load_config(self.write_config({
                        "budget": raw_budget,
                        "budget_declared": [key],
                        "budget_sources": sources,
                    }, snapshot="Snapshot: max 60 seconds or at most 3 trials."))

    def test_strict_contract_accepts_quote_as_a_contiguous_snapshot_substring(self):
        raw_budget = dict(self.raw["budget"], max_trials=3)
        loaded = load_config(self.write_config({
            "budget": raw_budget,
            "budget_declared": ["max_trials"],
            "budget_sources": {"max_trials": "at most 3 trials"},
        }, snapshot="Program note: at most 3 trials total."))
        self.assertEqual(loaded.budget["max_trials"], 3)

    def test_strict_contract_does_not_require_quotes_for_null_time_or_trial_budgets(self):
        raw_budget = dict(self.raw["budget"], max_seconds=None, max_trials=None)
        loaded = load_config(self.write_config({
            "budget": raw_budget,
            "budget_declared": ["max_seconds", "max_trials"],
            "budget_sources": {},
        }, snapshot="A readable snapshot."))
        self.assertIsNone(loaded.budget["max_seconds"])
        self.assertIsNone(loaded.budget["max_trials"])

    def test_strict_contract_skips_quotes_for_request_and_cost_caps(self):
        raw_budget = dict(self.raw["budget"], max_requests=20, max_cost_usd=4.5)
        loaded = load_config(self.write_config({
            "budget": raw_budget,
            "budget_declared": ["max_requests", "max_cost_usd"],
            "budget_sources": {},
        }, snapshot="A readable snapshot."))
        self.assertEqual(loaded.budget["max_requests"], 20)
        self.assertEqual(loaded.budget["max_cost_usd"], 4.5)

    def test_strict_contract_rejects_invalid_provenance_types_and_keys(self):
        cases = (
            ({"budget_declared": "max_trials", "budget_sources": {}},),
            ({"budget_declared": ["unexpected"], "budget_sources": {}},),
            ({"budget_declared": [], "budget_sources": []},),
        )
        for (updates,) in cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(LabError, "Invalid strict budget provenance"):
                    load_config(self.write_config(updates, snapshot="A readable snapshot."))

    def test_non_strict_config_does_not_require_baseline_provenance_or_snapshot(self):
        loaded = load_config(self.write_config({
            "strict_contract": False,
            "baseline_validation": None,
            "budget_declared": "not a list",
            "budget_sources": [],
        }))
        self.assertEqual(loaded.budget["max_trials"], 5)

    def test_strict_contract_flag_must_be_boolean(self):
        with self.assertRaisesRegex(LabError, "strict_contract must be a boolean"):
            load_config(self.write_config({"strict_contract": 1}))


if __name__ == "__main__":
    unittest.main()
