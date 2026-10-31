try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.bootstrap import prepare_lab
from autoresearch_runtime.lab import init_lab, load_config, load_state
from autoresearch_runtime import lab_compat
from autoresearch_runtime.lab_compat import CURRENT_LAB_FORMAT, audit_lab
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.store import _sha256


class LabCompatTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({'metrics': {'score': 1}, "
            "'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        self.state_dir = self.root / "state"
        self.config_path = self.root / "lab.json"

    def _write_lab(self, *, include_format: bool = False, extra: dict | None = None) -> None:
        payload = {
            "source_root": str(self.source),
            "state_dir": str(self.state_dir),
            "source_paths": ["value.txt"],
            "input_files": [],
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 1,
            "timeout_seconds": 30,
            "budget": {"max_seconds": None, "max_trials": None, "max_requests": 0, "max_cost_usd": 0},
        }
        if include_format:
            payload["lab_format_version"] = CURRENT_LAB_FORMAT
        if extra:
            payload.update(extra)
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

    def _init_state(self, *, drop_epochs: bool = False, wrong_hash: bool = False) -> None:
        config = load_config(self.config_path, allow_pending=True)
        init_lab(config)
        state_path = self.state_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if drop_epochs:
            state.pop("epochs", None)
            state.pop("current_epoch", None)
        if wrong_hash:
            state["config_sha256"] = "0" * 64
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def test_audit_legacy_lab_missing_format_version(self):
        self._write_lab(include_format=False)
        self._init_state()
        config_before = self.config_path.read_bytes()
        state_before = (self.state_dir / "state.json").read_bytes()
        report = audit_lab(self.config_path)
        self.assertEqual(report["format_version"], 1)
        codes = [item["code"] for item in report["findings"]]
        self.assertIn("missing_format_version", codes)
        self.assertEqual(self.config_path.read_bytes(), config_before)
        self.assertEqual((self.state_dir / "state.json").read_bytes(), state_before)

    def test_format_migration_api_is_removed(self):
        self.assertFalse(hasattr(lab_compat, "migrate_lab"))

    def test_prepare_writes_lab_format_without_changing_contract_fingerprint(self):
        previous_home = os.environ.get("AUTORESEARCH_HOME")
        home = self.root / "home"
        home.mkdir()
        os.environ["AUTORESEARCH_HOME"] = str(home)
        self.addCleanup(lambda: os.environ.pop("AUTORESEARCH_HOME", None) if previous_home is None
                        else os.environ.__setitem__("AUTORESEARCH_HOME", previous_home))

        known_case_path = self.root / "known-case.json"
        known_case_path.write_text(json.dumps({
            "schema": 1, "id": "fixture-score-v1", "reproducible": True,
            "evaluator_sha256": hashlib.sha256(self.evaluator.read_bytes()).hexdigest(),
            "input_hashes": {}, "expected_metrics": {"score": 1},
            "observed_metrics": {"score": 1}, "tolerance": 1e-9,
        }), encoding="utf-8")
        program = self.root / "program.md"
        program.write_text("Goal: score\nMetric: score, max\n", encoding="utf-8")
        spec_path = self.root / "derived.json"
        spec_path.write_text(json.dumps({
            "project": str(self.source),
            "evaluator": ["{python}", str(self.evaluator), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(known_case_path),
                    "sha256": hashlib.sha256(known_case_path.read_bytes()).hexdigest(),
                },
            },
        }), encoding="utf-8")

        prepared = prepare_lab(program, spec_path, confirmed=True)
        config_path = Path(prepared["config"])
        store = SQLiteStore(home)
        lab_id = store.resolve_lab_id(config_path)
        self.assertEqual(str(lab_id), prepared["lab_id"])
        raw_config = store.load_config_bytes(lab_id)
        self.assertEqual(hashlib.sha256(raw_config).hexdigest(), store.load_projection(lab_id, "state")["config_sha256"])
        lab_raw = json.loads(raw_config)
        self.assertEqual(lab_raw.get("lab_format_version"), CURRENT_LAB_FORMAT)
        contract_like = {key: lab_raw[key] for key in lab_raw if key != "lab_format_version"}
        digest_with = hashlib.sha256(
            json.dumps(lab_raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        digest_without = hashlib.sha256(
            json.dumps(contract_like, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(digest_with, digest_without)
        loaded = load_config(config_path, allow_pending=True)
        self.assertEqual(str(loaded.lab_id), str(lab_id))

    def test_dependency_missing_is_not_auto_created(self):
        self.evaluator.write_text(
            "import openrouter_budget\n" + self.evaluator.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        self._write_lab(include_format=False)
        self._init_state()
        report = audit_lab(self.config_path)
        self.assertIn("dependency_missing", [item["code"] for item in report["findings"]])
        self.assertFalse((self.root / "openrouter_budget.py").is_file())
        self.assertFalse((self.state_dir / "openrouter_budget.py").is_file())
        report_after = audit_lab(self.config_path)
        self.assertIn("dependency_missing", [item["code"] for item in report_after["findings"]])


if __name__ == "__main__":
    unittest.main()
