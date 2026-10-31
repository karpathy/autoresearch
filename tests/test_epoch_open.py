try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch_runtime.errors import LabError
from autoresearch_runtime.lab import init_lab, load_config, load_state
from autoresearch_runtime.measurement import open_contract_epoch
from autoresearch_runtime.state import _check_frozen
from autoresearch_runtime.store import _sha256


class EpochOpenTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json\nfrom pathlib import Path\nimport sys\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({"
            "'metrics': {'score': 1}, 'usage': {'requests': 0, 'cost_usd': 0}}))\n",
            encoding="utf-8",
        )
        self.config_path = self.root / "lab.json"
        self.config_path.write_text(json.dumps({
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["value.txt"],
            "evaluator": ["{python}", str(self.evaluator.resolve()), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 2,
            "timeout_seconds": 30,
            "budget": {"max_seconds": 60, "max_trials": 5, "max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")
        config = load_config(self.config_path)
        init_lab(config)
        self.config = load_config(self.config_path)
        self.state_before = load_state(self.config)

    def _prepare_legacy_source_contract(self, *, include_candidate_copy=True):
        project = self.root / "legacy-migration"
        source = project / "source"
        source.mkdir(parents=True)
        (source / "value.txt").write_text("1", encoding="utf-8")
        legacy_bytes = b"immutable legacy workbook snapshot"
        (source / "legacy.xlsx").write_bytes(legacy_bytes)
        protected = project / "protected.xlsx"
        protected.write_bytes(legacy_bytes)
        evaluator = project / "evaluator.py"
        evaluator.write_text("pass\n", encoding="utf-8")
        config_path = project / "lab.json"
        config_path.write_text(json.dumps({
            "source_root": "source",
            "state_dir": "state",
            "source_paths": ["value.txt", "legacy.xlsx"],
            "editable_files": ["value.txt", "legacy.xlsx"],
            "evaluator": ["{python}", str(evaluator.resolve()), "{candidate}", "{run_dir}"],
            "metrics": {"primary": "score", "direction": "max", "guards": []},
            "repeats": 2,
            "timeout_seconds": 30,
            "budget": {"max_seconds": 60, "max_trials": 5, "max_requests": 0, "max_cost_usd": 0},
        }), encoding="utf-8")
        init_lab(load_config(config_path))
        state_before = load_state(load_config(config_path))
        candidate = project / "state" / "candidate"
        manifest_path = project / "state" / "runs" / "old" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_bytes = b'{"id":"old","status":"accepted","runs":[]}'
        manifest_path.write_bytes(manifest_bytes)

        if not include_candidate_copy:
            (candidate / "legacy.xlsx").unlink()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        raw["source_paths"] = ["value.txt"]
        raw["editable_files"] = ["value.txt"]
        raw["input_files"] = [str(protected.resolve())]
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        config = load_config(config_path)
        state_after_contract = {
            **state_before,
            "config_sha256": _sha256(config_path),
            "inputs": [{"path": str(item), "sha256": _sha256(item)} for item in config.input_files],
        }
        return {
            "project": project,
            "config_path": config_path,
            "config": config,
            "candidate": candidate,
            "protected": protected,
            "legacy_bytes": legacy_bytes,
            "state_before": state_before,
            "state_after_contract": state_after_contract,
            "manifest_path": manifest_path,
            "manifest_bytes": manifest_bytes,
        }

    def test_open_contract_epoch_updates_repeats_and_epoch(self):
        runs_dir = self.config.state_dir / "runs" / "baseline"
        runs_dir.mkdir(parents=True)
        manifest_path = runs_dir / "manifest.json"
        sibling_path = runs_dir / "keep-me.txt"
        manifest_bytes = b'{"id":"baseline","status":"accepted","runs":[]}'
        sibling_bytes = b"unchanged payload"
        manifest_path.write_bytes(manifest_bytes)
        sibling_path.write_bytes(sibling_bytes)

        before_epoch = int(self.state_before.get("current_epoch") or 1)
        result = open_contract_epoch(self.config_path, {"repeats": 3}, reason="new contract")

        self.assertEqual(result["epoch_id"], before_epoch + 1)
        self.assertEqual(result["repeats"], 3)
        config = load_config(self.config_path)
        self.assertEqual(config.repeats, 3)
        state = load_state(config)
        self.assertEqual(state["config_sha256"], _sha256(self.config_path))
        self.assertEqual(state["epochs"][-1]["repeats"], 3)
        self.assertIsNone(state["baseline_id"])
        self.assertIsNone(state["best_id"])
        self.assertFalse(state["baseline_verified"])
        self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
        self.assertEqual(sibling_path.read_bytes(), sibling_bytes)
        copy_path = config.state_dir / "lab.json"
        self.assertEqual(copy_path.read_bytes(), self.config_path.read_bytes())
        self.assertEqual(state["config_copy_sha256"], _sha256(copy_path))

    def test_empty_reason_leaves_lab_json_unchanged(self):
        before = self.config_path.read_bytes()
        with self.assertRaises(LabError):
            open_contract_epoch(self.config_path, {"repeats": 3}, reason="   ")
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_epoch_resnapshots_legacy_protected_copy_and_keeps_history(self):
        case = self._prepare_legacy_source_contract()
        with self.assertRaisesRegex(LabError, "Candidate contains a copy of a protected input"):
            _check_frozen(case["config"], case["state_after_contract"])

        prior_head = case["state_before"]["best_commit"]
        open_contract_epoch(case["config_path"], {"repeats": 3}, reason="strict source contract migration")

        config = load_config(case["config_path"])
        state = load_state(config)
        self.assertFalse((case["candidate"] / "legacy.xlsx").exists())
        self.assertEqual(
            {path.relative_to(case["candidate"]).as_posix() for path in case["candidate"].rglob("*")
             if path.is_file() and ".git" not in path.relative_to(case["candidate"]).parts},
            {"value.txt"},
        )
        self.assertEqual(state["source"], [{"path": "value.txt", "sha256": _sha256(case["candidate"] / "value.txt")}])
        self.assertEqual(state["source_snapshots"][-1]["files"], case["state_before"]["source"])
        self.assertEqual(state["best_commit"], prior_head)
        self.assertEqual(state["current_epoch"], 2)
        self.assertEqual(state["epochs"][-1]["repeats"], 3)
        self.assertEqual(case["protected"].read_bytes(), case["legacy_bytes"])
        self.assertEqual((case["project"] / "source" / "legacy.xlsx").read_bytes(), case["legacy_bytes"])
        self.assertEqual(case["manifest_path"].read_bytes(), case["manifest_bytes"])
        _check_frozen(config, state)

    def test_epoch_resnapshots_when_legacy_candidate_copy_is_missing(self):
        case = self._prepare_legacy_source_contract(include_candidate_copy=False)
        with self.assertRaises(FileNotFoundError):
            _check_frozen(case["config"], case["state_after_contract"])

        open_contract_epoch(case["config_path"], {"repeats": 3}, reason="strict source contract migration")
        config = load_config(case["config_path"])
        state = load_state(config)
        self.assertEqual(state["source_snapshots"][-1]["files"], case["state_before"]["source"])
        self.assertEqual(state["best_commit"], case["state_before"]["best_commit"])
        _check_frozen(config, state)

    def test_epoch_refuses_to_delete_changed_obsolete_candidate_file(self):
        case = self._prepare_legacy_source_contract()
        changed_bytes = b"changed after the previous source snapshot"
        (case["candidate"] / "legacy.xlsx").write_bytes(changed_bytes)
        state_path = case["project"] / "state" / "state.json"
        state_bytes = state_path.read_bytes()

        with self.assertRaisesRegex(LabError, "unchanged previous source snapshot"):
            open_contract_epoch(case["config_path"], {"repeats": 3}, reason="strict source contract migration")

        self.assertEqual((case["candidate"] / "legacy.xlsx").read_bytes(), changed_bytes)
        self.assertEqual(state_path.read_bytes(), state_bytes)


if __name__ == "__main__":
    unittest.main()
