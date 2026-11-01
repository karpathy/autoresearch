try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from durable_workflow_helpers import run_baseline
from autoresearch_runtime.evaluator import _evaluate_once
from autoresearch_runtime.lab import init_lab, load_config, load_state, run_trial
from autoresearch_runtime.measurement import incomparable_with_current, migrate_manifest, next_sequence_id
from durable_workflow_helpers import run_probe


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "value.txt").write_text("1", encoding="utf-8")
        self.evaluator = self.root / "evaluator.py"
        self.evaluator.write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "value = int((candidate / 'value.txt').read_text())\n"
            "att = {'probe_env': os.environ.get('PROBE_TAG', '')}\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps({\n"
            "  'metrics': {'score': value},\n"
            "  'usage': {'requests': 0, 'cost_usd': 0},\n"
            "  'attestations': att,\n"
            "}))\n",
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

    def _config(self, **raw_overrides):
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.update(raw_overrides)
        self.config_path.write_text(json.dumps(raw), encoding="utf-8")
        return load_config(self.config_path)

    def test_manifest_migration_adds_schema_without_touching_disk_fixture(self):
        legacy = {"id": "trial-0001", "status": "accepted", "runs": []}
        migrated = migrate_manifest(legacy)
        self.assertEqual(migrated["schema"], 1)
        self.assertEqual(migrated["epoch"], 1)

    def test_baseline_supersede_opens_new_epoch_and_records_noise_floor(self):
        config = self._config()
        init_lab(config)
        first = run_baseline(config)
        self.assertEqual(first["epoch"], 1)
        state = load_state(config)
        self.assertEqual(state["noise_floor"]["primary"], 0.0)
        second = run_baseline(config, supersede=True, reason="new evaluator contract")
        self.assertEqual(second["epoch"], 2)
        state = load_state(config)
        self.assertEqual(len(state["epochs"]), 2)
        self.assertEqual(state["epochs"][0]["supersede_reason"], "new evaluator contract")

    def test_trial_from_prior_epoch_is_incomparable_with_current(self):
        config = self._config()
        init_lab(config)
        run_baseline(config)
        run_baseline(config, supersede=True, reason="epoch bump")
        manifest = {
            "epoch": 1,
            "evaluator": list(config.evaluator),
            "input_hashes": {item["path"]: item["sha256"] for item in load_state(config)["inputs"]},
            "config_sha256": load_state(config)["config_sha256"],
        }
        reason = incomparable_with_current(manifest, load_state(config), config)
        self.assertIn("incomparable_with_current", reason or "")

    def test_expected_attestations_mismatch_invalidates_run(self):
        config = self._config(expected_attestations={"probe_env": "required"})
        init_lab(config)
        (config.state_dir / "candidate").mkdir(parents=True, exist_ok=True)
        status, _, reason = _evaluate_once(
            config, config.state_dir / "runs" / "probe-test" / "repeat_01", 10,
            extra_env={"PROBE_TAG": "wrong"},
        )
        self.assertEqual(status, "invalid")
        self.assertIn("attestation", reason)



    def test_acceptance_min_gain_requires_margin(self):
        config = self._config(acceptance={"min_gain": 2.5, "confirm_repeats": 1})
        init_lab(config)
        run_baseline(config)
        (config.state_dir / "candidate" / "value.txt").write_text("2", encoding="utf-8")
        trial = run_trial(config, {
            "title": "Small gain", "observation": "x", "expected_effect": "y",
        })
        self.assertEqual(trial["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
