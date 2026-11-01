try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import os
import unittest
from tests import test_agent_guidance as guidance
from durable_workflow_helpers import run_baseline
from autoresearch_runtime.sqlite_store import SQLiteStore
from autoresearch_runtime.cockpit import build_cockpit, get_recommended_actions
from autoresearch_runtime.snapshot import init_lab
from autoresearch_runtime.state import _save_manifest, load_state


class GuideFactsTest(unittest.TestCase):
    setUp = guidance.AgentGuidanceTest.setUp

    def _write_run(self, run_id, status, state, *, epoch=1, parent=None):
        run_dir = self.config.state_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": run_id,
            "status": status,
            "epoch": epoch,
            "parent": parent,
            "config_sha256": state["config_sha256"],
            "input_hashes": {item["path"]: item["sha256"] for item in state["inputs"]},
            "evaluator": list(self.config.evaluator),
            "runs": [],
        }
        support.save_manifest(self.config, manifest)
        return manifest

    def test_paths_usage_metric_range_and_origin_are_real_facts(self):
        init_lab(self.config)
        baseline = run_baseline(self.config)
        card = build_cockpit(self.config, load_state(self.config))
        self.assertEqual(card["state"]["paths"]["candidate"], str(self.config.state_dir / "candidate"))
        self.assertEqual(card["state"]["spent"]["requests"], 0)
        self.assertEqual(card["evidence"][0]["evidence_origin"], "fresh")
        self.assertTrue(card["evidence"][0]["source"].startswith(f"/api/v1/labs/{self.config.lab_id}/runs/"))
        stats = card["evidence"][0]["metrics"][self.config.primary]
        self.assertLessEqual(stats["min"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["max"])

    def test_failed_or_running_baseline_never_recommends_a_new_baseline(self):
        init_lab(self.config)
        state = load_state(self.config)
        support.save_manifest(self.config, {"id": "baseline", "status": "accepted", "epoch": 1, "runs": []})
        for status, expected in (("crash", "INSPECT_RUN"), ("running", "WAIT")):
            action = get_recommended_actions(self.config, state, {"id": "baseline", "status": status})[0]
            self.assertEqual(action["action"], expected)

    def test_selected_run_exposes_hypothesis_only_when_requested(self):
        init_lab(self.config)
        baseline = run_baseline(self.config)
        path = self.config.state_dir / "runs" / baseline["id"] / "manifest.json"
        baseline.update(observation="Need evidence", expected_effect="Lower error", falsification="Error does not fall")
        support.save_manifest(self.config, baseline)
        state = load_state(self.config)
        self.assertNotIn("run_detail", build_cockpit(self.config, state))
        detail = build_cockpit(self.config, state, run_id=baseline["id"])["run_detail"]
        self.assertEqual(detail["hypothesis"]["falsification"], "Error does not fall")
        self.assertTrue(detail["manifest"].startswith(f"/api/v1/labs/{self.config.lab_id}/runs/"))

    def test_selected_old_run_keeps_running_trial_lifecycle_visible(self):
        init_lab(self.config)
        baseline = run_baseline(self.config)
        old_path = self.config.state_dir / "runs" / baseline["id"] / "manifest.json"
        baseline.update(observation="Old observation", expected_effect="Old effect",
                        falsification="Old falsifier")
        support.save_manifest(self.config, baseline)
        state = load_state(self.config)
        active_id = "trial-0002"
        self._write_run(active_id, "running", state, parent=baseline["id"])

        card = build_cockpit(self.config, state, run_id=baseline["id"])

        self.assertEqual(card["selected_run_id"], baseline["id"])
        self.assertEqual(card["run_detail"]["hypothesis"]["falsification"], "Old falsifier")
        self.assertIsNotNone(card["active_run"])
        self.assertEqual(card["active_run"]["id"], active_id)
        self.assertEqual(card["active_run"]["status"], "running")
        self.assertEqual(card["next_steps"][0]["action"], "WAIT")
        self.assertEqual(card["next_steps"][0]["run_id"], active_id)



    def test_selected_old_run_does_not_hide_latest_crash_guidance(self):
        init_lab(self.config)
        baseline = run_baseline(self.config)
        old_path = self.config.state_dir / "runs" / baseline["id"] / "manifest.json"
        baseline.update(observation="Old observation", expected_effect="Old effect",
                        falsification="Old falsifier")
        support.save_manifest(self.config, baseline)
        state = load_state(self.config)
        latest_id = "trial-0002"
        latest = self._write_run(latest_id, "crash", state, parent=baseline["id"])

        card = build_cockpit(self.config, state, run_id=baseline["id"])

        self.assertEqual(card["selected_run_id"], baseline["id"])
        self.assertEqual(card["run_detail"]["hypothesis"]["falsification"], "Old falsifier")
        self.assertEqual(card["next_steps"][0]["action"], "INSPECT_RUN")
        self.assertEqual(card["next_steps"][0]["run_id"], str(SQLiteStore().resolve_run_id(self.config.lab_id, latest["id"])))


if __name__ == "__main__":
    unittest.main()
