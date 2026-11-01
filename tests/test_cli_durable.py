from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch_runtime.commands import baseline, epoch, recover, resume, step


class DurableCliDispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = SimpleNamespace(
            lab_id="b5ef9c01-3a58-4ca8-a63c-4eb79216a912",
            state_dir=Path(self.temp.name) / "state",
        )

    def test_sync_mutating_commands_use_the_common_durable_operation_path(self):
        with patch("autoresearch_runtime.commands.baseline.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.baseline.run_durable_operation", return_value={"status": "accepted"}) as run:
            baseline.run(SimpleNamespace(supersede=False, reason=""), None)
            run.assert_called_once_with(self.config, "baseline", {})

        run_id = "trial-0001"
        with patch("autoresearch_runtime.commands.resume.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.resume.run_durable_operation", return_value={"status": "rejected"}) as run:
            resume.run(SimpleNamespace(run_id=run_id), None)
            run.assert_called_once_with(self.config, "resume", {"run_id": run_id})

        with patch("autoresearch_runtime.commands.recover.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.recover.run_durable_operation", return_value={"interrupted": []}) as run:
            recover.run(SimpleNamespace(), None)
            run.assert_called_once_with(self.config, "recover", {"restore_candidate": True})

        spec_path = Path(self.temp.name) / "epoch.json"
        spec = {"metrics": {"primary": "score", "direction": "max"}}
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with patch("autoresearch_runtime.commands.epoch.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.epoch.run_durable_operation", return_value={"epoch": 2}) as run:
            epoch.run(SimpleNamespace(epoch_command="open", spec=spec_path, reason=""), None)
            run.assert_called_once_with(self.config, "epoch", {"action": "open", "spec": spec})

    def test_step_run_queues_recommended_baseline_and_trial(self):
        baseline_guide = {"status": "recommended", "recommended_action": {"action": "RUN_BASELINE"}}
        with patch("autoresearch_runtime.commands.step.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.step.run_step", return_value=dict(baseline_guide)), \
             patch("autoresearch_runtime.commands.step.run_durable_operation", return_value={"status": "accepted"}) as run, \
             patch("autoresearch_runtime.commands.step.load_state", return_value={}), \
             patch("autoresearch_runtime.commands.step.get_state_card", return_value={"stage": "ITERATING"}):
            result = step.run(SimpleNamespace(run=True, hypothesis=None), None)
            run.assert_called_once_with(self.config, "baseline", {})
        self.assertEqual(result["action_taken"], "baseline")

        hypothesis = {"title": "A", "observation": "B", "expected_effect": "C", "evidence": ["D"]}
        hypothesis_path = Path(self.temp.name) / "hypothesis.json"
        hypothesis_path.write_text(json.dumps(hypothesis), encoding="utf-8")
        normalized_hypothesis = {**hypothesis, "rationale": "B -> C"}
        trial_guide = {"status": "recommended", "recommended_action": {"action": "RUN_TRIAL"}}
        with patch("autoresearch_runtime.commands.step.resolve_config", return_value=self.config), \
             patch("autoresearch_runtime.commands.step.run_step", return_value=dict(trial_guide)), \
             patch("autoresearch_runtime.commands.step._git"), \
             patch("autoresearch_runtime.commands.step._git_bytes", return_value=b"diff"), \
             patch("autoresearch_runtime.commands.step.run_durable_operation", return_value={"status": "rejected"}) as run, \
             patch("autoresearch_runtime.commands.step.load_state", return_value={}), \
             patch("autoresearch_runtime.commands.step.get_state_card", return_value={"stage": "ITERATING"}):
            result = step.run(SimpleNamespace(run=True, hypothesis=hypothesis_path), None)
            run.assert_called_once_with(self.config, "trial", {"hypothesis": normalized_hypothesis})
        self.assertEqual(result["action_taken"], "trial")


if __name__ == "__main__":
    unittest.main()
