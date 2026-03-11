from __future__ import annotations

from pathlib import Path
import asyncio
import tempfile
import unittest
import sys

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.env import AutoResearchDiscoverEnv, AutoResearchState
from ttt_autoresearch.reward import AutoResearchRewardEvaluator
from ttt_autoresearch.runner import AutoResearchRunner

MINIMAL_VALID_TRAIN_PY = """from prepare import MAX_SEQ_LEN
TOTAL_BATCH_SIZE = 2048
DEVICE_BATCH_SIZE = 1
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0

class GPT:
    def forward(self, idx, targets=None, reduction='mean'):
        return 0

# val_bpb: 1.100000
print(f"val_bpb:          {1.1:.6f}")
"""


class EnvSmokeTests(unittest.TestCase):
    def test_state_prompt_shows_before_after_without_construction(self) -> None:
        state = AutoResearchState(
            timestep=1,
            construction=[],
            code="print('candidate')\n",
            value=-0.9,
            parent_values=[-1.1],
            observation="val_bpb: 0.900000\n",
            baseline_val_bpb=1.1,
            current_best_val_bpb=0.9,
            raw_score=0.9,
        )
        prompt = state.to_prompt(0.85, metric_name="val_bpb", maximize=False, language="python")
        self.assertIn("Here is the val_bpb before and after running the code above", prompt)
        self.assertIn("1.100000 -> 0.900000", prompt)

    def test_env_prompt_and_reward_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("Focus on val_bpb.", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text(MINIMAL_VALID_TRAIN_PY, encoding="utf-8")
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            fixture_src = Path(__file__).parent / "fixtures" / "fake_train.py"
            (fixtures / "fake_train.py").write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

            config = TTTAutoResearchConfig(
                execution_backend="local",
                max_concurrent_evaluations=1,
                timeout_sec=1,
                target_val_bpb=0.95,
                candidate_command_override=[sys.executable, "tests/fixtures/fake_train.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.1)
            AutoResearchDiscoverEnv.configure(bootstrap)
            AutoResearchRewardEvaluator.configure(bootstrap, runner)

            state = AutoResearchDiscoverEnv.create_initial_state("autoresearch")
            env = AutoResearchDiscoverEnv(renderer=None, initial_state=state, sampler=None, config=type("Cfg", (), {
                "problem_type": "autoresearch",
                "log_path": str(bootstrap.discover_log_dir),
                "eval_timeout": config.eval_timeout,
                "num_cpus_per_task": 0,
            })())

            prompt = env.get_question()
            self.assertIn("You are iteratively optimizing val_bpb.", prompt)
            self.assertIn("Current val_bpb (lower is better): 1.100000", prompt)
            self.assertIn("Target: 0.95", prompt)
            self.assertIn("Here is the last code we ran", prompt)
            self.assertIn("## Problem", prompt)
            self.assertIn("## Budget & Resources", prompt)
            self.assertIn("## Rules", prompt)
            self.assertIn("You may want to start your search from the current training script shown above.", prompt)
            self.assertIn("This is the current starting point selected by the search procedure.", prompt)
            self.assertIn("Reason about how you could further improve this training script under the fixed 5-minute training budget.", prompt)
            self.assertIn("Moderate increases in VRAM are acceptable if they lead to meaningful gains.", prompt)
            self.assertNotIn("Baseline val_bpb from the original script", prompt)
            self.assertNotIn("LOOP FOREVER", prompt)
            self.assertNotIn("results.tsv", prompt)
            self.assertNotIn("git reset", prompt)
            self.assertIn("TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0", prompt)
            payload = "<<<<<<< SEARCH\n# val_bpb: 1.100000\n=======\n# val_bpb: 0.900000\n>>>>>>> REPLACE"
            self.assertTrue(env.check_format(payload))

            verify = asyncio.run(env.check_answer(payload, 0))
            self.assertGreater(verify.reward, 0.0)
            next_state = env._create_next_state(0, payload, verify)
            self.assertAlmostEqual(next_state.current_best_val_bpb, 0.9)


if __name__ == "__main__":
    unittest.main()
