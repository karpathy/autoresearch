from __future__ import annotations

from pathlib import Path
import asyncio
import tempfile
import unittest
import sys

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.env import AutoResearchDiscoverEnv
from ttt_autoresearch.reward import AutoResearchRewardEvaluator
from ttt_autoresearch.runner import AutoResearchRunner


class EnvSmokeTests(unittest.TestCase):
    def test_env_prompt_and_reward_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("Focus on val_bpb.", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("# val_bpb: 1.100000\n", encoding="utf-8")
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            fixture_src = Path(__file__).parent / "fixtures" / "fake_train.py"
            (fixtures / "fake_train.py").write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

            config = TTTAutoResearchConfig(
                timeout_sec=1,
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
            self.assertIn("Current best val_bpb: 1.100000", prompt)
            self.assertTrue(env.check_format('{"summary":"s","rationale":"r","train_py":"# val_bpb: 0.900000\\n"}'))

            verify = asyncio.run(env.check_answer('{"summary":"s","rationale":"r","train_py":"# val_bpb: 0.900000\\n"}', 0))
            self.assertGreater(verify.reward, 0.0)
            next_state = env._create_next_state(0, '{"summary":"s","rationale":"r","train_py":"# val_bpb: 0.900000\\n"}', verify)
            self.assertAlmostEqual(next_state.current_best_val_bpb, 0.9)


if __name__ == "__main__":
    unittest.main()
