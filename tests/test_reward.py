from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import sys

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.env import AutoResearchState
from ttt_autoresearch.reward import AutoResearchRewardEvaluator, reward_for_result
from ttt_autoresearch.runner import AutoResearchRunner, RunResult


class RewardTests(unittest.TestCase):
    def test_reward_mapping(self) -> None:
        result = RunResult(
            status="success",
            val_bpb=0.9,
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            elapsed_sec=1.0,
            workspace_path=Path("."),
            metrics_path=None,
            command=["python", "train.py"],
            returncode=0,
        )
        reward, correctness = reward_for_result(1.0, result)
        self.assertAlmostEqual(reward, 0.1)
        self.assertEqual(correctness, 1.0)

        timeout_result = RunResult(
            status="timeout",
            val_bpb=None,
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            elapsed_sec=1.0,
            workspace_path=Path("."),
            metrics_path=None,
            command=["python", "train.py"],
            returncode=None,
        )
        reward, correctness = reward_for_result(1.0, timeout_result)
        self.assertEqual(reward, -0.5)
        self.assertEqual(correctness, 0.0)

    def test_evaluator_uses_inner_metric_as_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("# val_bpb: 1.000000\n", encoding="utf-8")
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            fixture_src = Path(__file__).parent / "fixtures" / "fake_train.py"
            (fixtures / "fake_train.py").write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

            config = TTTAutoResearchConfig(
                timeout_sec=1,
                baseline_command_override=[sys.executable, "tests/fixtures/fake_train.py"],
                candidate_command_override=[sys.executable, "tests/fixtures/fake_train.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.0)
            AutoResearchRewardEvaluator.configure(bootstrap, runner)
            evaluator = AutoResearchRewardEvaluator(problem_type="autoresearch", log_dir=str(bootstrap.run_dir))
            state = AutoResearchState(
                timestep=-1,
                construction=[],
                code=(root / "train.py").read_text(encoding="utf-8"),
                value=-1.0,
                baseline_val_bpb=1.0,
                current_best_val_bpb=1.0,
            )
            payload = '{"summary":"improve","rationale":"lower loss","train_py":"# val_bpb: 0.900000\\n"}'
            result = evaluator.get_reward(payload, state)
            self.assertGreater(result["reward"], 0.0)
            self.assertEqual(result["correctness"], 1.0)


if __name__ == "__main__":
    unittest.main()
