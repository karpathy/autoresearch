from __future__ import annotations

from pathlib import Path
import json
import tempfile
import threading
import time
import unittest
import sys

from ttt_autoresearch.config import BootstrapContext, TTTAutoResearchConfig
from ttt_autoresearch.env import AutoResearchState
from ttt_autoresearch.reward import AutoResearchRewardEvaluator, reward_for_result
from ttt_autoresearch.runner import AutoResearchRunner, RunResult, parse_patch_candidate_for_state

MINIMAL_VALID_TRAIN_PY = """from prepare import MAX_SEQ_LEN
TOTAL_BATCH_SIZE = 2048
DEVICE_BATCH_SIZE = 1
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0

class GPT:
    def forward(self, idx, targets=None, reduction='mean'):
        return 0

# val_bpb: 1.000000
print(f"val_bpb:          {1.0:.6f}")
"""


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
        reward, correctness = reward_for_result(result)
        self.assertAlmostEqual(reward, 1.0 / 0.9)
        self.assertEqual(correctness, 1.0)

        regression_result = RunResult(
            status="success",
            val_bpb=1.1,
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            elapsed_sec=1.0,
            workspace_path=Path("."),
            metrics_path=None,
            command=["python", "train.py"],
            returncode=0,
        )
        reward, correctness = reward_for_result(regression_result)
        self.assertAlmostEqual(reward, 1.0 / 1.1)
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
        reward, correctness = reward_for_result(timeout_result)
        self.assertEqual(reward, 0.0)
        self.assertEqual(correctness, 0.0)

    def test_evaluator_uses_inner_metric_as_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
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
            payload = "<<<<<<< SEARCH\n# val_bpb: 1.000000\n=======\n# val_bpb: 0.900000\n>>>>>>> REPLACE"
            result = evaluator.get_reward(payload, state)
            self.assertGreater(result["reward"], 0.0)
            self.assertEqual(result["correctness"], 1.0)
            manifest = json.loads((Path(config.run_dir) / "candidates").glob("*/rollout_manifest.json").__next__().read_text(encoding="utf-8"))
            self.assertEqual(manifest["starting_state"]["timestep"], -1)
            self.assertEqual(manifest["candidate"]["summary"], "search_replace_patch_candidate")
            self.assertEqual(manifest["evaluation"]["status"], "success")
            self.assertIn("Problem", manifest["prompt"])
            self.assertTrue((Path(manifest["prompt_path"])).exists())
            self.assertTrue((Path(manifest["raw_response_path"])).exists())

    def test_invalid_candidate_is_persisted_to_history_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text(MINIMAL_VALID_TRAIN_PY, encoding="utf-8")

            config = TTTAutoResearchConfig(
                execution_backend="local",
                max_concurrent_evaluations=1,
                timeout_sec=1,
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.0)
            AutoResearchRewardEvaluator.configure(bootstrap, runner)
            evaluator = AutoResearchRewardEvaluator(problem_type="autoresearch", log_dir=str(bootstrap.run_dir))
            state = AutoResearchState(
                timestep=2,
                construction=[],
                code=(root / "train.py").read_text(encoding="utf-8"),
                value=-1.0,
                baseline_val_bpb=1.0,
                current_best_val_bpb=1.0,
            )

            result = evaluator.get_reward("", state)

            self.assertEqual(result["metrics"]["candidate_status"], "invalid_candidate")
            history_path = Path(config.run_dir) / "history.jsonl"
            history_entries = history_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(history_entries), 1)
            history = json.loads(history_entries[0])
            self.assertEqual(history["status"], "invalid_candidate")
            manifest_path = next((Path(config.run_dir) / "candidates").glob("*/rollout_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["evaluation"]["status"], "invalid_candidate")
            self.assertEqual(manifest["raw_response"], "")
            self.assertIn("Problem", manifest["prompt"])

    def test_parse_patch_candidate_rejects_raw_code(self) -> None:
        with self.assertRaises(ValueError):
            parse_patch_candidate_for_state("print(1)\n", "print(0)\n")

    def test_parse_patch_candidate_accepts_search_replace_patch(self) -> None:
        candidate = parse_patch_candidate_for_state(
            "<<<<<<< SEARCH\nprint(0)\n=======\nprint(1)\n>>>>>>> REPLACE",
            "print(0)\n",
        )
        self.assertEqual(candidate.summary, "search_replace_patch_candidate")
        self.assertEqual(candidate.train_py, "print(1)\n")

    def test_invalid_batch_patch_runs_and_crashes_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = (
                "from prepare import MAX_SEQ_LEN\n"
                "TOTAL_BATCH_SIZE = 8\n"
                "DEVICE_BATCH_SIZE = 1\n"
                "tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN\n"
                "assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0\n"
                "class GPT:\n"
                "    def forward(self, idx, targets=None, reduction='mean'):\n"
                "        return 0\n"
                "print(f\"val_bpb:          {1.0:.6f}\")\n"
            )
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text(original, encoding="utf-8")

            config = TTTAutoResearchConfig(
                execution_backend="local",
                max_concurrent_evaluations=1,
                timeout_sec=1,
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.0)
            AutoResearchRewardEvaluator.configure(bootstrap, runner)
            evaluator = AutoResearchRewardEvaluator(problem_type="autoresearch", log_dir=str(bootstrap.run_dir))
            state = AutoResearchState(
                timestep=0,
                construction=[],
                code=original,
                value=-1.0,
                baseline_val_bpb=1.0,
                current_best_val_bpb=1.0,
            )

            payload = "<<<<<<< SEARCH\nTOTAL_BATCH_SIZE = 8\n=======\nTOTAL_BATCH_SIZE = 7\n>>>>>>> REPLACE"
            result = evaluator.get_reward(payload, state)

            self.assertEqual(result["metrics"]["candidate_status"], "crash")
            history_path = Path(config.run_dir) / "history.jsonl"
            history = json.loads(history_path.read_text(encoding="utf-8").strip())
            self.assertEqual(history["status"], "crash")
            self.assertEqual(history["failure_stage"], "runtime")
            manifest_path = next((Path(config.run_dir) / "candidates").glob("*/rollout_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["evaluation"]["status"], "crash")
            self.assertIn("stdout_path", manifest["evaluation"])
            self.assertIn("stderr_path", manifest["evaluation"])

    def test_concurrent_reward_calls_serialize_inner_evaluations(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                self.active = 0
                self.max_seen = 0
                self._workspace_index = 0

            def prepare_candidate_workspace(self, candidate, step: int, prefix: str = "candidate") -> Path:
                workspace = Path(tempfile.mkdtemp()) / f"{step:04d}_{prefix}_{self._workspace_index}"
                self._workspace_index += 1
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / "train.py").write_text(candidate.train_py, encoding="utf-8")
                return workspace

            def preflight_candidate(self, workspace: Path, candidate):
                from ttt_autoresearch.runner import PreflightResult

                return PreflightResult(ok=True, stage="ok", reason="ok", details={})

            def write_json_artifact(self, path: Path, payload: dict[str, object]) -> Path:
                path.write_text("{}", encoding="utf-8")
                return path

            def run_candidate(self, **kwargs: object) -> RunResult:
                workspace = kwargs["workspace"]
                with self.lock:
                    self.active += 1
                    self.max_seen = max(self.max_seen, self.active)
                try:
                    time.sleep(0.1)
                finally:
                    with self.lock:
                        self.active -= 1
                return RunResult(
                    status="success",
                    val_bpb=0.9,
                    stdout_path=Path("stdout.log"),
                    stderr_path=Path("stderr.log"),
                    elapsed_sec=0.1,
                    workspace_path=Path(workspace),
                    metrics_path=None,
                    command=["python", "train.py"],
                    returncode=0,
                )

            def update_best(self, **_: object) -> bool:
                return False

            def append_history(self, _: dict[str, object]) -> None:
                return None

            def write_rollout_manifest(self, workspace: Path, payload: dict[str, object]) -> Path:
                return workspace / "rollout_manifest.json"

            def read_text(self, _: Path, max_chars: int = 4000) -> str:
                return ""

        bootstrap = BootstrapContext(
            repo_root=Path("."),
            run_dir=Path("."),
            config=TTTAutoResearchConfig(execution_backend="local", max_concurrent_evaluations=1).normalized(Path(".")),
            program_text="program",
            baseline_train_py="train",
            baseline_val_bpb=1.0,
        )
        runner = FakeRunner()
        AutoResearchRewardEvaluator.configure(bootstrap, runner)  # type: ignore[arg-type]
        evaluator = AutoResearchRewardEvaluator(problem_type="autoresearch", log_dir=".")
        payload = "<<<<<<< SEARCH\nprint(0)\n=======\nprint(1)\n>>>>>>> REPLACE"

        def make_state() -> AutoResearchState:
            return AutoResearchState(
                timestep=-1,
                construction=[],
                code="print(0)\n",
                value=-1.0,
                baseline_val_bpb=1.0,
                current_best_val_bpb=1.0,
            )

        threads = [
            threading.Thread(target=evaluator.get_reward, args=(payload, make_state()))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(runner.max_seen, 1)

    def test_parallel_evaluations_require_explicit_gpu_devices(self) -> None:
        bootstrap = BootstrapContext(
            repo_root=Path("."),
            run_dir=Path("."),
            config=TTTAutoResearchConfig(execution_backend="local", max_concurrent_evaluations=2).normalized(Path(".")),
            program_text="program",
            baseline_train_py="train",
            baseline_val_bpb=1.0,
        )
        with self.assertRaises(ValueError):
            AutoResearchRewardEvaluator.configure(bootstrap, object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
