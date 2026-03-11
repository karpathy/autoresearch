from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest
import sys
import os

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.runner import (
    AutoResearchRunner,
    _find_top_level_undefined_name,
    parse_patch_candidate_for_state,
    parse_val_bpb,
)


class RunnerTests(unittest.TestCase):
    def test_parse_candidate_rejects_legacy_json(self) -> None:
        with self.assertRaises(ValueError):
            parse_patch_candidate_for_state('{"summary":"s","rationale":"r","train_py":"x"}', "x")

    def test_parse_candidate_rejects_raw_python(self) -> None:
        with self.assertRaises(ValueError):
            parse_patch_candidate_for_state("print(1)\n", "print(0)\n")

    def test_parse_candidate_accepts_search_replace_patch(self) -> None:
        candidate = parse_patch_candidate_for_state(
            "<<<<<<< SEARCH\nprint(0)\n=======\nprint(1)\n>>>>>>> REPLACE",
            "print(0)\n",
        )
        self.assertEqual(candidate.candidate_format, "search_replace_patch")
        self.assertEqual(candidate.patch_block_count, 1)
        self.assertEqual(candidate.train_py, "print(1)\n")

    def test_parse_candidate_extracts_patch_from_wrapper_text(self) -> None:
        candidate = parse_patch_candidate_for_state(
            "Here is the patch\n<<<<<<< SEARCH\nprint(0)\n=======\nprint(1)\n>>>>>>> REPLACE\nDone.",
            "print(0)\n",
        )
        self.assertEqual(candidate.candidate_format, "search_replace_patch_extracted")
        self.assertEqual(candidate.train_py, "print(1)\n")

    def test_parse_val_bpb(self) -> None:
        stdout = "---\nval_bpb:          0.997900\n"
        self.assertEqual(parse_val_bpb(stdout), 0.9979)

    def test_runner_reads_metric_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("# val_bpb: 1.250000\n", encoding="utf-8")
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            fixture_src = Path(__file__).parent / "fixtures" / "fake_train.py"
            (fixtures / "fake_train.py").write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

            config = TTTAutoResearchConfig(
                execution_backend="local",
                timeout_sec=1,
                baseline_command_override=[sys.executable, "tests/fixtures/fake_train.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.25)
            result = runner.run_baseline(bootstrap=bootstrap)
            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.val_bpb, 1.25)
            self.assertTrue((Path(config.run_dir) / "baseline" / "train.py").exists())

    def test_runner_ignores_malformed_metrics_json_when_stdout_has_val_bpb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("print('ok')\n", encoding="utf-8")
            fixture = (
                "from pathlib import Path\n"
                "print('val_bpb: 0.876543')\n"
                "Path('metrics.json').write_text('{not-json', encoding='utf-8')\n"
            )
            fixtures = root / "tests" / "fixtures"
            fixtures.mkdir(parents=True)
            (fixtures / "bad_metrics.py").write_text(fixture, encoding="utf-8")
            config = TTTAutoResearchConfig(
                execution_backend="local",
                timeout_sec=5,
                baseline_command_override=[sys.executable, "tests/fixtures/bad_metrics.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.0)
            result = runner.run_baseline(bootstrap=bootstrap)
            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.val_bpb, 0.876543)
            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(metrics["val_bpb"], 0.876543)
            self.assertIn("metrics_json_error", metrics)

    def test_preflight_rejects_invalid_batch_divisibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            original = (
                "from prepare import MAX_SEQ_LEN\n"
                "TOTAL_BATCH_SIZE = 8\n"
                "DEVICE_BATCH_SIZE = 2\n"
                "tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN\n"
                "assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0\n"
                "class GPT:\n"
                "    def forward(self, idx, targets=None, reduction='mean'):\n"
                "        return 0\n"
                "print(f\"val_bpb:          {1.0:.6f}\")\n"
            )
            (root / "train.py").write_text(original, encoding="utf-8")
            config = TTTAutoResearchConfig(execution_backend="local").normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            candidate = parse_patch_candidate_for_state(
                "<<<<<<< SEARCH\nTOTAL_BATCH_SIZE = 8\n=======\nTOTAL_BATCH_SIZE = 7\n>>>>>>> REPLACE",
                original,
            )
            workspace = runner.prepare_candidate_workspace(candidate, step=0)
            preflight = runner.preflight_candidate(workspace, candidate)
            self.assertFalse(preflight.ok)
            self.assertEqual(preflight.stage, "batch_divisibility")

    def test_top_level_undefined_name_does_not_treat_nested_bindings_as_module_scope(self) -> None:
        module = ast.parse(
            "if False:\n"
            "    x = 1\n"
            "print(x)\n"
        )
        self.assertEqual(_find_top_level_undefined_name(module), "x")

    def test_preflight_rejects_missing_val_bpb_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            original = (
                "from prepare import MAX_SEQ_LEN\n"
                "TOTAL_BATCH_SIZE = 8\n"
                "DEVICE_BATCH_SIZE = 1\n"
                "tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN\n"
                "assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0\n"
                "class GPT:\n"
                "    def forward(self, idx, targets=None, reduction='mean'):\n"
                "        return 0\n"
                "print('val_bpb:          1.0')\n"
            )
            (root / "train.py").write_text(original, encoding="utf-8")
            config = TTTAutoResearchConfig(execution_backend="local").normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            candidate = parse_patch_candidate_for_state(
                "<<<<<<< SEARCH\nprint('val_bpb:          1.0')\n=======\nprint('done')\n>>>>>>> REPLACE",
                original,
            )
            workspace = runner.prepare_candidate_workspace(candidate, step=0)
            preflight = runner.preflight_candidate(workspace, candidate)
            self.assertFalse(preflight.ok)
            self.assertEqual(preflight.stage, "summary_output")

    def test_build_bootstrap_prefers_stored_baseline_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "resume-run"
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "train.py").write_text("repo version\n", encoding="utf-8")
            (run_dir / "baseline").mkdir(parents=True)
            (run_dir / "baseline" / "train.py").write_text("stored baseline\n", encoding="utf-8")

            config = TTTAutoResearchConfig(run_dir=str(run_dir)).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.0)
            self.assertEqual(bootstrap.baseline_train_py, "stored baseline\n")

    def test_config_normalizes_relative_paths_and_overrides_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = TTTAutoResearchConfig(
                run_dir="relative-runs",
                data_path="data/custom",
                local_model_path="models/local",
                provider="forced-provider",
                api_base="https://example.invalid/v1",
            ).normalized(root)
            self.assertEqual(config.run_dir, str(root / "relative-runs"))
            self.assertEqual(config.data_path, str(root / "data/custom"))
            self.assertEqual(config.local_model_path, str(root / "models/local"))

            bootstrap = type("Bootstrap", (), {"config": config})()
            from ttt_autoresearch.config import BootstrapContext
            context = BootstrapContext(
                repo_root=root,
                run_dir=Path(config.run_dir),
                config=config,
                program_text="program",
                baseline_train_py="train",
                baseline_val_bpb=1.0,
            )
            old_provider = os.environ.get("TINKER_PROVIDER")
            old_base = os.environ.get("OPENAI_BASE_URL")
            os.environ["TINKER_PROVIDER"] = "wrong-provider"
            os.environ["OPENAI_BASE_URL"] = "https://wrong.invalid"
            env = context.subprocess_env()
            self.assertEqual(env["TINKER_PROVIDER"], "forced-provider")
            self.assertEqual(env["OPENAI_BASE_URL"], "https://example.invalid/v1")
            if old_provider is None:
                os.environ.pop("TINKER_PROVIDER", None)
            else:
                os.environ["TINKER_PROVIDER"] = old_provider
            if old_base is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = old_base

    def test_unknown_model_requires_explicit_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                TTTAutoResearchConfig(model_name="meta-llama/Meta-Llama-3-70B").normalized(root)

            config = TTTAutoResearchConfig(
                model_name="meta-llama/Meta-Llama-3-70B",
                renderer_name="gpt_oss_high_reasoning",
            ).normalized(root)
            self.assertEqual(config.renderer_name, "gpt_oss_high_reasoning")

    def test_kimi_model_is_primary_supported_renderer(self) -> None:
        config = TTTAutoResearchConfig(
            model_name="moonshotai/Kimi-K2.5",
            execution_backend="local",
        ).normalized(Path("."))
        self.assertEqual(config.renderer_name, "qwen3")

    def test_group_defaults_reflect_medium_preset(self) -> None:
        config = TTTAutoResearchConfig().normalized(Path("."))
        self.assertEqual(config.model_name, "openai/gpt-oss-120b")
        self.assertEqual(config.execution_backend, "hyperbolic")
        self.assertEqual(config.max_steps, 12)
        self.assertEqual(config.groups_per_step, 2)
        self.assertEqual(config.samples_per_step, 8)
        self.assertEqual(config.max_concurrent_evaluations, 8)
        self.assertEqual(config.renderer_name, "gpt_oss_high_reasoning")
        self.assertEqual(config.gpu_devices, ["0", "1", "2", "3", "4", "5", "6", "7"])
        self.assertIsNone(config.wandb_project)
        self.assertIn("HF_TOKEN", config.hyperbolic_forward_env_vars)
        self.assertNotIn("WANDB_API_KEY", config.hyperbolic_forward_env_vars)

    def test_gpu_devices_are_normalized(self) -> None:
        config = TTTAutoResearchConfig(gpu_devices=[0, 3, 7]).normalized(Path("."))
        self.assertEqual(config.gpu_devices, ["0", "3", "7"])


if __name__ == "__main__":
    unittest.main()
