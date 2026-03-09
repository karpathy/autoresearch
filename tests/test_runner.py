from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import sys
import os

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.runner import AutoResearchRunner, parse_patch_candidate, parse_val_bpb


class RunnerTests(unittest.TestCase):
    def test_parse_candidate_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_patch_candidate('{"summary":"s","rationale":"r","train_py":"x","prepare_py":"bad"}')

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
                timeout_sec=1,
                baseline_command_override=[sys.executable, "tests/fixtures/fake_train.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            bootstrap = runner.build_bootstrap(1.25)
            result = runner.run_baseline(bootstrap=bootstrap)
            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.val_bpb, 1.25)

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


if __name__ == "__main__":
    unittest.main()
