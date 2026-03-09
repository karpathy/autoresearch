from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import types
import unittest

from ttt_autoresearch import cli


class CliIntegrationTests(unittest.TestCase):
    def test_resolve_config_path_falls_back_to_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "configs").mkdir()
            expected = repo_root / "configs" / "ttt_discover_autoresearch.yaml"
            expected.write_text("model_name: Qwen/Qwen3.5-35B-A3B\n", encoding="utf-8")

            with tempfile.TemporaryDirectory() as other_tmp:
                old_cwd = Path.cwd()
                os.chdir(other_tmp)
                try:
                    resolved = cli._resolve_config_path("configs/ttt_discover_autoresearch.yaml", repo_root)
                finally:
                    os.chdir(old_cwd)

            self.assertEqual(resolved, expected.resolve())

    def test_cli_wires_baseline_and_discover_entrypoint(self) -> None:
        captured: dict[str, object] = {}

        fake_root = types.ModuleType("ttt_discover")
        fake_rl = types.ModuleType("ttt_discover.rl")
        fake_rl_train = types.ModuleType("ttt_discover.rl.train")
        fake_utils = types.ModuleType("ttt_discover.tinker_utils")
        fake_dataset_builder = types.ModuleType("ttt_discover.tinker_utils.dataset_builder")

        class FakeRLConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeDatasetConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                captured["dataset_config"] = kwargs

        def fake_get_single_problem_dataset_builder(config):
            async def builder():
                captured["dataset_builder_called"] = True
                return {"dataset_config": config}

            return builder

        async def fake_discover_main(cfg):
            captured["rl_config"] = cfg.__dict__.copy()
            await cfg.dataset_builder()

        fake_rl_train.Config = FakeRLConfig
        fake_rl_train.main = fake_discover_main
        fake_dataset_builder.DatasetConfig = FakeDatasetConfig
        fake_dataset_builder.get_single_problem_dataset_builder = fake_get_single_problem_dataset_builder

        previous_modules = {name: sys.modules.get(name) for name in (
            "ttt_discover",
            "ttt_discover.rl",
            "ttt_discover.rl.train",
            "ttt_discover.tinker_utils",
            "ttt_discover.tinker_utils.dataset_builder",
        )}
        sys.modules["ttt_discover"] = fake_root
        sys.modules["ttt_discover.rl"] = fake_rl
        sys.modules["ttt_discover.rl.train"] = fake_rl_train
        sys.modules["ttt_discover.tinker_utils"] = fake_utils
        sys.modules["ttt_discover.tinker_utils.dataset_builder"] = fake_dataset_builder

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                run_dir = tmp_path / "runs" / "cli-test"
                config_path = tmp_path / "config.yaml"
                config_path.write_text(
                    "\n".join(
                        [
                            "model_name: Qwen/Qwen3.5-35B-A3B",
                            f"run_dir: {run_dir}",
                            "max_steps: 3",
                            "samples_per_step: 2",
                            "baseline_command_override:",
                            f"  - {sys.executable}",
                            "  - -c",
                            '  - "print(\'---\'); print(\'val_bpb:          1.000000\')"',
                            "candidate_command_override:",
                            f"  - {sys.executable}",
                            "  - -c",
                            '  - "print(\'---\'); print(\'val_bpb:          0.900000\')"',
                            "wandb_project: null",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                exit_code = cli.main(["--config", str(config_path)])
                self.assertEqual(exit_code, 0)
                self.assertTrue((run_dir / "baseline.json").exists())
                self.assertTrue((run_dir / "best" / "metrics.json").exists())
                self.assertTrue((run_dir / "resolved_config.json").exists())
                self.assertTrue(captured.get("dataset_builder_called"))
                self.assertEqual(captured["rl_config"]["model_name"], "Qwen/Qwen3.5-35B-A3B")
                self.assertEqual(captured["rl_config"]["num_epochs"], 3)
                self.assertEqual(captured["dataset_config"]["group_size"], 2)
                self.assertEqual(captured["dataset_config"]["problem_type"], "autoresearch")
        finally:
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main()
