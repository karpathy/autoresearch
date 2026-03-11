from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import types
import unittest

from ttt_autoresearch import cli


class CliIntegrationTests(unittest.TestCase):
    def test_cli_launches_detached_hyperbolic_controller(self) -> None:
        captured: dict[str, object] = {}

        class FakeLauncher:
            def __init__(self, repo_root: Path, run_dir: Path, config) -> None:
                captured["repo_root"] = repo_root
                captured["run_dir"] = run_dir
                captured["config"] = config

            def launch_detached_controller(self) -> dict[str, str]:
                captured["launched"] = True
                return {
                    "remote_run_dir": "/home/ubuntu/autoresearch/runs/demo",
                    "remote_config_path": "/home/ubuntu/autoresearch/runs/launches/demo/remote_config.yaml",
                    "remote_log_path": "/home/ubuntu/autoresearch/runs/launches/demo/controller.log",
                    "remote_pid_path": "/home/ubuntu/autoresearch/runs/launches/demo/controller.pid",
                    "remote_exitcode_path": "/home/ubuntu/autoresearch/runs/launches/demo/controller.exitcode",
                    "remote_launch_dir": "/home/ubuntu/autoresearch/runs/launches/demo",
                }

        original_launcher = cli.HyperbolicPool
        original_mirror = cli._start_hyperbolic_mirror
        cli.HyperbolicPool = FakeLauncher  # type: ignore[assignment]
        cli._start_hyperbolic_mirror = lambda config, run_dir, launch_info: {  # type: ignore[assignment]
            "local_mirror_dir": str(run_dir / "mirror"),
            "local_mirror_log_path": str(run_dir / "hyperbolic_mirror.log"),
            "local_mirror_pid": "12345",
        }
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                run_dir = tmp_path / "runs" / "hyperbolic-demo"
                config_path = tmp_path / "config.yaml"
                config_path.write_text(
                    "\n".join(
                        [
                            "model_name: openai/gpt-oss-120b",
                            "renderer_name: gpt_oss_high_reasoning",
                            "execution_backend: hyperbolic",
                            "hyperbolic_ssh_host: 1.2.3.4",
                            f"run_dir: {run_dir}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                exit_code = cli.main(["--config", str(config_path)])
                self.assertEqual(exit_code, 0)
                self.assertTrue(captured.get("launched"))
                self.assertTrue((run_dir / "hyperbolic_launch.json").exists())
                self.assertTrue((run_dir / "resolved_config.json").exists())
                launch = (run_dir / "hyperbolic_launch.json").read_text(encoding="utf-8")
                self.assertIn("local_mirror_dir", launch)
        finally:
            cli.HyperbolicPool = original_launcher  # type: ignore[assignment]
            cli._start_hyperbolic_mirror = original_mirror  # type: ignore[assignment]

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
                            "execution_backend: local",
                            f"run_dir: {run_dir}",
                            "max_steps: 3",
                            "groups_per_step: 3",
                            "samples_per_step: 2",
                            "max_concurrent_evaluations: 1",
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
                self.assertEqual(captured["rl_config"]["adv_estimator"], "entropic_adaptive_beta")
                self.assertEqual(captured["rl_config"]["loss_fn"], "importance_sampling")
                self.assertEqual(captured["rl_config"]["num_substeps"], 1)
                self.assertTrue(captured["rl_config"]["remove_constant_reward_groups"])
                self.assertEqual(captured["dataset_config"]["batch_size"], 3)
                self.assertEqual(captured["dataset_config"]["group_size"], 2)
                self.assertEqual(captured["dataset_config"]["problem_type"], "autoresearch")
        finally:
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def test_cli_reuses_existing_baseline_when_resuming(self) -> None:
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
                run_dir = tmp_path / "runs" / "resume-test"
                (run_dir / "baseline" / "workspace").mkdir(parents=True)
                (run_dir / "baseline" / "workspace" / "train.py").write_text("# stored baseline\n", encoding="utf-8")
                (run_dir / "baseline" / "train.py").write_text("# stored baseline\n", encoding="utf-8")
                (run_dir / "best").mkdir(parents=True)
                (run_dir / "baseline.json").write_text(
                    "\n".join(
                        [
                            "{",
                            '  "status": "success",',
                            '  "val_bpb": 1.0,',
                            f'  "stdout_path": "{run_dir / "baseline" / "workspace" / "stdout.log"}",',
                            f'  "stderr_path": "{run_dir / "baseline" / "workspace" / "stderr.log"}",',
                            '  "elapsed_sec": 1.0,',
                            f'  "workspace_path": "{run_dir / "baseline" / "workspace"}",',
                            '  "metrics_path": null,',
                            '  "command": ["python", "train.py"],',
                            '  "returncode": 0',
                            "}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                config_path = tmp_path / "config.yaml"
                config_path.write_text(
                    "\n".join(
                        [
                            "model_name: Qwen/Qwen3.5-35B-A3B",
                            "execution_backend: local",
                            f"run_dir: {run_dir}",
                            "max_steps: 3",
                            "groups_per_step: 2",
                            "samples_per_step: 2",
                            "max_concurrent_evaluations: 1",
                            "baseline_command_override:",
                            f"  - {sys.executable}",
                            "  - -c",
                            '  - "import sys; sys.exit(7)"',
                            "candidate_command_override:",
                            f"  - {sys.executable}",
                            "  - -c",
                            '  - "print(\'val_bpb: 0.900000\')"',
                            "wandb_project: null",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                exit_code = cli.main(["--config", str(config_path)])
                self.assertEqual(exit_code, 0)
                self.assertTrue(captured.get("dataset_builder_called"))
                self.assertEqual(captured["dataset_config"]["batch_size"], 2)
        finally:
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main()
