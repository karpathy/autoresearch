from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.discover_compat import patch_ttt_discover_no_wandb_bug
from ttt_autoresearch.hyperbolic import HyperbolicError, HyperbolicPool, RemoteExecutionResult
from ttt_autoresearch.runner import AutoResearchRunner


class HyperbolicPoolTests(unittest.TestCase):
    def test_validate_requires_host(self) -> None:
        config = TTTAutoResearchConfig(execution_backend="hyperbolic", hyperbolic_ssh_host=None).normalized(Path("."))
        pool = object.__new__(HyperbolicPool)
        pool.config = config
        with self.assertRaises(HyperbolicError):
            pool._validate_config()

    def test_validate_ssh_key_rejects_missing_key(self) -> None:
        config = TTTAutoResearchConfig(
            execution_backend="hyperbolic",
            hyperbolic_ssh_host="1.2.3.4",
            hyperbolic_ssh_private_key_path="/nonexistent/path/to/key",
        ).normalized(Path("."))
        pool = object.__new__(HyperbolicPool)
        pool.config = config
        with self.assertRaises(HyperbolicError):
            pool._validate_ssh_key()

    def test_runner_close_shuts_down_hyperbolic_pool(self) -> None:
        class FakePool:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = TTTAutoResearchConfig(execution_backend="local").normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            pool = FakePool()
            runner._hyperbolic_pool = pool  # type: ignore[assignment]
            runner.close()
            self.assertTrue(pool.closed)

    def test_runner_uses_hyperbolic_backend(self) -> None:
        class FakePool:
            def __init__(self) -> None:
                self.last_env = None

            def execute_workspace(self, workspace: Path, command: list[str], env: dict[str, str], timeout_sec: int, label: str) -> RemoteExecutionResult:
                self.last_env = dict(env)
                return RemoteExecutionResult(stdout="val_bpb: 0.900000\n", stderr="", returncode=0, elapsed_sec=1.0)

            def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("print('ok')\n", encoding="utf-8")
            config = TTTAutoResearchConfig(
                execution_backend="hyperbolic",
                hyperbolic_ssh_host="1.2.3.4",
                gpu_devices=["0", "1"],
                baseline_command_override=["python3", "train.py"],
            ).normalized(root)
            runner = AutoResearchRunner(root, config, Path(config.run_dir))
            fake_pool = FakePool()
            runner._hyperbolic_pool = fake_pool  # type: ignore[assignment]
            bootstrap = runner.build_bootstrap(1.0)
            result = runner.run_baseline(bootstrap=bootstrap)
            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.val_bpb, 0.9)

    def test_detached_launch_aliases_openai_key_to_tinker_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "demo"
            run_dir.mkdir(parents=True)
            (root / "program.md").write_text("program", encoding="utf-8")
            (root / "prepare.py").write_text("TIME_BUDGET = 1\n", encoding="utf-8")
            (root / "train.py").write_text("print('ok')\n", encoding="utf-8")
            config = TTTAutoResearchConfig(
                execution_backend="hyperbolic",
                hyperbolic_ssh_host="1.2.3.4",
                run_dir=str(run_dir),
            ).normalized(root)
            pool = object.__new__(HyperbolicPool)
            pool.repo_root = root
            pool.run_dir = run_dir
            pool.config = config
            pool.repo_archive_path = run_dir / "bundle.tar.gz"
            pool.repo_archive_lock = None
            pool.bootstrap_lock = None
            pool.bootstrap_complete = True

            uploaded_scripts: list[str] = []

            def fake_run_ssh(command: str, timeout: int, check: bool):
                return RemoteExecutionResult(stdout="", stderr="", returncode=0, elapsed_sec=0.0)

            def fake_upload(local_path: Path, remote_path: str):
                if remote_path.endswith("start_controller.sh"):
                    uploaded_scripts.append(Path(local_path).read_text(encoding="utf-8"))

            pool._ensure_node_ready = lambda: None  # type: ignore[method-assign]
            pool._run_ssh = fake_run_ssh  # type: ignore[method-assign]
            pool._upload_file = fake_upload  # type: ignore[method-assign]
            pool._build_remote_controller_config = lambda remote_run_dir: {"run_dir": remote_run_dir}  # type: ignore[method-assign]

            with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "abc123"}, clear=False):
                pool.launch_detached_controller()

            self.assertEqual(len(uploaded_scripts), 1)
            self.assertIn("export OPENAI_API_KEY=abc123", uploaded_scripts[0])
            self.assertIn("export TINKER_API_KEY=abc123", uploaded_scripts[0])

    def test_no_wandb_patch_pads_logger_list(self) -> None:
        try:
            from ttt_discover.tinker_utils import ml_log
        except ImportError:
            self.skipTest("ttt_discover not installed in local environment")
        patch_ttt_discover_no_wandb_bug()
        logger = ml_log.setup_logging(log_dir=tempfile.mkdtemp(), wandb_project="demo", wandb_name="demo", config=None)
        self.assertGreaterEqual(len(logger.loggers), 3)

    def test_detached_launch_refuses_active_remote_runs(self) -> None:
        config = TTTAutoResearchConfig(
            execution_backend="hyperbolic",
            hyperbolic_ssh_host="1.2.3.4",
        ).normalized(Path("."))
        pool = object.__new__(HyperbolicPool)
        pool.config = config

        def fake_run_ssh(command: str, timeout: int, check: bool):
            if "pgrep -af" in command:
                return subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=12,
                    stdout="Detected active AutoResearch processes already running on the Hyperbolic node.\nControllers:\n123 python run_ttt_discover.py",
                    stderr="",
                )
            raise AssertionError("unexpected remote command")

        pool._run_ssh = fake_run_ssh  # type: ignore[method-assign]
        with self.assertRaises(HyperbolicError):
            pool._assert_no_active_remote_runs()


if __name__ == "__main__":
    unittest.main()
