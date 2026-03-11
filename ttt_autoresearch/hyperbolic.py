from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

from ttt_autoresearch.config import TTTAutoResearchConfig


class HyperbolicError(RuntimeError):
    pass


@dataclass(slots=True)
class RemoteExecutionResult:
    stdout: str
    stderr: str
    returncode: int | None
    elapsed_sec: float


class HyperbolicPool:
    def __init__(self, repo_root: Path, run_dir: Path, config: TTTAutoResearchConfig) -> None:
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.config = config
        self.repo_archive_path = self.run_dir / "hyperbolic_repo_bundle.tar.gz"
        self.repo_archive_lock = threading.Lock()
        self.bootstrap_lock = threading.Lock()
        self.bootstrap_complete = False
        self._validate_config()
        self._validate_ssh_key()
        self._write_repo_archive()

    def execute_workspace(
        self,
        workspace: Path,
        command: list[str],
        env: dict[str, str],
        timeout_sec: int,
        label: str,
    ) -> RemoteExecutionResult:
        self._ensure_node_ready()
        return self._run_workspace_on_node(workspace, command, env, timeout_sec, label)

    def close(self) -> None:
        return None

    def launch_detached_controller(self) -> dict[str, str]:
        self._ensure_node_ready()
        self._assert_no_active_remote_runs()
        run_name = Path(self.run_dir).name
        remote_run_dir = self.config.hyperbolic_remote_run_dir or f"{self.config.hyperbolic_repo_root}/runs/{run_name}"
        remote_launch_dir = f"{self.config.hyperbolic_repo_root}/runs/launches/{run_name}"
        remote_config_path = f"{remote_launch_dir}/remote_config.yaml"
        remote_log_path = f"{remote_launch_dir}/controller.log"
        remote_pid_path = f"{remote_launch_dir}/controller.pid"
        remote_exitcode_path = f"{remote_launch_dir}/controller.exitcode"
        remote_metadata_path = f"{remote_launch_dir}/launch.json"
        remote_start_path = f"{remote_launch_dir}/start_controller.sh"

        remote_config = self._build_remote_controller_config(remote_run_dir)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
            handle.write(_dump_yaml_like(remote_config))
            local_config_path = Path(handle.name)
        try:
            self._run_ssh(f"mkdir -p {shlex.quote(remote_launch_dir)} {shlex.quote(remote_run_dir)}", timeout=60, check=True)
            self._upload_file(local_config_path, remote_config_path)
        finally:
            local_config_path.unlink(missing_ok=True)

        forwarded_env = {}
        for name in self.config.hyperbolic_forward_env_vars or []:
            value = os.environ.get(name)
            if value:
                forwarded_env[name] = value
        # Tinker clients expect TINKER_API_KEY, but many local shells are configured
        # with OPENAI_API_KEY only. Mirror them so detached remote runs keep working.
        if "TINKER_API_KEY" not in forwarded_env and os.environ.get("OPENAI_API_KEY"):
            forwarded_env["TINKER_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if "OPENAI_API_KEY" not in forwarded_env and os.environ.get("TINKER_API_KEY"):
            forwarded_env["OPENAI_API_KEY"] = os.environ["TINKER_API_KEY"]
        if self.config.provider:
            forwarded_env.setdefault("TINKER_PROVIDER", self.config.provider)
        if self.config.api_base:
            forwarded_env.setdefault("OPENAI_BASE_URL", self.config.api_base)
            forwarded_env.setdefault("OPENAI_API_BASE", self.config.api_base)
            forwarded_env.setdefault("TINKER_BASE_URL", self.config.api_base)
        start_script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(remote_launch_dir)}",
                f"if [ -f {shlex.quote(remote_pid_path)} ] && kill -0 \"$(cat {shlex.quote(remote_pid_path)})\" 2>/dev/null; then",
                f"  echo 'Controller already running at {remote_pid_path}'",
                "  exit 1",
                "fi",
                f'export PATH="{self._remote_uv_bin_dir()}:$HOME/.local/bin:$PATH"',
                *[f"export {name}={shlex.quote(value)}" for name, value in sorted(forwarded_env.items())],
                f"cd {shlex.quote(self.config.hyperbolic_repo_root)}",
                "nohup bash -lc "
                + shlex.quote(
                    f"cd {self.config.hyperbolic_repo_root} && "
                    f"uv run python run_ttt_discover.py --config {remote_config_path} "
                    f"> {remote_log_path} 2>&1; "
                    f"rc=$?; printf '%s' \"$rc\" > {remote_exitcode_path}"
                )
                + " < /dev/null > /dev/null 2>&1 &",
                f"printf '%s' \"$!\" > {shlex.quote(remote_pid_path)}",
                f"cat > {shlex.quote(remote_metadata_path)} <<'JSON'",
                json.dumps(
                    {
                        "remote_run_dir": remote_run_dir,
                        "remote_config_path": remote_config_path,
                        "remote_log_path": remote_log_path,
                        "remote_pid_path": remote_pid_path,
                        "remote_exitcode_path": remote_exitcode_path,
                        "remote_start_path": remote_start_path,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "JSON",
            ]
        ) + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as handle:
            handle.write(start_script)
            local_start_path = Path(handle.name)
        try:
            self._upload_file(local_start_path, remote_start_path)
        finally:
            local_start_path.unlink(missing_ok=True)
        self._run_ssh(
            f"chmod +x {shlex.quote(remote_start_path)} && {shlex.quote(remote_start_path)}",
            timeout=30,
            check=True,
        )
        return {
            "remote_run_dir": remote_run_dir,
            "remote_config_path": remote_config_path,
            "remote_log_path": remote_log_path,
            "remote_pid_path": remote_pid_path,
            "remote_exitcode_path": remote_exitcode_path,
            "remote_launch_dir": remote_launch_dir,
            "remote_start_path": remote_start_path,
        }

    def _assert_no_active_remote_runs(self) -> None:
        repo_root = self.config.hyperbolic_repo_root
        script = "\n".join(
            [
                "set -euo pipefail",
                "matches=$(python3 - <<'PY'\n"
                "import subprocess\n"
                f"repo_root = {repo_root!r}\n"
                "rows = subprocess.run(['ps', '-eo', 'pid=,args='], text=True, capture_output=True, check=True).stdout.splitlines()\n"
                "hits = []\n"
                "for row in rows:\n"
                "    row = row.strip()\n"
                "    if not row:\n"
                "        continue\n"
                "    pid, _, args = row.partition(' ')\n"
                "    if repo_root not in args:\n"
                "        continue\n"
                "    is_controller = 'run_ttt_discover.py' in args and f'{repo_root}/runs/launches/' in args\n"
                "    is_train = 'train.py' in args and 'run_ttt_discover.py' not in args\n"
                "    if is_controller or is_train:\n"
                "        hits.append(row)\n"
                "print('\\n'.join(hits))\n"
                "PY\n)",
                'if [ -n "$matches" ]; then',
                "  echo 'Detected active AutoResearch processes already running on the Hyperbolic node.'",
                "  echo \"$matches\"",
                "  exit 12",
                "fi",
            ]
        )
        completed = self._run_ssh(script, timeout=30, check=False)
        if completed.returncode != 0:
            details = completed.stdout.strip() or completed.stderr.strip()
            raise HyperbolicError(
                "Refusing to launch because another detached AutoResearch run appears to still be active on the "
                f"Hyperbolic node.\n{details}"
            )

    def _validate_config(self) -> None:
        if not self.config.hyperbolic_ssh_host:
            raise HyperbolicError(
                "hyperbolic_ssh_host is not set. Create an on-demand Hyperbolic H100 node and set its SSH host in the config."
            )

    def _build_remote_controller_config(self, remote_run_dir: str) -> dict[str, object]:
        config_dict = self.config.to_dict()
        config_dict["execution_backend"] = "local"
        config_dict["run_dir"] = remote_run_dir
        config_dict["hyperbolic_detached_controller"] = False
        config_dict["gpu_devices"] = self.config.gpu_devices or [str(index) for index in range(8)]
        config_dict["max_concurrent_evaluations"] = min(
            int(self.config.max_concurrent_evaluations),
            len(config_dict["gpu_devices"]),
        )
        return config_dict

    def _validate_ssh_key(self) -> None:
        key_path = self.config.hyperbolic_ssh_private_key_path
        if key_path:
            if not Path(key_path).exists():
                raise HyperbolicError(f"SSH private key not found at {key_path}")
            return
        default_keys = [Path.home() / ".ssh" / name for name in ("id_ed25519", "id_rsa", "id_ecdsa")]
        has_agent = bool(os.environ.get("SSH_AUTH_SOCK"))
        has_default = any(key.exists() for key in default_keys)
        if not has_agent and not has_default:
            raise HyperbolicError(
                "No SSH private key configured for Hyperbolic. "
                "Set hyperbolic_ssh_private_key_path, ensure a default SSH key exists, or run an ssh-agent."
            )

    def _ensure_node_ready(self) -> None:
        with self.bootstrap_lock:
            if self.bootstrap_complete:
                return
            self._wait_for_ssh()
            self._bootstrap_node()
            self.bootstrap_complete = True

    def _wait_for_ssh(self) -> None:
        deadline = time.time() + self.config.hyperbolic_bootstrap_timeout_sec
        while time.time() < deadline:
            try:
                completed = self._run_ssh("true", timeout=30, check=False)
            except HyperbolicError:
                time.sleep(5)
                continue
            if completed.returncode == 0:
                return
            time.sleep(5)
        raise HyperbolicError("Hyperbolic node never accepted SSH before the bootstrap timeout.")

    def _bootstrap_node(self) -> None:
        remote_archive = "/tmp/autoresearch_repo_bundle.tar.gz"
        self._upload_file(self.repo_archive_path, remote_archive)
        repo_root = self.config.hyperbolic_repo_root
        bootstrap_commands = self.config.hyperbolic_bootstrap_commands or [
            f'if ! command -v uv >/dev/null 2>&1 && [ ! -x "{self._remote_uv_bin_dir()}/uv" ]; then '
            "curl -LsSf https://astral.sh/uv/install.sh | sh; "
            "fi",
            "cd {repo_root} && uv sync",
            "cd {repo_root} && uv run prepare.py --num-shards {prepare_num_shards}",
        ]
        rendered = [
            command.format(
                repo_root=repo_root,
                prepare_num_shards=self.config.hyperbolic_prepare_num_shards,
            )
            for command in bootstrap_commands
        ]
        script_lines = [
            "set -euo pipefail",
            f'export PATH="{self._remote_uv_bin_dir()}:$HOME/.local/bin:$PATH"',
            f"rm -rf {shlex.quote(repo_root)}",
            f"mkdir -p {shlex.quote(repo_root)}",
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(repo_root)}",
            f"if [ ! -f {shlex.quote(repo_root)}/pyproject.toml ]; then rm -rf {shlex.quote(repo_root)}/* && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(repo_root)} --strip-components=1; fi",
            f"test -f {shlex.quote(repo_root)}/pyproject.toml",
            f"test -f {shlex.quote(repo_root)}/prepare.py",
        ]
        script_lines.extend(rendered)
        self._run_ssh(
            "\n".join(script_lines),
            timeout=self.config.hyperbolic_bootstrap_timeout_sec,
            check=True,
        )

    @staticmethod
    def _remote_uv_root() -> str:
        return "$HOME/.local"

    @classmethod
    def _remote_uv_bin_dir(cls) -> str:
        return f"{cls._remote_uv_root()}/bin"

    def _run_workspace_on_node(
        self,
        workspace: Path,
        command: list[str],
        env: dict[str, str],
        timeout_sec: int,
        label: str,
    ) -> RemoteExecutionResult:
        remote_workspace = f"{self.config.hyperbolic_repo_root}/../jobs/{label}-{uuid.uuid4().hex[:8]}"
        remote_archive = f"/tmp/{uuid.uuid4().hex}.tar.gz"
        local_archive = self._build_workspace_archive(workspace)
        try:
            self._upload_file(local_archive, remote_archive)
            env_lines = []
            for key in sorted(env):
                env_lines.append(f"export {key}={shlex.quote(env[key])}")
            env_lines.append("export PYTHONUNBUFFERED=1")
            command_str = " ".join(shlex.quote(part) for part in command)
            script = "\n".join(
                [
                    "set -uo pipefail",
                    f"rm -rf {shlex.quote(remote_workspace)}",
                    f"mkdir -p {shlex.quote(remote_workspace)}",
                    f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_workspace)}",
                    f"if [ ! -f {shlex.quote(remote_workspace)}/train.py ]; then rm -rf {shlex.quote(remote_workspace)}/* && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_workspace)} --strip-components=1; fi",
                    f"test -f {shlex.quote(remote_workspace)}/train.py",
                    f"cd {shlex.quote(remote_workspace)}",
                    *env_lines,
                    f"timeout --kill-after=30s {timeout_sec}s {command_str} > stdout.log 2> stderr.log",
                    "rc=$?",
                    'printf "%s" "$rc" > .exit_code',
                    "exit 0",
                ]
            )
            start = time.time()
            self._run_ssh(script, timeout=timeout_sec + 180, check=True)
            elapsed = time.time() - start
            stdout = self._download_text_file(f"{remote_workspace}/stdout.log")
            stderr = self._download_text_file(f"{remote_workspace}/stderr.log")
            exit_text = self._download_text_file(f"{remote_workspace}/.exit_code").strip()
            if exit_text:
                try:
                    returncode = int(exit_text)
                except ValueError:
                    returncode = 1
            else:
                returncode = 1
            metrics_json = self._download_text_file(f"{remote_workspace}/metrics.json")
            if metrics_json:
                (workspace / "metrics.json").write_text(metrics_json, encoding="utf-8")
            return RemoteExecutionResult(stdout=stdout, stderr=stderr, returncode=returncode, elapsed_sec=elapsed)
        finally:
            try:
                self._run_ssh(
                    "\n".join(
                        [
                            "set -e",
                            f"rm -rf {shlex.quote(remote_workspace)}",
                            f"rm -f {shlex.quote(remote_archive)}",
                        ]
                    ),
                    timeout=60,
                    check=False,
                )
            except HyperbolicError:
                pass
            local_archive.unlink(missing_ok=True)

    def _upload_file(self, local_path: Path, remote_path: str) -> None:
        destination = f"{self.config.hyperbolic_ssh_user}@{self.config.hyperbolic_ssh_host}:{remote_path}"
        try:
            completed = subprocess.run(
                self._scp_base_args() + [str(local_path), destination],
                text=True,
                capture_output=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HyperbolicError(f"Timed out uploading {local_path.name} to the Hyperbolic node.") from exc
        if completed.returncode != 0:
            raise HyperbolicError(completed.stderr.strip() or f"scp upload failed for {local_path.name}.")

    def _download_text_file(self, remote_path: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / Path(remote_path).name
            source = f"{self.config.hyperbolic_ssh_user}@{self.config.hyperbolic_ssh_host}:{remote_path}"
            try:
                completed = subprocess.run(
                    self._scp_base_args() + [source, str(local_path)],
                    text=True,
                    capture_output=True,
                    timeout=600,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise HyperbolicError(f"Timed out downloading {remote_path} from the Hyperbolic node.") from exc
            if completed.returncode != 0:
                return ""
            return local_path.read_text(encoding="utf-8")

    def _run_ssh(self, script: str, timeout: int, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                self._ssh_base_args() + [f"bash -lc {shlex.quote(script)}"],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HyperbolicError("Timed out waiting for a remote SSH command on the Hyperbolic node.") from exc
        if check and completed.returncode != 0:
            raise HyperbolicError(completed.stderr.strip() or completed.stdout.strip() or "Remote command failed on the Hyperbolic node.")
        return completed

    def _ssh_base_args(self) -> list[str]:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
        ]
        if self.config.hyperbolic_ssh_private_key_path:
            args.extend(["-i", self.config.hyperbolic_ssh_private_key_path])
        args.extend(
            [
                "-p",
                str(self.config.hyperbolic_ssh_port),
                f"{self.config.hyperbolic_ssh_user}@{self.config.hyperbolic_ssh_host}",
            ]
        )
        return args

    def _scp_base_args(self) -> list[str]:
        args = [
            "scp",
            "-O",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
        ]
        if self.config.hyperbolic_ssh_private_key_path:
            args.extend(["-i", self.config.hyperbolic_ssh_private_key_path])
        args.extend(["-P", str(self.config.hyperbolic_ssh_port)])
        return args

    def _write_repo_archive(self) -> None:
        with self.repo_archive_lock:
            with tarfile.open(self.repo_archive_path, "w:gz") as archive:
                for path in self.repo_root.rglob("*"):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(self.repo_root)
                    if self._should_skip(rel):
                        continue
                    archive.add(path, arcname=str(rel))

    def _build_workspace_archive(self, workspace: Path) -> Path:
        fd, archive_path = tempfile.mkstemp(prefix="workspace_", suffix=".tar.gz")
        os.close(fd)
        archive_file = Path(archive_path)
        with tarfile.open(archive_file, "w:gz") as archive:
            for path in workspace.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(workspace)
                if self._should_skip(rel):
                    continue
                archive.add(path, arcname=str(rel))
        return archive_file

    @staticmethod
    def _should_skip(rel: Path) -> bool:
        parts = rel.parts
        if not parts:
            return False
        if parts[0] in {".git", "runs", "__pycache__", ".pytest_cache", ".venv"}:
            return True
        if rel.name in {"prompt.txt", "response.txt"}:
            return True
        return rel.suffix in {".pyc", ".pyo"}


def _dump_yaml_like(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if item is None:
                    rendered = "null"
                elif isinstance(item, bool):
                    rendered = "true" if item else "false"
                else:
                    rendered = json.dumps(item)
                lines.append(f"  - {rendered}")
            continue
        if value is None:
            rendered = "null"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"
