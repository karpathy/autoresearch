from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import queue
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from ttt_autoresearch.config import TTTAutoResearchConfig


class RunPodError(RuntimeError):
    pass


class RunPodAPIError(RunPodError):
    pass


class RunPodPodLostError(RunPodError):
    pass


@dataclass(slots=True)
class RemoteExecutionResult:
    stdout: str
    stderr: str
    returncode: int | None
    elapsed_sec: float


@dataclass(slots=True)
class RunPodPod:
    id: str
    name: str
    public_ip: str | None = None
    ssh_port: int | None = None
    desired_status: str | None = None
    machine_id: str | None = None
    ready: bool = False


class RunPodAPIClient:
    def __init__(self, config: TTTAutoResearchConfig) -> None:
        api_key = os.environ.get(config.runpod_api_key_env)
        if not api_key:
            raise RunPodAPIError(
                f"{config.runpod_api_key_env} is not set. Export your RunPod API key before starting a run."
            )
        self.api_key = api_key
        self.base_url = config.runpod_api_base.rstrip("/")

    def list_pods(self) -> list[dict[str, object]]:
        payload = self._request("GET", "/pods")
        pods = payload.get("pods", payload)
        if isinstance(pods, list):
            return [pod for pod in pods if isinstance(pod, dict)]
        raise RunPodAPIError(f"Unexpected /pods response: {payload!r}")

    def create_pod(self, body: dict[str, object]) -> dict[str, object]:
        payload = self._request("POST", "/pods", body)
        if isinstance(payload, dict):
            return payload
        raise RunPodAPIError(f"Unexpected create pod response: {payload!r}")

    def delete_pod(self, pod_id: str) -> None:
        try:
            self._request("DELETE", f"/pods/{pod_id}")
        except RunPodAPIError as exc:
            message = str(exc)
            if "404" in message:
                return
            raise

    def _request(self, method: str, path: str, body: dict[str, object] | None = None) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, method=method, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8") or "{}"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RunPodAPIError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RunPodAPIError(f"{method} {path} failed: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunPodAPIError(f"{method} {path} returned invalid JSON: {raw!r}") from exc
        if isinstance(parsed, dict):
            return parsed
        raise RunPodAPIError(f"{method} {path} returned unexpected payload: {parsed!r}")


class RunPodPool:
    def __init__(self, repo_root: Path, run_dir: Path, config: TTTAutoResearchConfig) -> None:
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.config = config
        self.client = RunPodAPIClient(config)
        self.lock = threading.Lock()
        self.available: queue.Queue[RunPodPod] = queue.Queue()
        self.created_pods: dict[str, RunPodPod] = {}
        self.repo_archive_path = self.run_dir / "runpod_repo_bundle.tar.gz"
        self.repo_archive_lock = threading.Lock()
        self.closed = False
        self.sequence = 0
        self._validate_ssh_key()
        self._cleanup_orphaned_pods()
        self._write_repo_archive()

    def _validate_ssh_key(self) -> None:
        key_path = self.config.runpod_ssh_private_key_path
        if key_path:
            if not Path(key_path).exists():
                raise RunPodError(f"SSH private key not found at {key_path}")
            return
        # No explicit key — check that the system default exists.
        default_keys = [Path.home() / ".ssh" / name for name in ("id_ed25519", "id_rsa", "id_ecdsa")]
        has_agent = bool(os.environ.get("SSH_AUTH_SOCK"))
        has_default = any(key.exists() for key in default_keys)
        if not has_agent and not has_default:
            raise RunPodError(
                "No SSH private key configured for RunPod. "
                "Set runpod_ssh_private_key_path in your config, or ensure a default key "
                "exists at ~/.ssh/id_ed25519 (or id_rsa), or run an ssh-agent."
            )

    def _cleanup_orphaned_pods(self) -> None:
        pool_state_path = self.run_dir / "runpod_pool.json"
        if not pool_state_path.exists():
            return
        try:
            entries = json.loads(pool_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(entries, list):
            return
        orphan_ids = [str(entry["id"]) for entry in entries if isinstance(entry, dict) and "id" in entry]
        if not orphan_ids:
            return
        cleaned = 0
        for pod_id in orphan_ids:
            try:
                self.client.delete_pod(pod_id)
                cleaned += 1
            except RunPodAPIError:
                pass
        if cleaned:
            print(f"[RunPodPool] Cleaned up {cleaned} orphaned pod(s) from a previous run.")
        pool_state_path.unlink(missing_ok=True)

    def execute_workspace(
        self,
        workspace: Path,
        command: list[str],
        env: dict[str, str],
        timeout_sec: int,
        label: str,
    ) -> RemoteExecutionResult:
        last_error: Exception | None = None
        for _ in range(self.config.runpod_retry_limit):
            pod = self._acquire_pod()
            reusable = True
            try:
                self._ensure_pod_ready(pod)
                result = self._run_workspace_on_pod(pod, workspace, command, env, timeout_sec, label)
                self._release_pod(pod, reusable=True)
                return result
            except RunPodPodLostError as exc:
                reusable = False
                last_error = exc
                self._release_pod(pod, reusable=False)
            except Exception:
                self._release_pod(pod, reusable=reusable)
                raise
        raise RunPodPodLostError(f"RunPod spot pod was interrupted too many times while running {label}.") from last_error

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            pods = list(self.created_pods.values())
            self.created_pods.clear()
        if not self.config.runpod_terminate_on_close:
            return
        for pod in pods:
            try:
                self.client.delete_pod(pod.id)
            except RunPodAPIError:
                continue

    def _acquire_pod(self) -> RunPodPod:
        try:
            return self.available.get_nowait()
        except queue.Empty:
            pass

        with self.lock:
            if self.closed:
                raise RunPodError("RunPod pool is closed.")
            if len(self.created_pods) < self.config.max_concurrent_evaluations:
                pod = self._create_pod()
                self.created_pods[pod.id] = pod
                self._write_pool_state()
                return pod

        return self.available.get()

    def _release_pod(self, pod: RunPodPod, reusable: bool) -> None:
        if not reusable:
            with self.lock:
                self.created_pods.pop(pod.id, None)
                self._write_pool_state()
            try:
                self.client.delete_pod(pod.id)
            except RunPodAPIError:
                pass
            return
        if not self.closed:
            self.available.put(pod)

    def _create_pod(self) -> RunPodPod:
        ordinal = self.sequence
        self.sequence += 1
        pod_name = f"{self.config.runpod_name_prefix}-{Path(self.run_dir).name}-{ordinal:02d}"
        body: dict[str, object] = {
            "name": pod_name,
            "cloudType": self.config.runpod_cloud_type,
            "interruptible": self.config.runpod_interruptible,
            "supportPublicIp": self.config.runpod_support_public_ip,
            "ports": self.config.runpod_ports,
            "containerDiskInGb": self.config.runpod_container_disk_gb,
        }
        if self.config.runpod_volume_gb > 0:
            body["volumeInGb"] = self.config.runpod_volume_gb
            body["volumeMountPath"] = self.config.runpod_volume_mount_path
        if self.config.runpod_template_id:
            body["templateId"] = self.config.runpod_template_id
        else:
            body["imageName"] = self.config.runpod_image_name
            body["gpuTypeIds"] = self.config.runpod_gpu_type_ids
        payload = self.client.create_pod(body)
        pod = self._pod_from_payload(payload)
        if pod.id == "":
            raise RunPodAPIError(f"Could not parse pod id from create response: {payload!r}")
        return pod

    def _ensure_pod_ready(self, pod: RunPodPod) -> None:
        refreshed = self._wait_for_pod_network(pod.id)
        pod.public_ip = refreshed.public_ip
        pod.ssh_port = refreshed.ssh_port
        pod.desired_status = refreshed.desired_status
        pod.machine_id = refreshed.machine_id
        self._wait_for_ssh(pod)
        if pod.ready:
            return
        self._bootstrap_pod(pod)
        pod.ready = True

    def _wait_for_pod_network(self, pod_id: str) -> RunPodPod:
        deadline = time.time() + self.config.runpod_bootstrap_timeout_sec
        while time.time() < deadline:
            payload = self._lookup_pod(pod_id)
            if payload is None:
                raise RunPodPodLostError(f"Pod {pod_id} disappeared before it became ready.")
            pod = self._pod_from_payload(payload)
            if pod.public_ip and pod.ssh_port:
                return pod
            time.sleep(self.config.runpod_poll_interval_sec)
        raise RunPodPodLostError(f"Pod {pod_id} never exposed SSH before the bootstrap timeout.")

    def _wait_for_ssh(self, pod: RunPodPod) -> None:
        deadline = time.time() + self.config.runpod_bootstrap_timeout_sec
        while time.time() < deadline:
            try:
                completed = self._run_ssh(pod, "true", timeout=30, check=False)
            except RunPodPodLostError:
                raise
            if completed.returncode == 0:
                return
            time.sleep(self.config.runpod_poll_interval_sec)
        raise RunPodPodLostError(f"Pod {pod.id} exposed a public IP but never accepted SSH.")

    def _bootstrap_pod(self, pod: RunPodPod) -> None:
        remote_archive = "/tmp/autoresearch_repo_bundle.tar.gz"
        self._upload_file(pod, self.repo_archive_path, remote_archive)
        repo_root = self.config.runpod_repo_root
        bootstrap_commands = self.config.runpod_bootstrap_commands or [
            "python3 -m pip install --upgrade uv",
            "cd {repo_root} && uv sync",
            "cd {repo_root} && uv run prepare.py --num-shards {prepare_num_shards}",
        ]
        rendered = [
            command.format(
                repo_root=repo_root,
                prepare_num_shards=self.config.runpod_prepare_num_shards,
            )
            for command in bootstrap_commands
        ]
        script_lines = [
            "set -euo pipefail",
            f"rm -rf {shlex.quote(repo_root)}",
            f"mkdir -p {shlex.quote(repo_root)}",
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(repo_root)}",
            f"if [ ! -f {shlex.quote(repo_root)}/pyproject.toml ]; then rm -rf {shlex.quote(repo_root)}/* && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(repo_root)} --strip-components=1; fi",
            f"test -f {shlex.quote(repo_root)}/pyproject.toml",
            f"test -f {shlex.quote(repo_root)}/prepare.py",
        ]
        script_lines.extend(rendered)
        self._run_ssh(
            pod,
            "\n".join(script_lines),
            timeout=self.config.runpod_bootstrap_timeout_sec,
            check=True,
        )

    def _run_workspace_on_pod(
        self,
        pod: RunPodPod,
        workspace: Path,
        command: list[str],
        env: dict[str, str],
        timeout_sec: int,
        label: str,
    ) -> RemoteExecutionResult:
        remote_workspace = f"{self.config.runpod_repo_root}/../jobs/{label}-{uuid.uuid4().hex[:8]}"
        remote_archive = f"/tmp/{uuid.uuid4().hex}.tar.gz"
        local_archive = self._build_workspace_archive(workspace)
        try:
            self._upload_file(pod, local_archive, remote_archive)
            env_lines = []
            for key in sorted(env):
                if key.startswith("RUNPOD_"):
                    continue
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
            self._run_ssh(pod, script, timeout=timeout_sec + 180, check=True)
            elapsed = time.time() - start
            stdout = self._download_text_file(pod, f"{remote_workspace}/stdout.log")
            stderr = self._download_text_file(pod, f"{remote_workspace}/stderr.log")
            exit_text = self._download_text_file(pod, f"{remote_workspace}/.exit_code").strip()
            # If .exit_code is missing or empty, assume the process crashed rather
            # than silently treating it as success (returncode=0).
            if exit_text:
                try:
                    returncode = int(exit_text)
                except ValueError:
                    returncode = 1
            else:
                returncode = 1
            metrics_json = self._download_text_file(pod, f"{remote_workspace}/metrics.json")
            if metrics_json:
                metrics_dest = workspace / "metrics.json"
                metrics_dest.write_text(metrics_json, encoding="utf-8")
            return RemoteExecutionResult(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                elapsed_sec=elapsed,
            )
        finally:
            try:
                self._run_ssh(
                    pod,
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
            except RunPodPodLostError:
                pass
            local_archive.unlink(missing_ok=True)

    def _upload_file(self, pod: RunPodPod, local_path: Path, remote_path: str) -> None:
        destination = f"{self.config.runpod_ssh_user}@{pod.public_ip}:{remote_path}"
        try:
            completed = subprocess.run(
                self._scp_base_args(pod) + [str(local_path), destination],
                text=True,
                capture_output=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if self._pod_missing(pod.id):
                raise RunPodPodLostError(f"Pod {pod.id} disappeared while uploading {local_path.name}.") from exc
            raise RunPodError(f"Timed out uploading {local_path.name} to pod {pod.id}.") from exc
        if completed.returncode != 0:
            if self._pod_missing(pod.id):
                raise RunPodPodLostError(f"Pod {pod.id} disappeared while uploading {local_path.name}.")
            raise RunPodError(completed.stderr.strip() or f"scp upload to {pod.id} failed.")

    def _download_text_file(self, pod: RunPodPod, remote_path: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / Path(remote_path).name
            source = f"{self.config.runpod_ssh_user}@{pod.public_ip}:{remote_path}"
            try:
                completed = subprocess.run(
                    self._scp_base_args(pod) + [source, str(local_path)],
                    text=True,
                    capture_output=True,
                    timeout=600,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                if self._pod_missing(pod.id):
                    raise RunPodPodLostError(f"Pod {pod.id} disappeared while downloading artifacts.") from exc
                raise RunPodError(f"Timed out downloading {remote_path} from pod {pod.id}.") from exc
            if completed.returncode != 0:
                if self._pod_missing(pod.id):
                    raise RunPodPodLostError(f"Pod {pod.id} disappeared while downloading artifacts.")
                return ""
            return local_path.read_text(encoding="utf-8")

    def _run_ssh(self, pod: RunPodPod, script: str, timeout: int, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                self._ssh_base_args(pod) + [f"bash -lc {shlex.quote(script)}"],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if self._pod_missing(pod.id):
                raise RunPodPodLostError(f"Pod {pod.id} was interrupted during remote execution.") from exc
            raise RunPodError(f"Timed out waiting for remote SSH command on pod {pod.id}.") from exc
        if completed.returncode == 255 and self._pod_missing(pod.id):
            raise RunPodPodLostError(f"Pod {pod.id} was interrupted during remote execution.")
        if check and completed.returncode != 0:
            raise RunPodError(completed.stderr.strip() or completed.stdout.strip() or f"Remote command failed on pod {pod.id}.")
        return completed

    def _ssh_base_args(self, pod: RunPodPod) -> list[str]:
        if not pod.public_ip or pod.ssh_port is None:
            raise RunPodPodLostError(f"Pod {pod.id} does not have a reachable SSH endpoint.")
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
        if self.config.runpod_ssh_private_key_path:
            args.extend(["-i", self.config.runpod_ssh_private_key_path])
        args.extend(
            [
                "-p",
                str(pod.ssh_port),
                f"{self.config.runpod_ssh_user}@{pod.public_ip}",
            ]
        )
        return args

    def _scp_base_args(self, pod: RunPodPod) -> list[str]:
        if not pod.public_ip or pod.ssh_port is None:
            raise RunPodPodLostError(f"Pod {pod.id} does not have a reachable SSH endpoint.")
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
        if self.config.runpod_ssh_private_key_path:
            args.extend(["-i", self.config.runpod_ssh_private_key_path])
        args.extend(["-P", str(pod.ssh_port)])
        return args

    def _lookup_pod(self, pod_id: str) -> dict[str, object] | None:
        for payload in self.client.list_pods():
            if str(payload.get("id", "")) == pod_id:
                return payload
        return None

    def _pod_missing(self, pod_id: str) -> bool:
        return self._lookup_pod(pod_id) is None

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

    def _write_pool_state(self) -> None:
        payload = [
            {
                "id": pod.id,
                "name": pod.name,
                "public_ip": pod.public_ip,
                "ssh_port": pod.ssh_port,
                "desired_status": pod.desired_status,
                "machine_id": pod.machine_id,
                "ready": pod.ready,
            }
            for pod in self.created_pods.values()
        ]
        (self.run_dir / "runpod_pool.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _pod_from_payload(payload: dict[str, object]) -> RunPodPod:
        runtime = payload.get("runtime")
        public_ip: str | None = None
        ssh_port: int | None = None

        if isinstance(runtime, dict):
            ports = runtime.get("ports")
            if isinstance(ports, list):
                for port in ports:
                    if not isinstance(port, dict):
                        continue
                    private = str(port.get("privatePort", ""))
                    protocol = str(port.get("type", port.get("protocol", ""))).lower()
                    if private == "22" and "tcp" in protocol:
                        ip_value = port.get("ip")
                        if ip_value is not None:
                            public_ip = str(ip_value)
                        public_port = port.get("publicPort")
                        if public_port is not None:
                            ssh_port = int(public_port)
                        break
            if public_ip is None:
                ip_value = runtime.get("publicIp") or runtime.get("ip")
                if ip_value is not None:
                    public_ip = str(ip_value)
            if ssh_port is None:
                mappings = runtime.get("portMappings")
                if isinstance(mappings, dict):
                    for key, value in mappings.items():
                        if str(key) == "22" and value is not None:
                            ssh_port = int(value)
                            break

        return RunPodPod(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            public_ip=public_ip,
            ssh_port=ssh_port,
            desired_status=str(payload.get("desiredStatus", payload.get("status", ""))),
            machine_id=str(payload.get("machineId", "")) or None,
        )
