from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import shlex
from typing import Any


DISCOVER_GIT_REV = "5df1a0ee9b04272ca33de0101ae64dd499e63f29"
SUPPORTED_RENDERERS = (
    "kimi_k25",
    "qwen3",
    "qwen3_instruct",
    "gpt_oss_no_sysprompt",
    "gpt_oss_low_reasoning",
    "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
)


@dataclass(slots=True)
class TTTAutoResearchConfig:
    model_name: str = "openai/gpt-oss-120b"
    provider: str | None = None
    api_base: str | None = None
    target_val_bpb: float | None = 0.97
    max_steps: int = 12
    groups_per_step: int = 2
    samples_per_step: int = 8
    temperature: float = 1.0
    timeout_sec: int = 2700
    run_dir: str | None = None
    data_path: str | None = None
    baseline_command_override: list[str] | None = None
    candidate_command_override: list[str] | None = None
    experiment_name: str | None = None
    renderer_name: str | None = None
    learning_rate: float = 4e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000
    save_every: int = 2
    wandb_project: str | None = None
    num_cpus_per_task: int = 0
    eval_timeout: int | None = None
    local_model_path: str | None = None
    keep_history: int = 6
    max_concurrent_evaluations: int = 8
    gpu_devices: list[str] | None = None
    execution_backend: str = "hyperbolic"
    hyperbolic_ssh_host: str | None = None
    hyperbolic_ssh_port: int = 22
    hyperbolic_ssh_user: str = "ubuntu"
    hyperbolic_ssh_private_key_path: str | None = None
    hyperbolic_repo_root: str = "/home/ubuntu/autoresearch"
    hyperbolic_prepare_num_shards: int = 10
    hyperbolic_bootstrap_timeout_sec: int = 7200
    hyperbolic_bootstrap_commands: list[str] | None = None
    hyperbolic_detached_controller: bool = True
    hyperbolic_remote_run_dir: str | None = None
    hyperbolic_forward_env_vars: list[str] | None = None
    hyperbolic_local_mirror: bool = True
    hyperbolic_sync_interval_sec: int = 30
    hyperbolic_local_mirror_dir: str | None = None
    runpod_api_key_env: str = "RUNPOD_API_KEY"
    runpod_api_base: str = "https://rest.runpod.io/v1"
    runpod_cloud_type: str = "COMMUNITY"
    runpod_interruptible: bool = True
    runpod_gpu_type_ids: list[str] | None = None
    runpod_template_id: str | None = None
    runpod_image_name: str | None = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    runpod_name_prefix: str = "autoresearch-ttt"
    runpod_support_public_ip: bool = True
    runpod_ports: list[str] | None = None
    runpod_container_disk_gb: int = 50
    runpod_volume_gb: int = 0
    runpod_volume_mount_path: str = "/workspace"
    runpod_ssh_user: str = "root"
    runpod_ssh_private_key_path: str | None = None
    runpod_repo_root: str = "/workspace/autoresearch"
    runpod_prepare_num_shards: int = 10
    runpod_bootstrap_timeout_sec: int = 7200
    runpod_retry_limit: int = 3
    runpod_poll_interval_sec: int = 5
    runpod_bootstrap_commands: list[str] | None = None
    runpod_terminate_on_close: bool = True

    def normalized(self, repo_root: Path) -> "TTTAutoResearchConfig":
        run_dir = _resolve_path(self.run_dir, repo_root) if self.run_dir else repo_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.experiment_name or run_dir.name
        execution_backend = self.execution_backend.lower()
        if execution_backend not in {"local", "runpod", "hyperbolic"}:
            raise ValueError("execution_backend must be one of 'local', 'runpod', or 'hyperbolic'.")
        gpu_devices = _normalize_string_list(self.gpu_devices)
        if execution_backend == "hyperbolic" and not gpu_devices:
            gpu_devices = [str(index) for index in range(8)]
        return TTTAutoResearchConfig(
            model_name=self.model_name,
            provider=self.provider,
            api_base=self.api_base,
            target_val_bpb=self.target_val_bpb,
            max_steps=self.max_steps,
            groups_per_step=max(1, int(self.groups_per_step)),
            samples_per_step=self.samples_per_step,
            temperature=self.temperature,
            timeout_sec=self.timeout_sec,
            run_dir=str(run_dir),
            data_path=_resolve_optional_path_str(self.data_path, repo_root),
            baseline_command_override=_normalize_command(self.baseline_command_override),
            candidate_command_override=_normalize_command(self.candidate_command_override),
            experiment_name=experiment_name,
            renderer_name=resolve_renderer_name(self.model_name, self.renderer_name),
            learning_rate=self.learning_rate,
            lora_rank=self.lora_rank,
            kl_penalty_coef=self.kl_penalty_coef,
            phase1_max_tokens=self.phase1_max_tokens,
            save_every=self.save_every,
            wandb_project=self.wandb_project,
            num_cpus_per_task=self.num_cpus_per_task,
            eval_timeout=self.eval_timeout or self.timeout_sec,
            local_model_path=_resolve_optional_path_str(self.local_model_path, repo_root),
            keep_history=self.keep_history,
            max_concurrent_evaluations=max(1, int(self.max_concurrent_evaluations)),
            gpu_devices=gpu_devices,
            execution_backend=execution_backend,
            hyperbolic_ssh_host=self.hyperbolic_ssh_host,
            hyperbolic_ssh_port=max(1, int(self.hyperbolic_ssh_port)),
            hyperbolic_ssh_user=self.hyperbolic_ssh_user,
            hyperbolic_ssh_private_key_path=_resolve_optional_path_str(self.hyperbolic_ssh_private_key_path, repo_root),
            hyperbolic_repo_root=self.hyperbolic_repo_root.rstrip("/"),
            hyperbolic_prepare_num_shards=max(2, int(self.hyperbolic_prepare_num_shards)),
            hyperbolic_bootstrap_timeout_sec=max(300, int(self.hyperbolic_bootstrap_timeout_sec)),
            hyperbolic_bootstrap_commands=_normalize_command(self.hyperbolic_bootstrap_commands),
            hyperbolic_detached_controller=bool(self.hyperbolic_detached_controller),
            hyperbolic_remote_run_dir=self.hyperbolic_remote_run_dir,
            hyperbolic_forward_env_vars=_normalize_string_list(self.hyperbolic_forward_env_vars)
            or [
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "TINKER_API_KEY",
                "TINKER_BASE_URL",
                "TINKER_PROVIDER",
                "TM_API_KEY",
                "HF_TOKEN",
            ],
            hyperbolic_local_mirror=bool(self.hyperbolic_local_mirror),
            hyperbolic_sync_interval_sec=max(5, int(self.hyperbolic_sync_interval_sec)),
            hyperbolic_local_mirror_dir=_resolve_optional_path_str(self.hyperbolic_local_mirror_dir, repo_root),
            runpod_api_key_env=self.runpod_api_key_env,
            runpod_api_base=self.runpod_api_base.rstrip("/"),
            runpod_cloud_type=self.runpod_cloud_type.upper(),
            runpod_interruptible=bool(self.runpod_interruptible),
            runpod_gpu_type_ids=_normalize_string_list(self.runpod_gpu_type_ids) or ["NVIDIA H100 PCIe"],
            runpod_template_id=self.runpod_template_id,
            runpod_image_name=self.runpod_image_name,
            runpod_name_prefix=self.runpod_name_prefix,
            runpod_support_public_ip=bool(self.runpod_support_public_ip),
            runpod_ports=_normalize_string_list(self.runpod_ports) or ["22/tcp"],
            runpod_container_disk_gb=max(20, int(self.runpod_container_disk_gb)),
            runpod_volume_gb=max(0, int(self.runpod_volume_gb)),
            runpod_volume_mount_path=self.runpod_volume_mount_path,
            runpod_ssh_user=self.runpod_ssh_user,
            runpod_ssh_private_key_path=_resolve_optional_path_str(self.runpod_ssh_private_key_path, repo_root),
            runpod_repo_root=self.runpod_repo_root.rstrip("/"),
            runpod_prepare_num_shards=max(2, int(self.runpod_prepare_num_shards)),
            runpod_bootstrap_timeout_sec=max(300, int(self.runpod_bootstrap_timeout_sec)),
            runpod_retry_limit=max(1, int(self.runpod_retry_limit)),
            runpod_poll_interval_sec=max(1, int(self.runpod_poll_interval_sec)),
            runpod_bootstrap_commands=_normalize_command(self.runpod_bootstrap_commands),
            runpod_terminate_on_close=bool(self.runpod_terminate_on_close),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapContext:
    repo_root: Path
    run_dir: Path
    config: TTTAutoResearchConfig
    program_text: str
    baseline_train_py: str
    baseline_val_bpb: float

    @property
    def history_path(self) -> Path:
        return self.run_dir / "history.jsonl"

    @property
    def best_dir(self) -> Path:
        return self.run_dir / "best"

    @property
    def discover_log_dir(self) -> Path:
        return self.run_dir / "discover_log"

    @property
    def candidates_dir(self) -> Path:
        return self.run_dir / "candidates"

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.provider:
            env["TINKER_PROVIDER"] = self.config.provider
        if self.config.api_base:
            env["OPENAI_BASE_URL"] = self.config.api_base
            env["OPENAI_API_BASE"] = self.config.api_base
            env["TINKER_BASE_URL"] = self.config.api_base
        if self.config.data_path:
            env["AUTORESEARCH_DATA_PATH"] = self.config.data_path
        return env


def infer_renderer_name(model_name: str) -> str | None:
    lowered = model_name.lower()
    if "kimi-k2" in lowered or "moonshotai/kimi" in lowered:
        return "kimi_k25"
    if "qwen" in lowered:
        if "instruct" in lowered:
            return "qwen3_instruct"
        return "qwen3"
    if "gpt-oss" in lowered:
        return "gpt_oss_high_reasoning"
    return None


def resolve_renderer_name(model_name: str, renderer_name: str | None) -> str:
    if renderer_name is not None:
        if renderer_name not in SUPPORTED_RENDERERS:
            supported = ", ".join(SUPPORTED_RENDERERS)
            raise ValueError(f"Unsupported renderer_name={renderer_name!r}. Supported values: {supported}.")
        return renderer_name

    inferred = infer_renderer_name(model_name)
    if inferred is None:
        supported = ", ".join(SUPPORTED_RENDERERS)
        raise ValueError(
            f"Could not infer a renderer for model_name={model_name!r}. "
            f"Set renderer_name explicitly to one of: {supported}."
        )
    return inferred


def load_config(path: str | os.PathLike[str], repo_root: str | os.PathLike[str] | None = None) -> TTTAutoResearchConfig:
    raw = _load_yaml_like(Path(path))
    config = TTTAutoResearchConfig(**raw)
    return config.normalized(Path(repo_root) if repo_root else Path.cwd())


def write_resolved_config(path: Path, config: TTTAutoResearchConfig) -> None:
    path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_command(command: list[str] | str | None) -> list[str] | None:
    if command is None:
        return None
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]


def _normalize_string_list(values: list[str] | list[int] | tuple[str, ...] | tuple[int, ...] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def _resolve_path(path_value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_optional_path_str(path_value: str | os.PathLike[str] | None, repo_root: Path) -> str | None:
    if path_value is None:
        return None
    return str(_resolve_path(path_value, repo_root))


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_yaml_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a top-level mapping.")
        return loaded
    except ImportError:
        return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key is not None:
            if current_list is None:
                current_list = []
                result[current_key] = current_list
            current_list.append(_coerce_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        current_key = key.strip()
        current_list = None
        value = value.strip()
        if value == "":
            result[current_key] = []
            current_list = result[current_key]
        else:
            result[current_key] = _coerce_scalar(value)
    return result
