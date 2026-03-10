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
    "qwen3",
    "qwen3_instruct",
    "gpt_oss_no_sysprompt",
    "gpt_oss_low_reasoning",
    "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
)


@dataclass(slots=True)
class TTTAutoResearchConfig:
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    provider: str | None = None
    api_base: str | None = None
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
    wandb_project: str | None = "autoresearch-ttt-discover"
    num_cpus_per_task: int = 0
    eval_timeout: int | None = None
    local_model_path: str | None = None
    keep_history: int = 6
    max_concurrent_evaluations: int = 1
    gpu_devices: list[str] | None = None

    def normalized(self, repo_root: Path) -> "TTTAutoResearchConfig":
        run_dir = _resolve_path(self.run_dir, repo_root) if self.run_dir else repo_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.experiment_name or run_dir.name
        return TTTAutoResearchConfig(
            model_name=self.model_name,
            provider=self.provider,
            api_base=self.api_base,
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
            gpu_devices=_normalize_string_list(self.gpu_devices),
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
