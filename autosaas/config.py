from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CommandsConfig:
    lint: str
    typecheck: str
    test: str
    dev: str
    smoke: str


@dataclass(frozen=True)
class TargetConfig:
    commands: CommandsConfig
    app_boot_url: str | None = None


def load_target_config(config_path: Path) -> TargetConfig:
    """
    Load a project's `project.autosaas.yaml` configuration.

    Intentionally narrow: only the `commands.*` keys used by the early scaffolding
    are supported/validated here.
    """
    raw_data = yaml.safe_load(config_path.read_text())
    data = raw_data if raw_data is not None else {}
    if not isinstance(data, dict):
        raise ValueError("project.autosaas.yaml must be a mapping")
    commands = data.get("commands") or {}
    if not isinstance(commands, dict):
        raise ValueError("project.autosaas.yaml 'commands' must be a mapping")

    required = ("lint", "typecheck", "test", "dev", "smoke")
    missing = [k for k in required if k not in commands]
    if missing:
        raise KeyError(f"Missing required commands: {', '.join(missing)}")

    cmd_cfg = CommandsConfig(
        lint=str(commands["lint"]),
        typecheck=str(commands["typecheck"]),
        test=str(commands["test"]),
        dev=str(commands["dev"]),
        smoke=str(commands["smoke"]),
    )
    app_boot_raw = data.get("app_boot_url")
    app_boot_url: str | None = None
    if app_boot_raw is not None:
        app_boot_url = str(app_boot_raw)
    return TargetConfig(commands=cmd_cfg, app_boot_url=app_boot_url)
