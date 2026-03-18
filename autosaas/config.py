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


def load_target_config(config_path: Path) -> TargetConfig:
    """
    Load a project's `project.autosaas.yaml` configuration.

    Intentionally narrow: only the `commands.*` keys used by the early scaffolding
    are supported/validated here.
    """
    data = yaml.safe_load(config_path.read_text()) or {}
    commands = (data or {}).get("commands") or {}

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
    return TargetConfig(commands=cmd_cfg)

