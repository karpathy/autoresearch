"""`init` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..locking import lab_lock
from ..snapshot import init_lab
from .common import resolve_config


def register(subparsers: Any) -> None:
    subparsers.add_parser("init")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False, mutating_action="init")
    with lab_lock(config.state_dir):
        return init_lab(config)
