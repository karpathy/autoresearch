"""`recover` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..locking import lab_lock
from ..trial import recover_lab
from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    subparsers.add_parser("recover")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False, mutating_action="recover")
    if config.storage_mode == "sqlite":
        return run_durable_operation(
            config, "recover", {"restore_candidate": True, **explicit_confirmation(args)}
        )
    with lab_lock(config.state_dir):
        return {"interrupted": recover_lab(config)}
