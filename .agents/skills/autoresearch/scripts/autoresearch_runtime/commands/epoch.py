"""`epoch open` subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import LabError
from ..locking import lab_lock
from ..measurement import open_contract_epoch
from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("epoch")
    epoch_commands = parser.add_subparsers(dest="epoch_command", required=True)
    open_parser = epoch_commands.add_parser("open")
    open_parser.add_argument("--reason", default="")
    open_parser.add_argument("--spec", type=Path, required=True)


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    if getattr(args, "epoch_command", None) != "open":
        raise LabError(f"Unknown epoch subcommand: {getattr(args, 'epoch_command', None)!r}")
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabError(f"Invalid epoch spec: {error}") from error
    config = resolve_config(args, labs_root, pending_ok=True, mutating_action="epoch open")
    reason = str(getattr(args, "reason", "") or "")
    if config.storage_mode == "sqlite":
        params = {"action": "open", "spec": spec}
        if reason:
            params["reason"] = reason
        params.update(explicit_confirmation(args))
        return run_durable_operation(
            config, "epoch", params
        )
    with lab_lock(config.state_dir):
        return open_contract_epoch(config.path, spec, reason=reason)
