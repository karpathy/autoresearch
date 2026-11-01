"""`baseline` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("baseline")
    parser.add_argument("--supersede", action="store_true")
    parser.add_argument("--reason", default="")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False)
    params: dict[str, Any] = {}
    if bool(getattr(args, "supersede", False)):
        params["supersede"] = True
    reason = str(getattr(args, "reason", "") or "")
    if reason:
        params["reason"] = reason
    params.update(explicit_confirmation(args))
    return run_durable_operation(config, "baseline", params)
