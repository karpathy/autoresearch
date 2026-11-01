"""Cancel one durable background operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import LabError
from ..runtimeops import cancel_run
from .common import resolve_config


def register(subparsers: Any) -> None:
    command = subparsers.add_parser("cancel", help="stop and recover an active run")
    command.add_argument("run_id", help="active run UUID or alias")
    command.add_argument("--operation-id", help="optional operation UUID assertion")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    operation = cancel_run(
        str(config.lab_id), args.run_id,
        expected_operation_id=getattr(args, "operation_id", None),
    )
    error = operation.get("error")
    progress = operation.get("progress")
    return {
        "operation_id": str(operation["operation_id"]),
        "run_id": str(args.run_id),
        "status": operation["status"],
        "cancel_requested": bool(isinstance(progress, dict) and progress.get("cancel_requested") is True),
        "cancel_pending_reason": progress.get("cancel_pending_reason") if isinstance(progress, dict) else None,
        "error_code": error.get("code") if isinstance(error, dict) else None,
    }
