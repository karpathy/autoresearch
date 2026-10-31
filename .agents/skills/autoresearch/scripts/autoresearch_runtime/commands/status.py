"""`status` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..trial import status_lab
from .common import legacy_read_only_status, resolve_config, resolve_legacy_metadata


def register(subparsers: Any) -> None:
    subparsers.add_parser("status")


def run(args: Any, labs_root: Path | None = None) -> dict[str, Any]:
    legacy = resolve_legacy_metadata(args, labs_root)
    if legacy is not None:
        return legacy_read_only_status(legacy)
    config = resolve_config(args, labs_root, pending_ok=True)
    if getattr(args, "agent", False) and not getattr(args, "brief", False):
        return {}
    return status_lab(config)
