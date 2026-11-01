"""`status` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..trial import status_lab
from .common import resolve_config


def register(subparsers: Any) -> None:
    subparsers.add_parser("status")


def run(args: Any, labs_root: Path | None = None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    if getattr(args, "agent", False) and not getattr(args, "brief", False):
        return {}
    return status_lab(config)
