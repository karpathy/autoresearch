"""`report` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state import _refresh_dashboard, load_state
from .common import resolve_config


def register(subparsers: Any) -> None:
    subparsers.add_parser("report")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    load_state(config)
    _refresh_dashboard(config)
    return {"dashboard": str(config.state_dir / "dashboard.html")}
