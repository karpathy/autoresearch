"""`doctor` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..doctor import run_doctor


def register(subparsers: Any) -> None:
    subparsers.add_parser("doctor")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    project = args.project.resolve() if args.project else None
    return run_doctor(project)
