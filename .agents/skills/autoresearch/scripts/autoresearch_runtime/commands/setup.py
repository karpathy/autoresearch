"""`setup` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..snapshot import setup_project


def register(subparsers: Any) -> None:
    setup = subparsers.add_parser("setup")
    setup.add_argument("--project", type=Path, dest="setup_project")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    project = (args.setup_project or args.project or Path.cwd()).resolve()
    return setup_project(project, labs_root)
