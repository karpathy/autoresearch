"""`catalog repair` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    catalog = subparsers.add_parser("catalog")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    repair = catalog_commands.add_parser("repair")
    repair.add_argument("--root", type=Path, default=None)
    repair.add_argument("--apply", action="store_true", help="Request catalog changes (legacy writes are refused)")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    from ..catalog import repair_catalog
    from ..paths import resolve_home

    return repair_catalog(args.root or resolve_home(), apply=args.apply)
