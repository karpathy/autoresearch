"""`archive` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import LabError


def register(subparsers: Any) -> None:
    archive = subparsers.add_parser("archive")
    archive.add_argument("--root", type=Path, default=None)
    archive.add_argument("--apply", action="store_true", help="retired; legacy folders are read-only")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    from ..archive import archive_idle_labs
    from ..paths import resolve_home

    root = args.root or resolve_home()
    if args.apply:
        raise LabError(
            "Physical archive moves for legacy lab folders are no longer available",
            code="E_LEGACY_READ_ONLY",
            hint=(
                "Export a legacy lab with `--config <legacy-lab.json> storage export --output archive.zip`, "
                "then start fresh with `prepare` and a new AUTORESEARCH_HOME."
            ),
        )
    return archive_idle_labs(root, apply=args.apply)
