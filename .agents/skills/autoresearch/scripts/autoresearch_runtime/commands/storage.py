"""Export and reconcile SQLite-backed labs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..errors import LabError
from ..storage_ops import export_lab_json, reconcile_storage


def register(subparsers: Any) -> None:
    storage = subparsers.add_parser("storage", help="export or reconcile SQLite-backed labs")
    commands = storage.add_subparsers(dest="storage_command", required=True)

    export = commands.add_parser("export", help="export a SQLite lab as JSON")
    export.add_argument("--lab", dest="storage_lab", default=argparse.SUPPRESS,
                        help="SQLite lab UUID or unambiguous lab reference")
    export.add_argument("--output", required=True, type=Path, help="new export file; existing files are not overwritten")

    reconcile = commands.add_parser("reconcile", help="preview terminal run/repeat status repairs")
    reconcile.add_argument("--lab", dest="storage_lab", default=argparse.SUPPRESS,
                           help="filter by lab UUID or unambiguous lab reference")
    reconcile.add_argument("--apply", action="store_true", help="apply only repairs proven by terminal manifests")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    global_lab = getattr(args, "lab", None)
    command_lab = getattr(args, "storage_lab", None)
    if global_lab is not None and command_lab is not None:
        raise LabError(
            "Pass the storage lab selector once",
            code="E_USAGE_SELECTOR",
            hint="Use either global --lab <sqlite-ref> or storage export/reconcile --lab <sqlite-ref>.",
        )
    lab = command_lab if command_lab is not None else global_lab
    if args.storage_command == "export":
        if lab is None:
            raise LabError(
                "An export source is required",
                code="E_USAGE_SELECTOR",
                hint="Pass --lab <sqlite-ref>.",
            )
        return export_lab_json(lab=lab, output=args.output)
    if args.storage_command == "reconcile":
        return reconcile_storage(lab=lab, apply=bool(args.apply))
    raise ValueError(f"Unknown storage command: {args.storage_command}")

