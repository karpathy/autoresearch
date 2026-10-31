"""`compat` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import LabError
from ..lab_compat import audit_lab
from ..resolve import resolve_config_path


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("compat")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--apply", action="store_true")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    if args.migrate:
        raise LabError(
            "`compat --migrate` is no longer available; legacy labs stay read-only",
            code="E_OPERATION_REMOVED",
            hint=(
                "Export a legacy lab with `--config <legacy-lab.json> storage export --output archive.zip`, "
                "then start fresh with `prepare` and a new AUTORESEARCH_HOME."
            ),
        )
    config_path = resolve_config_path(
        config=getattr(args, "config", None),
        lab=getattr(args, "lab", None),
        project=getattr(args, "project", None),
        labs_root=labs_root,
    )
    return audit_lab(config_path)
