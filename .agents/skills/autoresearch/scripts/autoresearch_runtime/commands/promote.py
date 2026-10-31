"""`promote` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..promote import export_best, promote_in_place, promote_to_branch
from .common import resolve_config


def register(subparsers: Any) -> None:
    promote = subparsers.add_parser("promote")
    promote.add_argument("--to-branch", dest="branch")
    promote.add_argument("--in-place", action="store_true")
    promote.add_argument("--allow-drift", action="store_true")
    promote.add_argument("--mode", choices=["patch", "in-place", "branch"], default="patch")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False)
    in_place = args.in_place or getattr(args, "mode", None) == "in-place"
    branch = args.branch
    if in_place and branch:
        from ..errors import LabError

        raise LabError("Use either --in-place or --to-branch, not both")
    if in_place:
        return promote_in_place(config, allow_drift=args.allow_drift)
    if branch:
        return promote_to_branch(config, branch, allow_drift=args.allow_drift)
    return export_best(config)
