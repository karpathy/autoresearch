"""`cases diff` subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..case_repro import compare_repeat_dirs
from ..errors import LabError


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("cases")
    cases_commands = parser.add_subparsers(dest="cases_command", required=True)
    diff_parser = cases_commands.add_parser("diff")
    diff_parser.add_argument("--left", type=Path, required=True)
    diff_parser.add_argument("--right", type=Path, required=True)
    diff_parser.add_argument("--out", type=Path, default=None)


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    if getattr(args, "cases_command", None) != "diff":
        raise LabError(f"Unknown cases subcommand: {getattr(args, 'cases_command', None)!r}")
    left = args.left.resolve()
    right = args.right.resolve()
    if not left.is_dir():
        raise LabError(f"Left repeat directory not found: {left}")
    if not right.is_dir():
        raise LabError(f"Right repeat directory not found: {right}")
    result = compare_repeat_dirs(left, right)
    out_path = getattr(args, "out", None)
    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
