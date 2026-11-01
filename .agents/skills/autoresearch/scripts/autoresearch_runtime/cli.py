"""Small, data-agnostic experiment runner.

An evaluator owns the domain data and writes one result.json per repeat. This
module only manages files, comparable metrics, and experiment history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .dashboard import render_report_page
from .catalog import update_lab_status

from .commands import register_all, run_command
from .envelope import emit, fail
from .errors import LabError
from .fuzzy_args import normalize_cli_args


def build_parser() -> argparse.ArgumentParser:
    """Build the lab CLI parser without parsing or executing anything."""
    parser = argparse.ArgumentParser(description="Generic hypothesis lab")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lab", type=str, default=None, help="Catalog lab id when --config is omitted")
    parser.add_argument("--agent", action="store_true", help="Return a compact command outcome and canonical guide")
    parser.add_argument("--headless", action="store_true", help="Do not start or open the local dashboard")
    parser.add_argument("--confirmed", action="store_true",
                        help="Confirm this evaluator or write operation for the current contract")
    register_all(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None, *, labs_root: Path | None = None) -> int:
    parser = build_parser()
    if argv is not None:
        argv_list = list(argv)
    else:
        argv_list = list(sys.argv[1:])
    argv_list = normalize_cli_args(argv_list)
    try:
        args = parser.parse_args(argv_list)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 2
        return code if code != 0 else 0
    dashboard_warning = None
    try:
        if args.agent and args.command == "trial" and getattr(args, "trial_command", None) in (None, "start"):
            hint_msg = (
                "Run: python scripts/lab.py --agent step --config <path> --hypothesis-json '{\"title\":\"...\",\"observation\":\"...\",\"expected_effect\":\"...\"}'"
            )
            has_inline_hypothesis = bool(
                getattr(args, "title", None)
                and getattr(args, "observation", None)
                and getattr(args, "expected_effect", None)
            )
            has_hypothesis = bool(
                getattr(args, "hypothesis", None)
                or getattr(args, "hypothesis_json", None)
                or has_inline_hypothesis
            )
            if not has_hypothesis:
                raise LabError(
                    "Agent trial requires --hypothesis with title, observation, and a prediction in expected_effect",
                    code="E_CONTRACT_HYPOTHESIS",
                    hint=hint_msg,
                )
            from .step import resolve_hypothesis_args, load_and_validate_hypothesis
            hypothesis_input = resolve_hypothesis_args(args)
            try:
                hypothesis = load_and_validate_hypothesis(hypothesis_input)
            except LabError as error:
                if error.code == "E_CONTRACT_HYPOTHESIS" and str(error).startswith("Hypothesis requires non-empty"):
                    raise LabError(
                        "Agent trial requires --hypothesis with title, observation, and a prediction in expected_effect",
                        code="E_CONTRACT_HYPOTHESIS",
                        hint=hint_msg,
                    ) from error
                raise
    except (LabError, RuntimeError) as error:
        command = getattr(args, "command", "lab")
        return fail(command, error)

    if not args.headless and not args.agent and not os.environ.get("AUTORESEARCH_NO_DASHBOARD") and args.command != "dashboard":
        try:
            from .dashboard_server import ensure_dashboard_server
            from .paths import resolve_home

            dashboard = ensure_dashboard_server(resolve_home(), open_browser=False)
            dashboard_warning = dashboard.get("warning") if isinstance(dashboard, dict) else None
        except Exception:
            dashboard_warning = "Dashboard did not start; the CLI command will continue and a later command can retry."

    try:
        output = run_command(args, labs_root)
    except (LabError, RuntimeError) as error:
        if dashboard_warning:
            print(f"Warning: {dashboard_warning}", file=sys.stderr)
        command = getattr(args, "command", "lab")
        return fail(command, error)
    warnings = output.pop("_warnings", None) if isinstance(output, dict) else None
    warnings = list(warnings or [])
    if dashboard_warning:
        warnings.append(str(dashboard_warning))
    cockpit_payload = None
    if getattr(args, "agent", False) and args.command == "guide":
        # guide already returns the canonical compact cockpit document.
        # Preserve it as command data instead of adding the same projection again.
        cockpit_payload = None
    elif getattr(args, "agent", False):
        try:
            from .commands.common import resolve_config
            from .state import load_state
            from .cockpit import build_cockpit

            cfg = resolve_config(args, labs_root, pending_ok=True)
            st = load_state(cfg)
            # Status and guide payloads are summaries, not run manifests.
            run = output if isinstance(output, dict) and output.get("id") and output.get("status") else None
            cockpit_payload = build_cockpit(cfg, st, last_run=run,
                                            run_id=getattr(args, "run_id", None) if args.command == "guide" else None)
            if isinstance(output, dict):
                output = {key: output[key] for key in (
                    "id", "status", "reason", "executed", "run_id",
                    "patch", "applied", "conflicts", "project_prefix", "apply_check", "branch",
                    "worktree", "apply_error",
                ) if key in output}
        except Exception:
            cockpit_payload = None
    return emit(args.command, output, warnings=warnings, cockpit=cockpit_payload)

def _print_console_safe(message: str, stream: Any) -> None:
    """Avoid Windows legacy code pages raising while printing non-ASCII diagnostics."""
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        message.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        message = message.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
    print(message, file=stream)
