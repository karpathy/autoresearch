"""`guide` subcommand for Agent Cockpit state, impact, and next actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..cockpit import build_cockpit, render_ascii_card
from ..state import load_state
from .common import resolve_config


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("guide")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human/agent readable card")
    parser.add_argument("--run", dest="run_id", help="Guide around one recorded run ID")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    state = load_state(config)
    cockpit = build_cockpit(config, state, run_id=getattr(args, "run_id", None))
    current_epoch = state.get("current_epoch")
    if isinstance(current_epoch, int) and not isinstance(current_epoch, bool) and current_epoch >= 1:
        from ..harness_policy import contract_fingerprint
        cockpit["contract_fingerprint"] = contract_fingerprint(config)
    if not getattr(args, "json", False):
        card = render_ascii_card(cockpit)
        cockpit["card"] = card
    return cockpit
