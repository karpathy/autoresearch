"""`dashboard` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    dashboard = subparsers.add_parser("dashboard")
    dashboard_commands = dashboard.add_subparsers(dest="dashboard_command", required=True)
    for name in ("start", "ensure", "status", "stop"):
        command = dashboard_commands.add_parser(name)
        command.add_argument("--root", type=Path, default=None)
        if name in ("start", "ensure"):
            command.add_argument("--open-browser", action="store_true",
                                 help="Open a browser tab even if this labs root opened one before")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    from ..dashboard_server import dashboard_server_status, ensure_dashboard_server, stop_dashboard
    from ..paths import adopt_legacy_stores, resolve_home

    root = args.root or resolve_home()
    if args.root is None:
        adopt_legacy_stores(root)
    if args.dashboard_command in ("start", "ensure"):
        return ensure_dashboard_server(root, open_browser=args.open_browser, force_open=args.open_browser)
    if args.dashboard_command == "status":
        return dashboard_server_status(root)
    return stop_dashboard(root)
