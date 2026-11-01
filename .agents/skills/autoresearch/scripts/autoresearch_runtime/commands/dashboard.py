"""`dashboard` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register(subparsers: Any) -> None:
    dashboard = subparsers.add_parser("dashboard")
    dashboard_commands = dashboard.add_subparsers(dest="dashboard_command", required=True)
    for name in ("start", "status", "stop"):
        command = dashboard_commands.add_parser(name)
        command.add_argument("--root", type=Path, default=None)
        if name == "start":
            command.add_argument("--open-browser", action="store_true",
                                 help="Open the dashboard URL in a browser tab")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    from ..dashboard_server import dashboard_server_status, start_dashboard, stop_dashboard
    from ..paths import resolve_home

    root = args.root or resolve_home()
    if args.dashboard_command == "start":
        return start_dashboard(root, open_browser=args.open_browser)
    if args.dashboard_command == "status":
        return dashboard_server_status(root)
    return stop_dashboard(root)
