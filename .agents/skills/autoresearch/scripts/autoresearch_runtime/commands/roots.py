"""Register and list extra lab storage roots under AUTORESEARCH_HOME."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..paths import add_root, home_root_id, load_roots, resolve_home


def list_roots() -> dict[str, Any]:
    home = resolve_home()
    entries: list[dict[str, Any]] = [{"id": home_root_id(home), "path": str(home), "home": True}]
    for item in load_roots(home):
        entries.append({"id": str(item.get("id")), "path": str(item.get("path")), "home": False})
    return {"home": str(home), "roots": entries}


def add_root_command(path: Path) -> dict[str, Any]:
    record = add_root(path)
    return {"added": record, "roots": list_roots()["roots"]}


def register(subparsers: Any) -> None:
    roots = subparsers.add_parser("roots")
    commands = roots.add_subparsers(dest="roots_command", required=True)
    commands.add_parser("list")
    add = commands.add_parser("add")
    add.add_argument("path", type=Path)


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    if args.roots_command == "add":
        return add_root_command(args.path)
    return list_roots()
