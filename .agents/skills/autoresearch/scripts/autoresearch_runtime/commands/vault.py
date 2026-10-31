"""`vault restore` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..vault import restore_vault
from .common import resolve_config


def register(subparsers: Any) -> None:
    vault = subparsers.add_parser("vault")
    vault_commands = vault.add_subparsers(dest="vault_command", required=True)
    restore = vault_commands.add_parser("restore")
    restore.add_argument("--path", required=True, type=Path)
    restore.add_argument("--allow-source-write", action="store_true")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    return restore_vault(
        resolve_config(args, labs_root, pending_ok=False), args.path,
        allow_source_write=args.allow_source_write,
    )
