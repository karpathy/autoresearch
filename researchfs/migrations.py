"""Inspect and upgrade ResearchFS SQLite schema versions."""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from typing import TextIO

from .db import (
    CURRENT_SCHEMA_VERSION,
    get_connection,
    get_pending_migrations,
    get_schema_version,
    migrate_db,
)


def _migration_payload(migrations):
    return [
        {"version": migration.version, "name": migration.name}
        for migration in migrations
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or upgrade a ResearchFS SQLite database."
    )
    parser.add_argument("command", choices=("status", "check", "upgrade"))
    parser.add_argument(
        "--db-path",
        help="SQLite database path (defaults to RESEARCHFS_DB_PATH or researchfs.db).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        conn = get_connection(args.db_path)
        try:
            before_version = get_schema_version(conn)
            pending_before = get_pending_migrations(conn)
            if args.command == "upgrade":
                applied = migrate_db(conn)
                payload = {
                    "ok": True,
                    "command": "upgrade",
                    "from_version": before_version,
                    "to_version": get_schema_version(conn),
                    "latest_version": CURRENT_SCHEMA_VERSION,
                    "applied": _migration_payload(applied),
                }
                exit_code = 0
            else:
                payload = {
                    "ok": args.command == "status" or not pending_before,
                    "command": args.command,
                    "current_version": before_version,
                    "latest_version": CURRENT_SCHEMA_VERSION,
                    "up_to_date": not pending_before,
                    "pending": _migration_payload(pending_before),
                }
                exit_code = 1 if args.command == "check" and pending_before else 0
        finally:
            conn.close()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=stderr)
        return 2

    print(json.dumps(payload, sort_keys=True), file=stdout)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
