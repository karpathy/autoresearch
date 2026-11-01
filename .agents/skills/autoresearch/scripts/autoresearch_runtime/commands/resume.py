"""Continue one pinned baseline or trial under the lab's existing lock."""
from pathlib import Path
from typing import Any

from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("resume")
    parser.add_argument("run_id", help="Original interrupted baseline or trial ID")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    return run_durable_operation(
        config, "resume", {"run_id": str(args.run_id), **explicit_confirmation(args)}
    )
