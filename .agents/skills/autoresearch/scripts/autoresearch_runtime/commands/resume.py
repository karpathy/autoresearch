"""Continue one pinned baseline or trial under the lab's existing lock."""
from pathlib import Path
from typing import Any

from ..locking import lab_lock
from ..resume import resume_run
from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("resume")
    parser.add_argument("run_id", help="Original interrupted baseline or trial ID")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True, mutating_action="resume")
    if config.storage_mode == "sqlite":
        return run_durable_operation(
            config, "resume", {"run_id": str(args.run_id), **explicit_confirmation(args)}
        )
    with lab_lock(config.state_dir):
        return resume_run(config, args.run_id)
