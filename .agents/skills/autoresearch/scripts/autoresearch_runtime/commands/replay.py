"""`replay` subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import LabError
from ..locking import lab_lock
from ..replay import replay_plan, run_replay
from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    replay = subparsers.add_parser("replay")
    replay.add_argument("--plan", action="store_true")
    replay.add_argument("--trial", dest="replay_trial")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    if args.plan == bool(args.replay_trial):
        raise LabError("Use either replay --plan or replay --trial <id>")
    config = resolve_config(
        args, labs_root, pending_ok=False,
        mutating_action=None if args.plan else "replay --trial",
    )
    if args.plan:
        return replay_plan(config)
    if config.storage_mode == "sqlite":
        return run_durable_operation(config, "replay", {
            "trial_id": args.replay_trial, **explicit_confirmation(args),
        })
    with lab_lock(config.state_dir):
        return run_replay(config, args.replay_trial)
