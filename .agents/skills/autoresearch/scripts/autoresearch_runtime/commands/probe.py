"""`probe` subcommand (registered later by coordinator)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..locking import lab_lock
from ..probe import adopt_probe
from .common import explicit_confirmation, resolve_config, run_durable_operation


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("probe")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--runs", dest="repeats", type=int, default=None, help="Alias for --repeats")
    parser.add_argument("--env-json", default="{}")
    parser.add_argument("--adopt", help="Probe ID or commit to adopt as the new best baseline")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=False)
    if getattr(args, "adopt", None):
        with lab_lock(config.state_dir):
            return adopt_probe(config, args.adopt)

    from ..state import load_state
    state = load_state(config)
    commit = args.commit or state.get("best_commit") or "HEAD"
    group = args.group or "probes"
    label = args.label or "Agent Probe"

    repeats = args.repeats if args.repeats is not None else config.repeats
    try:
        env = json.loads(args.env_json)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid --env-json: {error}") from error
    if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise ValueError("--env-json must be an object of string keys and string values")
    return run_durable_operation(config, "probe", {
        "commit": commit, "env": env, "group": group, "label": label, "repeats": repeats,
        **explicit_confirmation(args),
    })
