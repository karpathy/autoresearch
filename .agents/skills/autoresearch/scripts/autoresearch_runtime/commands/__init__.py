"""One module per CLI subcommand. cli.py only builds the parser and prints JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import (
    baseline,
    cancel,
    cases,
    dashboard,
    doctor,
    epoch,
    guide,
    ingest,
    probe,
    promote,
    preflight,
    recover,
    resume,
    replay,
    status,
    storage,
    step,
    trial,
    vault,
)

COMMANDS = (
    doctor, baseline, epoch, status, recover, trial, replay, vault, cancel,
    dashboard, promote, probe, cases, guide, ingest, step, preflight, resume, storage,
)


def register_all(subparsers: Any) -> None:
    for command in COMMANDS:
        command.register(subparsers)


def run_command(args: Any, labs_root: Path | None) -> dict[str, Any]:
    by_name = {
        "baseline": baseline, "epoch": epoch, "status": status,
        "recover": recover, "trial": trial, "replay": replay,
        "vault": vault, "dashboard": dashboard, "doctor": doctor, "promote": promote,
        "probe": probe, "cases": cases, "storage": storage, "cancel": cancel,
        "guide": guide, "ingest": ingest, "step": step, "preflight": preflight, "resume": resume,
    }
    return by_name[args.command].run(args, labs_root)
