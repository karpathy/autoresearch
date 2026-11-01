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
    promote,
    preflight,
    recover,
    resume,
    status,
    storage,
    step,
    trial,
    vault,
)

COMMANDS = (
    doctor, baseline, epoch, status, recover, trial, vault, cancel,
    dashboard, promote, cases, guide, step, preflight, resume, storage,
)


def register_all(subparsers: Any) -> None:
    for command in COMMANDS:
        command.register(subparsers)


def run_command(args: Any, labs_root: Path | None) -> dict[str, Any]:
    by_name = {
        "baseline": baseline, "epoch": epoch, "status": status,
        "recover": recover, "trial": trial,
        "vault": vault, "dashboard": dashboard, "doctor": doctor, "promote": promote,
        "cases": cases, "storage": storage, "cancel": cancel,
        "guide": guide, "step": step, "preflight": preflight, "resume": resume,
    }
    return by_name[args.command].run(args, labs_root)
