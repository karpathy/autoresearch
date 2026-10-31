"""One module per CLI subcommand. cli.py only builds the parser and prints JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import (
    archive,
    baseline,
    cancel,
    catalog_cmd,
    cases,
    compat,
    dashboard,
    doctor,
    epoch,
    guide,
    history_import,
    ingest,
    init,
    probe,
    promote,
    preflight,
    roots,
    recover,
    resume,
    replay,
    report,
    setup,
    status,
    storage,
    step,
    trial,
    vault,
)

COMMANDS = (
    doctor, setup, init, baseline, epoch, status, report, recover, trial, replay, vault, cancel,
    history_import, dashboard, archive, catalog_cmd, promote, probe, roots, compat, cases,
    guide, ingest, step, preflight, resume, storage,
)


def register_all(subparsers: Any) -> None:
    for command in COMMANDS:
        command.register(subparsers)


def run_command(args: Any, labs_root: Path | None) -> dict[str, Any]:
    for command in COMMANDS:
        # The parser dest is the subcommand name, which matches the module's parser.
        pass
    by_name = {
        "setup": setup, "init": init, "baseline": baseline, "epoch": epoch, "status": status,
        "report": report, "recover": recover, "trial": trial, "replay": replay,
        "vault": vault, "history-import": history_import, "dashboard": dashboard,
        "archive": archive, "catalog": catalog_cmd, "doctor": doctor, "promote": promote,
        "probe": probe, "roots": roots, "compat": compat, "cases": cases, "storage": storage,
        "cancel": cancel,
        "guide": guide, "ingest": ingest, "step": step, "preflight": preflight, "resume": resume,
    }
    return by_name[args.command].run(args, labs_root)
