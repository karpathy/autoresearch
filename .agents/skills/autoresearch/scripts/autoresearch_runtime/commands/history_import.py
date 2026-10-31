"""Compatibility stub for the retired legacy history writer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..errors import LabError


def register(subparsers: Any) -> None:
    historical = subparsers.add_parser("history-import")
    historical.add_argument("--file", type=Path, help=argparse.SUPPRESS)


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    raise LabError(
        "`history-import` is no longer available; legacy history stays read-only",
        code="E_OPERATION_REMOVED",
        hint=(
            "Export a legacy lab with `--config <legacy-lab.json> storage export --output archive.zip`, "
            "then start fresh with `prepare` and a new AUTORESEARCH_HOME."
        ),
    )
