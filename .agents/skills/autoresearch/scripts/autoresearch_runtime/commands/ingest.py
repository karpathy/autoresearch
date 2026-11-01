"""`ingest` subcommand for universal report/run import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ingest import RESULT_TYPES, ingest_reports
from .common import resolve_config


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("ingest")
    parser.add_argument("--source", "--from-report", "--input", dest="source", required=True, type=Path,
                        help="Path to report file (JSON, JSONL, CSV, TSV, MD)")
    parser.add_argument("--group", default="imported", help="Probe group name")
    parser.add_argument("--as-type", choices=(*RESULT_TYPES, "probe"), default="experiment",
                        help="Recorded evidence type; legacy probe means experiment, never a verified baseline")
    parser.add_argument("--dry-run", action="store_true", help="Preview extracted records without writing")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    return ingest_reports(
        state_dir=config.state_dir,
        source_path=args.source,
        group=args.group,
        result_type="experiment" if args.as_type == "probe" else args.as_type,
        dry_run=args.dry_run,
    )
