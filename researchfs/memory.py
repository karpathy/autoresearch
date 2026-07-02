"""Operational CLI for portable compressed ResearchFS memory bundles."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from researchfs.bundle import (
    export_corpus_memory_bundle,
    export_paper_memory_bundle,
    import_memory_bundle,
)
from researchfs.evaluation import evaluate_extraction
from researchfs.db import get_connection, initialize_db
from researchfs import retrieve as rfs

JSONDict = dict[str, Any]


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant with injectable stderr for tests and embedders."""

    def __init__(self, *args: Any, stderr: TextIO | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stderr = stderr or sys.stderr

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._stderr.write(message)
        raise SystemExit(status)

    def error(self, message: str) -> None:
        _write_json({"error": message, "ok": False}, self._stderr)
        raise SystemExit(2)


@contextmanager
def _open_initialized_connection(db_path: str | None) -> Iterator[Any]:
    conn = get_connection(db_path)
    try:
        initialize_db(conn)
        yield conn
    finally:
        conn.close()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(value: Any, stream: TextIO) -> None:
    stream.write(_json_text(value))


def _write_json_file(path_text: str, value: Any) -> Path:
    path = Path(path_text).expanduser()
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(value), encoding="utf-8")
    return path


def _read_json_file(path_text: str) -> Any:
    return json.loads(Path(path_text).expanduser().read_text(encoding="utf-8"))


def _int_value(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc


def _non_negative_int(value: str) -> int:
    parsed = _int_value(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed

def _positive_int(value: str) -> int:
    parsed = _int_value(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed

def _metric_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _non_empty_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("must be non-empty")
    return text


def _source_type_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source-type must not be empty")
    return label


def _parse_paper_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("--paper-ids must be a comma-separated list of integers")
    parsed: list[int] = []
    for part in parts:
        paper_id = _int_value(part)
        if paper_id < 0:
            raise argparse.ArgumentTypeError("--paper-ids must contain non-negative integers")
        if paper_id not in parsed:
            parsed.append(paper_id)
    return parsed


def _export_summary(bundle: JSONDict, output: Path) -> JSONDict:
    summary: JSONDict = {
        "action": "export",
        "kind": bundle.get("kind"),
        "ok": True,
        "output": str(output),
        "version": bundle.get("version"),
    }
    if "paper" in bundle:
        source_ids = bundle.get("source_ids") or {}
        summary["paper_id"] = source_ids.get("paper_id")
    else:
        papers = bundle.get("papers") or []
        paper_ids = []
        for paper_bundle in papers:
            if isinstance(paper_bundle, dict):
                source_ids = paper_bundle.get("source_ids") or {}
                paper_ids.append(source_ids.get("paper_id"))
        summary["paper_count"] = len(papers)
        summary["paper_ids"] = paper_ids
    return summary


def _export(args: argparse.Namespace, stdout: TextIO) -> int:
    with _open_initialized_connection(args.db_path) as conn:
        if args.paper_id is not None:
            bundle = export_paper_memory_bundle(args.paper_id, conn=conn, max_chars=args.max_chars)
        elif args.paper_ids is not None:
            bundle = export_corpus_memory_bundle(
                paper_ids=args.paper_ids,
                conn=conn,
                max_chars=args.max_chars,
            )
        else:
            bundle = export_corpus_memory_bundle(conn=conn, max_chars=args.max_chars)

    if args.output:
        output = _write_json_file(args.output, bundle)
        _write_json(_export_summary(bundle, output), stdout)
    else:
        _write_json(bundle, stdout)
    return 0


def _import(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    if args.input:
        bundle = _read_json_file(args.input)
    else:
        bundle = json.loads(stdin.read())

    with _open_initialized_connection(args.db_path) as conn:
        result = import_memory_bundle(bundle, conn=conn)

    summary = {
        "action": "import",
        "kind": bundle.get("kind") if isinstance(bundle, dict) else None,
        "ok": True,
        **result,
    }
    _write_json(summary, stdout)
    return 0

def _json_error(code: str, message: str, stream: TextIO) -> None:
    _write_json({"error": {"code": code, "message": message}, "ok": False}, stream)


def _load_expected_methods(args: argparse.Namespace) -> Any:
    if args.expected_methods is not None:
        return json.loads(args.expected_methods)
    return _read_json_file(args.expected_methods_file)


def _evaluate(args: argparse.Namespace, stdout: TextIO) -> int:
    expected_methods = _load_expected_methods(args)
    if not isinstance(expected_methods, list) or not expected_methods:
        raise ValueError("'expected_methods' must be a non-empty list")

    kwargs: JSONDict = {}
    if args.min_recall is not None:
        kwargs["min_recall"] = args.min_recall
    if args.min_evidence_coverage is not None:
        kwargs["min_evidence_coverage"] = args.min_evidence_coverage
    if args.notes is not None:
        kwargs["notes"] = args.notes

    with _open_initialized_connection(args.db_path) as conn:
        result = evaluate_extraction(conn, args.paper_id, expected_methods, **kwargs)

    _write_json({"ok": True, "evaluation": result}, stdout)
    return 0

def _query(args: argparse.Namespace, stdout: TextIO) -> int:
    with _open_initialized_connection(args.db_path) as conn:
        body = rfs.get_research_brief(
            args.query,
            conn=conn,
            max_chars=args.max_chars,
            max_papers=args.max_papers,
            include_scores=args.include_scores,
            agent_run_id=args.agent_run_id,
            source_types=args.source_type,
        )
        if args.log:
            event_id = rfs.log_retrieval_event(
                args.event_type,
                args.query,
                body["context"],
                agent_run_id=args.agent_run_id,
                experiment_id=args.experiment_id,
                conn=conn,
                method_ids=body.get("method_ids") or [],
            )
            body["event_id"] = event_id

    _write_json(body, stdout)
    return 0


def build_parser(stderr: TextIO | None = None) -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="researchfs.memory",
        description="Export, import, query, and evaluate compressed ResearchFS memory.",
        stderr=stderr,
    )
    parser_factory = lambda *args, **kwargs: _ArgumentParser(*args, stderr=stderr, **kwargs)
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=parser_factory)

    export_parser = subparsers.add_parser(
        "export",
        help="Export a compressed paper or corpus memory bundle.",
    )
    export_choice = export_parser.add_mutually_exclusive_group(required=True)
    export_choice.add_argument("--paper-id", type=_non_negative_int, help="Export one paper by ID")
    export_choice.add_argument(
        "--paper-ids",
        type=_parse_paper_ids,
        help="Export a selected corpus as comma-separated paper IDs",
    )
    export_choice.add_argument("--all", action="store_true", help="Export all papers as a corpus bundle")
    export_parser.add_argument("--max-chars", type=_non_negative_int, default=None)
    export_parser.add_argument("--db-path", default=None, help="SQLite DB path")
    export_parser.add_argument("--output", default=None, help="Write bundle JSON to this file")

    import_parser = subparsers.add_parser(
        "import",
        help="Import a compressed paper or corpus memory bundle.",
    )
    import_parser.add_argument("--input", default=None, help="Read bundle JSON from this file; stdin if omitted")
    import_parser.add_argument("--db-path", default=None, help="SQLite DB path")

    query_parser = subparsers.add_parser(
        "query",
        help="Query compressed local ResearchFS memory.",
    )
    query_parser.add_argument("--query", required=True, type=_non_empty_text)
    query_parser.add_argument("--max-chars", type=_non_negative_int, default=None)
    query_parser.add_argument("--max-papers", type=_positive_int, default=3)
    query_parser.add_argument(
        "--source-type",
        action="append",
        type=_source_type_label,
        default=None,
        help="Restrict retrieval to a source type; repeat for multiple labels",
    )
    query_parser.add_argument("--include-scores", action="store_true")
    query_parser.add_argument("--log", action="store_true")
    query_parser.add_argument("--event-type", type=_non_empty_text, default="research_brief")
    query_parser.add_argument("--agent-run-id", type=_non_negative_int, default=None)
    query_parser.add_argument("--experiment-id", type=_non_negative_int, default=None)
    query_parser.add_argument("--db-path", default=None, help="SQLite DB path")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate compressed extraction quality for one paper.",
    )
    evaluate_parser.add_argument("--paper-id", required=True, type=_positive_int)
    expected_choice = evaluate_parser.add_mutually_exclusive_group(required=True)
    expected_choice.add_argument("--expected-methods", default=None, help="Expected methods JSON list")
    expected_choice.add_argument("--expected-methods-file", default=None, help="Read expected methods JSON list")
    evaluate_parser.add_argument("--min-recall", type=_metric_threshold, default=None)
    evaluate_parser.add_argument("--min-evidence-coverage", type=_metric_threshold, default=None)
    evaluate_parser.add_argument("--notes", type=_non_empty_text, default=None)
    evaluate_parser.add_argument("--db-path", default=None, help="SQLite DB path")

    return parser


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser(stderr=stderr)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.command == "evaluate":
        try:
            return _evaluate(args, stdout)
        except LookupError as exc:
            _json_error("paper_not_found", str(exc), stderr)
            return 1
        except json.JSONDecodeError as exc:
            _json_error("invalid_json", str(exc), stderr)
            return 1
        except ValueError as exc:
            _json_error("invalid_evaluation", str(exc), stderr)
            return 1
        except OSError as exc:
            _json_error("io_error", str(exc), stderr)
            return 1

    try:
        if args.command == "export":
            return _export(args, stdout)
        if args.command == "import":
            return _import(args, stdin, stdout)
        if args.command == "query":
            return _query(args, stdout)
    except (LookupError, ValueError, OSError, json.JSONDecodeError) as exc:
        _write_json({"error": str(exc), "ok": False}, stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
