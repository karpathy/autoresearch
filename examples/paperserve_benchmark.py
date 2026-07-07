"""PaperServe-Bench: deterministic literature-routing scaffold.

This benchmark is intentionally local and dependency-free. It loads small JSON
fixtures, converts papers into ``researchfs.context.ContextCandidate`` objects,
runs deterministic selection strategies, and reports JSON-safe metrics for
budgeted supported-answer routing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchfs.context import (
    ContextBudget,
    ContextCandidate,
    citation_f1,
    citation_precision,
    citation_recall,
    context_efficiency,
    gold_recall_at_k,
    recovery_success,
    route_top_k_context,
)

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "paperserve_bench"
STRATEGIES = (
    "bm25_top20",
    "long_context_all_docs",
    "context_frontier_router",
    "router_verifier_loop",
)
DEFAULT_CORPUS_SIZES = (5, 20, 50, 100)
TASK_TYPES = (
    "deep_paper_finding",
    "wide_literature_collection",
    "evidence_grounded_answer",
    "missing_evidence_no_answer",
    "loop_recovery",
)
DISTRACTOR_TYPES = (
    "closer_gold_like",
    "near_miss_citation",
    "partial_evidence",
    "conflicting_evidence",
)
DISTRACTOR_MODES = ("mixed", *DISTRACTOR_TYPES)
SCALING_CHART_METRICS = (
    "supported_answer_rate_at_budget",
    "paperserve_score",
    "context_efficiency",
    "average_selected_tokens",
    "average_selected_papers",
    "false_stop_rate",
    "recovery_rate",
    "gold_recall_at_20",
    "citation_f1",
    "wide_research_precision",
    "wide_research_recall",
)
DEFAULT_SCALING_THRESHOLDS = (
    {
        "strategy": "context_frontier_router",
        "corpus_size": 100,
        "metric": "supported_answer_rate_at_budget",
        "minimum": 0.5,
    },
    {
        "strategy": "router_verifier_loop",
        "corpus_size": 100,
        "metric": "supported_answer_rate_at_budget",
        "minimum": 0.8,
    },
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_fixtures(fixture_dir: str | Path = DEFAULT_FIXTURE_DIR) -> list[dict]:
    """Load PaperServe-Bench JSON fixtures in stable filename order."""

    path = Path(fixture_dir)
    fixtures: list[dict] = []
    for fixture_path in sorted(path.glob("*.json")):
        with fixture_path.open("r", encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        fixture["candidate_papers"] = [
            _candidate_from_paper(candidate) for candidate in fixture.get("candidate_papers", [])
        ]
        _validate_fixture(fixture, fixture_path)
        fixtures.append(fixture)
    return fixtures


def evaluate_fixture(fixture: dict, *, strategies: Iterable[str] | None = None, k: int | None = None) -> dict:
    """Evaluate one fixture under the requested deterministic strategies."""

    strategy_names = _resolve_strategies(strategies)
    resolved_k = _resolve_k(fixture, k)
    max_tokens = _fixture_int(fixture, "max_reader_tokens")
    candidates = list(fixture.get("candidate_papers", []))
    results: dict[str, dict] = {}

    for strategy in strategy_names:
        if strategy == "bm25_top20":
            routed = _bm25_top20(fixture, candidates)
            recovered = False
            attempt_count = 1
        elif strategy == "long_context_all_docs":
            routed = _long_context_all_docs(fixture, candidates)
            recovered = False
            attempt_count = 1
        elif strategy == "context_frontier_router":
            routed = _context_frontier_router(fixture, _first_attempt_candidates(candidates), resolved_k, max_tokens)
            recovered = False
            attempt_count = 1
        elif strategy == "router_verifier_loop":
            routed, recovered, attempt_count = _router_verifier_loop(fixture, candidates, resolved_k, max_tokens)
        else:  # pragma: no cover - guarded by _resolve_strategies
            raise ValueError(f"unknown strategy: {strategy}")
        results[strategy] = _strategy_result(fixture, routed, resolved_k, max_tokens, recovered, attempt_count)

    return {
        "fixture_id": str(fixture["id"]),
        "task_type": str(fixture["task_type"]),
        "question": str(fixture["question"]),
        "strategies": results,
    }


def run_benchmark(
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
    *,
    strategy: str | None = None,
    k: int | None = None,
) -> dict:
    """Run PaperServe-Bench and aggregate headline metrics per strategy."""

    strategy_names = _resolve_strategies(None if strategy is None else [strategy])
    fixtures = [evaluate_fixture(fixture, strategies=strategy_names, k=k) for fixture in load_fixtures(fixture_dir)]
    aggregate = {
        strategy_name: _aggregate_results(
            [fixture["strategies"][strategy_name] for fixture in fixtures], strategy_name=strategy_name
        )
        for strategy_name in strategy_names
    }
    return {
        "benchmark": "PaperServe-Bench",
        "fixtures": fixtures,
        "aggregate": aggregate,
    }

def expand_fixture_corpus(fixture: dict, target_size: int, *, distractor_mode: str = "mixed") -> dict:
    """Return a fixture copy padded with deterministic synthetic distractors."""

    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    if distractor_mode not in DISTRACTOR_MODES:
        raise ValueError(f"distractor_mode must be one of: {', '.join(DISTRACTOR_MODES)}")


    copied = dict(fixture)
    candidates = [_copy_candidate(candidate) for candidate in fixture.get("candidate_papers", [])]
    existing_count = len(candidates)
    if target_size < existing_count:
        raise ValueError("target_size must be at least the existing candidate count")

    fixture_id = str(fixture["id"])
    question = str(fixture["question"])
    for index in range(1, target_size - existing_count + 1):
        distractor_type = _distractor_type(index, distractor_mode)
        candidates.append(
            ContextCandidate(
                id=f"synthetic-distractor-{fixture_id}-{index}",
                title=f"Distractor {index} ({distractor_type}) for {fixture_id}",
                text=_distractor_text(fixture, distractor_type, index, question),
                source="synthetic-distractor",
                metadata={
                    "synthetic_distractor": True,
                    "retrieval_attempt": 1,
                    "distractor_type": distractor_type,
                    "distractor_mode": distractor_mode,
                },
                score=_distractor_score(fixture, index),
                token_estimate=35 + (index % 11),
            )
        )

    copied["candidate_papers"] = candidates
    copied["corpus_size"] = target_size
    copied["synthetic_distractor_count"] = target_size - existing_count
    return copied


def run_scaling_benchmark(
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
    *,
    strategy: str | None = None,
    k: int | None = None,
    corpus_sizes: Iterable[int] = DEFAULT_CORPUS_SIZES,
    distractor_mode: str = "mixed",
) -> dict:
    """Run PaperServe-Bench across deterministic expanded corpus sizes."""

    normalized_sizes = _normalize_corpus_sizes(corpus_sizes)
    strategy_names = _resolve_strategies(None if strategy is None else [strategy])
    loaded_fixtures = load_fixtures(fixture_dir)
    scaling = []
    chart_series: dict[str, list[dict]] = {strategy_name: [] for strategy_name in strategy_names}

    for size in normalized_sizes:
        fixture_results = [
            evaluate_fixture(expand_fixture_corpus(fixture, size, distractor_mode=distractor_mode), strategies=strategy_names, k=k)
            for fixture in loaded_fixtures
        ]
        aggregate = {
            strategy_name: _aggregate_results(
                [fixture["strategies"][strategy_name] for fixture in fixture_results],
                strategy_name=strategy_name,
            )
            for strategy_name in strategy_names
        }
        scaling.append(
            {
                "corpus_size": size,
                "fixtures": fixture_results,
                "aggregate": aggregate,
            }
        )
        for strategy_name in strategy_names:
            headline = aggregate[strategy_name]
            point = {"corpus_size": size}
            for metric in SCALING_CHART_METRICS:
                point[metric] = headline[metric]
            chart_series[strategy_name].append(point)

    report = {
        "benchmark": "PaperServe-Bench",
        "mode": "scaling",
        "corpus_sizes": list(normalized_sizes),
        "distractor_mode": distractor_mode,
        "scaling": scaling,
        "chart": {
            "x_axis": "corpus_size",
            "y_axis": "supported_answer_rate_at_budget",
            "series": chart_series,
        },
    }
    report["thresholds"] = check_scaling_thresholds(report)
    return report

def scaling_chart_rows(report: dict) -> list[dict]:
    """Return chart-series rows suitable for CSV export."""

    rows: list[dict] = []
    for strategy, points in report["chart"]["series"].items():
        for point in points:
            rows.append(
                {"strategy": strategy, **{metric: point[metric] for metric in ("corpus_size", *SCALING_CHART_METRICS)}}
            )
    return rows


def format_scaling_csv(report: dict) -> str:
    """Format the scaling chart series as CSV."""

    fieldnames = [
        "strategy",
        "corpus_size",
        *SCALING_CHART_METRICS,
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(scaling_chart_rows(report))
    return output.getvalue()



def check_scaling_thresholds(
    report: dict,
    thresholds: Iterable[dict] = DEFAULT_SCALING_THRESHOLDS,
) -> dict:
    """Evaluate configured scaling regression thresholds against a report."""

    points_by_key = {
        (strategy, point["corpus_size"]): point
        for strategy, points in report["chart"]["series"].items()
        for point in points
    }
    checks = []
    for threshold in thresholds:
        strategy = str(threshold["strategy"])
        corpus_size = int(threshold["corpus_size"])
        metric = str(threshold["metric"])
        minimum = float(threshold["minimum"])
        point = points_by_key.get((strategy, corpus_size))
        if point is None or metric not in point:
            checks.append(
                {
                    "strategy": strategy,
                    "corpus_size": corpus_size,
                    "metric": metric,
                    "minimum": minimum,
                    "value": None,
                    "status": "skipped",
                }
            )
            continue
        value = float(point[metric])
        checks.append(
            {
                "strategy": strategy,
                "corpus_size": corpus_size,
                "metric": metric,
                "minimum": minimum,
                "value": value,
                "status": "passed" if value >= minimum else "failed",
            }
        )
    failures = [check for check in checks if check["status"] == "failed"]
    skipped = [check for check in checks if check["status"] == "skipped"]
    return {
        "kind": "minimum_metric_thresholds",
        "passed": not failures,
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "checks": checks,
    }

def format_scaling_table(report: dict, metric: str = "supported_answer_rate_at_budget") -> str:
    """Format a scaling report as an aligned strategy-by-corpus-size table."""

    corpus_sizes = list(report["corpus_sizes"])
    headers = ["strategy", *(f"corpus={size}" for size in corpus_sizes)]
    rows = []
    for strategy, points in report["chart"]["series"].items():
        points_by_size = {point["corpus_size"]: point for point in points}
        rows.append(
            [
                strategy,
                *[
                    _format_table_value(points_by_size.get(size, {}).get(metric))
                    for size in corpus_sizes
                ],
            ]
        )

    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [_format_table_row(headers, widths)]
    lines.append(_format_table_row(["-" * width for width in widths], widths))
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic PaperServe-Bench scaffold.")
    parser.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR), help="Directory of JSON fixtures.")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None, help="Run one strategy instead of all.")
    parser.add_argument("--k", type=_positive_int, default=None, help="Override selected-paper router limit.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON instead of pretty JSON.")
    parser.add_argument("--scaling", action="store_true", help="Run corpus-size scaling report mode.")
    parser.add_argument("--table", action="store_true", help="Print a human-readable scaling table instead of JSON.")
    parser.add_argument("--distractor-mode", choices=DISTRACTOR_MODES, default="mixed", help="Synthetic distractor mode for --scaling.")
    parser.add_argument("--check-thresholds", action="store_true", help="Exit nonzero if any evaluated scaling threshold fails.")
    parser.add_argument("--fail-on-thresholds", action="store_true", dest="check_thresholds", help=argparse.SUPPRESS)
    parser.add_argument("--output", default=None, help="Write the JSON report to this path.")
    parser.add_argument("--csv-output", default=None, help="Write scaling chart series CSV to this path.")
    parser.add_argument(
        "--corpus-sizes",
        type=_parse_corpus_sizes,
        default=None,
        help="Comma-separated positive corpus sizes for --scaling.",
    )
    args = parser.parse_args(argv)
    if args.table and not args.scaling:
        parser.error("--table requires --scaling")
    if args.csv_output and not args.scaling:
        parser.error("--csv-output requires --scaling")
    if args.check_thresholds and not args.scaling:
        parser.error("--check-thresholds requires --scaling")
    if args.distractor_mode != "mixed" and not args.scaling:
        parser.error("--distractor-mode requires --scaling")

    if args.scaling:
        result = run_scaling_benchmark(
            args.fixture_dir,
            strategy=args.strategy,
            k=args.k,
            corpus_sizes=args.corpus_sizes or DEFAULT_CORPUS_SIZES,
            distractor_mode=args.distractor_mode,
        )
    else:
        result = run_benchmark(args.fixture_dir, strategy=args.strategy, k=args.k)

    if args.compact:
        json_text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    else:
        json_text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(f"{json_text}\n", encoding="utf-8")
    if args.csv_output:
        Path(args.csv_output).write_text(format_scaling_csv(result), encoding="utf-8")

    if args.table:
        print(format_scaling_table(result))
    else:
        print(json_text)
    return 1 if args.check_thresholds and result.get("thresholds", {}).get("failure_count", 0) else 0


def _candidate_from_paper(paper: dict) -> ContextCandidate:
    return ContextCandidate(
        id=str(paper["id"]),
        title=paper.get("title"),
        text=str(paper["text"]),
        source=paper.get("source"),
        metadata=dict(paper.get("metadata") or {}),
        score=paper.get("score"),
        token_estimate=paper.get("token_estimate"),
    )

def _copy_candidate(candidate: ContextCandidate | dict) -> ContextCandidate:
    if not isinstance(candidate, ContextCandidate):
        return _candidate_from_paper(candidate)
    return ContextCandidate(
        id=candidate.id,
        title=candidate.title,
        text=candidate.text,
        source=candidate.source,
        metadata=dict(candidate.metadata),
        score=candidate.score,
        token_estimate=candidate.token_estimate,
    )


def _normalize_corpus_sizes(corpus_sizes: Iterable[int]) -> tuple[int, ...]:
    seen = set()
    normalized = []
    for size in corpus_sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("corpus_sizes must contain positive integers")
        if size not in seen:
            seen.add(size)
            normalized.append(size)
    if not normalized:
        raise ValueError("corpus_sizes must contain at least one size")
    return tuple(normalized)

def _distractor_type(index: int, distractor_mode: str = "mixed") -> str:
    if distractor_mode == "mixed":
        return DISTRACTOR_TYPES[(index - 1) % len(DISTRACTOR_TYPES)]
    if distractor_mode in DISTRACTOR_TYPES:
        return distractor_mode
    raise ValueError(f"unknown distractor_mode: {distractor_mode}")


def _distractor_text(fixture: dict, distractor_type: str, index: int, question: str) -> str:
    fixture_id = str(fixture["id"])
    expected_points = "; ".join(_string_list(fixture.get("expected_answer_points", []))) or "the requested answer"
    required_citations = ", ".join(_required_citations(fixture)) or "the required citation"
    gold_ids = ", ".join(_gold_ids(fixture)) or "the gold evidence"
    prefix = (
        f"Synthetic distractor {index} for {fixture_id}. "
        f"Question terms: {question}. "
    )
    if distractor_type == "closer_gold_like":
        return (
            prefix
            + f"It repeats gold-like vocabulary about {expected_points} and mentions {gold_ids}, "
            + "but it is a background summary and provides no required citation evidence."
        )
    if distractor_type == "near_miss_citation":
        return (
            prefix
            + f"It cites adjacent-looking work near {required_citations}, "
            + "but the cited material is only related work and not the paper needed to support the answer."
        )
    if distractor_type == "partial_evidence":
        return (
            prefix
            + f"It contains partial evidence about {expected_points}, "
            + "but omits the decisive condition needed for a supported answer."
        )
    if distractor_type == "conflicting_evidence":
        return (
            prefix
            + f"It makes a conflicting claim about {expected_points}, "
            + "so it should be rejected rather than treated as supporting evidence."
        )
    raise ValueError(f"unknown distractor_type: {distractor_type}")


def _distractor_score(fixture: dict, index: int) -> float:
    base_score = max(0.05, 0.62 - 0.003 * (index % 100))
    gold_ids = set(_gold_ids(fixture))
    gold_scores = []
    for candidate in fixture.get("candidate_papers", []):
        if isinstance(candidate, ContextCandidate):
            candidate_id = candidate.id
            score = candidate.score
        else:
            candidate_id = str(candidate.get("id"))
            score = candidate.get("score")
        if candidate_id in gold_ids and score is not None:
            gold_scores.append(float(score))
    if gold_scores:
        base_score = min(base_score, min(gold_scores) - 0.05)
    return round(max(0.05, base_score), 6)


def _format_table_value(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _format_table_row(row: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))


def _bm25_top20(fixture: dict, candidates: list[ContextCandidate]) -> dict:
    scored = _bm25_scored_candidates(str(fixture["question"]), candidates)
    selection = route_top_k_context(
        scored,
        k=20,
        budget=ContextBudget(max_items=20, max_tokens=None, overflow_policy="truncate"),
    )
    return _read_selection(fixture, selection.selected, selection.rejected, selection.budget_used_tokens)


def _long_context_all_docs(fixture: dict, candidates: list[ContextCandidate]) -> dict:
    selected = list(candidates)
    return _read_selection(fixture, selected, [], sum(_candidate_tokens(candidate) for candidate in selected))


def _context_frontier_router(
    fixture: dict,
    candidates: list[ContextCandidate],
    k: int,
    max_tokens: int,
) -> dict:
    selection = route_top_k_context(
        candidates,
        k=k,
        budget=ContextBudget(max_items=k, max_tokens=max_tokens, overflow_policy="drop_lowest_score"),
    )
    return _read_selection(fixture, selection.selected, selection.rejected, selection.budget_used_tokens)


def _router_verifier_loop(
    fixture: dict,
    candidates: list[ContextCandidate],
    k: int,
    max_tokens: int,
) -> tuple[dict, bool, int]:
    attempts = sorted({_retrieval_attempt(candidate) for candidate in candidates}) or [1]
    first_supported = False
    final: dict | None = None
    attempt_count = 0

    for attempt in attempts:
        attempt_count += 1
        visible = [candidate for candidate in candidates if _retrieval_attempt(candidate) <= attempt]
        final = _context_frontier_router(fixture, visible, k, max_tokens)
        if attempt_count == 1:
            first_supported = bool(final["supported"])
        if final["supported"] or not _is_recoverable(fixture, final):
            break

    if final is None:
        final = _context_frontier_router(fixture, candidates, k, max_tokens)
        attempt_count = 1
    recovered = bool(recovery_success(first_attempt_supported=first_supported, final_attempt_supported=bool(final["supported"]))) and not first_supported and attempt_count > 1
    return final, recovered, attempt_count


def _read_selection(
    fixture: dict,
    selected: list[ContextCandidate],
    rejected: list[ContextCandidate],
    selected_tokens: int,
) -> dict:
    selected_ids = [candidate.id for candidate in selected]
    required = _required_citations(fixture)
    task_type = str(fixture["task_type"])
    selected_required = [citation for citation in required if citation in set(selected_ids)]

    if task_type == "missing_evidence_no_answer":
        return {
            "selected_ids": selected_ids,
            "selected_tokens": selected_tokens,
            "rejected_ids": [candidate.id for candidate in rejected],
            "status": "insufficient_evidence",
            "citations": [],
            "supported": True,
        }

    supported = bool(required) and len(selected_required) == len(required)
    return {
        "selected_ids": selected_ids,
        "selected_tokens": selected_tokens,
        "rejected_ids": [candidate.id for candidate in rejected],
        "status": "answered" if supported else "insufficient_evidence",
        "citations": selected_required,
        "supported": supported,
    }


def _strategy_result(
    fixture: dict,
    routed: dict,
    k: int,
    max_tokens: int,
    recovered: bool,
    attempt_count: int,
) -> dict:
    task_type = str(fixture["task_type"])
    gold_ids = _gold_ids(fixture)
    gold_set = set(gold_ids)
    required = _required_citations(fixture)
    selected_ids = list(routed["selected_ids"])
    selected_set = set(selected_ids)
    citations = list(routed["citations"])
    supported = bool(routed["supported"])
    within_token_budget = int(routed["selected_tokens"]) <= max_tokens
    target_id = str(fixture.get("target_paper_id") or "")
    found_target = bool(target_id and target_id in selected_ids)
    target_rank = selected_ids.index(target_id) + 1 if found_target else None
    selected_gold = selected_set & gold_set
    candidate_count = len(fixture["candidate_papers"])
    collection_iou = _iou(selected_ids, gold_ids)
    if selected_set:
        collection_precision = len(selected_gold) / len(selected_set)
    else:
        collection_precision = 1.0 if not gold_set else 0.0
    collection_recall = 1.0 if not gold_set else len(selected_gold) / len(gold_set)
    no_answer_correct = task_type == "missing_evidence_no_answer" and routed["status"] == "insufficient_evidence" and not citations
    false_stop = not supported and not recovered
    return {
        "task_type": task_type,
        "selected_ids": selected_ids,
        "rejected_ids": list(routed["rejected_ids"]),
        "citations": citations,
        "status": str(routed["status"]),
        "supported": supported,
        "within_token_budget": within_token_budget,
        "supported_at_budget": supported and within_token_budget,
        "found_target": found_target,
        "found": found_target,
        "target_rank": target_rank,
        "collection_iou": collection_iou,
        "iou": collection_iou,
        "collection_precision": collection_precision,
        "collection_recall": collection_recall,
        "no_answer_correct": no_answer_correct,
        "gold_recall_at_20": gold_recall_at_k(selected_ids, gold_ids, 20),
        "gold_recall_at_k": gold_recall_at_k(selected_ids, gold_ids, k),
        "citation_precision": citation_precision(citations, required),
        "citation_recall": citation_recall(citations, required),
        "citation_f1": citation_f1(citations, required),
        "selected_tokens": int(routed["selected_tokens"]),
        "selected_count": len(selected_ids),
        "candidate_count": candidate_count,
        "distractor_count": max(0, candidate_count - len(gold_set)),
        "context_efficiency": context_efficiency(
            selected_tokens=int(routed["selected_tokens"]),
            max_tokens=max_tokens,
            selected_items=len(selected_ids),
            max_items=k,
            answer_supported=supported,
        ),
        "recovered": recovered,
        "false_stop": false_stop,
        "attempt_count": attempt_count,
        "selected_gold_count": len(selected_gold),
    }


def _aggregate_results(results: list[dict], *, strategy_name: str) -> dict:
    deep = _task_results(results, "deep_paper_finding")
    wide = _task_results(results, "wide_literature_collection")
    answerable = [result for result in results if result["task_type"] != "missing_evidence_no_answer"]
    missing = _task_results(results, "missing_evidence_no_answer")
    loop = _task_results(results, "loop_recovery")

    headline = {
        "task_count": len(results),
        "strategy": strategy_name,
        "deep_research_accuracy": _average_bool(deep, "found_target"),
        "wide_research_iou": _average_field(wide, "collection_iou"),
        "supported_answer_rate_at_budget": _average_bool(answerable, "supported_at_budget"),
        "wide_research_precision": _average_field(wide, "collection_precision"),
        "wide_research_recall": _average_field(wide, "collection_recall"),
        "citation_f1": _average_field(results, "citation_f1"),
        "context_efficiency": _average_field(results, "context_efficiency"),
        "recovery_rate": _average_bool(loop, "recovered"),
        "no_answer_accuracy": _average_bool(missing, "no_answer_correct"),
        "gold_recall_at_20": _average_field(results, "gold_recall_at_20"),
        "average_selected_papers": _average_number([result["selected_count"] for result in results]),
        "average_selected_tokens": _average_number([result["selected_tokens"] for result in results]),
        "false_stop_rate": _average_bool(results, "false_stop"),
    }
    headline["paperserve_score"] = (
        0.35 * headline["supported_answer_rate_at_budget"]
        + 0.25 * headline["gold_recall_at_20"]
        + 0.20 * headline["citation_f1"]
        + 0.10 * headline["recovery_rate"]
        + 0.10 * headline["context_efficiency"]
    )
    return headline


def _bm25_scored_candidates(query: str, candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    query_terms = _tokens(query)
    documents = [_tokens(f"{candidate.title or ''} {candidate.text}") for candidate in candidates]
    doc_count = len(documents)
    avgdl = _average_number([len(document) for document in documents]) or 1.0
    dfs = Counter(term for document in documents for term in set(document))
    scored: list[ContextCandidate] = []
    for candidate, document in zip(candidates, documents):
        term_counts = Counter(document)
        score = 0.0
        doc_len = len(document) or 1
        for term in query_terms:
            if not term_counts[term]:
                continue
            idf = math.log(1.0 + (doc_count - dfs[term] + 0.5) / (dfs[term] + 0.5))
            tf = term_counts[term]
            score += idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_len / avgdl)))
        scored.append(
            ContextCandidate(
                id=candidate.id,
                title=candidate.title,
                text=candidate.text,
                source=candidate.source,
                metadata=dict(candidate.metadata),
                score=score,
                token_estimate=candidate.token_estimate,
            )
        )
    return scored


def _first_attempt_candidates(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    if any(_retrieval_attempt(candidate) > 1 for candidate in candidates):
        return [candidate for candidate in candidates if _retrieval_attempt(candidate) <= 1]
    return list(candidates)


def _is_recoverable(fixture: dict, routed: dict) -> bool:
    return str(fixture["task_type"]) in {"loop_recovery", "deep_paper_finding", "wide_literature_collection", "evidence_grounded_answer"} and not routed["supported"]


def _retrieval_attempt(candidate: ContextCandidate) -> int:
    value = candidate.metadata.get("retrieval_attempt", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def _required_citations(fixture: dict) -> list[str]:
    citations = _string_list(fixture.get("required_citations", []))
    return citations if citations else _gold_ids(fixture)


def _gold_ids(fixture: dict) -> list[str]:
    if fixture.get("gold_paper_ids"):
        return _string_list(fixture.get("gold_paper_ids", []))
    if fixture.get("target_paper_id"):
        return [str(fixture["target_paper_id"])]
    return []


def _candidate_tokens(candidate: ContextCandidate) -> int:
    if candidate.token_estimate is not None:
        return candidate.token_estimate
    return len(candidate.text.split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _iou(selected_ids: list[str], gold_ids: list[str]) -> float:
    selected = set(selected_ids)
    gold = set(gold_ids)
    if not selected and not gold:
        return 1.0
    union = selected | gold
    if not union:
        return 0.0
    return len(selected & gold) / len(union)


def _task_results(results: list[dict], task_type: str) -> list[dict]:
    return [result for result in results if result["task_type"] == task_type]


def _average_field(results: list[dict], field: str) -> float:
    return _average_number([float(result[field]) for result in results])


def _average_bool(results: list[dict], field: str) -> float:
    return _average_number([1.0 if result[field] else 0.0 for result in results])


def _average_number(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _resolve_strategies(strategies: Iterable[str] | None) -> tuple[str, ...]:
    if strategies is None:
        return STRATEGIES
    resolved = tuple(str(strategy) for strategy in strategies)
    unknown = [strategy for strategy in resolved if strategy not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategy: {unknown[0]}")
    return resolved


def _resolve_k(fixture: dict, k: int | None) -> int:
    if k is None:
        return _fixture_int(fixture, "max_selected_papers")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return k


def _fixture_int(fixture: dict, key: str) -> int:
    value = fixture[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fixture {fixture.get('id', '<unknown>')} {key} must be an integer")
    return value


def _string_list(values: Iterable[object]) -> list[str]:
    return [str(value) for value in values]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

def _parse_corpus_sizes(value: str) -> tuple[int, ...]:
    message = "must be comma-separated positive integers"
    seen = set()
    parsed = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise argparse.ArgumentTypeError(message)
        try:
            size = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(message) from exc
        if size <= 0:
            raise argparse.ArgumentTypeError(message)
        if size not in seen:
            seen.add(size)
            parsed.append(size)
    return tuple(parsed)


def _validate_fixture(fixture: dict, path: Path) -> None:
    missing = [
        key
        for key in ("id", "task_type", "question", "candidate_papers", "max_selected_papers", "max_reader_tokens")
        if key not in fixture
    ]
    if missing:
        raise ValueError(f"fixture {path} missing required field: {missing[0]}")
    if fixture["task_type"] not in TASK_TYPES:
        raise ValueError(f"fixture {path} has unknown task_type: {fixture['task_type']}")


if __name__ == "__main__":
    raise SystemExit(main())
