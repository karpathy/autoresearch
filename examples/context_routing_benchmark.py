"""Deterministic context-routing benchmark over JSON fixtures.

The runner is intentionally dependency-free. It routes fixture candidates through
``researchfs.context`` and simulates a reader that answers only when every
required citation is present in the selected context frontier.
"""

import argparse
import json
import sys
from pathlib import Path

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
    route_top_k_context,
)

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "context_routing"


def load_fixtures(fixture_dir: str | Path = DEFAULT_FIXTURE_DIR) -> list[dict]:
    """Load context-routing fixtures with papers as ``ContextCandidate`` objects."""

    path = Path(fixture_dir)
    fixtures = []
    for fixture_path in sorted(path.glob("*.json")):
        with fixture_path.open("r", encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        fixture["candidate_papers"] = [
            _candidate_from_paper(candidate) for candidate in fixture.get("candidate_papers", [])
        ]
        fixtures.append(fixture)
    return fixtures


def evaluate_fixture(fixture: dict, *, k: int | None = None) -> dict:
    """Evaluate one fixture and return serializable routing and metric fields."""

    resolved_k = _fixture_int(fixture, "max_selected_papers") if k is None else k
    max_tokens = _fixture_int(fixture, "max_reader_tokens")
    candidates = list(fixture.get("candidate_papers", []))
    first_attempt = _attempt_candidates(candidates, 1)
    later_candidates = [candidate for candidate in candidates if _retrieval_attempt(candidate) > 1]

    if first_attempt and later_candidates:
        first = _route_and_read(fixture, first_attempt, resolved_k, max_tokens)
        if not first["supported"]:
            final = _route_and_read(fixture, candidates, resolved_k, max_tokens)
            recovered = final["supported"]
            attempt_count = 2
        else:
            final = first
            recovered = False
            attempt_count = 1
    else:
        final = _route_and_read(fixture, candidates, resolved_k, max_tokens)
        recovered = False
        attempt_count = 1

    return _fixture_result(fixture, final, resolved_k, max_tokens, recovered, attempt_count)


def run_benchmark(fixture_dir: str | Path = DEFAULT_FIXTURE_DIR, *, k: int | None = None) -> dict:
    """Run every fixture and aggregate metrics by averaging per-fixture values."""

    results = [evaluate_fixture(fixture, k=k) for fixture in load_fixtures(fixture_dir)]
    return {
        "fixtures": results,
        "aggregate": _aggregate_results(results),
    }


def evaluate_fixture_ablation(fixture: dict, *, k: int | None = None) -> dict:
    """Evaluate one fixture under all deterministic context-routing strategies."""

    resolved_k = _fixture_int(fixture, "max_selected_papers") if k is None else k
    max_tokens = _fixture_int(fixture, "max_reader_tokens")
    candidates = list(fixture.get("candidate_papers", []))

    all_ids = [candidate.id for candidate in candidates]
    all_citations = _reader_citations(fixture, all_ids)
    all_context = _fixture_result(
        fixture,
        {
            "selected_ids": all_ids,
            "selected_tokens": sum(_candidate_token_estimate(candidate) for candidate in candidates),
            "rejected_ids": [],
            "status": "answered" if all_citations else "insufficient_evidence",
            "citations": all_citations,
            "supported": bool(all_citations),
        },
        resolved_k,
        max_tokens,
        False,
        1,
    )

    top_k_selection = route_top_k_context(
        candidates,
        k=resolved_k,
        budget=ContextBudget(max_items=resolved_k, max_tokens=None, overflow_policy="truncate"),
    )
    top_k_ids = [candidate.id for candidate in top_k_selection.selected]
    top_k_citations = _reader_citations(fixture, top_k_ids)
    top_k_score = _fixture_result(
        fixture,
        {
            "selected_ids": top_k_ids,
            "selected_tokens": top_k_selection.budget_used_tokens,
            "rejected_ids": [candidate.id for candidate in top_k_selection.rejected],
            "status": "answered" if top_k_citations else "insufficient_evidence",
            "citations": top_k_citations,
            "supported": bool(top_k_citations),
        },
        resolved_k,
        max_tokens,
        False,
        1,
    )

    score_budget_route = _fixture_result(
        fixture,
        _route_and_read(fixture, candidates, resolved_k, max_tokens),
        resolved_k,
        max_tokens,
        False,
        1,
    )
    route_recover = evaluate_fixture(fixture, k=resolved_k)

    return {
        "fixture_id": str(fixture["id"]),
        "strategies": {
            "all_context": all_context,
            "top_k_score": top_k_score,
            "score_budget_route": score_budget_route,
            "route_recover": route_recover,
        },
    }


def run_ablation_benchmark(fixture_dir: str | Path = DEFAULT_FIXTURE_DIR, *, k: int | None = None) -> dict:
    """Run every fixture under each ablation strategy and aggregate metrics."""

    fixtures = [evaluate_fixture_ablation(fixture, k=k) for fixture in load_fixtures(fixture_dir)]
    strategy_names = ["all_context", "top_k_score", "score_budget_route", "route_recover"]
    aggregate = {
        strategy_name: _aggregate_results(
            [fixture["strategies"][strategy_name] for fixture in fixtures]
        )
        for strategy_name in strategy_names
    }
    return {
        "fixtures": fixtures,
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic context-routing benchmark.")
    parser.add_argument("--fixture-dir", default=str(DEFAULT_FIXTURE_DIR), help="Directory of JSON fixtures.")
    parser.add_argument("--k", type=int, default=None, help="Override selected-paper limit for every fixture.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON instead of pretty JSON.")
    parser.add_argument("--ablation", action="store_true", help="Run four-strategy ablation benchmark.")
    args = parser.parse_args(argv)

    if args.ablation:
        result = run_ablation_benchmark(args.fixture_dir, k=args.k)
    else:
        result = run_benchmark(args.fixture_dir, k=args.k)
    if args.compact:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


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


def _route_and_read(fixture: dict, candidates: list[ContextCandidate], k: int, max_tokens: int) -> dict:
    budget = ContextBudget(max_items=k, max_tokens=max_tokens, overflow_policy="drop_lowest_score")
    selection = route_top_k_context(candidates, k=k, budget=budget)
    selected_ids = [candidate.id for candidate in selection.selected]
    rejected_ids = [candidate.id for candidate in selection.rejected]
    citations = _reader_citations(fixture, selected_ids)
    supported = bool(citations)
    return {
        "selected_ids": selected_ids,
        "selected_tokens": selection.budget_used_tokens,
        "rejected_ids": rejected_ids,
        "status": "answered" if supported else "insufficient_evidence",
        "citations": citations,
        "supported": supported,
    }


def _reader_citations(fixture: dict, selected_ids: list[str]) -> list[str]:
    required = _required_citations(fixture)
    selected = set(selected_ids)
    if required and all(citation in selected for citation in required):
        return required
    return []


def _fixture_result(
    fixture: dict,
    routed: dict,
    k: int,
    max_tokens: int,
    recovered: bool,
    attempt_count: int,
) -> dict:
    gold_ids = _string_list(fixture.get("gold_paper_ids", []))
    required = _required_citations(fixture)
    precision = citation_precision(routed["citations"], required)
    recall = citation_recall(routed["citations"], required)
    f1 = citation_f1(routed["citations"], required)
    supported = routed["supported"]
    false_stop = not supported and not recovered
    selected_ids = routed["selected_ids"]
    return {
        "fixture_id": str(fixture["id"]),
        "selected_ids": selected_ids,
        "selected_tokens": routed["selected_tokens"],
        "rejected_ids": routed["rejected_ids"],
        "status": routed["status"],
        "citations": routed["citations"],
        "supported": supported,
        "gold_recall_at_k": gold_recall_at_k(selected_ids, gold_ids, k),
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "context_efficiency": context_efficiency(
            selected_tokens=routed["selected_tokens"],
            max_tokens=max_tokens,
            selected_items=len(selected_ids),
            max_items=k,
            answer_supported=supported,
        ),
        "recovered": recovered,
        "false_stop": false_stop,
        "attempt_count": attempt_count,
    }


def _aggregate_results(results: list[dict]) -> dict:
    return {
        "task_count": len(results),
        "supported_answer_rate": _average_bool(results, "supported"),
        "average_selected_papers": _average_number([len(result["selected_ids"]) for result in results]),
        "average_selected_tokens": _average_number([result["selected_tokens"] for result in results]),
        "gold_recall_at_k": _average_field(results, "gold_recall_at_k"),
        "citation_precision": _average_field(results, "citation_precision"),
        "citation_recall": _average_field(results, "citation_recall"),
        "citation_f1": _average_field(results, "citation_f1"),
        "context_efficiency": _average_field(results, "context_efficiency"),
        "recovery_rate": _average_bool(results, "recovered"),
        "false_stop_rate": _average_bool(results, "false_stop"),
    }


def _attempt_candidates(candidates: list[ContextCandidate], attempt: int) -> list[ContextCandidate]:
    return [candidate for candidate in candidates if _retrieval_attempt(candidate) == attempt]


def _retrieval_attempt(candidate: ContextCandidate) -> int:
    value = candidate.metadata.get("retrieval_attempt", 1)
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return value


def _required_citations(fixture: dict) -> list[str]:
    citations = _string_list(fixture.get("required_citations", []))
    return citations if citations else _string_list(fixture.get("gold_paper_ids", []))


def _string_list(values: list) -> list[str]:
    return [str(value) for value in values]


def _fixture_int(fixture: dict, key: str) -> int:
    value = fixture[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"fixture {fixture.get('id', '<unknown>')} {key} must be an integer")
    return value


def _candidate_token_estimate(candidate: ContextCandidate) -> int:
    if candidate.token_estimate is not None:
        return candidate.token_estimate
    return len(candidate.text.split())


def _average_field(results: list[dict], field: str) -> float:
    return _average_number([result[field] for result in results])


def _average_bool(results: list[dict], field: str) -> float:
    return _average_number([1.0 if result[field] else 0.0 for result in results])


def _average_number(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
