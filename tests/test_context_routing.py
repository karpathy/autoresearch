import csv
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _accepted_kwargs(callable_obj, values):
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(values)
    return {name: value for name, value in values.items() if name in signature.parameters}


def _candidate(candidate_id, *, score=None, tokens=None, text=None):
    from researchfs.context import ContextCandidate

    text = text if text is not None else f"Evidence text for {candidate_id}"
    values = {
        "id": candidate_id,
        "candidate_id": candidate_id,
        "source_id": candidate_id,
        "paper_id": candidate_id,
        "title": f"Paper {candidate_id}",
        "text": text,
        "content": text,
        "score": score,
        "token_estimate": tokens,
        "tokens": tokens,
        "metadata": {"fixture_id": candidate_id},
    }
    kwargs = _accepted_kwargs(ContextCandidate, values)
    try:
        return ContextCandidate(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            "ContextCandidate must be constructible from ordinary routing fields "
            "such as id/candidate_id, text/content, score, and token_estimate"
        ) from exc


def _budget(**overrides):
    from researchfs.context import ContextBudget

    values = {
        "max_items": None,
        "max_tokens": None,
        "overflow_policy": "truncate",
    }
    values.update(overrides)
    kwargs = _accepted_kwargs(ContextBudget, values)
    return ContextBudget(**kwargs)


def _candidate_id(candidate):
    if isinstance(candidate, dict):
        for key in ("id", "candidate_id", "source_id", "paper_id"):
            if key in candidate:
                return candidate[key]
        if "candidate" in candidate:
            return _candidate_id(candidate["candidate"])
    for attr in ("id", "candidate_id", "source_id", "paper_id"):
        if hasattr(candidate, attr):
            value = getattr(candidate, attr)
            if value is not None:
                return value
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("id", "candidate_id", "source_id", "paper_id", "fixture_id"):
            if key in metadata:
                return metadata[key]
    nested = getattr(candidate, "candidate", None)
    if nested is not None:
        return _candidate_id(nested)
    raise AssertionError(f"Could not extract candidate id from {candidate!r}")


def _token_count(candidate):
    if isinstance(candidate, dict):
        for key in ("token_estimate", "tokens"):
            if candidate.get(key) is not None:
                return candidate[key]
        if "candidate" in candidate:
            return _token_count(candidate["candidate"])
    for attr in ("token_estimate", "tokens"):
        if hasattr(candidate, attr):
            value = getattr(candidate, attr)
            if value is not None:
                return value
    text = getattr(candidate, "text", None) or getattr(candidate, "content", "")
    return len(str(text).split())


def _selection_items(selection, names):
    for name in names:
        if isinstance(selection, dict) and name in selection:
            return list(selection[name])
        if hasattr(selection, name):
            return list(getattr(selection, name))
    if isinstance(selection, (list, tuple)):
        return list(selection)
    raise AssertionError(f"Selection object does not expose any of {names}: {selection!r}")


def _selected(selection):
    return _selection_items(
        selection,
        ("selected", "selected_candidates", "candidates", "context", "items"),
    )


def _rejected(selection):
    for names in (("rejected", "rejected_candidates", "rejections"),):
        try:
            return _selection_items(selection, names)
        except AssertionError:
            continue
    return []


def _ids(candidates):
    return [_candidate_id(candidate) for candidate in candidates]


def _route(candidates, **kwargs):
    from researchfs.context import route_top_k_context

    return route_top_k_context(candidates, **kwargs)


def _answer_text(result):
    if isinstance(result, dict):
        for key in ("answer", "message", "status", "reason"):
            if result.get(key) is not None:
                return str(result[key])
    for attr in ("answer", "message", "status", "reason"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value is not None:
                return str(value)
    return str(result)


def _result_supported(result):
    if isinstance(result, dict):
        for key in ("supported", "answer_supported", "success", "verified"):
            if key in result:
                return bool(result[key])
        if "evidence" in result:
            return _result_supported(result["evidence"])
        if "verifier" in result:
            return _result_supported(result["verifier"])
    for attr in ("supported", "answer_supported", "success", "verified"):
        if hasattr(result, attr):
            return bool(getattr(result, attr))
    evidence = getattr(result, "evidence", None)
    if evidence is not None:
        return _result_supported(evidence)
    raise AssertionError(f"Could not determine support status from {result!r}")


def _result_citations(result):
    if isinstance(result, dict):
        for key in ("citations", "supported_citations", "required_citations"):
            if key in result:
                return list(result[key])
        if isinstance(result.get("answer"), dict):
            return _result_citations(result["answer"])
    for attr in ("citations", "supported_citations", "required_citations"):
        if hasattr(result, attr):
            return list(getattr(result, attr))
    answer = getattr(result, "answer", None)
    if answer is not None and answer is not result:
        return _result_citations(answer)
    return []


def _call_runner(module, **kwargs):
    injected_runner = getattr(module, "run_paperserve_workflow", None)
    if callable(injected_runner):
        accepted = _accepted_kwargs(injected_runner, kwargs)
        return injected_runner(**accepted)

    runner = getattr(module, "run_paper_routing_workflow", None)
    assert callable(runner), (
        "examples.context_routing_papers must export an executable workflow runner "
        "named run_paper_routing_workflow or run_paperserve_workflow"
    )
    runner_kwargs = {
        "question": kwargs["query"],
        "k": kwargs.get("k", 2),
    }
    if "max_retries" in kwargs:
        runner_kwargs["max_attempts"] = kwargs["max_retries"] + 1
    return runner(**runner_kwargs)


def _extract_context_from_call(args, kwargs):
    for key in ("context", "selected_context", "selected", "candidates"):
        if key in kwargs:
            return kwargs[key]
    for arg in reversed(args):
        if isinstance(arg, (list, tuple)):
            return arg
    return []

def _field(record, name):
    if isinstance(record, dict) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    raise AssertionError(f"{record!r} does not expose {name!r}")


def _benchmark_fixture_results(report):
    for name in ("fixtures", "fixture_results", "results", "tasks"):
        if isinstance(report, dict) and name in report:
            return list(report[name])
        if hasattr(report, name):
            return list(getattr(report, name))
    raise AssertionError(f"Benchmark report does not expose per-fixture results: {report!r}")


def _benchmark_aggregate(report):
    for name in ("aggregate", "metrics", "summary"):
        if isinstance(report, dict) and name in report:
            return report[name]
        if hasattr(report, name):
            return getattr(report, name)
    if isinstance(report, dict) and "task_count" in report:
        return report
    raise AssertionError(f"Benchmark report does not expose aggregate metrics: {report!r}")


def _benchmark_results_by_id(report):
    results = _benchmark_fixture_results(report)
    return {_field(result, "fixture_id"): result for result in results}


PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS = {
    "strategy",
    "corpus_size",
    "supported_answer_rate_at_budget",
    "paperserve_score",
    "context_efficiency",
    "average_selected_tokens",
    "average_selected_papers",
    "false_stop_rate",
    "recovery_rate",
    "wide_research_precision",
    "wide_research_recall",
    "gold_recall_at_20",
    "citation_f1",
}


def _paperserve_fixture(benchmark, fixture_id):
    return next(
        fixture
        for fixture in benchmark.load_fixtures(REPO_ROOT / "tests" / "fixtures" / "paperserve_bench")
        if fixture["id"] == fixture_id
    )


def _synthetic_distractors(fixture):
    return [
        candidate
        for candidate in fixture["candidate_papers"]
        if candidate.metadata.get("synthetic_distractor")
    ]


def _threshold_report_passed(threshold_report):
    if isinstance(threshold_report, dict):
        for key in ("passed", "ok", "success"):
            if key in threshold_report:
                return bool(threshold_report[key])
        if "failures" in threshold_report:
            return not threshold_report["failures"]
    raise AssertionError(f"Threshold report does not expose pass/fail status: {threshold_report!r}")


def _fake_paper_memory_card(paper_id=123):
    context = (
        "# Paper Memory: Retrieval-Augmented Compression\n\n"
        "The paper stores compact method evidence for routing."
    )
    extraction_status = {
        "status": "extracted",
        "confidence_score": 0.875,
        "source_char_count": 1200,
        "compressed_char_count": 180,
    }
    return {
        "paper_id": paper_id,
        "context": context,
        "method_ids": [7, 11],
        "chunk_ids": [101, 102],
        "lesson_ids": [5],
        "experiment_ids": [3],
        "confidence_score": 0.875,
        "source_char_count": 1200,
        "compressed_char_count": 180,
        "extraction_status": extraction_status,
    }


def test_paper_memory_to_context_candidate_maps_researchfs_card_fields():
    from researchfs.context import paper_memory_to_context_candidate

    card = _fake_paper_memory_card()

    candidate = paper_memory_to_context_candidate(card)

    assert candidate.id == "123"
    assert candidate.title == "Retrieval-Augmented Compression"
    assert candidate.text == card["context"]
    assert candidate.source == "researchfs://paper/123"
    assert candidate.score == pytest.approx(card["confidence_score"])
    assert candidate.token_estimate == len(card["context"].split())
    assert candidate.metadata["paper_id"] == 123
    assert candidate.metadata["method_ids"] == [7, 11]
    assert candidate.metadata["chunk_ids"] == [101, 102]
    assert candidate.metadata["lesson_ids"] == [5]
    assert candidate.metadata["experiment_ids"] == [3]
    assert candidate.metadata["confidence_score"] == pytest.approx(0.875)
    assert candidate.metadata["source_char_count"] == 1200
    assert candidate.metadata["compressed_char_count"] == 180
    assert candidate.metadata["extraction_status"] == card["extraction_status"]


def test_research_brief_to_context_candidates_uses_brief_paper_ids(monkeypatch):
    import researchfs.retrieve as retrieve
    import researchfs.context.researchfs as adapter
    from researchfs.context import research_brief_to_context_candidates

    calls = []
    cards = {
        41: _fake_paper_memory_card(41),
        42: _fake_paper_memory_card(42),
    }
    conn = object()

    def fake_get_paper_memory(paper_id, *, conn=None, max_chars=None):
        calls.append((paper_id, conn, max_chars))
        return cards[int(paper_id)]

    monkeypatch.setattr(retrieve, "get_paper_memory", fake_get_paper_memory)
    monkeypatch.setattr(adapter, "get_paper_memory", fake_get_paper_memory, raising=False)

    candidates = research_brief_to_context_candidates(
        {"paper_ids": [41, 42]},
        conn=conn,
        per_paper_max_chars=500,
        scores_by_paper_id={41: 0.91, "42": 0.82},
    )

    assert calls == [(41, conn, 500), (42, conn, 500)]
    assert [candidate.id for candidate in candidates] == ["41", "42"]
    assert [candidate.source for candidate in candidates] == [
        "researchfs://paper/41",
        "researchfs://paper/42",
    ]
    assert [candidate.score for candidate in candidates] == [pytest.approx(0.91), pytest.approx(0.82)]


def test_selection_audit_payload_reports_real_selection_ids_and_token_counts():
    from researchfs.context import ContextBudget, route_top_k_context, selection_audit_payload

    candidates = [
        _candidate("selected", score=0.9, tokens=4),
        _candidate("rejected", score=0.8, tokens=5),
    ]
    selection = route_top_k_context(
        candidates,
        k=2,
        budget=ContextBudget(max_tokens=4, overflow_policy="truncate"),
    )

    payload = selection_audit_payload(selection, query="How?", reason="unit test")

    assert payload["selected_ids"] == ["selected"]
    assert payload["rejected_ids"] == ["rejected"]
    assert payload["token_budget"] == 4
    assert payload["selected_tokens"] == 4
    assert payload["selected_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["query"] == "How?"
    assert payload["reason"] == "unit test"


def test_log_context_routing_audit_writes_retrieval_event_and_links_methods():
    from researchfs.context import ContextCandidate, ContextSelection, log_context_routing_audit
    from researchfs.db import get_connection, initialize_db

    conn = get_connection(":memory:")
    initialize_db(conn)
    method_ids = []
    for name in ("adapter-method-a", "adapter-method-b"):
        cursor = conn.execute("INSERT INTO methods (name) VALUES (?)", (name,))
        method_ids.append(cursor.lastrowid)
    conn.commit()

    selected = ContextCandidate(
        id="paper-1",
        title="Paper 1",
        text="selected evidence",
        metadata={"method_ids": method_ids},
        token_estimate=2,
    )
    rejected = ContextCandidate(
        id="paper-2",
        title="Paper 2",
        text="rejected evidence",
        metadata={"method_ids": [method_ids[1]]},
        token_estimate=3,
    )
    selection = ContextSelection(
        selected=[selected],
        rejected=[rejected],
        budget_used_tokens=2,
        budget_max_tokens=5,
        reason="max_tokens",
    )

    event_id = log_context_routing_audit("route this", selection, conn=conn)

    event = conn.execute(
        """
        SELECT event_type, query, retrieved_content
        FROM retrieval_events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    assert event["event_type"] == "context_routing_audit"
    assert event["query"] == "route this"
    payload = json.loads(event["retrieved_content"])
    assert payload["selected_ids"] == ["paper-1"]
    assert payload["rejected_ids"] == ["paper-2"]
    assert payload["selected_tokens"] == 2
    assert payload["token_budget"] == 5

    linked_method_ids = [
        row["method_id"]
        for row in conn.execute(
            "SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id",
            (event_id,),
        )
    ]
    assert linked_method_ids == method_ids


def test_researchfs_top_level_context_exports_are_lazy_and_public():
    import researchfs
    from researchfs import (
        ContextBudget,
        ContextCandidate,
        log_context_routing_audit,
        paper_memory_to_context_candidate,
        route_top_k_context,
    )

    expected = {
        "ContextBudget",
        "ContextCandidate",
        "log_context_routing_audit",
        "paper_memory_to_context_candidate",
        "route_top_k_context",
    }

    assert expected <= set(researchfs.__all__)
    assert ContextCandidate is researchfs.context.ContextCandidate
    assert ContextBudget is researchfs.context.ContextBudget
    assert route_top_k_context is researchfs.context.route_top_k_context
    assert paper_memory_to_context_candidate is researchfs.context.paper_memory_to_context_candidate
    assert log_context_routing_audit is researchfs.context.log_context_routing_audit


def test_top_k_uses_descending_scores_and_stable_ties_then_unsorted_fallback_keeps_input_order():
    scored = [
        _candidate("low", score=0.10, tokens=1),
        _candidate("tie-first", score=0.90, tokens=1),
        _candidate("tie-second", score=0.90, tokens=1),
        _candidate("middle", score=0.50, tokens=1),
    ]

    selection = _route(scored, k=3)

    assert _ids(_selected(selection)) == ["tie-first", "tie-second", "middle"]

    unscored = [
        _candidate("first", score=None, tokens=1),
        _candidate("second", score=None, tokens=1),
        _candidate("third", score=None, tokens=1),
    ]

    selection = _route(unscored, k=2)

    assert _ids(_selected(selection)) == ["first", "second"]


def test_token_budget_is_respected_and_overflow_candidates_are_tracked():
    candidates = [
        _candidate("best", score=0.90, tokens=5),
        _candidate("too-expensive-after-best", score=0.80, tokens=4),
        _candidate("small", score=0.70, tokens=2),
    ]
    budget = _budget(max_items=3, max_tokens=7, overflow_policy="truncate")

    selection = _route(candidates, k=3, budget=budget)
    selected = _selected(selection)
    rejected = _rejected(selection)

    assert sum(_token_count(candidate) for candidate in selected) <= 7
    assert "best" in _ids(selected)
    assert rejected, "router must expose rejected candidates when budget pressure drops context"
    assert set(_ids(selected)).isdisjoint(_ids(rejected))
    assert set(_ids(selected) + _ids(rejected)) == {"best", "too-expensive-after-best", "small"}


def test_min_score_rejects_hard_negatives_without_displacing_positive_context():
    candidates = [
        _candidate("strong", score=0.95, tokens=1),
        _candidate("weak-positive", score=0.05, tokens=1),
        _candidate("hard-negative", score=-0.75, tokens=1),
    ]

    selection = _route(candidates, k=3, min_score=0.0)

    assert _ids(_selected(selection)) == ["strong", "weak-positive"]
    assert "hard-negative" in _ids(_rejected(selection))


def test_raise_overflow_policy_raises_value_error_with_budget_wording():
    candidates = [
        _candidate("a", score=0.90, tokens=6),
        _candidate("b", score=0.80, tokens=6),
    ]
    budget = _budget(max_items=2, max_tokens=10, overflow_policy="raise")

    with pytest.raises(ValueError, match=r"(?i)(budget|max_tokens|token)"):
        _route(candidates, k=2, budget=budget)


def test_drop_lowest_score_overflow_policy_drops_deterministically_until_budget_fits():
    candidates = [
        _candidate("highest", score=0.95, tokens=5),
        _candidate("middle", score=0.70, tokens=3),
        _candidate("lowest", score=0.60, tokens=3),
    ]
    budget = _budget(max_items=3, max_tokens=8, overflow_policy="drop_lowest_score")

    selection = _route(candidates, k=3, budget=budget)

    assert _ids(_selected(selection)) == ["highest", "middle"]
    assert "lowest" in _ids(_rejected(selection))
    assert sum(_token_count(candidate) for candidate in _selected(selection)) == 8


def test_example_reader_receives_only_selected_context_and_verifier_accepts_required_citation(monkeypatch):
    import examples.context_routing_papers as workflow

    gold = _candidate("gold-paper", score=0.99, tokens=4, text="Gold paper proves the claim.")
    distractor = _candidate("distractor-paper", score=0.10, tokens=4, text="Distractor is irrelevant.")
    hard_negative = _candidate("hard-negative", score=-1.0, tokens=4, text="Rejected by min_score.")
    reader_context_ids = []

    def retriever(*_args, **_kwargs):
        return [gold, distractor, hard_negative]

    def reader(*args, **kwargs):
        context = list(_extract_context_from_call(args, kwargs))
        reader_context_ids.append(_ids(context))
        assert _ids(context) == ["gold-paper"]
        return {"status": "answered", "answer": "The claim is supported by the gold paper.", "citations": ["gold-paper"]}

    def verifier(*args, **kwargs):
        context_ids = _ids(_extract_context_from_call(args, kwargs))
        citations = []
        for arg in args:
            if isinstance(arg, dict) and "citations" in arg:
                citations = list(arg["citations"])
        citations = list(kwargs.get("citations", citations))
        supported = context_ids == ["gold-paper"] and citations == ["gold-paper"]
        return {"supported": supported, "recoverable": not supported, "citations": citations, "required_citations": ["gold-paper"]}

    monkeypatch.setattr(workflow, "retrieve_candidates", retriever)
    monkeypatch.setattr(workflow, "answer_from_selected_context", reader)
    monkeypatch.setattr(workflow, "verify_answer", verifier)

    result = _call_runner(
        workflow,
        query="Which paper proves the claim?",
        retriever=retriever,
        reader=reader,
        verifier=verifier,
        k=1,
        min_score=0.0,
        budget=_budget(max_items=1, max_tokens=8, overflow_policy="truncate"),
    )

    assert reader_context_ids == [["gold-paper"]]
    assert _result_supported(result) is True
    assert _result_citations(result) == ["gold-paper"]


def test_example_missing_gold_returns_insufficient_evidence_instead_of_hallucinated_answer(monkeypatch):
    import examples.context_routing_papers as workflow

    distractor = _candidate("distractor-paper", score=0.91, tokens=4, text="Looks related but lacks the gold fact.")

    def retriever(*_args, **_kwargs):
        return [distractor]

    def reader(*args, **kwargs):
        assert _ids(_extract_context_from_call(args, kwargs)) == ["distractor-paper"]
        return {"status": "answered", "answer": "Hallucinated unsupported answer", "citations": []}

    def verifier(*_args, **_kwargs):
        return {"supported": False, "recoverable": False, "citations": [], "required_citations": ["gold-paper"]}

    monkeypatch.setattr(workflow, "retrieve_candidates", retriever)
    monkeypatch.setattr(workflow, "answer_from_selected_context", reader)
    monkeypatch.setattr(workflow, "verify_answer", verifier)

    result = _call_runner(
        workflow,
        query="Which paper proves the claim?",
        retriever=retriever,
        reader=reader,
        verifier=verifier,
        k=1,
        min_score=0.0,
        budget=_budget(max_items=1, max_tokens=8, overflow_policy="truncate"),
    )

    assert _result_supported(result) is False
    answer_text = _answer_text(result).lower()
    assert "hallucinated unsupported answer" not in answer_text
    assert "insufficient" in answer_text or "not enough" in answer_text or "unsupported" in answer_text


def test_example_verifier_retry_recovers_after_first_retrieval_misses_gold(monkeypatch):
    import examples.context_routing_papers as workflow

    gold = _candidate("gold-paper", score=0.99, tokens=4, text="Gold paper proves the claim.")
    distractor = _candidate("distractor-paper", score=0.88, tokens=4, text="Distractor misses the required citation.")
    retrieval_calls = []
    reader_context_ids = []

    def retriever(*_args, **_kwargs):
        retrieval_calls.append(len(retrieval_calls) + 1)
        if len(retrieval_calls) == 1:
            return [distractor]
        return [gold, distractor]

    def reader(*args, **kwargs):
        context_ids = _ids(_extract_context_from_call(args, kwargs))
        reader_context_ids.append(context_ids)
        if "gold-paper" in context_ids:
            return {"status": "answered", "answer": "Recovered supported answer", "citations": ["gold-paper"]}
        return {"status": "answered", "answer": "First attempt lacks support", "citations": []}

    def verifier(*args, **kwargs):
        context_ids = _ids(_extract_context_from_call(args, kwargs))
        citations = []
        for arg in args:
            if isinstance(arg, dict) and "citations" in arg:
                citations = list(arg["citations"])
        citations = list(kwargs.get("citations", citations))
        supported = "gold-paper" in context_ids and "gold-paper" in citations
        return {"supported": supported, "recoverable": not supported, "citations": citations, "required_citations": ["gold-paper"]}

    monkeypatch.setattr(workflow, "retrieve_candidates", retriever)
    monkeypatch.setattr(workflow, "answer_from_selected_context", reader)
    monkeypatch.setattr(workflow, "verify_answer", verifier)

    result = _call_runner(
        workflow,
        query="Which paper proves the claim?",
        retriever=retriever,
        reader=reader,
        verifier=verifier,
        k=1,
        min_score=0.0,
        budget=_budget(max_items=1, max_tokens=8, overflow_policy="truncate"),
        max_retries=1,
    )

    assert retrieval_calls == [1, 2]
    assert reader_context_ids[0] == ["distractor-paper"]
    assert reader_context_ids[-1] == ["gold-paper"]
    assert _result_supported(result) is True
    assert _result_citations(result) == ["gold-paper"]



def test_paperserve_env_reward_components_cover_requested_step_contracts():
    import examples.paperserve_env as paperserve_env

    requested_components = {
        "correct_supported_answer",
        "gold_paper_selected",
        "valid_citation",
        "wrong_answer",
        "unsupported_citation",
        "too_many_selected_tokens",
        "unnecessary_retrieval_calls",
        "stopping_without_enough_evidence",
        "recovery_success",
    }
    assert requested_components <= set(paperserve_env.REWARD_COMPONENT_NAMES)

    gold_candidate = next(candidate for candidate in paperserve_env.DEFAULT_CANDIDATES if candidate.is_gold)
    gold_id = gold_candidate.id

    env = paperserve_env.make_env("Which paper explains PaperServe context frontiers?")
    _, _, _, select_info = env.step(paperserve_env.ACTION_SELECT_CANDIDATE, gold_id)
    _, _, _, reader_info = env.step(paperserve_env.ACTION_ASK_READER_NOW)

    over_budget_env = paperserve_env.make_env(
        "Which paper explains PaperServe context frontiers?",
        reader_token_budget=gold_candidate.token_estimate - 1,
    )
    over_budget_observation, _, over_budget_done, over_budget_info = over_budget_env.step(
        paperserve_env.ACTION_SELECT_CANDIDATE,
        gold_id,
    )

    no_evidence_env = paperserve_env.make_env("Which paper explains PaperServe context frontiers?")
    _, _, no_answer_done, no_answer_info = no_evidence_env.step(paperserve_env.ACTION_STOP_WITH_ANSWER)

    unsupported_answer_env = paperserve_env.make_env("Which paper explains PaperServe context frontiers?")
    _, _, unsupported_done, unsupported_info = unsupported_answer_env.step(
        paperserve_env.ACTION_STOP_WITH_ANSWER,
        answer="A non-selected paper supposedly supports the answer.",
    )

    retrieve_env = paperserve_env.make_env("Which paper explains PaperServe context frontiers?")
    _, _, _, retrieve_info = retrieve_env.step(paperserve_env.ACTION_RETRIEVE_MORE)

    for info in (select_info, reader_info, over_budget_info, no_answer_info, unsupported_info, retrieve_info):
        assert requested_components <= set(info["reward_component_names"])
        assert requested_components <= set(info["reward_components"])

    assert select_info["reward_components"]["gold_paper_selected"] == pytest.approx(1.0)
    assert over_budget_info["reward_components"]["too_many_selected_tokens"] == pytest.approx(-1.0)
    assert over_budget_done is False
    assert gold_id not in over_budget_observation["selected_ids"]

    reader_components = reader_info["reward_components"]
    assert reader_components["correct_supported_answer"] == pytest.approx(1.0)
    assert reader_components["valid_citation"] == pytest.approx(1.0)

    assert no_answer_done is True
    no_answer_components = no_answer_info["reward_components"]
    assert no_answer_components["stopping_without_enough_evidence"] == pytest.approx(-1.0)

    assert unsupported_done is True
    unsupported_components = unsupported_info["reward_components"]
    assert unsupported_components["stopping_without_enough_evidence"] == pytest.approx(-1.0)
    assert unsupported_components["wrong_answer"] == pytest.approx(-1.0)
    assert unsupported_components["unsupported_citation"] == pytest.approx(-1.0)

    assert retrieve_info["reward_components"]["unnecessary_retrieval_calls"] == pytest.approx(-0.25)

def test_context_routing_fixtures_have_required_schema_keys():
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "context_routing"
    files = sorted(fixture_dir.glob("*.json"))

    assert len(files) == 5, "context routing fixture suite must contain exactly five JSON fixtures"

    required_top_level = {
        "id",
        "question",
        "gold_paper_ids",
        "candidate_papers",
        "hard_negative_ids",
        "expected_answer_points",
        "required_citations",
        "max_selected_papers",
        "max_reader_tokens",
    }
    required_candidate_keys = {
        "id",
        "title",
        "text",
        "source",
        "score",
        "token_estimate",
        "metadata",
    }

    for path in files:
        payload = json.loads(path.read_text())
        assert isinstance(payload, dict), f"{path.name} must be a JSON object"
        assert required_top_level <= set(payload), f"{path.name} is missing required fixture keys"
        assert isinstance(payload["id"], str) and payload["id"]
        assert isinstance(payload["question"], str) and payload["question"]
        assert payload["gold_paper_ids"], f"{path.name} must name at least one gold paper"
        assert payload["candidate_papers"], f"{path.name} must include candidate papers"
        assert payload["required_citations"], f"{path.name} must include required citations"
        assert set(payload["required_citations"]) <= set(payload["gold_paper_ids"])
        assert isinstance(payload["max_selected_papers"], int) and payload["max_selected_papers"] > 0
        assert isinstance(payload["max_reader_tokens"], int) and payload["max_reader_tokens"] > 0
        assert isinstance(payload["hard_negative_ids"], list)

        candidate_ids = set()
        for candidate in payload["candidate_papers"]:
            assert required_candidate_keys <= set(candidate), f"{path.name} has an incomplete candidate"
            assert isinstance(candidate["id"], str) and candidate["id"]
            assert isinstance(candidate["text"], str) and candidate["text"].strip()
            assert isinstance(candidate["score"], (int, float))
            assert isinstance(candidate["token_estimate"], int) and candidate["token_estimate"] >= 0
            assert isinstance(candidate["metadata"], dict)
            candidate_ids.add(candidate["id"])

        assert set(payload["gold_paper_ids"]) & candidate_ids or "missing" in payload["id"]


def test_context_routing_benchmark_loads_default_fixtures_and_reports_aggregate_metrics():
    import examples.context_routing_benchmark as benchmark

    report = benchmark.run_benchmark()
    results = _benchmark_fixture_results(report)
    aggregate = _benchmark_aggregate(report)

    assert len(results) == 5
    assert _field(aggregate, "task_count") == 5
    assert set(_benchmark_results_by_id(report)) == {
        "single_gold_easy",
        "single_gold_hard_negatives",
        "multi_paper_synthesis",
        "missing_gold_no_answer",
        "loop_recovery_initial_retrieval_miss",
    }

    expected_metrics = {
        "task_count",
        "supported_answer_rate",
        "average_selected_papers",
        "average_selected_tokens",
        "gold_recall_at_k",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "context_efficiency",
        "recovery_rate",
        "false_stop_rate",
    }
    assert expected_metrics <= set(aggregate)


def test_context_routing_benchmark_missing_gold_returns_insufficient_evidence():
    import examples.context_routing_benchmark as benchmark

    result = _benchmark_results_by_id(benchmark.run_benchmark())["missing_gold_no_answer"]

    assert _field(result, "status") == "insufficient_evidence"
    assert _field(result, "supported") is False
    assert _field(result, "citations") == []


def test_context_routing_benchmark_loop_recovery_recovers_supported_answer():
    import examples.context_routing_benchmark as benchmark

    result = _benchmark_results_by_id(benchmark.run_benchmark())["loop_recovery_initial_retrieval_miss"]

    assert _field(result, "recovered") is True
    assert _field(result, "supported") is True


def test_context_routing_benchmark_cli_emits_parseable_json():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.context_routing_benchmark", "--compact"],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert _field(_benchmark_aggregate(report), "task_count") == 5
    assert len(_benchmark_fixture_results(report)) == 5



def test_paperserve_bench_fixtures_cover_contract_task_types_and_harder_cases():
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "paperserve_bench"
    files = sorted(fixture_dir.glob("*.json"))
    assert files, "PaperServe-Bench fixtures must live in tests/fixtures/paperserve_bench"

    required_task_types = {
        "deep_paper_finding",
        "wide_literature_collection",
        "evidence_grounded_answer",
        "missing_evidence_no_answer",
        "loop_recovery",
    }
    required_harder_fixture_ids = {
        "answer_near_miss_partial_evidence",
        "wide_conflicting_gold_like_distractors",
        "multi_hop_evidence_pressure",
        "deep_near_duplicate_pressure",
        "wide_conflicting_citation_pressure",
        "loop_recovery_evidence_pressure",
    }
    seen_task_types = set()
    seen_fixture_ids = set()
    required_fixture_keys = {
        "id",
        "task_type",
        "question",
        "candidate_papers",
        "required_citations",
        "expected_answer_points",
        "max_selected_papers",
        "max_reader_tokens",
    }
    required_candidate_keys = {"id", "title", "text", "score", "token_estimate", "metadata"}

    for path in files:
        payload = json.loads(path.read_text())
        assert required_fixture_keys <= set(payload), f"{path.name} missing PaperServe-Bench fixture keys"
        assert (
            "gold_paper_ids" in payload or "target_paper_id" in payload
        ), f"{path.name} must define gold_paper_ids or target_paper_id"
        assert payload["task_type"] in required_task_types, f"{path.name} has unknown task type"
        seen_task_types.add(payload["task_type"])
        seen_fixture_ids.add(payload["id"])
        assert isinstance(payload["id"], str) and payload["id"].strip()
        assert isinstance(payload["question"], str) and payload["question"].strip()
        assert isinstance(payload["candidate_papers"], list)
        assert isinstance(payload["required_citations"], list)
        assert isinstance(payload["expected_answer_points"], list)
        assert isinstance(payload["max_selected_papers"], int) and payload["max_selected_papers"] > 0
        assert isinstance(payload["max_reader_tokens"], int) and payload["max_reader_tokens"] > 0

        candidate_ids = set()
        for candidate in payload["candidate_papers"]:
            assert required_candidate_keys <= set(candidate), f"{path.name} has incomplete candidate"
            assert isinstance(candidate["id"], str) and candidate["id"].strip()
            assert isinstance(candidate["title"], str) and candidate["title"].strip()
            assert isinstance(candidate["text"], str) and candidate["text"].strip()
            assert isinstance(candidate["score"], (int, float))
            assert isinstance(candidate["token_estimate"], int) and candidate["token_estimate"] >= 0
            assert isinstance(candidate["metadata"], dict)
            candidate_ids.add(candidate["id"])

        if payload["task_type"] == "missing_evidence_no_answer":
            assert not set(payload.get("gold_paper_ids", [])) & candidate_ids
        elif "target_paper_id" in payload:
            assert payload["target_paper_id"] in candidate_ids
        else:
            assert set(payload["gold_paper_ids"]) & candidate_ids

        if payload["task_type"] == "loop_recovery":
            attempts = payload.get("retrieval_attempts") or payload.get("retrieval_attempt") or []
            assert attempts, "loop_recovery fixture must include retrieval attempt metadata"

    assert required_task_types <= seen_task_types
    assert required_harder_fixture_ids <= seen_fixture_ids


def test_paperserve_bench_harder_fixture_cases_load_with_default_corpus_size():
    import examples.paperserve_benchmark as benchmark

    fixtures = {
        fixture["id"]: fixture
        for fixture in benchmark.load_fixtures(REPO_ROOT / "tests" / "fixtures" / "paperserve_bench")
    }
    harder_fixture_ids = {
        "answer_near_miss_partial_evidence",
        "wide_conflicting_gold_like_distractors",
        "multi_hop_evidence_pressure",
        "deep_near_duplicate_pressure",
        "wide_conflicting_citation_pressure",
        "loop_recovery_evidence_pressure",
    }

    assert harder_fixture_ids <= set(fixtures)
    for fixture_id in harder_fixture_ids:
        assert len(fixtures[fixture_id]["candidate_papers"]) <= 5


def test_paperserve_bench_evidence_pressure_fixture_ids_map_to_required_cases():
    import examples.paperserve_benchmark as benchmark

    fixtures = {
        fixture["id"]: fixture
        for fixture in benchmark.load_fixtures(REPO_ROOT / "tests" / "fixtures" / "paperserve_bench")
    }

    required_cases = {
        "multi_hop_evidence_pressure": {
            "task_type": "evidence_grounded_answer",
            "minimum_gold_ids": 2,
            "pressure_terms": ("prism latch", "ion balance"),
        },
        "deep_near_duplicate_pressure": {
            "task_type": "deep_paper_finding",
            "target_required": True,
            "pressure_terms": ("near-duplicate", "microburst"),
            "distractor_types": {"closer_gold_like", "conflicting_evidence", "near_miss_citation"},
        },
        "wide_conflicting_citation_pressure": {
            "task_type": "wide_literature_collection",
            "minimum_gold_ids": 2,
            "pressure_terms": ("conflicting citation", "timing drift"),
            "distractor_types": {"conflicting_evidence", "near_miss_citation"},
        },
        "loop_recovery_evidence_pressure": {
            "task_type": "loop_recovery",
            "requires_retrieval_attempts": True,
            "pressure_terms": ("recovery", "second retrieval"),
        },
    }

    assert set(required_cases) <= fixtures.keys()
    for fixture_id, expected in required_cases.items():
        fixture = fixtures[fixture_id]
        assert fixture["task_type"] == expected["task_type"]
        assert len(fixture["candidate_papers"]) <= 5

        gold_ids = set(fixture.get("gold_paper_ids", []))
        candidate_ids = {candidate.id for candidate in fixture["candidate_papers"]}
        if expected.get("minimum_gold_ids"):
            assert len(gold_ids) >= expected["minimum_gold_ids"], fixture_id
            assert gold_ids <= candidate_ids, fixture_id
        if expected.get("target_required"):
            assert fixture["target_paper_id"] in candidate_ids
            assert len([candidate for candidate in fixture["candidate_papers"] if candidate.id != fixture["target_paper_id"]]) >= 2
        if expected.get("requires_retrieval_attempts"):
            attempts = fixture.get("retrieval_attempts") or fixture.get("retrieval_attempt") or []
            assert attempts, fixture_id

        joined_text = " ".join(
            [fixture["question"].lower()]
            + [point.lower() for point in fixture["expected_answer_points"]]
            + [candidate.text.lower() for candidate in fixture["candidate_papers"]]
        )
        for term in expected["pressure_terms"]:
            assert term in joined_text, fixture_id
        if expected.get("distractor_types"):
            distractor_types = {
                candidate.metadata.get("distractor_type")
                for candidate in fixture["candidate_papers"]
                if candidate.metadata.get("distractor_type")
            }
            assert expected["distractor_types"] <= distractor_types

def test_paperserve_bench_run_benchmark_reports_strategies_and_headline_metrics():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_benchmark()
    results = _benchmark_fixture_results(report)
    aggregate = _benchmark_aggregate(report)

    required_strategies = {
        "bm25_top20",
        "long_context_all_docs",
        "context_frontier_router",
        "router_verifier_loop",
    }
    headline_metrics = {
        "deep_research_accuracy",
        "wide_research_iou",
        "wide_research_precision",
        "wide_research_recall",
        "supported_answer_rate_at_budget",
        "citation_f1",
        "context_efficiency",
        "recovery_rate",
        "no_answer_accuracy",
        "gold_recall_at_20",
        "paperserve_score",
    }

    assert len(results) >= 5
    result_task_types = {_field(result, "task_type") for result in results}
    result_fixture_ids = {_field(result, "fixture_id") for result in results}
    assert {
        "deep_paper_finding",
        "wide_literature_collection",
        "evidence_grounded_answer",
        "missing_evidence_no_answer",
        "loop_recovery",
    } <= result_task_types
    assert {
        "answer_near_miss_partial_evidence",
        "wide_conflicting_gold_like_distractors",
    } <= result_fixture_ids
    assert required_strategies <= set(aggregate)
    for strategy_name in required_strategies:
        assert headline_metrics <= set(aggregate[strategy_name]), strategy_name
        expected_score = (
            0.35 * aggregate[strategy_name]["supported_answer_rate_at_budget"]
            + 0.25 * aggregate[strategy_name]["gold_recall_at_20"]
            + 0.20 * aggregate[strategy_name]["citation_f1"]
            + 0.10 * aggregate[strategy_name]["recovery_rate"]
            + 0.10 * aggregate[strategy_name]["context_efficiency"]
        )
        assert aggregate[strategy_name]["paperserve_score"] == pytest.approx(expected_score)

    for result in results:
        strategies = _field(result, "strategies")
        assert required_strategies <= set(strategies)
        assert _field(result, "task_type") in {
            "deep_paper_finding",
            "wide_literature_collection",
            "evidence_grounded_answer",
            "missing_evidence_no_answer",
            "loop_recovery",
        }
        for strategy_name in required_strategies:
            strategy_result = strategies[strategy_name]
            for key in (
                "selected_ids",
                "citations",
                "supported",
                "selected_tokens",
                "context_efficiency",
                "citation_f1",
                "within_token_budget",
                "supported_at_budget",
                "target_rank",
                "collection_precision",
                "collection_recall",
                "recovered",
                "false_stop",
                "attempt_count",
                "candidate_count",
                "distractor_count",
            ):
                assert key in strategy_result, f"{strategy_name} missing {key}"


def test_paperserve_bench_supported_answer_rate_respects_token_budget():
    import examples.paperserve_benchmark as benchmark

    fixtures = benchmark.load_fixtures(REPO_ROOT / "tests" / "fixtures" / "paperserve_bench")
    fixture = next(fixture for fixture in fixtures if fixture["id"] == "deep_hidden_target")
    expanded = benchmark.expand_fixture_corpus(fixture, 20)

    result = benchmark.evaluate_fixture(expanded, strategies=["long_context_all_docs"])
    strategy_result = result["strategies"]["long_context_all_docs"]

    assert strategy_result["supported"] is True
    assert strategy_result["within_token_budget"] is False
    assert strategy_result["supported_at_budget"] is False


def test_paperserve_bench_scaling_reports_chart_ready_series():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_scaling_benchmark(corpus_sizes=(5, 20), strategy="context_frontier_router")

    assert report["mode"] == "scaling"
    assert report["corpus_sizes"] == [5, 20]
    assert len(report["scaling"]) == 2

    chart = report["chart"]
    assert chart["x_axis"] == "corpus_size"
    assert chart["y_axis"] == "supported_answer_rate_at_budget"
    assert set(chart["series"]) == {"context_frontier_router"}

    series_points = chart["series"]["context_frontier_router"]
    assert [point["corpus_size"] for point in series_points] == [5, 20]
    for point in series_points:
        assert PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS - {"strategy"} <= set(point)


def test_paperserve_bench_scaling_preserves_router_and_verifier_support_as_distractors_grow():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_scaling_benchmark(
        corpus_sizes=(5, 20, 50, 100),
        strategy=None,
    )

    for strategy_name in ("context_frontier_router", "router_verifier_loop"):
        support_rates = [
            point["supported_answer_rate_at_budget"]
            for point in report["chart"]["series"][strategy_name]
        ]
        assert [point["corpus_size"] for point in report["chart"]["series"][strategy_name]] == [
            5,
            20,
            50,
            100,
        ]
        assert all(rate > 0.0 for rate in support_rates), strategy_name


def test_paperserve_bench_router_strategies_keep_support_for_every_distractor_mode_at_100():
    import examples.paperserve_benchmark as benchmark

    for distractor_mode in benchmark.DISTRACTOR_MODES:
        report = benchmark.run_scaling_benchmark(
            corpus_sizes=(100,),
            strategy=None,
            distractor_mode=distractor_mode,
        )

        assert report["distractor_mode"] == distractor_mode
        assert report["thresholds"]["passed"] is True
        assert report["thresholds"]["failure_count"] == 0
        assert all(check["status"] == "passed" for check in report["thresholds"]["checks"])
        for strategy_name in ("context_frontier_router", "router_verifier_loop"):
            series = report["chart"]["series"][strategy_name]
            assert [point["corpus_size"] for point in series] == [100]
            assert series[0]["supported_answer_rate_at_budget"] > 0.0, (
                strategy_name,
                distractor_mode,
            )


def test_paperserve_bench_scaling_table_formats_strategy_rows_and_corpus_columns():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_scaling_benchmark(
        corpus_sizes=(5, 20),
        strategy=None,
    )
    table = benchmark.format_scaling_table(report)

    assert isinstance(table, str)
    assert table.strip()
    assert not table.lstrip().startswith("{")

    lines = [line for line in table.splitlines() if line.strip()]
    header = next(line for line in lines if "strategy" in line.lower())
    assert "5" in header
    assert "20" in header
    for strategy_name in ("context_frontier_router", "router_verifier_loop"):
        row = next(line for line in lines if strategy_name in line)
        assert len(row.split()) >= 3


def test_paperserve_bench_scaling_csv_formats_required_export_columns():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_scaling_benchmark(
        corpus_sizes=(5, 20),
        strategy="context_frontier_router",
    )
    csv_text = benchmark.format_scaling_csv(report)

    rows = list(csv.DictReader(csv_text.splitlines()))
    assert rows
    assert csv_text.endswith("\n")
    assert set(rows[0]) == PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS
    assert {row["strategy"] for row in rows} == {"context_frontier_router"}
    assert [int(row["corpus_size"]) for row in rows] == [5, 20]

    series_points = report["chart"]["series"]["context_frontier_router"]
    for row, point in zip(rows, series_points):
        for column in PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS - {"strategy"}:
            if column == "corpus_size":
                assert int(row[column]) == point[column]
            else:
                assert float(row[column]) == pytest.approx(point[column])


@pytest.mark.parametrize(
    ("distractor_mode", "expected_types", "expected_text"),
    [
        (
            "mixed",
            {"closer_gold_like", "near_miss_citation", "partial_evidence", "conflicting_evidence"},
            "Question terms:",
        ),
        ("closer_gold_like", {"closer_gold_like"}, "gold-like vocabulary"),
        ("near_miss_citation", {"near_miss_citation"}, "adjacent-looking work"),
        ("partial_evidence", {"partial_evidence"}, "partial evidence"),
        ("conflicting_evidence", {"conflicting_evidence"}, "conflicting claim"),
    ],
)
def test_paperserve_bench_expanded_distractor_modes_have_mode_specific_metadata_and_text(
    distractor_mode,
    expected_types,
    expected_text,
):
    import examples.paperserve_benchmark as benchmark

    fixture = _paperserve_fixture(benchmark, "deep_hidden_target")
    expanded = benchmark.expand_fixture_corpus(fixture, 50, distractor_mode=distractor_mode)

    generated = _synthetic_distractors(expanded)
    assert generated
    assert {candidate.metadata.get("distractor_type") for candidate in generated} == expected_types
    assert all(candidate.metadata.get("distractor_type") in expected_types for candidate in generated)
    assert all(candidate.metadata.get("distractor_mode") == distractor_mode for candidate in generated)
    assert any(expected_text in candidate.text for candidate in generated)


def test_paperserve_bench_scaling_threshold_report_passes_default_gates():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_scaling_benchmark(corpus_sizes=(100,), strategy=None)

    threshold_report = report["thresholds"]
    assert threshold_report["kind"] == "minimum_metric_thresholds"
    assert _threshold_report_passed(threshold_report) is True
    assert threshold_report["checks"]
    assert all(check["corpus_size"] == 100 for check in threshold_report["checks"])
    assert all(check["status"] == "passed" for check in threshold_report["checks"])


def test_paperserve_bench_threshold_helper_fails_impossible_gate_when_exposed():
    import examples.paperserve_benchmark as benchmark

    helper = getattr(benchmark, "check_scaling_thresholds", None)
    if helper is None:
        pytest.skip("PaperServe-Bench does not expose a direct threshold helper")

    report = benchmark.run_scaling_benchmark(corpus_sizes=(100,), strategy="context_frontier_router")
    impossible_thresholds = [
        {
            "strategy": "context_frontier_router",
            "corpus_size": 100,
            "metric": "supported_answer_rate_at_budget",
            "minimum": 2.0,
        }
    ]
    threshold_report = helper(report, thresholds=impossible_thresholds)

    assert _threshold_report_passed(threshold_report) is False
    assert threshold_report["failure_count"] == 1


def test_paperserve_bench_cli_output_writes_json_file_and_preserves_stdout(tmp_path):
    output_path = tmp_path / "scaling-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--compact",
            "--scaling",
            "--corpus-sizes",
            "100",
            "--distractor-mode",
            "partial_evidence",
            "--strategy",
            "context_frontier_router",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_report = json.loads(completed.stdout)
    file_report = json.loads(output_path.read_text())
    assert stdout_report == file_report
    assert stdout_report["mode"] == "scaling"
    assert stdout_report["distractor_mode"] == "partial_evidence"
    assert stdout_report["corpus_sizes"] == [100]
    assert set(stdout_report["chart"]["series"]) == {"context_frontier_router"}
    series_point = stdout_report["chart"]["series"]["context_frontier_router"][0]
    assert PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS - {"strategy"} <= set(series_point)
    assert stdout_report["thresholds"]["checks"]
    assert {
        "strategy",
        "corpus_size",
        "metric",
        "minimum",
        "value",
        "status",
    } <= set(stdout_report["thresholds"]["checks"][0])


def test_paperserve_bench_cli_csv_output_writes_required_columns(tmp_path):
    csv_path = tmp_path / "scaling-report.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--compact",
            "--scaling",
            "--corpus-sizes",
            "100",
            "--distractor-mode",
            "near_miss_citation",
            "--strategy",
            "router_verifier_loop",
            "--csv-output",
            str(csv_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert report["mode"] == "scaling"
    assert report["distractor_mode"] == "near_miss_citation"
    assert rows
    assert set(rows[0]) == PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS
    assert {row["strategy"] for row in rows} == {"router_verifier_loop"}
    assert [int(row["corpus_size"]) for row in rows] == report["corpus_sizes"] == [100]
    for column in PAPERSERVE_SCALING_DIAGNOSTIC_COLUMNS - {"strategy"}:
        assert column in rows[0]


def test_paperserve_bench_cli_scaling_compact_emits_parseable_chart_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--compact",
            "--scaling",
            "--corpus-sizes",
            "5,20",
            "--strategy",
            "router_verifier_loop",
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["mode"] == "scaling"
    assert report["corpus_sizes"] == [5, 20]
    assert set(report["chart"]["series"]) == {"router_verifier_loop"}


def test_paperserve_bench_cli_fail_on_thresholds_exits_zero_for_default_scaling():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--compact",
            "--scaling",
            "--corpus-sizes",
            "100",
            "--distractor-mode",
            "conflicting_evidence",
            "--fail-on-thresholds",
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["mode"] == "scaling"
    assert report["corpus_sizes"] == [100]
    assert report["distractor_mode"] == "conflicting_evidence"
    assert _threshold_report_passed(report["thresholds"]) is True


def test_paperserve_bench_cli_rejects_invalid_corpus_sizes():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--scaling",
            "--corpus-sizes",
            "5,0",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert "comma-separated positive integers" in completed.stderr

def test_paperserve_bench_missing_evidence_strategy_records_no_answer_behavior():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_benchmark()
    missing_results = [
        result for result in _benchmark_fixture_results(report) if _field(result, "fixture_id") == "missing_polymer_claim"
    ]

    assert len(missing_results) == 1
    strategies = _field(missing_results[0], "strategies")
    for strategy_name, strategy_result in strategies.items():
        assert strategy_result["status"] == "insufficient_evidence", strategy_name
        assert strategy_result["citations"] == [], strategy_name
        assert strategy_result.get("no_answer_correct") is True or strategy_result.get("no_answer_accuracy") == 1.0
        assert strategy_result["false_stop"] is False


def test_paperserve_bench_loop_recovery_reports_recovered_support_for_router_verifier_loop():
    import examples.paperserve_benchmark as benchmark

    report = benchmark.run_benchmark()
    loop_results = [
        result for result in _benchmark_fixture_results(report) if _field(result, "fixture_id") == "loop_recovery_hydrogel"
    ]

    assert len(loop_results) == 1
    loop_strategy = _field(loop_results[0], "strategies")["router_verifier_loop"]
    assert loop_strategy["supported"] is True
    assert loop_strategy["recovered"] is True
    assert loop_strategy["attempt_count"] >= 2
    assert loop_strategy["citations"]


def test_paperserve_bench_cli_compact_emits_parseable_json():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.paperserve_benchmark", "--compact"],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert len(_benchmark_fixture_results(report)) >= 5
    assert {
        "answer_near_miss_partial_evidence",
        "wide_conflicting_gold_like_distractors",
    } <= {_field(result, "fixture_id") for result in _benchmark_fixture_results(report)}
    assert {
        "bm25_top20",
        "long_context_all_docs",
        "context_frontier_router",
        "router_verifier_loop",
    } <= set(_benchmark_aggregate(report))


def test_paperserve_bench_cli_strategy_filters_to_context_frontier_router():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.paperserve_benchmark",
            "--compact",
            "--strategy",
            "context_frontier_router",
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert set(_benchmark_aggregate(report)) == {"context_frontier_router"}
    for result in _benchmark_fixture_results(report):
        assert set(_field(result, "strategies")) == {"context_frontier_router"}


def test_paperserve_bench_module_has_no_training_vector_provider_or_scheduler_imports():
    path = REPO_ROOT / "examples" / "paperserve_benchmark.py"
    source = path.read_text()
    forbidden_patterns = [
        r"^\s*(?:from|import)\s+torch\b",
        r"^\s*(?:from|import)\s+transformers\b",
        r"^\s*(?:from|import)\s+sentence_transformers\b",
        r"^\s*(?:from|import)\s+faiss\b",
        r"^\s*(?:from|import)\s+chromadb\b",
        r"^\s*(?:from|import)\s+openai\b",
        r"^\s*(?:from|import)\s+anthropic\b",
        r"^\s*(?:from|import)\s+[^\n#]*scheduler",
        r"^\s*(?:from|import)\s+[^\n#]*vector",
        r"^\s*(?:from|import)\s+[^\n#]*provider",
    ]

    for pattern in forbidden_patterns:
        assert not re.search(pattern, source, flags=re.MULTILINE), pattern
    assert "NodeKind" not in source

def test_default_reader_uses_selected_context_only():
    from researchfs.ask import default_reader

    gold = _candidate("gold", text="# Gold paper\nSelected gold evidence survives routing.")
    rejected = _candidate(
        "rejected",
        text="POISON_UNSELECTED_CONTEXT should never reach the default reader.",
    )

    result = default_reader("question", [gold])

    assert result["status"] == "answered"
    assert "Selected gold evidence survives routing." in result["answer"]
    assert "POISON_UNSELECTED_CONTEXT" not in result["answer"]
    assert result["citations"] == ["gold"]


def test_citation_verifier_accepts_selected_citations_and_rejects_external_citations():
    from researchfs.ask import citation_verifier

    selected = [_candidate("gold")]

    accepted = citation_verifier(
        {"answer": "grounded answer", "citations": ["gold"]},
        selected,
    )
    rejected = citation_verifier(
        {"answer": "ungrounded answer", "citations": ["outside"]},
        selected,
    )

    assert accepted["supported"] is True
    assert accepted["recoverable"] is False
    assert rejected["supported"] is False
    assert rejected["recoverable"] is False
    assert rejected["missing_paper_ids"] == ["outside"]


def test_run_routed_ask_with_injected_reader_verifier_recovers_after_first_attempt(monkeypatch):
    import researchfs.ask as ask_module

    distractor = _candidate("distractor", score=0.4, tokens=3, text="Distractor evidence")
    gold = _candidate("gold", score=0.9, tokens=3, text="Gold evidence")
    reader_context_ids = []
    requested_max_papers = []

    def fake_get_research_brief(_query, **kwargs):
        requested_max_papers.append(kwargs["max_papers"])
        return {"attempt": len(requested_max_papers), "paper_ids": [], "method_ids": [], "chunk_ids": [], "lesson_ids": []}

    def fake_brief_to_candidates(brief, **_kwargs):
        if brief["attempt"] == 1:
            return [distractor]
        return [distractor, gold]

    def reader(_query, selected_context):
        ids = [candidate.id for candidate in selected_context]
        reader_context_ids.append(ids)
        assert ids in (["distractor"], ["gold"])
        return {"status": "answered", "answer": "answer for " + ",".join(ids), "citations": ids}

    def verifier(answer_payload, _selected_context):
        if "gold" in answer_payload["citations"]:
            return {"supported": True}
        return {"supported": False, "recoverable": True, "feedback": "need gold"}

    monkeypatch.setattr(ask_module.rfs, "get_research_brief", fake_get_research_brief)
    monkeypatch.setattr(ask_module, "research_brief_to_context_candidates", fake_brief_to_candidates)

    result = ask_module.run_routed_ask(
        "recover gold",
        reader=reader,
        verifier=verifier,
        max_papers=1,
        k=1,
        max_retries=1,
    )

    assert result["supported"] is True
    assert result["status"] == "answered"
    assert result["metrics"]["recovered"] is True
    assert len(result["attempts"]) == 2
    assert result["reader_context_ids_by_attempt"] == [["distractor"], ["gold"]]
    assert "gold" not in result["attempts"][0]["reader_context_ids"]
    assert "gold" in result["attempts"][1]["reader_context_ids"]
    assert reader_context_ids == [["distractor"], ["gold"]]
    assert requested_max_papers == [1, 2]


def test_run_routed_ask_unsupported_answer_is_not_returned(monkeypatch):
    import researchfs.ask as ask_module

    selected = _candidate("selected", score=1.0, tokens=2)

    monkeypatch.setattr(
        ask_module.rfs,
        "get_research_brief",
        lambda _query, **_kwargs: {"paper_ids": [], "method_ids": [], "chunk_ids": [], "lesson_ids": []},
    )
    monkeypatch.setattr(
        ask_module,
        "research_brief_to_context_candidates",
        lambda _brief, **_kwargs: [selected],
    )

    result = ask_module.run_routed_ask(
        "unsupported",
        reader=lambda _query, _context: "hallucinated unsupported answer",
        verifier=lambda _answer, _context: {"supported": False, "recoverable": False},
        max_retries=0,
    )

    assert result["status"] == "unsupported"
    assert result["supported"] is False
    assert result["answer"] is None
    assert result["citations"] == []
    assert "hallucinated unsupported answer" not in json.dumps(result, sort_keys=True)


def test_run_routed_ask_logs_attempt_and_final_events(monkeypatch):
    from researchfs.db import get_connection, initialize_db
    import researchfs.ask as ask_module

    conn = get_connection(":memory:")
    initialize_db(conn)
    method_ids = []
    for name in ("routed-ask-method-a", "routed-ask-method-b"):
        method_ids.append(conn.execute("INSERT INTO methods (name) VALUES (?)", (name,)).lastrowid)
    conn.commit()

    selected = _candidate(
        "selected",
        score=1.0,
        tokens=2,
        text="Selected logged evidence",
    )
    selected.metadata["method_ids"] = method_ids
    rejected = _candidate("rejected", score=0.1, tokens=2, text="Rejected logged evidence")
    rejected.metadata["method_ids"] = [method_ids[1]]

    monkeypatch.setattr(
        ask_module.rfs,
        "get_research_brief",
        lambda _query, **_kwargs: {"paper_ids": [], "method_ids": method_ids, "chunk_ids": [], "lesson_ids": []},
    )
    monkeypatch.setattr(
        ask_module,
        "research_brief_to_context_candidates",
        lambda _brief, **_kwargs: [selected, rejected],
    )

    result = ask_module.run_routed_ask(
        "log routed ask",
        conn=conn,
        reader=lambda _query, context: {"answer": "supported", "citations": [context[0].id]},
        verifier=lambda _answer, _context: True,
        k=1,
        log=True,
    )

    events = [
        dict(row)
        for row in conn.execute(
            "SELECT event_id, event_type, retrieved_content FROM retrieval_events ORDER BY event_id"
        )
    ]
    assert [event["event_type"] for event in events] == ["routed_ask_routing_audit", "routed_ask"]
    assert result["routing_event_ids"] == [events[0]["event_id"]]
    assert result["event_id"] == events[1]["event_id"]

    audit_payload = json.loads(events[0]["retrieved_content"])
    assert audit_payload["selected_ids"] == ["selected"]
    assert audit_payload["rejected_ids"] == ["rejected"]
    final_payload = json.loads(events[1]["retrieved_content"])
    assert final_payload["selected_ids"] == ["selected"]
    assert final_payload["rejected_ids"] == ["rejected"]

    final_links = [
        row["method_id"]
        for row in conn.execute(
            "SELECT method_id FROM retrieved_methods WHERE event_id = ? ORDER BY method_id",
            (result["event_id"],),
        )
    ]
    assert final_links == method_ids


def test_export_routing_traces_reconstructs_candidate_rows_from_routed_ask_logs(monkeypatch):
    """Logged routed asks export one labeled row per candidate in each attempt."""
    from researchfs.db import get_connection, initialize_db
    from researchfs.routing_traces import export_routing_traces
    import researchfs.ask as ask_module

    conn = get_connection(":memory:")
    initialize_db(conn)

    recoverable_miss = _candidate(
        "recoverable-miss",
        score=0.5,
        tokens=3,
        text="Recoverable miss evidence",
    )
    gold = _candidate("gold", score=0.9, tokens=3, text="Gold evidence")
    requested_max_papers = []

    def fake_get_research_brief(_query, **kwargs):
        requested_max_papers.append(kwargs["max_papers"])
        return {
            "attempt": len(requested_max_papers),
            "paper_ids": [],
            "method_ids": [],
            "chunk_ids": [],
            "lesson_ids": [],
        }

    def fake_brief_to_candidates(brief, **_kwargs):
        if brief["attempt"] == 1:
            return [recoverable_miss]
        return [recoverable_miss, gold]

    def reader(_query, selected_context):
        selected_ids = [candidate.id for candidate in selected_context]
        assert selected_ids in (["recoverable-miss"], ["gold"])
        return {
            "status": "answered",
            "answer": "answer for " + ",".join(selected_ids),
            "citations": selected_ids,
        }

    def verifier(answer_payload, _selected_context):
        if answer_payload["citations"] == ["gold"]:
            return {"supported": True, "recoverable": False}
        return {"supported": False, "recoverable": True, "feedback": "need gold"}

    monkeypatch.setattr(ask_module.rfs, "get_research_brief", fake_get_research_brief)
    monkeypatch.setattr(ask_module, "research_brief_to_context_candidates", fake_brief_to_candidates)

    ask_result = ask_module.run_routed_ask(
        "recover gold",
        conn=conn,
        reader=reader,
        verifier=verifier,
        max_papers=1,
        k=1,
        max_retries=1,
        log=True,
    )

    export_result = export_routing_traces(conn=conn, include_candidate_text=False)
    traces = export_result["traces"]
    traces_by_attempt_candidate = {
        (trace["attempt"], trace["candidate_id"]): trace for trace in traces
    }

    assert requested_max_papers == [1, 2]
    assert ask_result["supported"] is True
    assert ask_result["metrics"]["recovered"] is True
    assert export_result["ok"] is True
    assert export_result["event_count"] == 1
    assert export_result["skipped_event_count"] == 0
    assert export_result["trace_count"] == sum(
        len(attempt["candidate_ids"]) for attempt in ask_result["attempts"]
    )
    assert set(traces_by_attempt_candidate) == {
        (1, "recoverable-miss"),
        (2, "recoverable-miss"),
        (2, "gold"),
    }

    first_miss = traces_by_attempt_candidate[(1, "recoverable-miss")]
    assert first_miss["trace_id"] == (
        f"event:{ask_result['event_id']}:attempt:1:candidate:recoverable-miss"
    )
    assert first_miss["selected"] is True
    assert first_miss["attempt_supported"] is False
    assert first_miss["attempt_recoverable"] is True
    assert first_miss["final_supported"] is True
    assert first_miss["final_citations"] == ["gold"]
    assert first_miss["label"] == "selected_recoverable_miss"
    assert first_miss["candidate"]["text"] is None

    second_gold = traces_by_attempt_candidate[(2, "gold")]
    assert second_gold["trace_id"] == (
        f"event:{ask_result['event_id']}:attempt:2:candidate:gold"
    )
    assert second_gold["reader_context_ids"] == ["gold"]
    assert second_gold["selected"] is True
    assert second_gold["cited"] is True
    assert second_gold["citation_label"] is True
    assert second_gold["label"] == "cited_supported"

    second_miss = traces_by_attempt_candidate[(2, "recoverable-miss")]
    assert second_miss["selected"] is False
    assert second_miss["rejected"] is True
    assert second_miss["reader_context_ids"] == ["gold"]
    assert second_miss["label"] == "candidate_not_selected"


def test_export_routing_traces_skips_truncated_or_invalid_events():
    """Malformed and truncated final logs are skipped instead of producing unsafe labels."""
    from researchfs.db import get_connection, initialize_db
    from researchfs.routing_traces import export_routing_traces

    conn = get_connection(":memory:")
    initialize_db(conn)
    conn.execute(
        "INSERT INTO retrieval_events (event_type, query, retrieved_content) VALUES (?, ?, ?)",
        ("routed_ask", "bad json", "{not-json"),
    )
    conn.execute(
        "INSERT INTO retrieval_events (event_type, query, retrieved_content) VALUES (?, ?, ?)",
        ("routed_ask", "truncated", json.dumps({"truncated": True, "attempts": []})),
    )
    conn.commit()

    result = export_routing_traces(conn=conn, include_candidate_text=False)

    assert result["ok"] is True
    assert result["event_count"] == 2
    assert result["trace_count"] == 0
    assert result["skipped_event_count"] == 2
    reasons = [event["reason"] for event in result["skipped_events"]]
    assert any("invalid JSON" in reason for reason in reasons)
    assert any("truncated payload" in reason for reason in reasons)


def test_write_routing_traces_jsonl_emits_one_compact_object_per_line():
    """The JSONL writer emits parseable compact rows and returns the row count."""
    import io
    from researchfs.routing_traces import write_routing_traces_jsonl

    stream = io.StringIO()
    count = write_routing_traces_jsonl(
        [{"b": 2, "a": 1}, {"trace_id": "two", "selected": False}],
        stream,
    )

    lines = stream.getvalue().splitlines()
    assert count == 2
    assert lines == [
        '{"a":1,"b":2}',
        '{"selected":false,"trace_id":"two"}',
    ]
    assert [json.loads(line) for line in lines] == [
        {"a": 1, "b": 2},
        {"trace_id": "two", "selected": False},
    ]


def test_context_routing_ablation_reports_four_strategies():
    import examples.context_routing_benchmark as benchmark

    report = benchmark.run_ablation_benchmark()

    assert len(report["fixtures"]) == 5
    assert set(report["aggregate"]) == {
        "all_context",
        "top_k_score",
        "score_budget_route",
        "route_recover",
    }


def test_context_routing_ablation_cli_compact_json():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.context_routing_benchmark", "--compact", "--ablation"],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert len(report["fixtures"]) == 5
    assert set(report["aggregate"]) == {
        "all_context",
        "top_k_score",
        "score_budget_route",
        "route_recover",
    }

def test_metrics_cover_recall_citation_support_efficiency_and_recovery_contracts():
    from researchfs.context import metrics

    assert metrics.gold_recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == pytest.approx(0.5)
    assert metrics.gold_recall_at_k(["a", "a", "gold"], ["gold"], k=2) == pytest.approx(0.0)
    assert metrics.gold_recall_at_k(["a", "gold"], ["gold"], k=2) == pytest.approx(1.0)
    assert metrics.citation_precision(["a", "b", "x"], {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert metrics.citation_recall(["a", "b", "x"], {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert metrics.citation_f1(["a", "b", "x"], {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert metrics.supported_answer_rate([True, False, True]) == pytest.approx(2 / 3)
    assert metrics.context_efficiency(answer_supported=True, tokens_used=200) < metrics.context_efficiency(
        answer_supported=True,
        tokens_used=20,
    )
    assert metrics.context_efficiency(answer_supported=False, tokens_used=200) == 0
    assert metrics.recovery_success([False, True]) == 1.0
    assert metrics.recovery_success([False, False]) == 0.0
    if hasattr(metrics, "recovery_rate"):
        assert metrics.recovery_rate([True, False, True]) == pytest.approx(2 / 3)
    class FalseStopRecord:
        def __init__(self, stopped_without_enough_evidence):
            self.stopped_without_enough_evidence = stopped_without_enough_evidence

    assert "false_stop_rate" in getattr(metrics, "__all__", ())
    assert metrics.false_stop_rate(
        [
            True,
            False,
            {"false_stop": True},
            {"stopped_without_enough_evidence": False},
            FalseStopRecord(True),
        ]
    ) == pytest.approx(3 / 5)
    assert metrics.false_stop_rate([]) == 0.0


def test_context_routing_modules_do_not_touch_scheduler_symbols():
    module_paths = [
        REPO_ROOT / "researchfs" / "context" / "__init__.py",
        REPO_ROOT / "researchfs" / "context" / "routing.py",
        REPO_ROOT / "researchfs" / "context" / "metrics.py",
        REPO_ROOT / "researchfs" / "context" / "researchfs.py",
        REPO_ROOT / "researchfs" / "ask.py",
    ]

    for path in module_paths:
        source = path.read_text()
        assert "NodeKind" not in source, f"{path} must not depend on scheduler node kinds"
        assert "pending" not in source, f"{path} must not inspect or mutate scheduler pending loops"
        assert not re.search(
            r"^\s*(?:from|import)\s+[^\n#]*scheduler",
            source,
            flags=re.MULTILINE,
        ), f"{path} must not import scheduler modules"
