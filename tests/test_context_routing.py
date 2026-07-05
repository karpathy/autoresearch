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

def test_metrics_cover_recall_citation_support_efficiency_and_recovery_contracts():
    from researchfs.context import metrics

    assert metrics.gold_recall_at_k(["a", "b", "c"], {"b", "d"}, k=2) == pytest.approx(0.5)
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
