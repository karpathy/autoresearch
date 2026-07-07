"""First-class routed ask workflow over local ResearchFS memory."""

import json
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

import researchfs.retrieve as rfs
from researchfs.context import (
    ContextBudget,
    ContextCandidate,
    ContextSelection,
    EvidenceResult,
    citation_f1,
    citation_precision,
    citation_recall,
    context_efficiency,
    log_context_routing_audit,
    recovery_success,
    research_brief_to_context_candidates,
    route_top_k_context,
    selection_audit_payload,
)

ReaderFn = Callable[[str, list[ContextCandidate]], Mapping[str, Any] | str]
VerifierFn = Callable[[Mapping[str, Any], list[ContextCandidate]], Mapping[str, Any] | bool]

_AUDIT_JSON_MAX_CHARS = 32_000
_ALLOWED_OVERFLOW_POLICIES = {"truncate", "raise", "drop_lowest_score"}


def default_reader(query: str, selected_context: list[ContextCandidate]) -> dict[str, Any]:
    """Return a deterministic extractive answer from selected context only."""

    _validate_query(query)
    selected = _validate_context_candidates(selected_context, "selected_context")
    if not selected:
        return {"status": "insufficient_evidence", "answer": None, "citations": []}

    excerpts: list[str] = []
    for candidate in selected:
        excerpt = _first_evidence_line(candidate.text)
        if excerpt:
            excerpts.append(_truncate_text(excerpt, 500))
    answer = "Selected evidence:\n" + "\n\n".join(excerpts)
    return {
        "status": "answered",
        "answer": answer,
        "citations": [candidate.id for candidate in selected],
    }


def citation_verifier(
    answer_payload: Mapping[str, Any], selected_context: list[ContextCandidate]
) -> dict[str, Any]:
    """Verify that answer citations stay within the selected context frontier."""

    if not isinstance(answer_payload, Mapping):
        raise ValueError("answer_payload must be a mapping")
    selected = _validate_context_candidates(selected_context, "selected_context")
    citations = _require_citations(answer_payload.get("citations"))
    answer = answer_payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {
            "supported": False,
            "recoverable": True,
            "feedback": "Answer is empty or missing.",
            "missing_evidence": ["answer"],
            "missing_paper_ids": [],
        }
    if not citations:
        return {
            "supported": False,
            "recoverable": True,
            "feedback": "Answer has no citations.",
            "missing_evidence": ["citations"],
            "missing_paper_ids": [],
        }

    selected_ids = {candidate.id for candidate in selected}
    missing_paper_ids = [citation for citation in citations if citation not in selected_ids]
    if missing_paper_ids:
        return {
            "supported": False,
            "recoverable": False,
            "feedback": "Answer cited evidence outside the selected context frontier.",
            "missing_evidence": [],
            "missing_paper_ids": missing_paper_ids,
        }
    return {
        "supported": True,
        "recoverable": False,
        "feedback": "Answer citations are grounded in selected context.",
        "missing_evidence": [],
        "missing_paper_ids": [],
    }


def run_routed_ask(
    query: str,
    *,
    conn=None,
    reader: ReaderFn | None = None,
    verifier: VerifierFn | None = None,
    max_chars: int | None = None,
    max_papers: int = 3,
    per_paper_max_chars: int | None = None,
    k: int | None = 2,
    max_reader_tokens: int | None = 1200,
    min_score: float | None = None,
    overflow_policy: Literal["truncate", "raise", "drop_lowest_score"] = "drop_lowest_score",
    max_retries: int = 1,
    include_scores: bool = False,
    source_types=None,
    source_type=None,
    agent_run_id: int | None = None,
    experiment_id: int | None = None,
    log: bool = False,
    event_type: str = "routed_ask",
) -> dict[str, Any]:
    """Run deterministic routed ask over local compressed ResearchFS memory."""

    valid_query = _validate_query(query)
    valid_max_chars = _validate_optional_int(max_chars, "max_chars", minimum=0)
    valid_max_papers = _validate_int(max_papers, "max_papers", minimum=1)
    valid_per_paper_max_chars = _validate_optional_int(
        per_paper_max_chars, "per_paper_max_chars", minimum=0
    )
    valid_k = _validate_optional_int(k, "k", minimum=1)
    valid_max_reader_tokens = _validate_optional_int(
        max_reader_tokens, "max_reader_tokens", minimum=1
    )
    valid_max_retries = _validate_int(max_retries, "max_retries", minimum=0)
    valid_min_score = _validate_optional_finite_float(min_score, "min_score")
    valid_overflow_policy = _validate_overflow_policy(overflow_policy)
    valid_agent_run_id = _validate_optional_int(
        agent_run_id, "agent_run_id", minimum=1
    )
    valid_experiment_id = _validate_optional_int(
        experiment_id, "experiment_id", minimum=1
    )
    valid_event_type = _validate_event_type(event_type)
    if reader is None:
        reader = default_reader
    elif not callable(reader):
        raise ValueError("reader must be callable")
    if verifier is None:
        verifier = citation_verifier
    elif not callable(verifier):
        raise ValueError("verifier must be callable")
    if not isinstance(include_scores, bool):
        raise ValueError("include_scores must be a bool")
    if not isinstance(log, bool):
        raise ValueError("log must be a bool")

    attempts: list[dict[str, Any]] = []
    routing_event_ids: list[int] = []
    latest_brief: Mapping[str, Any] = {}
    latest_candidates: list[ContextCandidate] = []
    latest_selection = ContextSelection([], [], 0, valid_max_reader_tokens, None)
    final_answer: str | None = None
    final_citations: list[str] = []
    final_status = "insufficient_evidence"
    final_supported = False
    final_verifier_payload: dict[str, Any] = {
        "supported": False,
        "recoverable": True,
        "feedback": "",
        "missing_evidence": [],
        "missing_paper_ids": [],
    }
    first_attempt_supported: bool | None = None

    for attempt_number in range(1, valid_max_retries + 2):
        effective_max_papers = valid_max_papers * attempt_number
        latest_brief = rfs.get_research_brief(
            valid_query,
            conn=conn,
            max_chars=valid_max_chars,
            max_papers=effective_max_papers,
            include_scores=include_scores,
            agent_run_id=valid_agent_run_id,
            source_types=source_types,
            source_type=source_type,
        )
        latest_candidates = research_brief_to_context_candidates(
            latest_brief,
            conn=conn,
            per_paper_max_chars=valid_per_paper_max_chars,
        )
        budget = ContextBudget(
            max_items=valid_k,
            max_tokens=valid_max_reader_tokens,
            overflow_policy=valid_overflow_policy,
        )
        latest_selection = route_top_k_context(
            latest_candidates,
            k=valid_k,
            budget=budget,
            min_score=valid_min_score,
        )
        routing_event_id = None
        if log:
            routing_event_id = log_context_routing_audit(
                valid_query,
                latest_selection,
                agent_run_id=valid_agent_run_id,
                experiment_id=valid_experiment_id,
                conn=conn,
                event_type="routed_ask_routing_audit",
            )
            routing_event_ids.append(routing_event_id)

        answer_payload = _normalize_reader_payload(
            reader(valid_query, latest_selection.selected)
        )
        verifier_payload = _normalize_verifier_payload(
            verifier(answer_payload, latest_selection.selected)
        )
        final_verifier_payload = verifier_payload
        supported = verifier_payload["supported"]
        if first_attempt_supported is None:
            first_attempt_supported = supported

        attempts.append(
            {
                "attempt": attempt_number,
                "query": valid_query,
                "max_papers": effective_max_papers,
                "candidate_ids": [candidate.id for candidate in latest_candidates],
                "reader_context_ids": [
                    candidate.id for candidate in latest_selection.selected
                ],
                "selection_audit": selection_audit_payload(
                    latest_selection, query=valid_query
                ),
                "answer_status": answer_payload["status"],
                "verifier": verifier_payload,
                "routing_event_id": routing_event_id,
            }
        )

        if supported:
            final_supported = True
            final_status = str(answer_payload.get("status") or "answered")
            raw_answer = answer_payload.get("answer")
            final_answer = raw_answer if isinstance(raw_answer, str) else None
            final_citations = _require_citations(answer_payload.get("citations"))
            break
        if not verifier_payload["recoverable"]:
            final_status = "unsupported"
            final_answer = None
            final_citations = []
            break
        if attempt_number == valid_max_retries + 1:
            final_status = "insufficient_evidence"
            final_answer = None
            final_citations = []
            break

    selected_ids = [candidate.id for candidate in latest_selection.selected]
    rejected_ids = [candidate.id for candidate in latest_selection.rejected]
    selected_tokens = latest_selection.budget_used_tokens
    selected_count = len(latest_selection.selected)
    recovered = bool(
        final_supported and first_attempt_supported is False and len(attempts) > 1
    )
    false_stop = bool(
        final_status in {"unsupported", "insufficient_evidence"}
        and any(attempt["reader_context_ids"] for attempt in attempts)
    )
    metrics = {
        "citation_precision": citation_precision(final_citations, selected_ids),
        "citation_recall": citation_recall(final_citations, selected_ids),
        "citation_f1": citation_f1(final_citations, selected_ids),
        "context_efficiency": context_efficiency(
            selected_tokens=selected_tokens,
            max_tokens=valid_max_reader_tokens,
            selected_items=selected_count,
            max_items=valid_k,
            answer_supported=final_supported,
        ),
        "recovery_success": recovery_success(
            first_attempt_supported=bool(first_attempt_supported),
            final_attempt_supported=final_supported,
        ),
        "recovered": recovered,
        "false_stop": false_stop,
        "attempt_count": len(attempts),
        "selected_tokens": selected_tokens,
        "selected_count": selected_count,
    }

    routing_payload = selection_audit_payload(latest_selection, query=valid_query)
    routing_payload["reason"] = latest_selection.reason

    response = {
        "ok": True,
        "query": valid_query,
        "status": final_status,
        "answer": final_answer if final_supported else None,
        "citations": final_citations if final_supported else [],
        "supported": final_supported,
        "context": _selected_context_text(latest_selection.selected),
        "selected_context": [
            _candidate_summary(candidate) for candidate in latest_selection.selected
        ],
        "selected_ids": selected_ids,
        "rejected_ids": rejected_ids,
        "attempts": attempts,
        "reader_context_ids_by_attempt": [
            attempt["reader_context_ids"] for attempt in attempts
        ],
        "all_candidate_ids_by_attempt": [attempt["candidate_ids"] for attempt in attempts],
        "retrieval": _retrieval_payload(latest_brief),
        "routing": routing_payload,
        "verifier": final_verifier_payload,
        "metrics": metrics,
        "event_id": None,
        "routing_event_ids": routing_event_ids,
        "max_chars": _json_safe(latest_brief.get("max_chars", valid_max_chars)),
        "max_papers": _json_safe(latest_brief.get("max_papers", valid_max_papers)),
        "source_types": _json_safe(latest_brief.get("source_types", [])),
    }

    if log:
        event_id = rfs.log_retrieval_event(
            valid_event_type.strip(),
            valid_query.strip(),
            _bounded_final_summary(response),
            agent_run_id=valid_agent_run_id,
            experiment_id=valid_experiment_id,
            conn=conn,
            method_ids=_selected_method_ids(latest_selection),
        )
        response["event_id"] = event_id
    return response


def _validate_query(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query must be a non-empty string")
    return value.strip()


def _validate_context_candidates(value: Any, field_name: str) -> list[ContextCandidate]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of ContextCandidate")
    for index, candidate in enumerate(value):
        if not isinstance(candidate, ContextCandidate):
            raise ValueError(f"{field_name}[{index}] must be a ContextCandidate")
    return value


def _validate_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{field_name} must be non-negative")
        if minimum == 1:
            raise ValueError(f"{field_name} must be positive")
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _validate_optional_int(value: Any, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _validate_int(value, field_name, minimum=minimum)


def _validate_optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{field_name} must be finite")
    return value_float


def _validate_overflow_policy(value: Any) -> Literal["truncate", "raise", "drop_lowest_score"]:
    if value not in _ALLOWED_OVERFLOW_POLICIES:
        raise ValueError(
            "overflow_policy must be one of 'truncate', 'raise', or 'drop_lowest_score'"
        )
    return value


def _validate_event_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("event_type must be a non-empty string")
    return value


def _first_evidence_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def _require_citations(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(citation, str) for citation in value
    ):
        raise ValueError("answer citations must be a list of strings")
    return list(value)


def _normalize_reader_payload(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"status": "answered", "answer": value, "citations": []}
    if not isinstance(value, Mapping):
        raise ValueError("reader must return a mapping or string")
    payload = dict(value)
    payload.setdefault("status", "answered")
    payload.setdefault("answer", None)
    payload.setdefault("citations", [])
    if not isinstance(payload["status"], str) or not payload["status"].strip():
        raise ValueError("answer status must be a non-empty string")
    payload["citations"] = _require_citations(payload["citations"])
    return payload


def _normalize_verifier_payload(value: Mapping[str, Any] | bool) -> dict[str, Any]:
    if isinstance(value, bool):
        return {
            "supported": value,
            "recoverable": not value,
            "feedback": "",
            "missing_evidence": [],
            "missing_paper_ids": [],
        }
    if not isinstance(value, Mapping):
        raise ValueError("verifier must return a mapping or bool")
    payload = dict(value)
    if not isinstance(payload.get("supported"), bool):
        raise ValueError("verifier supported must be a bool")
    payload.setdefault("recoverable", not payload["supported"])
    payload.setdefault("feedback", "")
    payload.setdefault("missing_evidence", [])
    payload.setdefault("missing_paper_ids", [])
    if not isinstance(payload["recoverable"], bool):
        raise ValueError("verifier recoverable must be a bool")
    if not isinstance(payload["feedback"], str):
        raise ValueError("verifier feedback must be a string")
    payload["missing_evidence"] = _validate_string_list(
        payload["missing_evidence"], "verifier missing_evidence"
    )
    payload["missing_paper_ids"] = _validate_string_list(
        payload["missing_paper_ids"], "verifier missing_paper_ids"
    )
    return payload


def _validate_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return list(value)


def _selected_context_text(selected: list[ContextCandidate]) -> str:
    if not selected:
        return "# Selected context"
    blocks = []
    for candidate in selected:
        title = candidate.title or "Untitled"
        source = candidate.source or "unknown"
        blocks.append(
            f"## {candidate.id}: {title}\n"
            f"Source: {source}\n\n"
            f"{candidate.text}"
        )
    return "# Selected context\n\n" + "\n\n".join(blocks)


def _candidate_summary(candidate: ContextCandidate) -> dict[str, Any]:
    return _json_safe(
        {
            "id": candidate.id,
            "title": candidate.title,
            "source": candidate.source,
            "score": candidate.score,
            "token_estimate": candidate.token_estimate,
            "metadata": candidate.metadata,
            "text": candidate.text,
        }
    )


def _retrieval_payload(brief: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "paper_ids": brief.get("paper_ids", []),
        "method_ids": brief.get("method_ids", []),
        "chunk_ids": brief.get("chunk_ids", []),
        "lesson_ids": brief.get("lesson_ids", []),
        "paper_memory_provenance": brief.get("paper_memory_provenance", []),
    }
    if "scores" in brief:
        payload["scores"] = brief.get("scores")
    return _json_safe(payload)


def _selected_method_ids(selection: ContextSelection) -> list[int]:
    seen: set[int] = set()
    method_ids: list[int] = []
    for candidate in selection.selected:
        raw_ids = candidate.metadata.get("method_ids", [])
        if raw_ids is None or isinstance(raw_ids, (str, bytes)):
            continue
        try:
            iterable = list(raw_ids)
        except TypeError:
            continue
        for raw_id in iterable:
            if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id <= 0:
                continue
            if raw_id not in seen:
                seen.add(raw_id)
                method_ids.append(raw_id)
    return method_ids


def _bounded_final_summary(response: Mapping[str, Any]) -> str:
    payload = {
        "query": response.get("query"),
        "status": response.get("status"),
        "supported": response.get("supported"),
        "selected_ids": response.get("selected_ids"),
        "rejected_ids": response.get("rejected_ids"),
        "citations": response.get("citations"),
        "verifier": response.get("verifier"),
        "attempts": response.get("attempts"),
        "selected_tokens": response.get("metrics", {}).get("selected_tokens")
        if isinstance(response.get("metrics"), Mapping)
        else None,
        "selected_count": response.get("metrics", {}).get("selected_count")
        if isinstance(response.get("metrics"), Mapping)
        else None,
        "metrics": response.get("metrics"),
    }
    return _bounded_json(payload)


def _bounded_json(payload: Mapping[str, Any]) -> str:
    safe_payload = _json_safe(payload)
    content = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    bounded_payload = dict(safe_payload)
    bounded_payload["truncated"] = True
    bounded_payload["original_json_chars"] = len(content)
    bounded_payload["selected_ids"] = _fit_id_list(bounded_payload.get("selected_ids", []))
    bounded_payload["rejected_ids"] = _fit_id_list(bounded_payload.get("rejected_ids", []))
    bounded_payload["citations"] = _fit_id_list(bounded_payload.get("citations", []))
    bounded_payload["attempts"] = _fit_attempts(bounded_payload.get("attempts", []))
    content = json.dumps(bounded_payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    bounded_payload["selected_ids"] = []
    bounded_payload["rejected_ids"] = []
    bounded_payload["citations"] = []
    bounded_payload["ids_truncated"] = True
    if isinstance(bounded_payload.get("query"), str):
        bounded_payload["query"] = _truncate_string_to_json_budget(
            bounded_payload["query"], _AUDIT_JSON_MAX_CHARS // 4
        )
    content = json.dumps(bounded_payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    minimal_payload = {
        "ids_truncated": True,
        "original_json_chars": bounded_payload["original_json_chars"],
        "query": _truncate_string_to_json_budget(str(safe_payload.get("query") or ""), 256),
        "status": safe_payload.get("status"),
        "supported": safe_payload.get("supported"),
        "selected_count": safe_payload.get("selected_count"),
        "selected_ids": [],
        "rejected_ids": [],
        "citations": [],
        "truncated": True,
    }
    return json.dumps(minimal_payload, sort_keys=True, separators=(",", ":"))


def _fit_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    kept: list[dict[str, Any]] = []
    for attempt in value:
        if not isinstance(attempt, Mapping):
            continue
        candidate_ids = attempt.get("candidate_ids", [])
        reader_context_ids = attempt.get("reader_context_ids", [])
        selection_audit = attempt.get("selection_audit", {})
        kept.append(
            {
                "attempt": attempt.get("attempt"),
                "max_papers": attempt.get("max_papers"),
                "candidate_count": len(candidate_ids) if isinstance(candidate_ids, list) else None,
                "candidate_ids": _fit_id_list(candidate_ids),
                "reader_context_ids": _fit_id_list(reader_context_ids),
                "answer_status": attempt.get("answer_status"),
                "verifier": attempt.get("verifier"),
                "routing_event_id": attempt.get("routing_event_id"),
                "selection_audit": {
                    "selected_count": selection_audit.get("selected_count"),
                    "selected_tokens": selection_audit.get("selected_tokens"),
                    "token_budget": selection_audit.get("token_budget"),
                    "reason": selection_audit.get("reason"),
                }
                if isinstance(selection_audit, Mapping)
                else {},
            }
        )
    return kept


def _fit_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    kept: list[str] = []
    total = 0
    for item in value:
        text = str(item)
        total += len(text)
        if total > _AUDIT_JSON_MAX_CHARS // 4:
            break
        kept.append(text)
    return kept


def _truncate_string_to_json_budget(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return "…"
    return value[: max_chars - 1] + "…"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "ReaderFn",
    "VerifierFn",
    "citation_verifier",
    "default_reader",
    "run_routed_ask",
]
