"""Adapters between ResearchFS retrieval results and context routing primitives."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any


_AUDIT_JSON_MAX_CHARS = 32_000
_TITLE_RE = re.compile(r"^#\s+Paper Memory:\s*(?P<title>.+?)\s*$")
_METADATA_KEYS = (
    "method_ids",
    "chunk_ids",
    "lesson_ids",
    "experiment_ids",
    "confidence_score",
    "source_char_count",
    "compressed_char_count",
    "extraction_status",
)


def paper_memory_to_context_candidate(
    card: Mapping[str, Any],
    *,
    score: float | None = None,
    token_estimate: int | None = None,
) -> ContextCandidate:
    """Convert a ``get_paper_memory(...)`` result into a ``ContextCandidate``."""

    if not isinstance(card, Mapping):
        raise ValueError("card must be a mapping")

    paper_id = _validate_paper_id(card.get("paper_id"))
    context = card.get("context")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("card.context must be a non-empty string")

    candidate_score = _resolve_score(score, card.get("confidence_score"))
    candidate_token_estimate = _resolve_token_estimate(token_estimate, context)

    metadata: dict[str, Any] = {"paper_id": paper_id}
    for key in _METADATA_KEYS:
        if key in card:
            metadata[key] = _json_safe(card[key])

    from .routing import ContextCandidate
    paper_id_text = str(paper_id)
    return ContextCandidate(
        id=paper_id_text,
        title=_paper_memory_title(context),
        text=context,
        source=f"researchfs://paper/{paper_id_text}",
        metadata=metadata,
        score=candidate_score,
        token_estimate=candidate_token_estimate,
    )


def paper_memories_to_context_candidates(
    cards: Iterable[Mapping[str, Any]],
    *,
    scores_by_paper_id: Mapping[int | str, float] | None = None,
) -> list[ContextCandidate]:
    """Convert paper-memory cards into context candidates."""

    if isinstance(cards, (str, bytes)):
        raise ValueError("cards must be an iterable of mappings")
    if scores_by_paper_id is not None and not isinstance(scores_by_paper_id, Mapping):
        raise ValueError("scores_by_paper_id must be a mapping or None")

    try:
        materialized = list(cards)
    except TypeError as exc:
        raise ValueError("cards must be an iterable of mappings") from exc

    candidates: list[ContextCandidate] = []
    for index, card in enumerate(materialized):
        if not isinstance(card, Mapping):
            raise ValueError(f"cards[{index}] must be a mapping")
        paper_id = _validate_paper_id(card.get("paper_id"))
        score = _lookup_score(scores_by_paper_id, paper_id)
        candidates.append(paper_memory_to_context_candidate(card, score=score))
    return candidates


def research_brief_to_context_candidates(
    brief: Mapping[str, Any],
    *,
    conn=None,
    per_paper_max_chars: int | None = None,
    scores_by_paper_id: Mapping[int | str, float] | None = None,
) -> list[ContextCandidate]:
    """Build one context candidate per paper referenced by a research brief."""

    if not isinstance(brief, Mapping):
        raise ValueError("brief must be a mapping")
    if per_paper_max_chars is not None:
        per_paper_max_chars = _validate_non_negative_int(
            per_paper_max_chars, "per_paper_max_chars"
        )
    if scores_by_paper_id is not None and not isinstance(scores_by_paper_id, Mapping):
        raise ValueError("scores_by_paper_id must be a mapping or None")

    embedded_cards = _embedded_paper_memory_cards(brief)
    if embedded_cards is not None:
        return paper_memories_to_context_candidates(
            embedded_cards,
            scores_by_paper_id=scores_by_paper_id,
        )

    paper_ids = brief.get("paper_ids")
    if isinstance(paper_ids, (str, bytes)):
        raise ValueError("brief.paper_ids must be an iterable of paper ids")
    try:
        materialized_ids = list(paper_ids)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("brief.paper_ids must be an iterable of paper ids") from exc

    from researchfs.retrieve import get_paper_memory

    cards = [
        get_paper_memory(
            _validate_paper_id(paper_id),
            conn=conn,
            max_chars=per_paper_max_chars,
        )
        for paper_id in materialized_ids
    ]
    return paper_memories_to_context_candidates(
        cards,
        scores_by_paper_id=scores_by_paper_id,
    )


def selection_audit_payload(
    selection: ContextSelection,
    *,
    query: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic JSON-serializable audit payload for a selection."""

    from .routing import ContextSelection
    if not isinstance(selection, ContextSelection):
        raise ValueError("selection must be a ContextSelection")
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a string or None")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason must be a string or None")

    selected_ids = [_candidate_id(candidate, "selected") for candidate in selection.selected]
    rejected_ids = [_candidate_id(candidate, "rejected") for candidate in selection.rejected]
    selected_tokens = _validate_non_negative_int(
        selection.budget_used_tokens, "selection.budget_used_tokens"
    )
    if selection.budget_max_tokens is None:
        token_budget = None
    else:
        token_budget = _validate_non_negative_int(
            selection.budget_max_tokens, "selection.budget_max_tokens"
        )

    payload = {
        "query": query,
        "reason": reason if reason is not None else selection.reason,
        "rejected_count": len(rejected_ids),
        "rejected_ids": rejected_ids,
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "selected_tokens": selected_tokens,
        "token_budget": token_budget,
    }
    return _json_safe(payload)


def log_context_routing_audit(
    query: str,
    selection: ContextSelection,
    *,
    agent_run_id: int | None = None,
    experiment_id: int | None = None,
    conn=None,
    event_type: str = "context_routing_audit",
) -> int:
    """Log an opt-in context-routing audit event through retrieval logging."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("event_type must be a non-empty string")

    from researchfs.retrieve import log_retrieval_event

    payload = selection_audit_payload(selection, query=query)
    retrieved_content = _bounded_json(payload)
    return log_retrieval_event(
        event_type.strip(),
        query.strip(),
        retrieved_content,
        agent_run_id=agent_run_id,
        experiment_id=experiment_id,
        conn=conn,
        method_ids=_selected_method_ids(selection),
    )


def _validate_paper_id(value: Any) -> int | str:
    if isinstance(value, bool):
        raise ValueError("paper_id must be a positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("paper_id must be positive")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.isdecimal():
            raise ValueError("paper_id must be a positive integer")
        if int(stripped) <= 0:
            raise ValueError("paper_id must be positive")
        return stripped
    raise ValueError("paper_id must be a positive integer")


def _validate_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _resolve_score(explicit: Any, confidence_score: Any) -> float | None:
    if explicit is not None:
        return _validate_finite_score(explicit, "score")
    if confidence_score is None or isinstance(confidence_score, bool):
        return None
    if not isinstance(confidence_score, (int, float)):
        return None
    confidence = float(confidence_score)
    return confidence if math.isfinite(confidence) else None


def _validate_finite_score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number or None")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite")
    return score


def _resolve_token_estimate(explicit: Any, context: str) -> int:
    if explicit is not None:
        return _validate_non_negative_int(explicit, "token_estimate")
    return len(context.split())


def _lookup_score(
    scores_by_paper_id: Mapping[int | str, float] | None,
    paper_id: int | str,
) -> float | None:
    if scores_by_paper_id is None:
        return None
    if paper_id in scores_by_paper_id:
        return _validate_finite_score(scores_by_paper_id[paper_id], "score")
    paper_id_text = str(paper_id)
    if paper_id_text in scores_by_paper_id:
        return _validate_finite_score(scores_by_paper_id[paper_id_text], "score")
    if isinstance(paper_id, str) and paper_id.isdecimal():
        paper_id_int = int(paper_id)
        if paper_id_int in scores_by_paper_id:
            return _validate_finite_score(scores_by_paper_id[paper_id_int], "score")
    return None


def _paper_memory_title(context: str) -> str | None:
    for line in context.splitlines():
        match = _TITLE_RE.match(line.strip())
        if match:
            title = match.group("title").strip()
            return title or None
    return None


def _embedded_paper_memory_cards(brief: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    for key in ("paper_memory_cards", "paper_memories", "cards"):
        value = brief.get(key)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            raise ValueError(f"brief.{key} must be an iterable of mappings")
        try:
            cards = list(value)
        except TypeError as exc:
            raise ValueError(f"brief.{key} must be an iterable of mappings") from exc
        if not all(isinstance(card, Mapping) for card in cards):
            raise ValueError(f"brief.{key} must contain mappings")
        return cards
    return None


def _candidate_id(candidate: ContextCandidate, list_name: str) -> str:
    from .routing import ContextCandidate
    if not isinstance(candidate, ContextCandidate):
        raise ValueError(f"selection.{list_name} must contain ContextCandidate objects")
    if not isinstance(candidate.id, str) or not candidate.id.strip():
        raise ValueError("ContextCandidate.id must be a non-empty string")
    return candidate.id


def _selected_method_ids(selection: ContextSelection) -> list[int]:
    seen: set[int] = set()
    method_ids: list[int] = []
    for candidate in selection.selected:
        raw_ids = candidate.metadata.get("method_ids", [])
        if raw_ids is None:
            continue
        if isinstance(raw_ids, (str, bytes)):
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


def _bounded_json(payload: Mapping[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    bounded_payload = dict(payload)
    bounded_payload["truncated"] = True
    bounded_payload["original_json_chars"] = len(content)
    bounded_payload["rejected_ids"] = _fit_id_list(bounded_payload.get("rejected_ids", []))
    bounded_payload["selected_ids"] = _fit_id_list(bounded_payload.get("selected_ids", []))
    content = json.dumps(bounded_payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    bounded_payload["rejected_ids"] = []
    bounded_payload["selected_ids"] = []
    bounded_payload["ids_truncated"] = True
    for key in ("query", "reason"):
        if isinstance(bounded_payload.get(key), str):
            bounded_payload[key] = _truncate_string_to_json_budget(
                bounded_payload[key],
                _AUDIT_JSON_MAX_CHARS // 4,
            )
    content = json.dumps(bounded_payload, sort_keys=True, separators=(",", ":"))
    if len(content) <= _AUDIT_JSON_MAX_CHARS:
        return content

    minimal_payload = {
        "ids_truncated": True,
        "original_json_chars": bounded_payload["original_json_chars"],
        "query": _truncate_string_to_json_budget(str(payload.get("query") or ""), 256),
        "reason": "audit payload exceeded bounded JSON limit",
        "rejected_count": bounded_payload.get("rejected_count"),
        "rejected_ids": [],
        "selected_count": bounded_payload.get("selected_count"),
        "selected_ids": [],
        "selected_tokens": bounded_payload.get("selected_tokens"),
        "token_budget": bounded_payload.get("token_budget"),
        "truncated": True,
    }
    return json.dumps(minimal_payload, sort_keys=True, separators=(",", ":"))


def _truncate_string_to_json_budget(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return "…"
    return value[: max_chars - 1] + "…"


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
