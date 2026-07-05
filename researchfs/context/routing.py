"""Dependency-free context routing primitives for ResearchFS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, NamedTuple

OverflowPolicy = Literal["truncate", "raise", "drop_lowest_score"]
_ALLOWED_POLICIES = {"truncate", "raise", "drop_lowest_score"}


@dataclass(slots=True)
class ContextCandidate:
    """A single context item that can be routed into an answer frontier."""

    id: str
    title: str | None
    text: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    token_estimate: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ContextCandidate.id must be a non-empty string")
        if self.title is not None and not isinstance(self.title, str):
            raise ValueError("ContextCandidate.title must be a string or None")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("ContextCandidate.text must be a non-empty string")
        if self.source is not None and not isinstance(self.source, str):
            raise ValueError("ContextCandidate.source must be a string or None")
        if not isinstance(self.metadata, dict):
            raise ValueError("ContextCandidate.metadata must be a dict")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("ContextCandidate.score must be a finite number or None")
            if not math.isfinite(float(self.score)):
                raise ValueError("ContextCandidate.score must be finite")
            self.score = float(self.score)
        if self.token_estimate is not None:
            self.token_estimate = _validate_optional_int(
                self.token_estimate,
                "ContextCandidate.token_estimate",
                minimum=0,
            )


@dataclass(slots=True)
class ContextSelection:
    """Result of a deterministic routing pass."""

    selected: list[ContextCandidate]
    rejected: list[ContextCandidate]
    budget_used_tokens: int
    budget_max_tokens: int | None
    reason: str | None = None


@dataclass(slots=True)
class ContextBudget:
    """Hard context limits and overflow behavior for routing."""

    max_items: int | None = None
    max_tokens: int | None = None
    overflow_policy: OverflowPolicy = "truncate"

    def __post_init__(self) -> None:
        if self.max_items is not None:
            self.max_items = _validate_optional_int(
                self.max_items,
                "ContextBudget.max_items",
                minimum=0,
            )
        if self.max_tokens is not None:
            self.max_tokens = _validate_optional_int(
                self.max_tokens,
                "ContextBudget.max_tokens",
                minimum=0,
            )
        if self.overflow_policy not in _ALLOWED_POLICIES:
            raise ValueError(
                "ContextBudget.overflow_policy must be one of "
                "'truncate', 'raise', or 'drop_lowest_score'"
            )


@dataclass(slots=True)
class EvidenceResult:
    """Provider-agnostic answer/evidence verification result."""

    answer: str
    citations: list[str]
    supported: bool
    missing_evidence: list[str] = field(default_factory=list)
    verifier_notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str):
            raise ValueError("EvidenceResult.answer must be a string")
        if not isinstance(self.citations, list) or not all(
            isinstance(citation, str) for citation in self.citations
        ):
            raise ValueError("EvidenceResult.citations must be a list of strings")
        if not isinstance(self.supported, bool):
            raise ValueError("EvidenceResult.supported must be a bool")
        if not isinstance(self.missing_evidence, list) or not all(
            isinstance(item, str) for item in self.missing_evidence
        ):
            raise ValueError("EvidenceResult.missing_evidence must be a list of strings")
        if self.verifier_notes is not None and not isinstance(self.verifier_notes, str):
            raise ValueError("EvidenceResult.verifier_notes must be a string or None")


class _IndexedCandidate(NamedTuple):
    index: int
    candidate: ContextCandidate
    tokens: int


def route_top_k_context(
    candidates: Iterable[ContextCandidate],
    *,
    k: int | None = None,
    budget: ContextBudget | None = None,
    min_score: float | None = None,
) -> ContextSelection:
    """Select a deterministic top-k context frontier under optional hard budgets.

    Scored candidates rank ahead of unscored candidates, scores sort descending,
    and ties use input order. If no candidate has a score, input order is
    preserved. Candidates are never partially included or text-truncated.
    """

    if k is not None:
        k = _validate_optional_int(k, "k", minimum=0)
    if budget is None:
        budget = ContextBudget()
    elif not isinstance(budget, ContextBudget):
        raise ValueError("budget must be a ContextBudget or None")
    budget.__post_init__()
    if min_score is not None:
        min_score = _validate_finite_number(min_score, "min_score")

    indexed = _materialize_candidates(candidates)
    ranked = _rank_candidates(indexed)

    eligible: list[_IndexedCandidate] = []
    rejected_by_score: list[_IndexedCandidate] = []
    for item in ranked:
        score = item.candidate.score
        if min_score is not None and (score is None or score < min_score):
            rejected_by_score.append(item)
        else:
            eligible.append(item)

    if budget.overflow_policy == "raise":
        selected, rejected, reasons = _select_raise(eligible, k, budget)
    elif budget.overflow_policy == "drop_lowest_score":
        selected, rejected, reasons = _select_drop_lowest_score(eligible, k, budget)
    else:
        selected, rejected, reasons = _select_truncate(eligible, k, budget)

    rejected_items = [item.candidate for item in rejected_by_score] + [
        item.candidate for item in rejected
    ]
    if rejected_by_score:
        reasons.insert(0, "min_score")

    selected_candidates = [item.candidate for item in selected]
    reason = _join_reasons(reasons)
    return ContextSelection(
        selected=selected_candidates,
        rejected=rejected_items,
        budget_used_tokens=sum(item.tokens for item in selected),
        budget_max_tokens=budget.max_tokens,
        reason=reason,
    )


def _validate_optional_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{field_name} must be non-negative")
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _validate_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{field_name} must be finite")
    return value_float


def _materialize_candidates(candidates: Iterable[ContextCandidate]) -> list[_IndexedCandidate]:
    if isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be an iterable of ContextCandidate objects")
    try:
        raw_candidates = list(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be an iterable of ContextCandidate objects") from exc

    indexed: list[_IndexedCandidate] = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, ContextCandidate):
            raise ValueError(f"candidates[{index}] must be a ContextCandidate")
        candidate.__post_init__()
        indexed.append(_IndexedCandidate(index, candidate, _candidate_tokens(candidate)))
    return indexed


def _candidate_tokens(candidate: ContextCandidate) -> int:
    if candidate.token_estimate is not None:
        return candidate.token_estimate
    return len(candidate.text.split())


def _rank_candidates(indexed: list[_IndexedCandidate]) -> list[_IndexedCandidate]:
    if not any(item.candidate.score is not None for item in indexed):
        return list(indexed)
    return sorted(
        indexed,
        key=lambda item: (
            item.candidate.score is None,
            -(item.candidate.score or 0.0),
            item.index,
        ),
    )


def _item_limit(k: int | None, budget: ContextBudget) -> int | None:
    limits = [limit for limit in (k, budget.max_items) if limit is not None]
    if not limits:
        return None
    return min(limits)


def _select_truncate(
    eligible: list[_IndexedCandidate],
    k: int | None,
    budget: ContextBudget,
) -> tuple[list[_IndexedCandidate], list[_IndexedCandidate], list[str]]:
    selected: list[_IndexedCandidate] = []
    rejected: list[_IndexedCandidate] = []
    reasons: list[str] = []
    limit = _item_limit(k, budget)
    used_tokens = 0

    for position, item in enumerate(eligible):
        limit_reason = _limit_reason(len(selected), limit, k, budget)
        if limit_reason is not None:
            rejected.extend(eligible[position:])
            reasons.append(limit_reason)
            break
        if budget.max_tokens is not None and used_tokens + item.tokens > budget.max_tokens:
            rejected.extend(eligible[position:])
            reasons.append("max_tokens")
            break
        selected.append(item)
        used_tokens += item.tokens
    return selected, rejected, reasons


def _select_raise(
    eligible: list[_IndexedCandidate],
    k: int | None,
    budget: ContextBudget,
) -> tuple[list[_IndexedCandidate], list[_IndexedCandidate], list[str]]:
    selected: list[_IndexedCandidate] = []
    rejected: list[_IndexedCandidate] = []
    reasons: list[str] = []
    used_tokens = 0

    for position, item in enumerate(eligible):
        if k is not None and len(selected) >= k:
            rejected.extend(eligible[position:])
            reasons.append("k")
            break
        if budget.max_items is not None and len(selected) >= budget.max_items:
            raise ValueError(
                "ContextBudget.overflow_policy='raise' exceeded "
                f"ContextBudget.max_items={budget.max_items}"
            )
        if budget.max_tokens is not None and used_tokens + item.tokens > budget.max_tokens:
            raise ValueError(
                "ContextBudget.overflow_policy='raise' exceeded "
                f"ContextBudget.max_tokens={budget.max_tokens} for candidate id={item.candidate.id!r}"
            )
        selected.append(item)
        used_tokens += item.tokens
    return selected, rejected, reasons


def _select_drop_lowest_score(
    eligible: list[_IndexedCandidate],
    k: int | None,
    budget: ContextBudget,
) -> tuple[list[_IndexedCandidate], list[_IndexedCandidate], list[str]]:
    limit = _item_limit(k, budget)
    if limit is None:
        preliminary = list(eligible)
        unselected: list[_IndexedCandidate] = []
    else:
        preliminary = eligible[:limit]
        unselected = eligible[limit:]

    selected = list(preliminary)
    dropped: list[_IndexedCandidate] = []
    reasons: list[str] = []
    if unselected:
        reasons.append(_limit_name(k, budget))

    if budget.max_tokens is not None:
        while selected and sum(item.tokens for item in selected) > budget.max_tokens:
            drop_index = min(
                range(len(selected)),
                key=lambda position: _drop_key(selected[position]),
            )
            dropped.append(selected.pop(drop_index))
        if dropped:
            reasons.append("drop_lowest_score:max_tokens")

    rejected = dropped + unselected
    return selected, rejected, reasons


def _drop_key(item: _IndexedCandidate) -> tuple[float, int]:
    score = item.candidate.score
    score_value = float("-inf") if score is None else score
    return (score_value, -item.index)


def _limit_reason(
    selected_count: int,
    limit: int | None,
    k: int | None,
    budget: ContextBudget,
) -> str | None:
    if limit is None or selected_count < limit:
        return None
    return _limit_name(k, budget)


def _limit_name(k: int | None, budget: ContextBudget) -> str:
    if k is not None and budget.max_items is not None and k == budget.max_items:
        return "k/max_items"
    if k is not None and (budget.max_items is None or k <= budget.max_items):
        return "k"
    return "max_items"


def _join_reasons(reasons: list[str]) -> str | None:
    if not reasons:
        return None
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return "; ".join(unique)


__all__ = [
    "ContextBudget",
    "ContextCandidate",
    "ContextSelection",
    "EvidenceResult",
    "route_top_k_context",
]
