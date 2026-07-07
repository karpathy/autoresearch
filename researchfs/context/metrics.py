"""Dependency-free context-routing evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Hashable
from typing import Any


def gold_recall_at_k(selected_ids: Iterable[Any], gold_ids: Iterable[Any], k: int) -> float:
    """Return recall of unique gold IDs found in the first ``k`` selections."""

    k = _validate_int(k, "k", minimum=0)
    selected_window = list(selected_ids)[:k]
    selected = _unique_ids(selected_window, "selected_ids")
    gold = set(_unique_ids(list(gold_ids), "gold_ids"))
    if not gold:
        return 1.0
    return len(set(selected) & gold) / len(gold)


def citation_precision(predicted_citations: Iterable[Any], gold_ids: Iterable[Any]) -> float:
    """Return unique citation precision against supported/gold IDs."""

    predicted = set(_unique_ids(list(predicted_citations), "predicted_citations"))
    gold = set(_unique_ids(list(gold_ids), "gold_ids"))
    if not predicted:
        return 1.0 if not gold else 0.0
    if not gold:
        return 0.0
    return len(predicted & gold) / len(predicted)


def citation_recall(predicted_citations: Iterable[Any], gold_ids: Iterable[Any]) -> float:
    """Return unique citation recall against supported/gold IDs."""

    predicted = set(_unique_ids(list(predicted_citations), "predicted_citations"))
    gold = set(_unique_ids(list(gold_ids), "gold_ids"))
    if not gold:
        return 1.0
    return len(predicted & gold) / len(gold)


def citation_f1(predicted_citations: Iterable[Any], gold_ids: Iterable[Any]) -> float:
    """Return harmonic mean of citation precision and recall."""

    predicted = list(predicted_citations)
    gold = list(gold_ids)
    precision = citation_precision(predicted, gold)
    recall = citation_recall(predicted, gold)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def supported_answer_rate(results: Iterable[Any]) -> float:
    """Return the fraction of answers marked supported.

    Items may be booleans, dictionaries with ``supported``, or objects exposing a
    boolean ``supported`` attribute. Empty input returns ``0.0`` because no
    supported answer was observed.
    """

    values = [_coerce_supported(result, "results") for result in results]
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)

def false_stop_rate(results: Iterable[Any]) -> float:
    """Return the fraction of runs that stopped without enough evidence.

    Items may be booleans, dictionaries, or objects exposing a boolean field
    such as ``false_stop`` or ``stopped_without_enough_evidence``. Empty input
    returns ``0.0`` because no false stop was observed.
    """

    values = [_coerce_false_stop(result, "results") for result in results]
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)



def context_efficiency(
    selected_tokens: int | None = None,
    max_tokens: int | None = None,
    selected_items: int | None = None,
    max_items: int | None = None,
    *,
    answer_supported: bool | None = None,
    tokens_used: int | None = None,
) -> float:
    """Return a bounded score where less context use is better.

    Within a supplied budget, scores are in ``[0.5, 1.0]`` and decrease as usage
    rises. Over-budget usage falls below ``0.5``. Without a budget, the score is
    ``1 / (1 + used)`` for each supplied usage dimension. If
    ``answer_supported`` is supplied and false, efficiency is ``0.0``.
    """

    if answer_supported is not None and not isinstance(answer_supported, bool):
        raise ValueError("answer_supported must be a bool or None")
    if answer_supported is False:
        return 0.0

    if selected_tokens is None and tokens_used is not None:
        selected_tokens = tokens_used
    elif selected_tokens is not None and tokens_used is not None and selected_tokens != tokens_used:
        raise ValueError("selected_tokens and tokens_used must agree when both are provided")

    dimensions: list[float] = []
    if selected_tokens is not None or max_tokens is not None:
        used_tokens = _validate_int(
            0 if selected_tokens is None else selected_tokens,
            "selected_tokens",
            minimum=0,
        )
        token_budget = _validate_optional_int(max_tokens, "max_tokens", minimum=0)
        dimensions.append(_usage_efficiency(used_tokens, token_budget))
    if selected_items is not None or max_items is not None:
        used_items = _validate_int(
            0 if selected_items is None else selected_items,
            "selected_items",
            minimum=0,
        )
        item_budget = _validate_optional_int(max_items, "max_items", minimum=0)
        dimensions.append(_usage_efficiency(used_items, item_budget))

    if not dimensions:
        return 1.0
    return _bounded(sum(dimensions) / len(dimensions))


def recovery_success(
    attempts: Iterable[Any] | None = None,
    *,
    first_attempt_supported: bool | None = None,
    final_attempt_supported: bool | None = None,
) -> float:
    """Return ``1.0`` when recovery produced a supported final state.

    ``attempts`` may contain booleans, dictionaries, or objects with recovery- or
    success-like boolean fields. Keyword arguments support before/after recovery
    checks used by callers that track first and final answer support directly.
    """

    if first_attempt_supported is not None or final_attempt_supported is not None:
        if not isinstance(first_attempt_supported, bool):
            raise ValueError("first_attempt_supported must be provided as a bool")
        if not isinstance(final_attempt_supported, bool):
            raise ValueError("final_attempt_supported must be provided as a bool")
        return 1.0 if final_attempt_supported else 0.0

    if attempts is None:
        raise ValueError("attempts must be provided")
    values = _recovery_values(attempts, "attempts")
    return 1.0 if any(values) else 0.0


def recovery_rate(attempts: Iterable[Any]) -> float:
    """Return the fraction of recovery attempts that succeeded."""

    values = _recovery_values(attempts, "attempts")
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _unique_ids(values: list[Any], field_name: str) -> list[Hashable]:
    unique: list[Hashable] = []
    seen: set[Hashable] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Hashable):
            raise ValueError(f"{field_name}[{index}] must be hashable")
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _validate_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{field_name} must be non-negative")
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _validate_optional_int(value: Any, field_name: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _validate_int(value, field_name, minimum=minimum)


def _usage_efficiency(used: int, maximum: int | None) -> float:
    if maximum is None:
        return 1.0 / (1.0 + used)
    if maximum == 0:
        return 1.0 if used == 0 else 0.0
    ratio = used / maximum
    if ratio <= 1.0:
        return 1.0 - (0.5 * ratio)
    return 0.5 / ratio


def _bounded(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _recovery_values(attempts: Any, field_name: str) -> list[bool]:
    if isinstance(attempts, (bool, dict)) or _has_recovery_field(attempts):
        return [_coerce_recovery(attempts, field_name)]
    try:
        return [_coerce_recovery(attempt, field_name) for attempt in attempts]
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be a recovery attempt or an iterable of recovery attempts"
        ) from exc


def _has_recovery_field(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    for name in _RECOVERY_FIELD_NAMES:
        if hasattr(value, name):
            return True
    return False


_FALSE_STOP_FIELD_NAMES = (
    "false_stop",
    "false_stopped",
    "stopped_without_enough_evidence",
    "stopped_without_evidence",
    "stopped_without_support",
    "insufficient_evidence_stop",
)


_RECOVERY_FIELD_NAMES = (
    "recovery",
    "recovered",
    "recovery_success",
    "succeeded",
    "success",
    "supported",
    "final_attempt_supported",
)


def _coerce_supported(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and "supported" in value:
        supported = value["supported"]
    elif hasattr(value, "supported"):
        supported = getattr(value, "supported")
    else:
        raise ValueError(f"{field_name} items must be bools or expose .supported")
    if not isinstance(supported, bool):
        raise ValueError(f"{field_name}.supported must be a bool")
    return supported


def _coerce_false_stop(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    for name in _FALSE_STOP_FIELD_NAMES:
        if isinstance(value, dict) and name in value:
            return _coerce_bool(value[name], f"{field_name}.{name}")
        if hasattr(value, name):
            return _coerce_bool(getattr(value, name), f"{field_name}.{name}")
    raise ValueError(f"{field_name} items must be bools or expose a false-stop field")


def _coerce_recovery(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    for name in _RECOVERY_FIELD_NAMES:
        if isinstance(value, dict) and name in value:
            return _coerce_bool(value[name], f"{field_name}.{name}")
        if hasattr(value, name):
            return _coerce_bool(getattr(value, name), f"{field_name}.{name}")
    raise ValueError(f"{field_name} items must be bools or expose a recovery/succeeded field")


def _coerce_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


__all__ = [
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "context_efficiency",
    "false_stop_rate",
    "gold_recall_at_k",
    "recovery_rate",
    "recovery_success",
    "supported_answer_rate",
]
