"""Deterministic extraction evaluation helpers for ResearchFS."""

from __future__ import annotations

import re
from typing import Any

JSONDict = dict[str, Any]
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NAME_MATCH_THRESHOLD = 0.5


def _fetch_dicts(conn: Any, query: str, params: tuple[Any, ...] = ()) -> list[JSONDict]:
    cur = conn.execute(query, params)
    columns = [column[0] for column in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(name.lower()))


def _normalize_name(name: str) -> str:
    return " ".join(_name_tokens(name))


def _name_slug(name: str) -> str:
    return "-".join(_name_tokens(name))


def _name_match_score(expected_name: str, extracted_name: str) -> float:
    expected_slug = _name_slug(expected_name)
    extracted_slug = _name_slug(extracted_name)
    if expected_slug and expected_slug == extracted_slug:
        return 2.0

    expected_tokens = set(_name_tokens(expected_name))
    extracted_tokens = set(_name_tokens(extracted_name))
    if not expected_tokens or not extracted_tokens:
        return 0.0

    overlap = expected_tokens & extracted_tokens
    if not overlap:
        return 0.0

    jaccard = len(overlap) / len(expected_tokens | extracted_tokens)
    smaller_coverage = len(overlap) / min(len(expected_tokens), len(extracted_tokens))
    expected_coverage = len(overlap) / len(expected_tokens)
    extracted_coverage = len(overlap) / len(extracted_tokens)
    if jaccard >= _NAME_MATCH_THRESHOLD and smaller_coverage >= (2 / 3):
        return 1.0 + jaccard
    if len(expected_tokens) == 1 and expected_tokens <= extracted_tokens and extracted_coverage >= 0.5:
        return 1.0 + jaccard
    if len(extracted_tokens) == 1 and extracted_tokens <= expected_tokens and expected_coverage >= 0.5:
        return 1.0 + jaccard
    return 0.0


def _ratio(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
    if denominator <= 0:
        return empty_value
    return numerator / denominator


def _coerce_int_list(raw: Any, field: str, index: int, *, positive: bool = False) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"expected_methods[{index}].{field} must be a list")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"expected_methods[{index}].{field} must contain integers")
        if positive and value <= 0:
            raise ValueError(f"expected_methods[{index}].{field} must contain positive integers")
        if not positive and value < 0:
            raise ValueError(f"expected_methods[{index}].{field} must contain non-negative integers")
        if value not in values:
            values.append(value)
    return values


def _as_expected_method(raw: Any, index: int) -> JSONDict:
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise ValueError(f"expected_methods[{index}] must be a non-empty method name")
        return {"name": name, "source_chunk_indices": [], "source_chunk_ids": []}
    if not isinstance(raw, dict):
        raise ValueError(f"expected_methods[{index}] must be a string or object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"expected_methods[{index}].name must be a non-empty string")

    expected = dict(raw)
    expected["name"] = name.strip()
    expected["source_chunk_indices"] = _coerce_int_list(
        raw.get("source_chunk_indices"), "source_chunk_indices", index
    )
    expected["source_chunk_ids"] = _coerce_int_list(
        raw.get("source_chunk_ids"), "source_chunk_ids", index, positive=True
    )
    for key in ("source_contains", "evidence_contains"):
        if key not in expected or expected[key] is None:
            expected[key] = []
            continue
        if not isinstance(expected[key], list):
            raise ValueError(f"expected_methods[{index}].{key} must be a list")
        needles: list[str] = []
        for value in expected[key]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"expected_methods[{index}].{key} must contain non-empty strings")
            needles.append(value.strip())
        expected[key] = needles
    return expected


def _load_paper_extractions(conn: Any, paper_id: int) -> tuple[list[JSONDict], dict[int, list[JSONDict]]]:
    paper = conn.execute("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if paper is None:
        raise LookupError(f"No paper found for id {paper_id}")

    chunk_rows = _fetch_dicts(
        conn,
        """
        SELECT chunk_id
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        """,
        (paper_id,),
    )
    chunk_index_by_id = {int(row["chunk_id"]): index for index, row in enumerate(chunk_rows)}

    methods = _fetch_dicts(
        conn,
        """
        SELECT DISTINCT
            m.method_id,
            m.name,
            m.description,
            m.implementation_hint,
            m.risks,
            m.category
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        JOIN methods m ON m.method_id = ms.method_id
        WHERE c.paper_id = ?
        ORDER BY m.method_id
        """,
        (paper_id,),
    )
    sources = _fetch_dicts(
        conn,
        """
        SELECT
            ms.method_id,
            ms.chunk_id,
            c.content,
            ms.evidence_text,
            ms.evidence_start,
            ms.evidence_end,
            ms.evidence_strategy
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        ORDER BY ms.method_id, ms.chunk_id
        """,
        (paper_id,),
    )
    by_method: dict[int, list[JSONDict]] = {}
    for source in sources:
        source["source_chunk_index"] = chunk_index_by_id.get(int(source["chunk_id"]))
        by_method.setdefault(int(source["method_id"]), []).append(source)
    return methods, by_method


def _contains_all(haystacks: list[str], needles: list[str]) -> bool:
    lowered = [haystack.lower() for haystack in haystacks]
    for needle in needles:
        needle_lower = needle.lower()
        if not any(needle_lower in haystack for haystack in lowered):
            return False
    return True


def _source_expectation_counts(expected: JSONDict, sources: list[JSONDict]) -> tuple[int, int, int, int]:
    expected_indices = set(expected.get("source_chunk_indices") or [])
    expected_ids = set(expected.get("source_chunk_ids") or [])
    actual_indices = {
        int(source["source_chunk_index"])
        for source in sources
        if source.get("source_chunk_index") is not None
    }
    actual_ids = {int(source["chunk_id"]) for source in sources}
    evidence_indices = {
        int(source["source_chunk_index"])
        for source in sources
        if source.get("source_chunk_index") is not None and _source_has_evidence(source)
    }
    evidence_ids = {int(source["chunk_id"]) for source in sources if _source_has_evidence(source)}

    source_expectations = len(expected_indices) + len(expected_ids)
    source_hits = len(expected_indices & actual_indices) + len(expected_ids & actual_ids)
    evidence_expectations = source_expectations
    evidence_hits = len(expected_indices & evidence_indices) + len(expected_ids & evidence_ids)

    if expected.get("source_contains"):
        source_expectations += 1
        chunk_texts = [str(source.get("content") or "") for source in sources]
        if _contains_all(chunk_texts, expected["source_contains"]):
            source_hits += 1
    if expected.get("evidence_contains"):
        evidence_expectations += 1
        evidence_texts = [str(source.get("evidence_text") or "") for source in sources]
        if _contains_all(evidence_texts, expected["evidence_contains"]):
            evidence_hits += 1

    return source_expectations, source_hits, evidence_expectations, evidence_hits


def _source_has_evidence(source: JSONDict) -> bool:
    if str(source.get("evidence_text") or "").strip():
        return True
    start = source.get("evidence_start")
    end = source.get("evidence_end")
    return start is not None and end is not None and int(end) > int(start)


def _source_text_expectation_met(expected: JSONDict, sources: list[JSONDict]) -> bool | None:
    checks: list[bool] = []
    if expected.get("source_contains"):
        chunk_texts = [str(source.get("content") or "") for source in sources]
        checks.append(_contains_all(chunk_texts, expected["source_contains"]))
    if expected.get("evidence_contains"):
        evidence_texts = [str(source.get("evidence_text") or "") for source in sources]
        checks.append(_contains_all(evidence_texts, expected["evidence_contains"]))
    if not checks:
        return None
    return all(checks)


def _match_methods(expected: list[JSONDict], methods: list[JSONDict]) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for expected_index, item in enumerate(expected):
        for method_index, method in enumerate(methods):
            score = _name_match_score(item["name"], str(method["name"]))
            if score > 0.0:
                candidates.append((-score, expected_index, method_index))
    candidates.sort()

    matched_expected: set[int] = set()
    matched_methods: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _negative_score, expected_index, method_index in candidates:
        if expected_index in matched_expected or method_index in matched_methods:
            continue
        matched_expected.add(expected_index)
        matched_methods.add(method_index)
        matches.append((expected_index, method_index))
    matches.sort()
    return matches


def _fetch_confidence(conn: Any, paper_id: int) -> tuple[Any, Any]:
    row = conn.execute(
        """
        SELECT confidence_score, confidence_reason
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    if row is None:
        return None, None
    if hasattr(row, "keys"):
        return row["confidence_score"], row["confidence_reason"]
    return row[0], row[1]


def _threshold_result(threshold: float | None, observed: float) -> JSONDict | None:
    if threshold is None:
        return None
    return {"threshold": threshold, "observed": observed, "passed": observed >= threshold}


def evaluate_extraction(
    conn: Any,
    paper_id: int,
    expected_methods: list[Any],
    min_recall: float | None = 1.0,
    min_evidence_coverage: float | None = 0.5,
    notes: str | None = None,
    *,
    min_precision: float | None = None,
    min_f1: float | None = None,
    min_source_coverage: float | None = None,
    require_source_coverage: bool = False,
) -> JSONDict:
    """Compare expected extraction targets with methods stored for one paper.

    ``expected_methods`` must be a non-empty list of strings or dictionaries with
    ``name`` plus optional zero-based ``source_chunk_indices``.  The evaluator is
    deterministic, uses only SQLite rows already present in ResearchFS, and never
    recomputes or mutates extraction confidence scores.
    """
    if isinstance(paper_id, bool) or not isinstance(paper_id, int) or paper_id <= 0:
        raise ValueError("paper_id must be a positive integer")
    validated_paper_id = paper_id
    if not isinstance(expected_methods, list) or not expected_methods:
        raise ValueError("expected_methods must be a non-empty list")

    expected = [_as_expected_method(raw, index) for index, raw in enumerate(expected_methods)]
    methods, sources_by_method = _load_paper_extractions(conn, validated_paper_id)
    matches = _match_methods(expected, methods)

    matched_expected_indices = {expected_index for expected_index, _method_index in matches}
    matched_method_indices = {method_index for _expected_index, method_index in matches}
    matched_method_ids = {int(methods[method_index]["method_id"]) for method_index in matched_method_indices}

    explicit_source_expectations = 0
    explicit_source_hits = 0
    explicit_evidence_expectations = 0
    explicit_evidence_hits = 0
    matched_source_count = 0
    matched_evidence_count = 0
    matched_methods: list[JSONDict] = []

    for expected_index, method_index in matches:
        item = expected[expected_index]
        method = methods[method_index]
        method_id = int(method["method_id"])
        sources = sources_by_method.get(method_id, [])
        source_indices = [
            int(source["source_chunk_index"])
            for source in sources
            if source.get("source_chunk_index") is not None
        ]
        if source_indices:
            matched_source_count += 1
        if any(_source_has_evidence(source) for source in sources):
            matched_evidence_count += 1
        source_total, source_hits, evidence_total, evidence_hits = _source_expectation_counts(item, sources)
        explicit_source_expectations += source_total
        explicit_source_hits += source_hits
        explicit_evidence_expectations += evidence_total
        explicit_evidence_hits += evidence_hits
        text_expectation_met = _source_text_expectation_met(item, sources)
        source_expectation_met = None
        if source_total or evidence_total or text_expectation_met is not None:
            source_expectation_met = (source_hits + evidence_hits) == (source_total + evidence_total) and text_expectation_met is not False
        matched_methods.append(
            {
                "expected_name": item["name"],
                "method_id": method_id,
                "method_name": method["name"],
                "source_chunk_indices": source_indices,
                "source_chunk_ids": [int(source["chunk_id"]) for source in sources],
                "source_expectation_met": source_expectation_met,
            }
        )

    missing_methods = [
        item["name"] for index, item in enumerate(expected) if index not in matched_expected_indices
    ]
    extra_methods = [
        str(method["name"]) for index, method in enumerate(methods) if index not in matched_method_indices
    ]
    unexpected_methods = [
        {"method_id": int(method["method_id"]), "name": method["name"]}
        for method in methods
        if int(method["method_id"]) not in matched_method_ids
    ]

    true_positive = len(matches)
    extracted_count = len(methods)
    expected_count = len(expected)
    precision = _ratio(true_positive, extracted_count, 0.0)
    recall = _ratio(true_positive, expected_count, 0.0)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)

    if explicit_source_expectations:
        source_link_coverage = explicit_source_hits / explicit_source_expectations
    else:
        source_link_coverage = _ratio(matched_source_count, true_positive, 0.0)
    if explicit_evidence_expectations:
        evidence_coverage = explicit_evidence_hits / explicit_evidence_expectations
    else:
        evidence_coverage = _ratio(matched_evidence_count, true_positive, 0.0)
    confidence_score, confidence_reason = _fetch_confidence(conn, validated_paper_id)

    threshold_results: JSONDict = {}
    thresholds = {
        "min_precision": _threshold_result(min_precision, precision),
        "min_recall": _threshold_result(min_recall, recall),
        "min_f1": _threshold_result(min_f1, f1),
        "min_source_coverage": _threshold_result(min_source_coverage, source_link_coverage),
        "min_evidence_coverage": _threshold_result(min_evidence_coverage, evidence_coverage),
    }
    for key, value in thresholds.items():
        if value is not None:
            threshold_results[key] = value
    if require_source_coverage:
        threshold_results["require_source_coverage"] = {
            "threshold": 1.0,
            "observed": source_link_coverage,
            "passed": source_link_coverage >= 1.0,
        }
    passed = all(result["passed"] for result in threshold_results.values())

    result: JSONDict = {
        "paper_id": validated_paper_id,
        "expected_count": expected_count,
        "extracted_count": extracted_count,
        "matched_count": true_positive,
        "missing_methods": missing_methods,
        "extra_methods": extra_methods,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "source_link_coverage": source_link_coverage,
        "evidence_coverage": evidence_coverage,
        "confidence_score": confidence_score,
        "confidence_reason": confidence_reason,
        "passed": passed,
        "true_positive": true_positive,
        "false_negative": len(missing_methods),
        "false_positive": len(extra_methods),
        "expected_source_coverage": source_link_coverage,
        "explicit_source_expectations": explicit_source_expectations,
        "explicit_evidence_expectations": explicit_evidence_expectations,
        "matched_methods": matched_methods,
        "unexpected_methods": unexpected_methods,
        "thresholds": threshold_results,
        "notes": notes,
    }
    return result
