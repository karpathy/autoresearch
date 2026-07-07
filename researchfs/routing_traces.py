"""Export routed ask retrieval events as deterministic per-candidate traces."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TextIO

from researchfs.db import get_connection
import researchfs.retrieve as rfs


ROUTING_TRACE_SCHEMA_VERSION = 1
DEFAULT_ROUTING_TRACE_EVENT_TYPES = ("routed_ask",)




def iter_routing_traces(
    conn=None,
    *,
    event_types: tuple[str, ...] | list[str] | None = None,
    agent_run_id: int | None = None,
    experiment_id: int | None = None,
    since_event_id: int | None = None,
    limit: int | None = None,
    include_candidate_text: bool = True,
    max_candidate_chars: int = 2000,
) -> Iterator[dict[str, Any]]:
    """Return an iterator of per-candidate routing trace rows."""

    options = _validate_export_options(
        event_types=event_types,
        agent_run_id=agent_run_id,
        experiment_id=experiment_id,
        since_event_id=since_event_id,
        limit=limit,
        include_candidate_text=include_candidate_text,
        max_candidate_chars=max_candidate_chars,
    )
    traces, _event_count, _skipped_events = _collect_routing_traces(conn, options)
    return iter(traces)


def export_routing_traces(
    conn=None,
    *,
    event_types: tuple[str, ...] | list[str] | None = None,
    agent_run_id: int | None = None,
    experiment_id: int | None = None,
    since_event_id: int | None = None,
    limit: int | None = None,
    include_candidate_text: bool = True,
    max_candidate_chars: int = 2000,
) -> dict[str, Any]:
    """Return routing trace rows plus skipped-event summaries."""

    options = _validate_export_options(
        event_types=event_types,
        agent_run_id=agent_run_id,
        experiment_id=experiment_id,
        since_event_id=since_event_id,
        limit=limit,
        include_candidate_text=include_candidate_text,
        max_candidate_chars=max_candidate_chars,
    )
    traces, event_count, skipped_events = _collect_routing_traces(conn, options)
    return {
        "ok": True,
        "schema_version": ROUTING_TRACE_SCHEMA_VERSION,
        "trace_count": len(traces),
        "event_count": event_count,
        "skipped_event_count": len(skipped_events),
        "skipped_events": skipped_events,
        "traces": traces,
    }


def write_routing_traces_jsonl(
    traces: Iterable[Mapping[str, Any]], stream: TextIO
) -> int:
    """Write routing traces as compact JSONL and return the row count."""

    count = 0
    for trace in traces:
        json.dump(trace, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        count += 1
    return count


def _validate_export_options(
    *,
    event_types: tuple[str, ...] | list[str] | None,
    agent_run_id: int | None,
    experiment_id: int | None,
    since_event_id: int | None,
    limit: int | None,
    include_candidate_text: bool,
    max_candidate_chars: int,
) -> dict[str, Any]:
    normalized_event_types = _normalize_event_types(event_types)
    normalized_agent_run_id = _validate_positive_int_or_none(
        agent_run_id, "agent_run_id"
    )
    normalized_experiment_id = _validate_positive_int_or_none(
        experiment_id, "experiment_id"
    )
    normalized_since_event_id = _validate_positive_int_or_none(
        since_event_id, "since_event_id"
    )
    normalized_limit = _validate_positive_int_or_none(limit, "limit")
    if not isinstance(include_candidate_text, bool):
        raise ValueError("include_candidate_text must be a bool")
    normalized_max_candidate_chars = _validate_non_negative_int(
        max_candidate_chars, "max_candidate_chars"
    )
    return {
        "event_types": normalized_event_types,
        "agent_run_id": normalized_agent_run_id,
        "experiment_id": normalized_experiment_id,
        "since_event_id": normalized_since_event_id,
        "limit": normalized_limit,
        "include_candidate_text": include_candidate_text,
        "max_candidate_chars": normalized_max_candidate_chars,
    }


def _normalize_event_types(event_types: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if event_types is None:
        return DEFAULT_ROUTING_TRACE_EVENT_TYPES
    if not isinstance(event_types, (list, tuple)):
        raise ValueError("event_types must be a non-empty sequence of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for event_type in event_types:
        if not isinstance(event_type, str):
            raise ValueError("event_types must contain only strings")
        label = event_type.strip()
        if not label:
            raise ValueError("event_types must contain non-empty strings")
        if label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    if not normalized:
        raise ValueError("event_types must contain at least one event type")
    return tuple(normalized)


def _validate_positive_int_or_none(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _collect_routing_traces(conn, options: Mapping[str, Any]):
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        rows = _query_retrieval_events(conn, options)
        memory_cache: dict[tuple[int, int], dict[str, Any]] = {}
        traces: list[dict[str, Any]] = []
        skipped_events: list[dict[str, Any]] = []
        for row in rows:
            event_traces, skipped_reason = _trace_rows_for_event(
                row,
                include_candidate_text=options["include_candidate_text"],
                max_candidate_chars=options["max_candidate_chars"],
                memory_cache=memory_cache,
                conn=conn,
            )
            traces.extend(event_traces)
            if skipped_reason is not None:
                skipped_events.append(
                    {"event_id": _event_id_for_skip(row), "reason": skipped_reason}
                )
        return traces, len(rows), skipped_events
    finally:
        if close:
            conn.close()


def _query_retrieval_events(conn, options: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_types = options["event_types"]
    placeholders = ",".join("?" for _ in event_types)
    sql_parts = [
        """
        SELECT event_id, agent_run_id, experiment_id, event_type, query,
               retrieved_content, timestamp
        FROM retrieval_events
        """,
        f"WHERE event_type IN ({placeholders})",
    ]
    params: list[Any] = list(event_types)
    if options["agent_run_id"] is not None:
        sql_parts.append("AND agent_run_id = ?")
        params.append(options["agent_run_id"])
    if options["experiment_id"] is not None:
        sql_parts.append("AND experiment_id = ?")
        params.append(options["experiment_id"])
    if options["since_event_id"] is not None:
        sql_parts.append("AND event_id > ?")
        params.append(options["since_event_id"])
    sql_parts.append("ORDER BY event_id ASC")
    if options["limit"] is not None:
        sql_parts.append("LIMIT ?")
        params.append(options["limit"])

    cur = conn.cursor()
    cur.execute("\n".join(sql_parts), tuple(params))
    columns = [column[0] for column in cur.description]
    return [_row_to_dict(row, columns) for row in cur.fetchall()]


def _row_to_dict(row: Any, columns: list[str]) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    return {column: row[index] for index, column in enumerate(columns)}


def _trace_rows_for_event(
    row: Mapping[str, Any],
    *,
    include_candidate_text: bool,
    max_candidate_chars: int,
    memory_cache: dict[tuple[int, int], dict[str, Any]],
    conn,
) -> tuple[list[dict[str, Any]], str | None]:
    payload, skipped_reason = _load_event_payload(row.get("retrieved_content"))
    if skipped_reason is not None:
        return [], skipped_reason

    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        return [], "payload attempts is not a list"

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    attempt_count = metrics.get("attempt_count") or len(attempts)
    final_citations = _string_list(payload.get("citations"))
    final_supported = bool(payload.get("supported"))
    final_status = payload.get("status") if payload.get("status") is not None else None
    query = payload.get("query") if payload.get("query") is not None else row.get("query")
    recovered = bool(metrics.get("recovered"))
    false_stop = bool(metrics.get("false_stop"))

    traces: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        candidate_ids = attempt.get("candidate_ids")
        reader_context_ids = attempt.get("reader_context_ids")
        if not _is_string_list(candidate_ids) or not _is_string_list(reader_context_ids):
            continue
        attempt_number = attempt.get("attempt")
        routing_event_id = attempt.get("routing_event_id")
        selection_audit = (
            attempt.get("selection_audit")
            if isinstance(attempt.get("selection_audit"), Mapping)
            else {}
        )
        verifier = attempt.get("verifier") if isinstance(attempt.get("verifier"), Mapping) else {}
        attempt_supported = _bool_or_none(verifier.get("supported"))
        attempt_recoverable = _bool_or_none(verifier.get("recoverable"))

        for candidate_id in candidate_ids:
            selected = candidate_id in reader_context_ids
            cited = candidate_id in final_citations
            label = _trace_label(
                selected=selected,
                cited=cited,
                final_supported=final_supported,
                attempt_supported=attempt_supported,
                attempt_recoverable=attempt_recoverable,
            )
            traces.append(
                {
                    "schema_version": ROUTING_TRACE_SCHEMA_VERSION,
                    "trace_id": (
                        f"event:{row['event_id']}:attempt:{attempt_number}:"
                        f"candidate:{candidate_id}"
                    ),
                    "event_id": row["event_id"],
                    "routing_event_id": routing_event_id,
                    "agent_run_id": row.get("agent_run_id"),
                    "experiment_id": row.get("experiment_id"),
                    "timestamp": row.get("timestamp"),
                    "event_type": row.get("event_type"),
                    "query": query,
                    "attempt": attempt_number,
                    "attempt_count": attempt_count,
                    "max_papers": attempt.get("max_papers"),
                    "candidate_id": candidate_id,
                    "candidate_ids": list(candidate_ids),
                    "reader_context_ids": list(reader_context_ids),
                    "selected": selected,
                    "rejected": not selected,
                    "cited": cited,
                    "citation_label": final_supported and cited,
                    "answer_status": attempt.get("answer_status"),
                    "attempt_supported": attempt_supported,
                    "attempt_recoverable": attempt_recoverable,
                    "final_status": final_status,
                    "final_supported": final_supported,
                    "final_citations": list(final_citations),
                    "recovered": recovered,
                    "false_stop": false_stop,
                    "selected_tokens": selection_audit.get("selected_tokens"),
                    "token_budget": selection_audit.get("token_budget"),
                    "selection_reason": selection_audit.get("reason"),
                    "label": label,
                    "candidate": _candidate_payload(
                        candidate_id,
                        include_candidate_text=include_candidate_text,
                        max_candidate_chars=max_candidate_chars,
                        memory_cache=memory_cache,
                        conn=conn,
                    ),
                }
            )
    return traces, None


def _load_event_payload(retrieved_content: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(retrieved_content, str):
        return {}, "retrieved_content is not a JSON string"
    try:
        payload = json.loads(retrieved_content)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return {}, "payload is not a JSON object"
    if payload.get("truncated") is True:
        return {}, "truncated payload"
    if payload.get("ids_truncated") is True:
        return {}, "ids_truncated payload"
    if "attempts" not in payload:
        return {}, "payload missing attempts"
    return payload, None


def _trace_label(
    *,
    selected: bool,
    cited: bool,
    final_supported: bool,
    attempt_supported: bool | None,
    attempt_recoverable: bool | None,
) -> str:
    if not selected:
        return "candidate_not_selected"
    if final_supported and cited:
        return "cited_supported"
    if attempt_supported is True:
        return "selected_supported_uncited"
    if attempt_supported is False and attempt_recoverable is True:
        return "selected_recoverable_miss"
    return "selected_unsupported"


def _candidate_payload(
    candidate_id: str,
    *,
    include_candidate_text: bool,
    max_candidate_chars: int,
    memory_cache: dict[tuple[int, int], dict[str, Any]],
    conn,
) -> dict[str, Any]:
    paper_id = _positive_int_from_candidate_id(candidate_id)
    payload: dict[str, Any] = {
        "id": candidate_id,
        "paper_id": paper_id,
        "title": None,
        "source": None,
        "text": None,
        "method_ids": [],
        "chunk_ids": [],
        "lesson_ids": [],
        "lookup_error": None,
    }
    if paper_id is None or not include_candidate_text:
        return payload

    cache_key = (paper_id, max_candidate_chars)
    if cache_key not in memory_cache:
        memory_cache[cache_key] = _lookup_paper_memory(
            paper_id, conn=conn, max_candidate_chars=max_candidate_chars
        )
    memory = memory_cache[cache_key]
    payload.update(
        {
            "title": memory.get("title"),
            "source": memory.get("source"),
            "text": memory.get("text"),
            "method_ids": list(memory.get("method_ids", [])),
            "chunk_ids": list(memory.get("chunk_ids", [])),
            "lesson_ids": list(memory.get("lesson_ids", [])),
            "lookup_error": memory.get("lookup_error"),
        }
    )
    return payload


def _lookup_paper_memory(
    paper_id: int, *, conn, max_candidate_chars: int
) -> dict[str, Any]:
    try:
        memory = rfs.get_paper_memory(
            paper_id, conn=conn, max_chars=max_candidate_chars
        )
    except LookupError as exc:
        return {
            "title": None,
            "source": None,
            "text": None,
            "method_ids": [],
            "chunk_ids": [],
            "lesson_ids": [],
            "lookup_error": str(exc),
        }

    context = memory.get("context") if isinstance(memory.get("context"), str) else None
    extraction_status = (
        memory.get("extraction_status")
        if isinstance(memory.get("extraction_status"), Mapping)
        else {}
    )
    source = extraction_status.get("source_type")
    return {
        "title": _title_from_paper_memory_context(context),
        "source": source if isinstance(source, str) else None,
        "text": context,
        "method_ids": _int_list(memory.get("method_ids")),
        "chunk_ids": _int_list(memory.get("chunk_ids")),
        "lesson_ids": _int_list(memory.get("lesson_ids")),
        "lookup_error": None,
    }


def _positive_int_from_candidate_id(candidate_id: str) -> int | None:
    try:
        paper_id = int(candidate_id)
    except (TypeError, ValueError):
        return None
    if paper_id <= 0:
        return None
    return paper_id


def _title_from_paper_memory_context(context: str | None) -> str | None:
    if not context:
        return None
    first_line = context.splitlines()[0] if context.splitlines() else ""
    prefix = "# Paper Memory: "
    if not first_line.startswith(prefix):
        return None
    title = first_line[len(prefix) :].strip()
    return title or None


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _string_list(value: Any) -> list[str]:
    if not _is_string_list(value):
        return []
    return list(value)


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _event_id_for_skip(row: Mapping[str, Any]) -> int:
    event_id = row.get("event_id")
    if isinstance(event_id, int) and not isinstance(event_id, bool):
        return event_id
    return 0
