"""Compressed paper memory bundle import/export helpers for ResearchFS."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from researchfs.db import get_connection, initialize_db
from researchfs.retrieve import get_method_feedback_summary, get_paper_memory

BUNDLE_KIND = "researchfs.paper_memory_bundle"
CORPUS_BUNDLE_KIND = "researchfs.corpus_memory_bundle"
BUNDLE_VERSION = 1
MAX_EVIDENCE_CHARS = 1000

JSONDict = dict[str, Any]


def _compact_text(value: Any, max_chars: int = MAX_EVIDENCE_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _row_to_dict(row: Any | None) -> JSONDict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_source_int(source: Mapping[str, Any], field: str) -> int:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Bundle method source {field} must be an integer when present")
    return value


def _validate_source_provenance(source: Any) -> None:
    if not isinstance(source, Mapping):
        raise ValueError("Bundle method.sources must contain objects")
    if "content" in source:
        raise ValueError("Bundle method sources must not include raw content")
    evidence_text = source.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError("Bundle method source evidence_text must be a non-empty string")
    has_start = "evidence_start" in source
    has_end = "evidence_end" in source
    if has_start != has_end:
        raise ValueError(
            "Bundle method source evidence_start and evidence_end must both be present when either is present"
        )
    if has_start:
        start = _require_source_int(source, "evidence_start")
        end = _require_source_int(source, "evidence_end")
        if end <= start:
            raise ValueError("Bundle method source evidence_end must be greater than evidence_start")
    if "evidence_strategy" in source:
        strategy = source.get("evidence_strategy")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("Bundle method source evidence_strategy must be a non-empty string when present")


def _jsonish_dict(value: Any) -> JSONDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _jsonish_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _connect(conn: Any | None) -> tuple[Any, bool]:
    if conn is not None:
        return conn, False
    owned = get_connection()
    initialize_db(owned)
    return owned, True


def _load_paper(conn: Any, paper_id: int) -> JSONDict:
    row = conn.execute(
        """
        SELECT paper_id, title, authors, abstract, url, publish_date, created_at
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"No paper found for id {paper_id}")
    return dict(row)


def _load_extraction_status(conn: Any, paper_id: int) -> JSONDict | None:
    row = conn.execute(
        """
        SELECT
            paper_id,
            status,
            source_type,
            chunk_count,
            method_count,
            source_char_count,
            compressed_char_count,
            compression_ratio,
            confidence_score,
            confidence_reason,
            extractor_model,
            extractor_prompt_version,
            error_text,
            created_at,
            updated_at
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    return _row_to_dict(row)


def _feedback_notes(conn: Any, method_id: int, limit: int = 2) -> list[JSONDict]:
    rows = conn.execute(
        """
        SELECT feedback_id, rating, label, note, source, timestamp
        FROM method_feedback
        WHERE method_id = ?
        ORDER BY timestamp DESC, feedback_id DESC
        LIMIT ?
        """,
        (method_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def _source_rows(conn: Any, paper_id: int) -> list[JSONDict]:
    rows = conn.execute(
        """
        SELECT
            ms.method_id,
            ms.chunk_id,
            c.section,
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
    ).fetchall()
    sources: list[JSONDict] = []
    for row in rows:
        evidence_text = _compact_text(row["evidence_text"])
        if evidence_text is None:
            continue
        source = {
            "source_method_id": row["method_id"],
            "source_chunk_id": row["chunk_id"],
            "section": row["section"],
            "evidence_text": evidence_text,
            "evidence_start": row["evidence_start"],
            "evidence_end": row["evidence_end"],
            "evidence_strategy": row["evidence_strategy"],
        }
        sources.append({key: value for key, value in source.items() if value is not None})
    return sources


def _method_rows(conn: Any, paper_id: int) -> list[JSONDict]:
    rows = conn.execute(
        """
        SELECT DISTINCT
            m.method_id,
            m.name,
            m.description,
            m.implementation_hint,
            m.risks,
            m.category,
            m.created_at
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        JOIN methods m ON m.method_id = ms.method_id
        WHERE c.paper_id = ?
        ORDER BY m.method_id
        """,
        (paper_id,),
    ).fetchall()
    sources_by_method: dict[int, list[JSONDict]] = {}
    for source in _source_rows(conn, paper_id):
        source_method_id = source.get("source_method_id")
        if isinstance(source_method_id, int):
            sources_by_method.setdefault(source_method_id, []).append(source)

    methods: list[JSONDict] = []
    for row in rows:
        method_id = row["method_id"]
        method = {
            "source_method_id": method_id,
            "name": row["name"],
            "description": row["description"],
            "implementation_hint": row["implementation_hint"],
            "risks": row["risks"],
            "category": row["category"],
            "created_at": row["created_at"],
            "sources": sources_by_method.get(method_id, []),
            "feedback_summary": get_method_feedback_summary(method_id, conn=conn),
        }
        notes = _feedback_notes(conn, method_id)
        if notes:
            method["feedback_notes"] = notes
        methods.append(method)
    return methods


def _lesson_rows(conn: Any, method_ids: Iterable[int]) -> list[JSONDict]:
    source_method_ids = list(dict.fromkeys(int(method_id) for method_id in method_ids))
    if not source_method_ids:
        return []
    placeholders = ", ".join("?" for _ in source_method_ids)
    rows = conn.execute(
        f"""
        SELECT lesson_id, method_id, experiment_id, label, content, outcome, created_at
        FROM lessons
        WHERE method_id IN ({placeholders})
        ORDER BY method_id, lesson_id
        """,
        tuple(source_method_ids),
    ).fetchall()
    lessons: list[JSONDict] = []
    for row in rows:
        lesson = {
            "source_lesson_id": row["lesson_id"],
            "source_method_id": row["method_id"],
            "label": row["label"],
            "content": row["content"],
            "outcome": row["outcome"],
            "created_at": row["created_at"],
        }
        if row["experiment_id"] is not None:
            lesson["source_experiment_id"] = row["experiment_id"]
        lessons.append(lesson)
    return lessons


def _required_lesson_text(lesson: Mapping[str, Any], field: str) -> str:
    value = lesson.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Bundle lesson.{field} must be a non-empty string")
    return value


def _optional_lesson_text(lesson: Mapping[str, Any], field: str) -> str | None:
    value = lesson.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Bundle lesson.{field} must be a string when present")
    return value



def _required_positive_paper_id(paper_id: Any, context: str) -> int:
    if isinstance(paper_id, bool) or not isinstance(paper_id, int):
        raise ValueError(f"{context} paper IDs must be positive integers")
    if paper_id <= 0:
        raise ValueError(f"{context} paper IDs must be positive integers")
    return paper_id


def export_paper_memory_bundle(
    paper_id: int,
    conn: Any | None = None,
    max_chars: int | None = None,
) -> JSONDict:
    """Return a JSON-safe compressed bundle for one paper memory.

    The bundle intentionally excludes full paper text and full chunk content. It
    carries only paper metadata, bounded memory context, method metadata,
    provenance evidence snippets, feedback summaries, and source IDs.
    """
    source_paper_id = _required_positive_paper_id(paper_id, "Paper bundle")
    conn, close = _connect(conn)
    try:
        paper = _load_paper(conn, source_paper_id)
        memory = get_paper_memory(source_paper_id, conn=conn, max_chars=max_chars)
        extraction_status = _load_extraction_status(conn, source_paper_id)
        methods = _method_rows(conn, source_paper_id)
        lessons = _lesson_rows(conn, [method["source_method_id"] for method in methods])
        source_chunk_ids: list[int] = []
        for method in methods:
            for source in method.get("sources", []):
                source_chunk_id = source.get("source_chunk_id")
                if isinstance(source_chunk_id, int) and source_chunk_id not in source_chunk_ids:
                    source_chunk_ids.append(source_chunk_id)
        exported_paper = dict(paper)
        exported_paper["source_paper_id"] = exported_paper.pop("paper_id")
        return {
            "kind": BUNDLE_KIND,
            "version": BUNDLE_VERSION,
            "paper": exported_paper,
            "extraction_status": extraction_status,
            "memory": memory,
            "context": memory.get("context"),
            "methods": methods,
            "source_ids": {
                "paper_id": source_paper_id,
                "method_ids": [method["source_method_id"] for method in methods],
                "chunk_ids": source_chunk_ids,
                "lesson_ids": [lesson["source_lesson_id"] for lesson in lessons],
            },
            "lessons": lessons,
            "metadata": {
                "max_chars": max_chars,
                "compression": "evidence_snippets_only",
                "omitted": ["papers.content", "chunks.content"],
            },
        }
    finally:
        if close:
            conn.close()

def _paper_ids_for_corpus_export(conn: Any, paper_ids: Iterable[int] | None) -> list[int]:
    if paper_ids is None:
        rows = conn.execute("SELECT paper_id FROM papers ORDER BY paper_id").fetchall()
        return [int(row["paper_id"]) for row in rows]
    parsed_ids: list[int] = []
    for paper_id in paper_ids:
        parsed_id = _required_positive_paper_id(paper_id, "Corpus bundle")
        if parsed_id not in parsed_ids:
            parsed_ids.append(parsed_id)
    return parsed_ids


def export_corpus_memory_bundle(
    paper_ids: Iterable[int] | None = None,
    conn: Any | None = None,
    max_chars: int | None = None,
) -> JSONDict:
    """Return a JSON-safe compressed bundle for multiple paper memories."""
    conn, close = _connect(conn)
    try:
        resolved_paper_ids = _paper_ids_for_corpus_export(conn, paper_ids)
        papers = [
            export_paper_memory_bundle(paper_id, conn=conn, max_chars=max_chars)
            for paper_id in resolved_paper_ids
        ]
        return {
            "kind": CORPUS_BUNDLE_KIND,
            "version": BUNDLE_VERSION,
            "papers": papers,
            "metadata": {
                "paper_count": len(papers),
                "max_chars": max_chars,
                "compression": "paper_memory_bundles",
                "omitted": ["papers.content", "chunks.content"],
            },
        }
    finally:
        if close:
            conn.close()




def _validate_bundle(bundle: Any) -> JSONDict:
    if not isinstance(bundle, Mapping):
        raise ValueError("Bundle must be a JSON object")
    bundle_dict = dict(bundle)
    if bundle_dict.get("kind") != BUNDLE_KIND:
        raise ValueError(f"Bundle kind must be {BUNDLE_KIND!r}")
    if bundle_dict.get("version") != BUNDLE_VERSION:
        raise ValueError(f"Bundle version must be {BUNDLE_VERSION}")
    paper = bundle_dict.get("paper")
    if not isinstance(paper, Mapping):
        raise ValueError("Bundle paper must be an object")
    if "content" in paper:
        raise ValueError("Bundle paper.content is not allowed in compressed memory bundles")
    if "chunks" in bundle_dict or "chunk_contents" in bundle_dict:
        raise ValueError("Bundle must not include raw chunks.content or chunk_contents")
    title = paper.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Bundle paper.title must be a non-empty string")
    methods = bundle_dict.get("methods")
    if methods is not None and not isinstance(methods, list):
        raise ValueError("Bundle methods must be a list")
    for method in methods or []:
        if not isinstance(method, Mapping):
            raise ValueError("Bundle methods must contain objects")
        name = method.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Bundle method.name must be a non-empty string")
        sources = method.get("sources")
        if sources is not None and not isinstance(sources, list):
            raise ValueError("Bundle method.sources must be a list")
        for source in _jsonish_list(sources):
            _validate_source_provenance(source)
    lessons = bundle_dict.get("lessons")
    if lessons is not None and not isinstance(lessons, list):
        raise ValueError("Bundle lessons must be a list")
    for lesson in lessons or []:
        if not isinstance(lesson, Mapping):
            raise ValueError("Bundle lessons must contain objects")
        _required_lesson_text(lesson, "label")
        _required_lesson_text(lesson, "content")
    return bundle_dict

def _validate_corpus_bundle(bundle: Any) -> JSONDict:
    if not isinstance(bundle, Mapping):
        raise ValueError("Bundle must be a JSON object")
    bundle_dict = dict(bundle)
    if bundle_dict.get("kind") != CORPUS_BUNDLE_KIND:
        raise ValueError(f"Bundle kind must be {CORPUS_BUNDLE_KIND!r}")
    if bundle_dict.get("version") != BUNDLE_VERSION:
        raise ValueError(f"Bundle version must be {BUNDLE_VERSION}")
    papers = bundle_dict.get("papers")
    if not isinstance(papers, list):
        raise ValueError("Corpus bundle papers must be a list")
    bundle_dict["papers"] = [_validate_bundle(paper_bundle) for paper_bundle in papers]
    return bundle_dict




def _find_existing_paper(conn: Any, paper: Mapping[str, Any]) -> int | None:
    url = paper.get("url")
    if isinstance(url, str) and url.strip():
        row = conn.execute("SELECT paper_id FROM papers WHERE url = ?", (url.strip(),)).fetchone()
        if row is not None:
            return int(row["paper_id"])
    row = conn.execute(
        """
        SELECT paper_id
        FROM papers
        WHERE title = ?
          AND COALESCE(authors, '') = COALESCE(?, '')
          AND COALESCE(publish_date, '') = COALESCE(?, '')
        ORDER BY paper_id
        LIMIT 1
        """,
        (paper.get("title"), paper.get("authors"), paper.get("publish_date")),
    ).fetchone()
    return int(row["paper_id"]) if row is not None else None


def _insert_or_reuse_paper(conn: Any, paper: Mapping[str, Any]) -> tuple[int, bool]:
    existing_id = _find_existing_paper(conn, paper)
    if existing_id is not None:
        return existing_id, True
    cur = conn.execute(
        """
        INSERT INTO papers (title, authors, abstract, url, publish_date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(paper.get("title") or "").strip(),
            paper.get("authors"),
            paper.get("abstract"),
            paper.get("url"),
            paper.get("publish_date"),
        ),
    )
    return int(cur.lastrowid), False


def _insert_or_reuse_method(conn: Any, method: Mapping[str, Any]) -> int:
    name = str(method.get("name") or "").strip()
    if not name:
        raise ValueError("Bundle method.name must be a non-empty string")
    conn.execute(
        """
        INSERT OR IGNORE INTO methods (name, description, implementation_hint, risks, category)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            method.get("description"),
            method.get("implementation_hint"),
            method.get("risks"),
            method.get("category"),
        ),
    )
    row = conn.execute("SELECT method_id FROM methods WHERE name = ?", (name,)).fetchone()
    if row is None:  # pragma: no cover - guarded by INSERT/UNIQUE semantics
        raise RuntimeError(f"Could not insert or load method {name!r}")
    return int(row["method_id"])


def _source_iter(methods: Iterable[Any]) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for method in methods:
        if not isinstance(method, Mapping):
            raise ValueError("Bundle methods must contain objects")
        for source in _jsonish_list(method.get("sources")):
            if isinstance(source, Mapping):
                yield method, source


def _insert_or_reuse_chunk(conn: Any, paper_id: int, section: Any, content: str) -> int:
    row = conn.execute(
        """
        SELECT chunk_id
        FROM chunks
        WHERE paper_id = ?
          AND COALESCE(section, '') = COALESCE(?, '')
          AND content = ?
        ORDER BY chunk_id
        LIMIT 1
        """,
        (paper_id, section, content),
    ).fetchone()
    if row is not None:
        return int(row["chunk_id"])
    cur = conn.execute(
        "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
        (paper_id, section, content),
    )
    return int(cur.lastrowid)


def _insert_or_reuse_lesson(conn: Any, lesson: Mapping[str, Any], method_id: int) -> int:
    label = _required_lesson_text(lesson, "label")
    content = _required_lesson_text(lesson, "content")
    outcome = _optional_lesson_text(lesson, "outcome")
    row = conn.execute(
        """
        SELECT lesson_id
        FROM lessons
        WHERE method_id = ?
          AND label = ?
          AND content = ?
          AND (
            (outcome IS NULL AND ? IS NULL)
            OR outcome = ?
          )
        ORDER BY lesson_id
        LIMIT 1
        """,
        (method_id, label, content, outcome, outcome),
    ).fetchone()
    if row is not None:
        return int(row["lesson_id"])
    created_at = lesson.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        created_at = None
    cur = conn.execute(
        """
        INSERT INTO lessons (method_id, experiment_id, label, content, outcome, created_at)
        VALUES (?, NULL, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
        """,
        (method_id, label, content, outcome, created_at),
    )
    return int(cur.lastrowid)


def _import_remapped_lessons(
    conn: Any,
    lessons: Iterable[Any],
    method_id_map: Mapping[str, int],
) -> tuple[list[int], dict[str, int]]:
    lesson_ids: list[int] = []
    lesson_id_map: dict[str, int] = {}
    for lesson in lessons:
        if not isinstance(lesson, Mapping):
            raise ValueError("Bundle lessons must contain objects")
        source_method_id = _optional_int(lesson.get("source_method_id"))
        if source_method_id is None:
            continue
        local_method_id = method_id_map.get(str(source_method_id))
        if local_method_id is None:
            continue
        lesson_id = _insert_or_reuse_lesson(conn, lesson, local_method_id)
        if lesson_id not in lesson_ids:
            lesson_ids.append(lesson_id)
        source_lesson_id = _optional_int(lesson.get("source_lesson_id") or lesson.get("lesson_id"))
        if source_lesson_id is not None:
            lesson_id_map[str(source_lesson_id)] = lesson_id
    return lesson_ids, lesson_id_map



def _status_payload(bundle: Mapping[str, Any], paper_id: int, method_count: int, chunk_count: int) -> JSONDict:
    status = _jsonish_dict(bundle.get("extraction_status"))
    imported_status = status.get("status") if status.get("status") in {"skipped", "extracted", "failed"} else None
    if imported_status is None:
        imported_status = "extracted" if method_count else "skipped"
    source_char_count = _optional_int(status.get("source_char_count")) or 0
    compressed_char_count = _optional_int(status.get("compressed_char_count"))
    if compressed_char_count is None:
        context = bundle.get("context") or _jsonish_dict(bundle.get("memory")).get("context") or ""
        compressed_char_count = len(str(context))
    compression_ratio = status.get("compression_ratio")
    try:
        compression_ratio = float(compression_ratio) if compression_ratio is not None else None
    except (TypeError, ValueError):
        compression_ratio = None
    return {
        "paper_id": paper_id,
        "status": imported_status,
        "source_type": "memory_bundle",
        "chunk_count": chunk_count,
        "method_count": method_count,
        "source_char_count": source_char_count,
        "compressed_char_count": compressed_char_count,
        "compression_ratio": compression_ratio,
        "confidence_score": status.get("confidence_score"),
        "confidence_reason": status.get("confidence_reason"),
        "extractor_model": status.get("extractor_model"),
        "extractor_prompt_version": status.get("extractor_prompt_version"),
        "error_text": status.get("error_text"),
    }


def _upsert_extraction_status(conn: Any, payload: Mapping[str, Any]) -> JSONDict:
    conn.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, source_type, chunk_count, method_count,
            source_char_count, compressed_char_count, compression_ratio,
            confidence_score, confidence_reason,
            extractor_model, extractor_prompt_version, error_text
        )
        VALUES (
            :paper_id, :status, :source_type, :chunk_count, :method_count,
            :source_char_count, :compressed_char_count, :compression_ratio,
            :confidence_score, :confidence_reason,
            :extractor_model, :extractor_prompt_version, :error_text
        )
        ON CONFLICT(paper_id) DO UPDATE SET
            status = excluded.status,
            source_type = excluded.source_type,
            chunk_count = excluded.chunk_count,
            method_count = excluded.method_count,
            source_char_count = excluded.source_char_count,
            compressed_char_count = excluded.compressed_char_count,
            compression_ratio = excluded.compression_ratio,
            confidence_score = excluded.confidence_score,
            confidence_reason = excluded.confidence_reason,
            extractor_model = excluded.extractor_model,
            extractor_prompt_version = excluded.extractor_prompt_version,
            error_text = excluded.error_text,
            updated_at = CURRENT_TIMESTAMP
        """,
        dict(payload),
    )
    row = conn.execute(
        "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
        (payload["paper_id"],),
    ).fetchone()
    return dict(row) if row is not None else dict(payload)


def _append_mapping(mapping: dict[str, list[int]], source_id: Any, local_id: int) -> None:
    parsed_source_id = _optional_int(source_id)
    if parsed_source_id is None:
        return
    key = str(parsed_source_id)
    ids = mapping.setdefault(key, [])
    if local_id not in ids:
        ids.append(local_id)


def _source_id_maps(bundle: Mapping[str, Any], paper_id: int) -> tuple[dict[str, int], dict[str, list[int]], dict[str, int]]:
    source_ids = _jsonish_dict(bundle.get("source_ids"))
    paper = _jsonish_dict(bundle.get("paper"))
    source_paper_id = paper.get("source_paper_id") or source_ids.get("paper_id")
    paper_id_map = (
        {str(source_paper_id): paper_id}
        if _optional_int(source_paper_id) is not None
        else {}
    )
    return paper_id_map, {}, {}




def _existing_bundle_links(conn: Any, paper_id: int, methods: list[Any]) -> tuple[list[int], list[int], dict[str, int], dict[str, list[int]]]:
    method_ids: list[int] = []
    chunk_ids: list[int] = []
    method_id_map: dict[str, int] = {}
    chunk_id_map: dict[str, list[int]] = {}
    for method in methods:
        if not isinstance(method, Mapping):
            raise ValueError("Bundle methods must contain objects")
        name = str(method.get("name") or "").strip()
        if not name:
            raise ValueError("Bundle method.name must be a non-empty string")
        method_row = conn.execute("SELECT method_id FROM methods WHERE name = ?", (name,)).fetchone()
        if method_row is None:
            continue
        method_id = int(method_row["method_id"])
        if method_id not in method_ids:
            method_ids.append(method_id)
        source_method_id = _optional_int(method.get("source_method_id") or method.get("method_id"))
        if source_method_id is not None:
            method_id_map[str(source_method_id)] = method_id
        for source in _jsonish_list(method.get("sources")):
            if not isinstance(source, Mapping):
                continue
            source_chunk_id = _optional_int(source.get("source_chunk_id") or source.get("chunk_id"))
            row = None
            if source_chunk_id is not None:
                row = conn.execute(
                    """
                    SELECT c.chunk_id
                    FROM method_sources ms
                    JOIN chunks c ON c.chunk_id = ms.chunk_id
                    WHERE ms.method_id = ?
                      AND c.paper_id = ?
                      AND c.chunk_id = ?
                    """,
                    (method_id, paper_id, source_chunk_id),
                ).fetchone()
            if row is None:
                evidence_text = _compact_text(source.get("evidence_text"))
                section = source.get("section")
                if evidence_text is None:
                    continue
                row = conn.execute(
                    """
                    SELECT c.chunk_id
                    FROM method_sources ms
                    JOIN chunks c ON c.chunk_id = ms.chunk_id
                    WHERE ms.method_id = ?
                      AND c.paper_id = ?
                      AND COALESCE(c.section, '') = COALESCE(?, '')
                      AND c.content = ?
                    ORDER BY c.chunk_id
                    LIMIT 1
                    """,
                    (method_id, paper_id, section, evidence_text),
                ).fetchone()
            if row is None:
                continue
            chunk_id = int(row["chunk_id"])
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
            _append_mapping(chunk_id_map, source.get("source_chunk_id") or source.get("chunk_id"), chunk_id)
    return method_ids, chunk_ids, method_id_map, chunk_id_map


def _current_extraction_status(conn: Any, paper_id: int) -> JSONDict | None:
    row = conn.execute(
        "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def import_paper_memory_bundle(bundle: Mapping[str, Any], conn: Any | None = None, *, commit: bool = True) -> JSONDict:
    """Import a compressed paper memory bundle into the local database.

    Imported chunks are synthetic and contain only evidence snippets from the
    bundle, never original paper text. Existing papers and methods are reused
    when exact paper metadata or method-name uniqueness matches.
    """
    bundle_dict = _validate_bundle(bundle)
    conn, close = _connect(conn)
    try:
        paper = _jsonish_dict(bundle_dict["paper"])
        methods = _jsonish_list(bundle_dict.get("methods"))
        lessons = _jsonish_list(bundle_dict.get("lessons"))
        paper_id, duplicate = _insert_or_reuse_paper(conn, paper)
        if duplicate:
            existing_method_ids, existing_chunk_ids, existing_method_id_map, existing_chunk_id_map = _existing_bundle_links(
                conn, paper_id, methods
            )
            existing_status = _current_extraction_status(conn, paper_id)
            if existing_status is not None and existing_chunk_ids:
                lesson_ids, lesson_id_map = _import_remapped_lessons(conn, lessons, existing_method_id_map)
                if commit:
                    conn.commit()
                paper_id_map, _, _ = _source_id_maps(bundle_dict, paper_id)
                return {
                    "paper_id": paper_id,
                    "method_ids": existing_method_ids,
                    "chunk_ids": existing_chunk_ids,
                    "lesson_ids": lesson_ids,
                    "extraction_status": existing_status,
                    "duplicate": True,
                    "source_id_map": {
                        "paper_id": paper_id_map,
                        "method_ids": existing_method_id_map,
                        "chunk_ids": existing_chunk_id_map,
                        "lesson_ids": lesson_id_map,
                    },
                }
        method_ids: list[int] = []
        chunk_ids: list[int] = []
        method_id_map: dict[str, int] = {}
        chunk_id_map: dict[str, list[int]] = {}
        local_method_by_source_id: dict[int, int] = {}

        for method in methods:
            if not isinstance(method, Mapping):
                raise ValueError("Bundle methods must contain objects")
            local_method_id = _insert_or_reuse_method(conn, method)
            if local_method_id not in method_ids:
                method_ids.append(local_method_id)
            source_method_id = _optional_int(method.get("source_method_id") or method.get("method_id"))
            if source_method_id is not None:
                method_id_map[str(source_method_id)] = local_method_id
                local_method_by_source_id[source_method_id] = local_method_id

        for method, source in _source_iter(methods):
            evidence_text = _compact_text(source.get("evidence_text"))
            if evidence_text is None:
                continue
            source_method_id = _optional_int(source.get("source_method_id"))
            local_method_id = None
            if source_method_id is not None:
                local_method_id = local_method_by_source_id.get(source_method_id)
            if local_method_id is None:
                local_method_id = _insert_or_reuse_method(conn, method)
                if local_method_id not in method_ids:
                    method_ids.append(local_method_id)
            section = source.get("section")
            source_chunk_id = _optional_int(source.get("source_chunk_id") or source.get("chunk_id"))
            chunk_row = None
            if duplicate and source_chunk_id is not None:
                chunk_row = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE paper_id = ? AND chunk_id = ?",
                    (paper_id, source_chunk_id),
                ).fetchone()
            chunk_id = int(chunk_row["chunk_id"]) if chunk_row is not None else _insert_or_reuse_chunk(conn, paper_id, section, evidence_text)
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
            _append_mapping(chunk_id_map, source_chunk_id, chunk_id)
            conn.execute(
                """
                INSERT INTO method_sources (
                    method_id, chunk_id, evidence_text, evidence_start, evidence_end, evidence_strategy
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(method_id, chunk_id) DO UPDATE SET
                    evidence_text = excluded.evidence_text,
                    evidence_start = excluded.evidence_start,
                    evidence_end = excluded.evidence_end,
                    evidence_strategy = excluded.evidence_strategy
                """,
                (
                    local_method_id,
                    chunk_id,
                    evidence_text,
                    _optional_int(source.get("evidence_start")),
                    _optional_int(source.get("evidence_end")),
                    source.get("evidence_strategy") or "memory_bundle",
                ),
            )

        lesson_ids, lesson_id_map = _import_remapped_lessons(conn, lessons, method_id_map)

        extraction_status = _upsert_extraction_status(
            conn,
            _status_payload(bundle_dict, paper_id, len(method_ids), len(chunk_ids)),
        )
        if commit:
            conn.commit()
        paper_id_map, _, _ = _source_id_maps(bundle_dict, paper_id)
        return {
            "paper_id": paper_id,
            "method_ids": method_ids,
            "chunk_ids": chunk_ids,
            "lesson_ids": lesson_ids,
            "extraction_status": extraction_status,
            "duplicate": duplicate,
            "source_id_map": {
                "paper_id": paper_id_map,
                "method_ids": method_id_map,
                "chunk_ids": chunk_id_map,
                "lesson_ids": lesson_id_map,
            },
        }
    finally:
        if close:
            conn.close()


def _append_unique_ids(target: list[int], values: Iterable[Any]) -> None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and parsed not in target:
            target.append(parsed)


def import_corpus_memory_bundle(bundle: Mapping[str, Any], conn: Any | None = None, *, commit: bool = True) -> JSONDict:
    """Import a corpus memory bundle atomically by importing each contained paper bundle."""
    bundle_dict = _validate_corpus_bundle(bundle)
    conn, close = _connect(conn)
    savepoint = "researchfs_corpus_memory_import"
    conn.execute(f"SAVEPOINT {savepoint}")
    savepoint_open = True
    try:
        paper_results: list[JSONDict] = []
        paper_ids: list[int] = []
        method_ids: list[int] = []
        chunk_ids: list[int] = []
        lesson_ids: list[int] = []
        source_id_maps: list[JSONDict] = []

        for index, paper_bundle in enumerate(bundle_dict["papers"]):
            result = import_paper_memory_bundle(paper_bundle, conn=conn, commit=False)
            paper_results.append(result)
            _append_unique_ids(paper_ids, [result.get("paper_id")])
            _append_unique_ids(method_ids, _jsonish_list(result.get("method_ids")))
            _append_unique_ids(chunk_ids, _jsonish_list(result.get("chunk_ids")))
            _append_unique_ids(lesson_ids, _jsonish_list(result.get("lesson_ids")))
            source_id_maps.append(
                {
                    "paper_index": index,
                    "paper_id": result.get("paper_id"),
                    "source_id_map": _jsonish_dict(result.get("source_id_map")),
                }
            )

        result = {
            "paper_results": paper_results,
            "paper_ids": paper_ids,
            "method_ids": method_ids,
            "chunk_ids": chunk_ids,
            "lesson_ids": lesson_ids,
            "source_id_maps": source_id_maps,
        }
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        savepoint_open = False
        if commit:
            conn.commit()
        return result
    except Exception:
        if savepoint_open:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    finally:
        if close:
            conn.close()


def import_memory_bundle(bundle: Mapping[str, Any], conn: Any | None = None) -> JSONDict:
    """Import either one paper memory bundle or one corpus memory bundle."""
    if not isinstance(bundle, Mapping):
        raise ValueError("Bundle must be a JSON object")
    kind = bundle.get("kind")
    if kind == BUNDLE_KIND:
        return import_paper_memory_bundle(bundle, conn=conn)
    if kind == CORPUS_BUNDLE_KIND:
        return import_corpus_memory_bundle(bundle, conn=conn)
    raise ValueError(f"Bundle kind must be {BUNDLE_KIND!r} or {CORPUS_BUNDLE_KIND!r}")
