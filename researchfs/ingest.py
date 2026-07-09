"""Ingest research artifacts and extracted methods into ResearchFS."""

import argparse
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from typing import Callable, Optional

from researchfs import db as db_module
from researchfs import extractor
from researchfs.code_repos import enrich_paper_code_repositories
from researchfs.db import get_connection, initialize_db, migrate_db
from researchfs.fetcher import fetch_arxiv_source, fetch_arxiv_query, fetch_github_readme
from researchfs.pdf import extract_pdf_text

SUPPORTED_LOCAL_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".tex", ".pdf"}

_SOURCE_TYPE_EXPLICIT_META_KEY = "_researchfs_source_type_explicit"

def _read_only_sqlite_uri(db_uri: str) -> str | None:
    """Return a read-only SQLite URI for a configured file: target."""
    parsed = urlsplit(db_uri)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key == "mode" and value == "memory" for key, value in query):
        return None
    query = [(key, value) for key, value in query if key != "mode"]
    query.append(("mode", "ro"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _open_dry_run_target_connection():
    """Open an existing target DB read-only for dry-run duplicate checks."""
    try:
        target = db_module._resolve_db_path(None)
        target_str = str(target)
        if target_str in ("", ":memory:"):
            return None
        if target_str.startswith("file:"):
            uri = _read_only_sqlite_uri(target_str)
            if uri is None:
                return None
        else:
            target_path = Path(target_str).expanduser()
            if not target_path.exists():
                return None
            uri = f"file:{quote(str(target_path.resolve()), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except (OSError, sqlite3.Error):
        return None



def _mark_source_type_explicit(meta: dict) -> dict:
    """Mark source_type as a caller-supplied label rather than an inferred default."""
    meta[_SOURCE_TYPE_EXPLICIT_META_KEY] = True
    return meta


def _explicit_source_type(meta: dict) -> Optional[str]:
    """Return the caller-supplied source_type, excluding known inferred defaults."""
    if meta.get(_SOURCE_TYPE_EXPLICIT_META_KEY) is False:
        return None
    source_type = meta.get("source_type")
    if source_type is None:
        return None
    return _source_type_label(str(source_type))


def _title_from_file_path(path: Path) -> str:
    title = re.sub(r"[-_]+", " ", path.stem).strip()
    return title.title() if title else path.name


class LocalFileExtractionError(ValueError):
    """Raised when a supported local file cannot be converted to text."""


def _source_type_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source-type must not be empty")
    return label


def _apply_source_type_override(meta: dict, source_type: Optional[str]) -> dict:
    if source_type is not None:
        meta["source_type"] = _source_type_label(source_type)
        _mark_source_type_explicit(meta)
    return meta


def load_local_file_metadata(path: str | Path, overrides: Optional[dict] = None) -> dict:
    """Load a local text/markdown/TeX/PDF paper into ResearchFS metadata."""
    file_path = Path(path).expanduser()
    suffix = file_path.suffix.lower()
    supported = ", ".join(sorted(SUPPORTED_LOCAL_FILE_EXTENSIONS))
    if suffix not in SUPPORTED_LOCAL_FILE_EXTENSIONS:
        display_suffix = suffix or "<none>"
        raise ValueError(
            f"Unsupported local file type {display_suffix!r}; supported extensions are {supported}."
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"Local paper file not found: {file_path}")

    resolved_path = file_path.resolve()
    if suffix == ".pdf":
        try:
            content = extract_pdf_text(resolved_path)
        except ValueError as exc:
            raise LocalFileExtractionError(str(exc)) from exc
    else:
        content = resolved_path.read_text(encoding="utf-8")
    meta = dict(overrides or {})
    title = str(meta.get("title") or "").strip() or _title_from_file_path(resolved_path)
    source_type_is_explicit = bool(meta.get("source_type"))
    source_type = _source_type_label(str(meta["source_type"])) if source_type_is_explicit else "file"
    meta.update(
        {
            "title": title,
            "url": resolved_path.as_uri(),
            "source_type": source_type,
            "content": content,
        }
    )
    if source_type_is_explicit:
        _mark_source_type_explicit(meta)
    else:
        meta[_SOURCE_TYPE_EXPLICIT_META_KEY] = False
    meta.setdefault("authors", None)
    meta.setdefault("abstract", None)
    meta.setdefault("publish_date", None)
    return meta



def _build_llm_fn() -> Optional[Callable[[str], str]]:
    """Return a callable LLM client if one can be configured locally.

    Supports OpenAI-compatible endpoints via the ``OPENAI_API_KEY`` environment
    variable.  Returns ``None`` if no local LLM can be built so callers can fall
    back to ingestion without extraction.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:  # pragma: no cover - optional dependency
        return None

    client = openai.OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    return _llm


def _iter_extracted_methods(result: dict) -> list:
    """Return the extracted methods list only when the LLM schema is usable."""
    methods = result.get("methods", [])
    return methods if isinstance(methods, list) else []


def _normalize_extracted_method(method: object) -> Optional[dict]:
    """Validate and normalize one LLM-extracted method before ingestion."""
    if not isinstance(method, dict):
        return None
    name = str(method.get("name") or "").strip()
    if not name:
        return None
    source_indices = method.get("source_chunk_indices") or []
    if not isinstance(source_indices, (list, tuple)):
        source_indices = []
    return {
        "name": name,
        "description": str(method.get("description") or "").strip(),
        "implementation_hint": str(method.get("implementation_hint") or "").strip(),
        "risks": str(method.get("risks") or "").strip(),
        "category": _normalize_extracted_category(method.get("category")),
        "source_chunk_indices": [
            idx
            for idx in source_indices
            if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0
        ],
    }


def _normalize_extracted_category(value: object) -> str:
    """Return a canonical extractor category token for one LLM item."""
    category = str(value or "").strip().lower()
    if category in extractor.METHOD_CATEGORY_TOKENS:
        return category
    return "method"


_EVIDENCE_EXCERPT_CHARS = 360
_EVIDENCE_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "can",
    "for",
    "from",
    "into",
    "method",
    "none",
    "the",
    "this",
    "use",
    "uses",
    "using",
    "with",
}


def _method_evidence_terms(method: dict) -> list[str]:
    """Return deterministic candidate terms from normalized method metadata."""
    terms = []
    seen = set()
    for field in ("name", "description", "implementation_hint"):
        for term in re.findall(r"[A-Za-z0-9]+", method.get(field) or ""):
            normalized = term.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized in _EVIDENCE_STOPWORDS:
                continue
            if len(normalized) < 3 and not any(ch.isdigit() for ch in normalized):
                continue
            terms.append(term)
    return terms


def _bounded_evidence_window(content: str, anchor_start: int, anchor_end: int) -> tuple[str, int, int]:
    """Return a stable excerpt window containing the anchor span."""
    if not content:
        return "", 0, 0
    excerpt_len = min(_EVIDENCE_EXCERPT_CHARS, len(content))
    anchor_mid = (anchor_start + anchor_end) // 2
    start = max(0, anchor_mid - excerpt_len // 2)
    end = min(len(content), start + excerpt_len)
    start = max(0, end - excerpt_len)
    return content[start:end], start, end


def _build_method_source_evidence(content: str, method: dict) -> dict:
    """Build deterministic provenance evidence for a method-source link."""
    content = content or ""
    for term in _method_evidence_terms(method):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            evidence_text, start, end = _bounded_evidence_window(
                content, match.start(), match.end()
            )
            return {
                "evidence_text": evidence_text,
                "evidence_start": start,
                "evidence_end": end,
                "evidence_strategy": f"matched_term:{term.lower()}",
            }

    end = min(_EVIDENCE_EXCERPT_CHARS, len(content))
    return {
        "evidence_text": content[:end],
        "evidence_start": 0,
        "evidence_end": end,
        "evidence_strategy": "leading_excerpt" if content else "empty_chunk",
    }


def _build_method_fts_query(name: str) -> str:
    """Build an FTS5-safe OR query from a method name."""
    terms = re.findall(r"[A-Za-z0-9]+", name)
    return " OR ".join(f'"{term}"' for term in terms)


def _check_duplicate_method(
    method: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    conn=None,
) -> Optional[int]:
    """Check if the extracted method is a duplicate of an existing method.

    Checks:
    1. Case-insensitive name match.
    2. FTS5 query on method name keywords for similar candidates.
    3. LLM decision on the candidates if llm_fn is provided.
    """
    name = method.get("name")
    if not name:
        return None

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT method_id FROM methods WHERE LOWER(name) = LOWER(?)",
                (name,),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return None
        if row:
            return row[0]

        # 2. FTS5 search using keywords from the method's name
        fts_query = _build_method_fts_query(name)
        if not fts_query:
            return None
        try:
            cur.execute(
                "SELECT rowid, name, description FROM methods_fts WHERE methods_fts MATCH ? ORDER BY rank LIMIT 3",
                (fts_query,),
            )
            candidates = cur.fetchall()
        except sqlite3.OperationalError:
            candidates = []

        # 3. LLM comparison
        if candidates and llm_fn is not None:
            extracted_name = name
            extracted_desc = method.get("description", "")
            for cand in candidates:
                existing_name = cand["name"]
                existing_desc = cand["description"] or ""
                
                prompt = (
                    f"You are an expert machine learning database curator.\n"
                    f"We are ingesting a new method:\n"
                    f"Name: {extracted_name}\n"
                    f"Description: {extracted_desc}\n\n"
                    f"We found an existing method in the database:\n"
                    f"Name: {existing_name}\n"
                    f"Description: {existing_desc}\n\n"
                    f"Are these the exact same optimization method?\n"
                    f"Reply ONLY with 'YES' or 'NO'. Do not include explanation or markdown."
                )
                response = llm_fn(prompt)
                if response and response.strip().upper() == "YES":
                    return cand["rowid"]

        return None
    finally:
        if close:
            conn.close()


def _infer_source_type(meta: dict) -> Optional[str]:
    """Return source type metadata when it can be inferred cheaply."""
    source_type = meta.get("source_type")
    if source_type:
        return str(source_type)
    url = meta.get("url") or ""
    title = meta.get("title") or ""
    if "arxiv.org" in url:
        return "arxiv"
    if "github.com" in url or title.startswith("GitHub:"):
        return "github"
    return None


def _extractor_metadata(meta: dict, llm_fn: Optional[Callable[[str], str]]) -> tuple[Optional[str], Optional[str]]:
    """Return extractor model and prompt version when the caller exposes them."""
    model = meta.get("extractor_model")
    prompt_version = meta.get("extractor_prompt_version")
    if llm_fn is not None:
        model = model or getattr(llm_fn, "model", None) or getattr(llm_fn, "model_name", None)
        prompt_version = prompt_version or getattr(llm_fn, "prompt_version", None)
    prompt_version = prompt_version or getattr(extractor, "PROMPT_VERSION", None)
    return (str(model) if model else None, str(prompt_version) if prompt_version else None)


def _compressed_method_char_count(methods: list[dict]) -> int:
    """Count characters in the compressed method memory payload."""
    fields = ("name", "description", "implementation_hint", "risks", "category")
    return sum(len(str(method.get(field) or "")) for method in methods for field in fields)

def _compression_confidence_component(compression_ratio: Optional[float]) -> tuple[float, str]:
    """Return a bounded confidence component for compression sanity."""
    if compression_ratio is None:
        return 0.0, "missing"
    if compression_ratio <= 0:
        return 0.0, "nonpositive"
    if compression_ratio <= 1.0:
        return 1.0, "sane"
    if compression_ratio <= 2.0:
        return 0.5, "large"
    return 0.1, "oversized"


def _method_source_confidence_stats(methods: list[dict], chunks: list[str]) -> dict:
    """Measure observable source-link and evidence coverage for normalized methods."""
    method_count = len(methods)
    total_source_links = 0
    valid_source_links = 0
    methods_with_valid_source = 0
    evidence_links = 0
    methods_with_evidence = 0

    for method in methods:
        method_has_valid_source = False
        method_has_evidence = False
        for idx in method.get("source_chunk_indices") or []:
            total_source_links += 1
            if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(chunks)):
                continue
            valid_source_links += 1
            method_has_valid_source = True
            evidence = _build_method_source_evidence(chunks[idx], method)
            if (
                evidence.get("evidence_text")
                and isinstance(evidence.get("evidence_start"), int)
                and isinstance(evidence.get("evidence_end"), int)
                and evidence["evidence_end"] > evidence["evidence_start"]
            ):
                evidence_links += 1
                method_has_evidence = True
        if method_has_valid_source:
            methods_with_valid_source += 1
        if method_has_evidence:
            methods_with_evidence += 1

    return {
        "method_count": method_count,
        "total_source_links": total_source_links,
        "valid_source_links": valid_source_links,
        "methods_with_valid_source": methods_with_valid_source,
        "evidence_links": evidence_links,
        "methods_with_evidence": methods_with_evidence,
    }


def _extraction_confidence(
    status: str,
    methods: list[dict],
    chunks: list[str],
    compression_ratio: Optional[float],
    error_text: Optional[str] = None,
) -> tuple[float, str]:
    """Score deterministic extraction confidence from observable extraction quality."""
    if status == "skipped":
        return 0.0, "skipped"
    if status == "failed":
        reason = "failed"
        if error_text:
            reason = f"{reason}: {error_text}"
        return 0.0, reason

    stats = _method_source_confidence_stats(methods, chunks)
    method_count = stats["method_count"]
    total_source_links = stats["total_source_links"]
    valid_source_links = stats["valid_source_links"]
    methods_with_valid_source = stats["methods_with_valid_source"]
    evidence_links = stats["evidence_links"]
    methods_with_evidence = stats["methods_with_evidence"]

    method_score = min(method_count / 3.0, 1.0) if method_count else 0.0
    source_link_score = (
        valid_source_links / total_source_links if total_source_links else 0.0
    )
    source_coverage_score = (
        methods_with_valid_source / method_count if method_count else 0.0
    )
    evidence_coverage_score = (
        methods_with_evidence / method_count if method_count else 0.0
    )
    compression_score, compression_label = _compression_confidence_component(
        compression_ratio
    )
    score = (
        0.15
        + 0.20 * method_score
        + 0.20 * source_link_score
        + 0.20 * source_coverage_score
        + 0.15 * evidence_coverage_score
        + 0.10 * compression_score
    )
    bounded_score = round(max(0.0, min(1.0, score)), 3)
    ratio_text = "missing" if compression_ratio is None else f"{compression_ratio:.3f}"
    reason = (
        f"extracted: methods={method_count}; "
        f"valid_source_links={valid_source_links}/{total_source_links}; "
        f"method_source_coverage={methods_with_valid_source}/{method_count}; "
        f"evidence_coverage={methods_with_evidence}/{method_count}; "
        f"evidence_links={evidence_links}/{valid_source_links}; "
        f"compression_ratio={ratio_text}; compression={compression_label}"
    )
    return bounded_score, reason


def _build_extraction_status(
    meta: dict,
    status: str,
    chunks: list[str],
    methods: Optional[list[dict]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    error_text: Optional[str] = None,
) -> dict:
    """Build the durable extraction status payload returned and stored for a paper."""
    methods = methods or []
    source_text = meta.get("content", meta.get("abstract", "")) or ""
    source_char_count = len(source_text)
    compressed_char_count = _compressed_method_char_count(methods)
    compression_ratio = (
        compressed_char_count / source_char_count if source_char_count else None
    )
    extractor_model, extractor_prompt_version = _extractor_metadata(meta, llm_fn)
    confidence_score, confidence_reason = _extraction_confidence(
        status, methods, chunks, compression_ratio, error_text
    )
    return {
        "status": status,
        "source_type": _infer_source_type(meta),
        "chunk_count": len(chunks),
        "method_count": len(methods),
        "source_char_count": source_char_count,
        "compressed_char_count": compressed_char_count,
        "compression_ratio": compression_ratio,
        "confidence_score": confidence_score,
        "confidence_reason": confidence_reason,
        "external_id": meta.get("external_id"),
        "extractor_model": extractor_model,
        "extractor_prompt_version": extractor_prompt_version,
        "error_text": error_text,
    }


def _row_to_extraction_status(row) -> Optional[dict]:
    if row is None:
        return None
    return {
        "paper_id": row["paper_id"],
        "status": row["status"],
        "source_type": row["source_type"],
        "chunk_count": row["chunk_count"],
        "method_count": row["method_count"],
        "source_char_count": row["source_char_count"],
        "compressed_char_count": row["compressed_char_count"],
        "compression_ratio": row["compression_ratio"],
        "confidence_score": row["confidence_score"],
        "confidence_reason": row["confidence_reason"],
        "external_id": row["external_id"],
        "extractor_model": row["extractor_model"],
        "extractor_prompt_version": row["extractor_prompt_version"],
        "error_text": row["error_text"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _get_extraction_status(conn, paper_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    return _row_to_extraction_status(row)

def _paper_confidence_inputs(conn, paper_id: int) -> tuple[list[str], list[dict]]:
    """Reconstruct observable paper chunks and method links for legacy status confidence."""
    chunk_rows = conn.execute(
        """
        SELECT chunk_id, content
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        """,
        (paper_id,),
    ).fetchall()
    chunks = [(row["content"] or "") for row in chunk_rows]
    chunk_index_by_id = {
        row["chunk_id"]: idx for idx, row in enumerate(chunk_rows)
    }
    method_rows = conn.execute(
        """
        SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category, ms.chunk_id
        FROM method_sources ms
        JOIN methods m ON m.method_id = ms.method_id
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        ORDER BY m.method_id, ms.chunk_id
        """,
        (paper_id,),
    ).fetchall()
    methods_by_id = {}
    for row in method_rows:
        method = methods_by_id.setdefault(
            row["method_id"],
            {
                "name": row["name"] or "",
                "description": row["description"] or "",
                "implementation_hint": row["implementation_hint"] or "",
                "risks": row["risks"] or "",
                "category": row["category"] or "",
                "source_chunk_indices": [],
            },
        )
        chunk_index = chunk_index_by_id.get(row["chunk_id"])
        if chunk_index is not None:
            method["source_chunk_indices"].append(chunk_index)
    return chunks, list(methods_by_id.values())


def _status_payload_from_row(status: dict) -> dict:
    """Return an upsert payload from a stored extraction status dictionary."""
    return {
        "status": status["status"],
        "source_type": status["source_type"],
        "chunk_count": status["chunk_count"],
        "method_count": status["method_count"],
        "source_char_count": status["source_char_count"],
        "compressed_char_count": status["compressed_char_count"],
        "compression_ratio": status["compression_ratio"],
        "confidence_score": status["confidence_score"],
        "confidence_reason": status["confidence_reason"],
        "external_id": status["external_id"],
        "extractor_model": status["extractor_model"],
        "extractor_prompt_version": status["extractor_prompt_version"],
        "error_text": status["error_text"],
    }



def _upsert_extraction_status(conn, paper_id: int, status_payload: dict) -> dict:
    conn.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, source_type, external_id, chunk_count, method_count,
            source_char_count, compressed_char_count, compression_ratio,
            confidence_score, confidence_reason,
            extractor_model, extractor_prompt_version, error_text
        )
        VALUES (
            :paper_id, :status, :source_type, :external_id, :chunk_count, :method_count,
            :source_char_count, :compressed_char_count, :compression_ratio,
            :confidence_score, :confidence_reason,
            :extractor_model, :extractor_prompt_version, :error_text
        )
        ON CONFLICT(paper_id) DO UPDATE SET
            status = excluded.status,
            source_type = excluded.source_type,
            external_id = excluded.external_id,
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
        {"paper_id": paper_id, **status_payload},
    )
    return _get_extraction_status(conn, paper_id) or {"paper_id": paper_id, **status_payload}


def _update_existing_source_type_if_explicit(
    conn,
    paper_id: int,
    extraction_status: dict,
    meta: dict,
) -> tuple[dict, bool]:
    """Apply an explicit duplicate-ingest source_type correction without touching status metadata."""
    explicit_source_type = _explicit_source_type(meta)
    if (
        explicit_source_type is None
        or extraction_status["source_type"] == explicit_source_type
    ):
        return extraction_status, False
    conn.execute(
        "UPDATE paper_extraction_status SET source_type = ? WHERE paper_id = ?",
        (explicit_source_type, paper_id),
    )
    updated_status = _get_extraction_status(conn, paper_id)
    return (updated_status or {**extraction_status, "source_type": explicit_source_type}), True

def _extract_methods_from_existing_chunks(
    meta: dict,
    chunks: list[str],
    llm_fn: Callable[[str], str],
) -> dict:
    """Extract methods from already-stored paper chunks without rechunking them."""
    return extractor.extract_methods_from_chunks(meta, chunks, llm_fn)


def _stored_paper_inputs(conn, paper_id: int, fallback_meta: dict) -> tuple[dict, list[str], list[int]]:
    """Return stored paper metadata, chunk text, and chunk ids for duplicate extraction."""
    paper = conn.execute(
        """
        SELECT title, authors, abstract, url, publish_date
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    chunk_rows = conn.execute(
        """
        SELECT chunk_id, content
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        """,
        (paper_id,),
    ).fetchall()
    chunks = [(row["content"] or "") for row in chunk_rows]
    content = "\n\n".join(chunks) if chunks else (fallback_meta.get("content") or fallback_meta.get("abstract") or "")
    stored_meta = dict(fallback_meta)
    if paper is not None:
        stored_meta.update(
            {
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "url": paper["url"],
                "publish_date": paper["publish_date"],
            }
        )
    stored_meta["content"] = content
    return stored_meta, chunks, [row["chunk_id"] for row in chunk_rows]


def _paper_has_method_sources(conn, paper_id: int) -> bool:
    """Return whether a paper already has method provenance links."""
    row = conn.execute(
        """
        SELECT 1
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()
    return row is not None


def _insert_or_link_extracted_methods(
    conn,
    chunks: list[str],
    chunk_ids: list[int],
    raw_methods: list,
    llm_fn: Callable[[str], str],
) -> tuple[list[int], list[dict]]:
    """Insert or reuse extracted methods and link them to the supplied chunks."""
    cur = conn.cursor()
    method_ids = []
    normalized_methods = []
    for raw_method in raw_methods:
        method = _normalize_extracted_method(raw_method)
        if method is None:
            continue
        normalized_methods.append(method)
        resolved_id = _check_duplicate_method(method, llm_fn=llm_fn, conn=conn)
        if resolved_id is not None:
            method_id = resolved_id
        else:
            cur.execute(
                """
                INSERT OR IGNORE INTO methods
                (name, description, implementation_hint, risks, category)
                VALUES (:name, :description, :implementation_hint, :risks, :category)
                """,
                {
                    "name": method["name"],
                    "description": method["description"],
                    "implementation_hint": method["implementation_hint"],
                    "risks": method["risks"],
                    "category": method["category"],
                },
            )
            cur.execute(
                "SELECT method_id FROM methods WHERE name = ?",
                (method["name"],),
            )
            method_id = cur.fetchone()["method_id"]

        method_ids.append(method_id)

        for idx in method["source_chunk_indices"]:
            if 0 <= idx < len(chunk_ids):
                evidence = _build_method_source_evidence(chunks[idx], method)
                cur.execute(
                    """
                    INSERT INTO method_sources (
                        method_id, chunk_id, evidence_text, evidence_start, evidence_end, evidence_strategy
                    )
                    VALUES (
                        :method_id, :chunk_id, :evidence_text, :evidence_start, :evidence_end, :evidence_strategy
                    )
                    ON CONFLICT(method_id, chunk_id) DO UPDATE SET
                        evidence_text = COALESCE(method_sources.evidence_text, excluded.evidence_text),
                        evidence_start = COALESCE(method_sources.evidence_start, excluded.evidence_start),
                        evidence_end = COALESCE(method_sources.evidence_end, excluded.evidence_end),
                        evidence_strategy = COALESCE(method_sources.evidence_strategy, excluded.evidence_strategy)
                    """,
                    {"method_id": method_id, "chunk_id": chunk_ids[idx], **evidence},
                )

    return method_ids, normalized_methods


def _reextract_existing_paper(
    conn,
    paper_id: int,
    meta: dict,
    llm_fn: Callable[[str], str],
) -> dict:
    """Run extraction for an already-stored paper that has no usable compressed memory."""
    stored_meta, stored_chunks, stored_chunk_ids = _stored_paper_inputs(conn, paper_id, meta)
    extraction_chunks = stored_chunks or extractor.chunk_text(stored_meta.get("content", "") or "")
    try:
        result = _extract_methods_from_existing_chunks(stored_meta, extraction_chunks, llm_fn)
        method_ids, normalized_methods = _insert_or_link_extracted_methods(
            conn,
            extraction_chunks,
            stored_chunk_ids,
            _iter_extracted_methods(result),
            llm_fn,
        )
        extraction_status = _upsert_extraction_status(
            conn,
            paper_id,
            _build_extraction_status(
                stored_meta, "extracted", extraction_chunks, normalized_methods, llm_fn=llm_fn
            ),
        )
    except Exception as exc:
        method_ids = []
        extraction_status = _upsert_extraction_status(
            conn,
            paper_id,
            _build_extraction_status(
                stored_meta, "failed", extraction_chunks, llm_fn=llm_fn, error_text=str(exc)
            ),
        )
    return {
        "paper_id": paper_id,
        "chunk_ids": stored_chunk_ids,
        "method_ids": method_ids,
        "extraction_status": extraction_status,
        "duplicate": True,
    }

def _visual_index_failed_payload(error_text: object) -> dict:
    """Return a stable failed PixelRAG payload without raising into ingestion."""
    return {
        "provider": "pixelrag",
        "status": "failed",
        "source": None,
        "tiles_path": None,
        "embeddings_path": None,
        "index_path": None,
        "error_text": str(error_text),
    }


def _visual_plan_identifier(meta: dict) -> str:
    """Return a stable dry-run identifier for planned visual-index paths."""
    raw = meta.get("external_id") or meta.get("title") or meta.get("url") or "planned"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw)).strip("-")
    return f"planned-{(label or 'paper')[:80]}"


def _plan_visual_index_payload(meta: dict, output_root: str | Path | None) -> dict:
    """Plan an optional visual index without importing PixelRAG or running commands."""
    try:
        from researchfs import visual

        return visual.plan_pixelrag_visual_index(_visual_plan_identifier(meta), meta, output_root=output_root)
    except Exception as exc:
        return _visual_index_failed_payload(exc)


def _attach_planned_visual_index(
    result: dict,
    meta: dict,
    output_root: str | Path | None,
) -> dict:
    visual_index_payload = _plan_visual_index_payload(meta, output_root)
    result["visual_index"] = visual_index_payload
    planned = result.get("planned")
    if isinstance(planned, dict):
        planned["visual_index"] = visual_index_payload
    return result


def _attach_stored_visual_index(
    conn,
    result: dict,
    meta: dict,
    output_root: str | Path | None,
    runner=None,
) -> dict:
    paper_id = result.get("paper_id")
    if paper_id is None:
        return result
    try:
        from researchfs import visual

        result["visual_index"] = visual.build_and_store_pixelrag_visual_index(
            conn,
            paper_id,
            meta,
            output_root=output_root,
            runner=runner,
        )
        conn.commit()
    except Exception as exc:
        result["visual_index"] = _visual_index_failed_payload(exc)
    return result


def _resolve_code_repo_search_fn(code_repo_search_fn=None, code_repo_searcher=None):
    """Return the configured code repository search callable, accepting legacy aliases."""
    if code_repo_search_fn is not None and code_repo_searcher is not None:
        if code_repo_search_fn is not code_repo_searcher:
            raise ValueError("Pass either code_repo_search_fn or code_repo_searcher, not both")
    return code_repo_search_fn if code_repo_search_fn is not None else code_repo_searcher


def _code_repository_rows_from_enrichment(enrichment: object) -> list[dict]:
    """Return selected/stored repository rows from an enrichment result."""
    if isinstance(enrichment, list):
        return [row for row in enrichment if isinstance(row, dict)]
    if not isinstance(enrichment, dict):
        return []
    selected = enrichment.get("selected")
    if isinstance(selected, dict):
        return [selected]
    if isinstance(selected, list):
        return [row for row in selected if isinstance(row, dict)]
    stored = enrichment.get("stored")
    if isinstance(stored, list):
        return [row for row in stored if isinstance(row, dict)]
    return []


def _attach_code_repository_enrichment(
    conn,
    result: dict,
    *,
    enrich_code: bool,
    code_repo_search_fn=None,
) -> dict:
    """Attach automatic code repository enrichment results to an ingestion result."""
    if not enrich_code:
        return result
    paper_id = result.get("paper_id")
    if paper_id is None:
        return result
    try:
        enrichment = enrich_paper_code_repositories(
            conn,
            int(paper_id),
            search_fn=code_repo_search_fn,
        )
    except Exception as exc:
        result["code_repository_enrichment"] = {
            "paper_id": int(paper_id),
            "selected": None,
            "stored": [],
            "candidates": [],
            "error_text": str(exc),
        }
        result["code_repositories"] = []
        return result
    result["code_repository_enrichment"] = enrichment
    result["code_repositories"] = _code_repository_rows_from_enrichment(enrichment)
    return result


def ingest_paper(
    meta: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    conn=None,
    dry_run: bool = False,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> dict:
    """Insert (or plan) a paper, its chunks, methods, and extraction status."""
    code_repo_search_fn = _resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher)
    if dry_run:
        close_conn = False
        if conn is None:
            conn = _open_dry_run_target_connection()
            if conn is None:
                conn = sqlite3.connect(":memory:")
                conn.row_factory = sqlite3.Row
                initialize_db(conn)
            close_conn = True
        try:
            content = meta.get("content", meta.get("abstract", "")) or ""
            chunks = extractor.chunk_text(content)
            planned = {
                "paper": {
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "abstract": meta.get("abstract"),
                    "url": meta.get("url"),
                    "publish_date": meta.get("publish_date"),
                    "external_id": meta.get("external_id"),
                },
                "chunks": [
                    {"section": "extracted", "content": chunk} for chunk in chunks
                ],
                "methods": [],
                "merged_methods": [],
                "method_sources": [],
            }
            normalized_methods = []
            if llm_fn is None:
                extraction_status = _build_extraction_status(
                    meta, "skipped", chunks, llm_fn=llm_fn
                )
            else:
                try:
                    result = extractor.extract_methods_from_paper(meta, llm_fn)
                    for raw_method in _iter_extracted_methods(result):
                        method = _normalize_extracted_method(raw_method)
                        if method is None:
                            continue
                        normalized_methods.append(method)
                        resolved_id = _check_duplicate_method(method, llm_fn=llm_fn, conn=conn)
                        if resolved_id is not None:
                            planned["merged_methods"].append(
                                {
                                    "name": method["name"],
                                    "merged_into_id": resolved_id,
                                }
                            )
                        else:
                            planned["methods"].append(
                                {
                                    "name": method["name"],
                                    "description": method["description"],
                                    "implementation_hint": method["implementation_hint"],
                                    "risks": method["risks"],
                                    "category": method["category"],
                                }
                            )
                        for idx in method["source_chunk_indices"]:
                            if 0 <= idx < len(chunks):
                                evidence = _build_method_source_evidence(chunks[idx], method)
                                planned["method_sources"].append(
                                    {
                                        "method_name": method["name"],
                                        "chunk_index": idx,
                                        **evidence,
                                    }
                                )
                    extraction_status = _build_extraction_status(
                        meta, "extracted", chunks, normalized_methods, llm_fn=llm_fn
                    )
                except Exception as exc:  # pragma: no cover
                    planned["extraction_error"] = str(exc)
                    extraction_status = _build_extraction_status(
                        meta, "failed", chunks, llm_fn=llm_fn, error_text=str(exc)
                    )
            planned["extraction_status"] = extraction_status
            result = {"planned": planned, "extraction_status": extraction_status}
            if visual_index:
                result = _attach_planned_visual_index(result, meta, visual_index_root)
            return result
        finally:
            if close_conn:
                conn.close()

    close = False
    if conn is None:
        initialize_db()
        conn = get_connection()
        close = True

    try:
        migrate_db(conn)
        cur = conn.cursor()
        content = meta.get("content", meta.get("abstract", "")) or ""
        chunks = extractor.chunk_text(content)
        cur.execute(
            "SELECT paper_id FROM papers WHERE title = ?",
            (meta["title"],),
        )
        existing = cur.fetchone()
        if existing:
            paper_id = existing["paper_id"]
            extraction_status = _get_extraction_status(conn, paper_id)
            has_method_sources = _paper_has_method_sources(conn, paper_id)
            should_reextract = llm_fn is not None and (
                extraction_status is None
                or extraction_status["status"] in {"skipped", "failed"}
                or not has_method_sources
            )
            if should_reextract:
                result = _reextract_existing_paper(conn, paper_id, meta, llm_fn)
                conn.commit()
                if visual_index:
                    result = _attach_stored_visual_index(
                        conn,
                        result,
                        meta,
                        visual_index_root,
                        runner=visual_index_runner,
                    )
                result = _attach_code_repository_enrichment(
                    conn,
                    result,
                    enrich_code=enrich_code,
                    code_repo_search_fn=code_repo_search_fn,
                )
                return result
            if extraction_status is None:
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(meta, "skipped", chunks, llm_fn=llm_fn),
                )
                conn.commit()
            elif (
                extraction_status["confidence_score"] is None
                or extraction_status["confidence_reason"] is None
            ):
                stored_chunks, stored_methods = _paper_confidence_inputs(conn, paper_id)
                if not stored_chunks:
                    stored_chunks = chunks
                payload = _status_payload_from_row(extraction_status)
                confidence_score, confidence_reason = _extraction_confidence(
                    extraction_status["status"],
                    stored_methods,
                    stored_chunks,
                    extraction_status["compression_ratio"],
                    extraction_status["error_text"],
                )
                payload["confidence_score"] = confidence_score
                payload["confidence_reason"] = confidence_reason
                extraction_status = _upsert_extraction_status(conn, paper_id, payload)
                conn.commit()
            extraction_status, source_type_changed = _update_existing_source_type_if_explicit(
                conn,
                paper_id,
                extraction_status,
                meta,
            )
            if source_type_changed:
                conn.commit()
            result = {
                "paper_id": paper_id,
                "chunk_ids": [],
                "method_ids": [],
                "extraction_status": extraction_status,
                "duplicate": True,
            }
            if visual_index:
                result = _attach_stored_visual_index(
                    conn,
                    result,
                    meta,
                    visual_index_root,
                    runner=visual_index_runner,
                )
            result = _attach_code_repository_enrichment(
                conn,
                result,
                enrich_code=enrich_code,
                code_repo_search_fn=code_repo_search_fn,
            )
            return result

        cur.execute(
            """
            INSERT INTO papers (title, authors, abstract, url, publish_date)
            VALUES (:title, :authors, :abstract, :url, :publish_date)
            """,
            {
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "abstract": meta.get("abstract"),
                "url": meta.get("url"),
                "publish_date": meta.get("publish_date"),
            },
        )
        paper_id = cur.lastrowid

        chunk_ids = []
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
                (paper_id, "extracted", chunk),
            )
            chunk_ids.append(cur.lastrowid)

        method_ids = []
        normalized_methods = []
        if llm_fn is None:
            extraction_status = _upsert_extraction_status(
                conn,
                paper_id,
                _build_extraction_status(meta, "skipped", chunks, llm_fn=llm_fn),
            )
        else:
            try:
                result = extractor.extract_methods_from_paper(meta, llm_fn)
                method_ids, normalized_methods = _insert_or_link_extracted_methods(
                    conn,
                    chunks,
                    chunk_ids,
                    _iter_extracted_methods(result),
                    llm_fn,
                )
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(
                        meta, "extracted", chunks, normalized_methods, llm_fn=llm_fn
                    ),
                )
            except Exception as exc:
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(
                        meta, "failed", chunks, llm_fn=llm_fn, error_text=str(exc)
                    ),
                )
        conn.commit()
        result = {
            "paper_id": paper_id,
            "chunk_ids": chunk_ids,
            "method_ids": method_ids,
            "extraction_status": extraction_status,
        }
        if visual_index:
            result = _attach_stored_visual_index(
                conn,
                result,
                meta,
                visual_index_root,
                runner=visual_index_runner,
            )
        result = _attach_code_repository_enrichment(
            conn,
            result,
            enrich_code=enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        return result
    finally:
        if close:
            conn.close()


def _default_github_code_searcher(queries: list[str]) -> list[dict]:
    """Search GitHub repositories for automatic CLI code enrichment."""
    import requests

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    seen: set[str] = set()
    results: list[dict] = []
    for query_index, query in enumerate(queries):
        query = str(query or "").strip()
        if not query:
            continue
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        for rank, item in enumerate(items):
            html_url = item.get("html_url")
            if not html_url or html_url in seen:
                continue
            seen.add(html_url)
            results.append(
                {
                    "url": html_url,
                    "full_name": item.get("full_name"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "evidence": f"GitHub search query: {query}; rank={rank + 1}",
                    "confidence": max(0.35, 0.74 - 0.04 * query_index - 0.02 * rank),
                }
            )
    return results


def ingest_arxiv_id(
    arxiv_id: str,
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> dict:
    """Fetch and ingest a single arXiv paper.

    Wraps :func:`fetch_arxiv_source` and :func:`ingest_paper`, optionally
    overriding the ``source_type`` metadata label. Code repository enrichment
    runs automatically unless disabled. Returns both metadata and ingestion.
    """
    meta = fetch_arxiv_source(arxiv_id)
    _apply_source_type_override(meta, source_type)
    return {
        "meta": meta,
        "result": ingest_paper(
            meta,
            llm_fn=llm_fn,
            conn=conn,
            dry_run=dry_run,
            visual_index=visual_index,
            visual_index_root=visual_index_root,
            visual_index_runner=visual_index_runner,
            enrich_code=enrich_code,
            code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
        ),
    }


def ingest_arxiv_ids(
    arxiv_ids: list[str],
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> list[dict]:
    """Fetch and ingest arXiv papers with automatic code repository enrichment."""
    results: list[dict] = []
    for raw_id in arxiv_ids:
        arxiv_id = raw_id.strip()
        if not arxiv_id:
            continue
        try:
            outcome = ingest_arxiv_id(
                arxiv_id,
                llm_fn=llm_fn,
                conn=conn,
                dry_run=dry_run,
                source_type=source_type,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
            )
            meta = outcome.get("meta", {})
            result = outcome.get("result", {})
            title = meta.get("title", "")
            status = "unknown"
            paper_id = "planned" if dry_run else None
            if dry_run:
                planned = result.get("planned", {})
                status = (planned.get("extraction_status") or {}).get("status", "unknown")
            else:
                paper_id = result.get("paper_id")
                status = (result.get("extraction_status") or {}).get("status", "unknown")
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "paper_id": paper_id,
                    "status": status,
                    "error": None,
                    "result": result,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": "",
                    "paper_id": None,
                    "status": "error",
                    "error": str(exc),
                    "result": None,
                }
            )
    return results


def ingest_arxiv_query(
    query: str,
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    max_results: int = 10,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> list[dict]:
    """Search arXiv and ingest matches with automatic code repository enrichment."""
    arxiv_ids = fetch_arxiv_query(query, max_results=max_results)
    return ingest_arxiv_ids(
        arxiv_ids,
        llm_fn=llm_fn,
        conn=conn,
        dry_run=dry_run,
        source_type=source_type,
        visual_index=visual_index,
        visual_index_root=visual_index_root,
        visual_index_runner=visual_index_runner,
        enrich_code=enrich_code,
        code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
    )


def _print_arxiv_batch_summary(results: list[dict], dry_run: bool = False) -> None:
    """Print a concise summary line for each arXiv batch result."""
    if dry_run:
        print("DRY RUN - planned batch ingestion")
    else:
        print("Ingested batch:")
    has_visual_status = any((item.get("result") or {}).get("visual_index") for item in results)
    rule_width = 82 if has_visual_status else 70
    print("-" * rule_width)
    if has_visual_status:
        print(f"{'ID':<20}{'Paper ID':<14}{'Status':<14}{'Visual':<12}Title")
    else:
        print(f"{'ID':<20}{'Paper ID':<14}{'Status':<14}Title")
    print("-" * rule_width)
    for item in results:
        arxiv_id = item.get("arxiv_id", "")
        title = (item.get("title") or "").strip()
        paper_id = item.get("paper_id")
        if paper_id is None:
            paper_id_label = "error"
        elif paper_id == "planned":
            paper_id_label = "planned"
        else:
            paper_id_label = str(paper_id)
        status = item.get("status", "unknown")
        visual_index_payload = (item.get("result") or {}).get("visual_index") or {}
        visual_status = visual_index_payload.get("status", "")
        error = item.get("error")
        if has_visual_status:
            line = f"{arxiv_id:<20}{paper_id_label:<14}{status:<14}{visual_status:<12}{title}"
        else:
            line = f"{arxiv_id:<20}{paper_id_label:<14}{status:<14}{title}"
        print(line)
        if error:
            print(f"  -> error: {error}")


def _print_single_ingestion_result(result: dict, dry_run: bool = False) -> None:
    """Print the single-paper ingestion result, preserving the legacy format."""
    if dry_run:
        planned = result.get("planned", {})
        paper = planned.get("paper", {})
        print("DRY RUN - planned ingestion")
        print("=" * 40)
        print(f"Paper title : {paper.get('title')}")
        print(f"Authors     : {paper.get('authors')}")
        print(f"URL         : {paper.get('url')}")
        print(f"Publish date: {paper.get('publish_date')}")
        print(f"Chunks      : {len(planned.get('chunks', []))}")
        methods = planned.get("methods", [])
        merged_methods = planned.get("merged_methods", [])
        print(f"Methods     : {len(methods) + len(merged_methods)}")
        extraction_status = result.get("extraction_status") or planned.get("extraction_status", {})
        print(f"Extraction : {extraction_status.get('status')}")
        print(f"External ID : {extraction_status.get('external_id')}")
        print(f"Compression: {extraction_status.get('compressed_char_count')} / {extraction_status.get('source_char_count')} chars")
        print(f"Confidence : {extraction_status.get('confidence_score')} ({extraction_status.get('confidence_reason')})")
        visual_index_payload = result.get("visual_index") or planned.get("visual_index")
        if visual_index_payload:
            print(f"Visual index: {visual_index_payload.get('status')}")
            if visual_index_payload.get("error_text"):
                print(f"Visual error: {visual_index_payload.get('error_text')}")
        for method in methods:
            print(f"  - {method.get('name')} ({method.get('category')})")
        for method in merged_methods:
            print(f"  - {method.get('name')} (merged into existing)")
        if planned.get("extraction_error"):
            print(f"Extraction error: {planned['extraction_error']}")
    else:
        print("Ingested:")
        print(f"  paper_id : {result['paper_id']}")
        print(f"  chunk_ids: {result['chunk_ids']}")
        print(f"  method_ids: {result['method_ids']}")
        status = result.get("extraction_status", {})
        print(f"  extraction_status: {status.get('status')}")
        visual_index_payload = result.get("visual_index")
        if visual_index_payload:
            print(f"  visual_index: {visual_index_payload.get('status')}")
            if visual_index_payload.get("error_text"):
                print(f"  visual_error: {visual_index_payload.get('error_text')}")
        code_repositories = result.get("code_repositories") or []
        if code_repositories:
            print(f"  code_repositories: {len(code_repositories)} linked")
            for repo in code_repositories:
                url = repo.get("repository_url") or repo.get("url") or repo.get("html_url") or repo.get("repo_url")
                if url:
                    print(f"    selected: {url}")
        enrichment = result.get("code_repository_enrichment") or {}
        if enrichment.get("error_text"):
            print(f"  code_repository_error: {enrichment.get('error_text')}")


def main() -> None:
    """CLI entry point for ResearchFS ingestion."""
    parser = argparse.ArgumentParser(
        prog="researchfs.ingest",
        description="Ingest papers and methods into ResearchFS.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--arxiv", metavar="ID", help="Ingest an arXiv paper by ID")
    input_group.add_argument(
        "--arxiv-list",
        metavar="IDS",
        help="Ingest a comma-separated list of arXiv paper IDs",
    )
    input_group.add_argument(
        "--arxiv-query", metavar="QUERY", help="Ingest papers matching an arXiv search query"
    )
    input_group.add_argument("--repo", metavar="PATH", help="Ingest a GitHub README as a paper")
    input_group.add_argument(
        "--file", metavar="PATH", help="Ingest a local text/markdown/TeX/PDF paper note"
    )
    parser.add_argument(
        "--source-type",
        metavar="LABEL",
        type=_source_type_label,
        help="Set an explicit source_type metadata label for the ingested artifact",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract methods with a local LLM (only if one can be configured)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the records that would be created without writing to the DB",
    )
    parser.add_argument(
        "--enrich-code",
        action="store_true",
        help="Code repository enrichment is automatic; accepted for compatibility",
    )
    parser.add_argument(
        "--no-enrich-code",
        action="store_true",
        help="Disable automatic code repository enrichment and GitHub repository search",
    )
    parser.add_argument(
        "--visual-index",
        action="store_true",
        help="Build a PixelRAG visual index for the ingested paper",
    )
    parser.add_argument(
        "--visual-index-root",
        metavar="PATH",
        help="Directory for generated visual index artifacts",
    )
    parser.add_argument(
        "--max-results",
        metavar="N",
        type=int,
        default=10,
        help="Maximum arXiv results to ingest with --arxiv-query (default: 10)",
    )
    parser.add_argument(
        "--arxiv-proxy-list",
        metavar="PATH",
        help="Load arXiv request proxies from a local proxy-list file for this run",
    )
    args = parser.parse_args()
    if args.arxiv_proxy_list:
        os.environ["RESEARCHFS_ARXIV_PROXY_LIST_PATH"] = args.arxiv_proxy_list

    if not args.arxiv and not args.arxiv_list and not args.arxiv_query and not args.repo and not args.file:
        parser.print_help()
        return

    llm_fn: Optional[Callable[[str], str]] = None
    if args.extract:
        llm_fn = _build_llm_fn()
        if llm_fn is None:
            print(
                "Warning: --extract was requested but no local LLM could be built. "
                "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) "
                "or implement _build_llm_fn(). Ingesting without extraction."
            )

    effective_enrich_code = not args.no_enrich_code
    code_repo_search_fn = _default_github_code_searcher if effective_enrich_code else None

    if args.arxiv:
        meta = fetch_arxiv_source(args.arxiv)
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)
    elif args.arxiv_list:
        arxiv_ids = [id_.strip() for id_ in args.arxiv_list.split(",")]
        arxiv_ids = [id_ for id_ in arxiv_ids if id_]
        results = ingest_arxiv_ids(
            arxiv_ids,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            source_type=args.source_type,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_arxiv_batch_summary(results, dry_run=args.dry_run)
    elif args.arxiv_query:
        results = ingest_arxiv_query(
            args.arxiv_query,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            source_type=args.source_type,
            max_results=args.max_results,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_arxiv_batch_summary(results, dry_run=args.dry_run)
    elif args.repo:
        meta = fetch_github_readme(args.repo)
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)
    else:
        try:
            meta = load_local_file_metadata(args.file)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
