"""No-dependency lab-facing API for ResearchFS.

The module exposes an in-process dispatcher for tests and agent SDKs, plus a
small stdlib ``http.server`` entrypoint for labs that want a local JSON API
without adopting a web framework.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from researchfs.code_repos import enrich_paper_code_repositories
from researchfs.db import get_connection, initialize_db
from researchfs.evaluation import evaluate_extraction
from researchfs.fetcher import fetch_arxiv_source, fetch_github_readme
from researchfs.ingest import (
    LocalFileExtractionError,
    _code_repository_rows_from_enrichment,
    _mark_source_type_explicit,
    _source_type_label,
    ingest_paper,
    load_local_file_metadata,
)
from researchfs.bundle import export_corpus_memory_bundle, export_paper_memory_bundle, import_memory_bundle
from researchfs.paperstream import (
    get_arxiv_paper_chunks,
    get_arxiv_paper_state,
    get_paperstream_status,
    list_arxiv_jobs,
    list_missing_arxiv_assets,
    search_paperstream,
)
from researchfs import retrieve as rfs

JSONDict = dict[str, Any]
LLMFn = Callable[[str], str]
RoutedAskReader = Callable[[str, list[Any]], Mapping[str, Any] | str]
RoutedAskVerifier = Callable[[Mapping[str, Any], list[Any]], Mapping[str, Any] | bool]
ArxivFetcher = Callable[[str], dict[str, Any]]
RepoFetcher = Callable[[str], dict[str, Any]]
CodeRepoSearcher = Callable[[list[str]], list[Any]]

SERVICE_NAME = "ResearchFS lab API"
LOCAL_ONLY_CAVEAT = (
    "Local-only stdlib HTTP server for browser and local agent clients; "
    "bind to 127.0.0.1 unless you add your own network controls."
)
_ROUTE_METADATA: tuple[JSONDict, ...] = (
    {"method": "GET", "path": "/", "description": "Compact API index"},
    {"method": "GET", "path": "/health", "description": "Local service and database smoke check"},
    {"method": "GET", "path": "/papers", "description": "List papers with extraction status"},
    {"method": "GET", "path": "/paper/{paper_id}/memory", "description": "Retrieve one compact paper memory card"},
    {"method": "GET", "path": "/paper/{paper_id}/memory/export", "description": "Export a compressed paper memory bundle"},
    {"method": "GET", "path": "/memory/export", "description": "Export a compressed corpus memory bundle"},
    {"method": "GET", "path": "/paper/{paper_id}", "description": "Inspect one paper and provenance"},
    {"method": "GET", "path": "/method/{method_id}", "description": "Inspect one method and feedback"},
    {"method": "GET", "path": "/stats", "description": "Aggregate ResearchFS memory statistics"},
    {"method": "POST", "path": "/ingest", "description": "Ingest paper metadata and extracted methods"},
    {"method": "POST", "path": "/ingest/arxiv", "description": "Fetch and ingest arXiv source metadata"},
    {"method": "GET", "path": "/arxiv/search", "description": "Search PaperStream arXiv metadata and chunks"},
    {"method": "GET", "path": "/arxiv/papers/{arxiv_id}/chunks", "description": "Retrieve bounded PaperStream arXiv paper chunks"},
    {"method": "GET", "path": "/arxiv/papers/{arxiv_id}", "description": "Inspect one PaperStream arXiv paper state"},
    {"method": "GET", "path": "/arxiv/assets/missing", "description": "List PaperStream arXiv papers missing a PDF or source asset"},
    {"method": "GET", "path": "/arxiv/jobs", "description": "List PaperStream arXiv queue jobs"},
    {"method": "GET", "path": "/arxiv/sync/status", "description": "Inspect PaperStream sync and queue status"},
    {"method": "POST", "path": "/ingest/repo", "description": "Fetch and ingest GitHub repository README metadata"},
    {"method": "POST", "path": "/ingest/file", "description": "Ingest a local text/Markdown/TeX/PDF file"},
    {"method": "POST", "path": "/paper/{paper_id}/code-repositories/enrich", "description": "Enrich one paper with linked code repositories"},
    {"method": "POST", "path": "/evaluate/extraction", "description": "Evaluate stored extraction quality"},
    {"method": "POST", "path": "/retrieve", "description": "Retrieve compressed lab memory"},
    {"method": "POST", "path": "/brief", "description": "Retrieve a bounded compressed research brief"},
    {"method": "POST", "path": "/ask", "description": "Run a routed local ask workflow over compressed ResearchFS memory"},
    {"method": "POST", "path": "/memory/import", "description": "Import a compressed paper or corpus memory bundle"},
    {"method": "POST", "path": "/feedback", "description": "Record usefulness feedback for a method"},
    {"method": "POST", "path": "/experiment", "description": "Log a lab experiment"},
    {"method": "POST", "path": "/lesson", "description": "Log a reusable lab lesson"},
)
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "600",
}


def _implemented_routes() -> list[JSONDict]:
    return [dict(route) for route in _ROUTE_METADATA]



@dataclass(slots=True)
class LabAPIResponse:
    """JSON response returned by the in-process dispatcher."""

    status: int
    body: JSONDict
    headers: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/json; charset=utf-8"}
    )

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.body, sort_keys=True).encode("utf-8")


class LabAPIError(Exception):
    """Expected request/route error with an HTTP status code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json_error(status: int, code: str, message: str) -> LabAPIResponse:
    return LabAPIResponse(status, {"ok": False, "error": {"code": code, "message": message}})


def _parse_body(body: Any) -> JSONDict:
    if body is None or body == b"" or body == "":
        return {}
    if isinstance(body, dict):
        return body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LabAPIError(400, "invalid_json", f"Request body is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise LabAPIError(400, "invalid_json", "JSON request body must be an object")
        return parsed
    raise LabAPIError(400, "invalid_body", "Request body must be a JSON object")


def _required_str(payload: JSONDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LabAPIError(400, "missing_field", f"'{key}' must be a non-empty string")
    return value.strip()


def _optional_int(payload: JSONDict, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be an integer")
    parsed = value
    if parsed < 0:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be non-negative")
    return parsed


def _query_optional_int(query: str, key: str) -> int | None:
    values = parse_qs(query, keep_blank_values=True).get(key)
    if not values:
        return None
    value = values[-1]
    if value == "" or isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be an integer") from exc
    if parsed < 0:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be non-negative")
    return parsed

def _query_optional_bool(query: str, key: str) -> bool | None:
    values = parse_qs(query, keep_blank_values=True).get(key)
    if not values:
        return None
    value = values[-1]
    if not isinstance(value, str):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a boolean")
    normalized = value.strip().lower()
    if not normalized:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a boolean")
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise LabAPIError(400, "invalid_field", f"'{key}' must be a boolean")

def _query_optional_str(query: str, key: str) -> str | None:
    values = parse_qs(query, keep_blank_values=True).get(key)
    if not values:
        return None
    value = values[-1]
    if not isinstance(value, str):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a string")
    value = value.strip()
    if not value:
        raise LabAPIError(400, "invalid_field", f"'{key}' must not be blank")
    return value


def _query_optional_str_list(query: str, key: str) -> list[str] | None:
    values = parse_qs(query, keep_blank_values=True).get(key)
    if not values:
        return None
    parsed_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise LabAPIError(400, "invalid_field", f"'{key}' must contain non-empty strings")
        for part in value.split(","):
            part = part.strip()
            if not part:
                raise LabAPIError(400, "invalid_field", f"'{key}' must contain non-empty strings")
            if part not in parsed_values:
                parsed_values.append(part)
    return parsed_values

def _url_positive_int(value: str, name: str, code: str) -> int:
    if not value.isdecimal():
        raise LabAPIError(400, code, f"{name} id must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise LabAPIError(400, code, f"{name} id must be a positive integer")
    return parsed


def _query_optional_int_list(query: str, key: str) -> list[int] | None:
    values = parse_qs(query, keep_blank_values=True).get(key)
    if not values:
        return None
    parsed_values: list[int] = []
    for value in values:
        if value == "":
            raise LabAPIError(400, "invalid_field", f"'{key}' must contain integers")
        for part in value.split(","):
            part = part.strip()
            if not part:
                raise LabAPIError(400, "invalid_field", f"'{key}' must contain integers")
            try:
                parsed = int(part)
            except (TypeError, ValueError) as exc:
                raise LabAPIError(400, "invalid_field", f"'{key}' must contain integers") from exc
            if parsed <= 0:
                raise LabAPIError(400, "invalid_field", f"'{key}' must contain positive integers")
            if parsed not in parsed_values:
                parsed_values.append(parsed)
    return parsed_values




def _optional_float(payload: JSONDict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a number")
    return parsed

def _optional_positive_int(payload: JSONDict, key: str) -> int | None:
    value = _optional_int(payload, key)
    if value is not None and value <= 0:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be positive")
    return value


def _optional_non_negative_int(payload: JSONDict, key: str) -> int | None:
    return _optional_int(payload, key)


def _optional_finite_float(payload: JSONDict, key: str) -> float | None:
    return _optional_float(payload, key)

def _optional_metric_threshold(payload: JSONDict, key: str) -> float | None:
    value = _optional_float(payload, key)
    if value is None:
        return None
    if value < 0.0 or value > 1.0:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be between 0.0 and 1.0")
    return value


def _optional_bool(payload: JSONDict, key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a boolean")
    return value

def _optional_source_types(payload: JSONDict):
    if payload.get("source_types") is not None:
        return payload.get("source_types")
    if "source_type" in payload:
        return payload.get("source_type")
    return None





def _required_int(payload: JSONDict, key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise LabAPIError(400, "missing_field", f"'{key}' must be an integer")
    if not isinstance(value, int) or isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be an integer")
    parsed = value
    if parsed < 0:
        raise LabAPIError(400, "invalid_field", f"'{key}' must be non-negative")
    return parsed


def _required_feedback_rating(payload: JSONDict) -> float:
    if "rating" not in payload:
        raise LabAPIError(400, "missing_field", "'rating' must be a number between -1.0 and 1.0")
    value = payload.get("rating")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LabAPIError(400, "invalid_field", "'rating' must be a number between -1.0 and 1.0")
    rating = float(value)
    if not math.isfinite(rating):
        raise LabAPIError(400, "invalid_field", "'rating' must be a number between -1.0 and 1.0")
    if rating < -1.0 or rating > 1.0:
        raise LabAPIError(400, "invalid_field", "'rating' must be between -1.0 and 1.0")
    return rating


def _optional_str(payload: JSONDict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LabAPIError(400, "invalid_field", f"'{key}' must be a string")
    return value

def _visual_index_options(payload: JSONDict) -> tuple[bool, str | None]:
    return (
        _optional_bool(payload, "visual_index", False),
        _optional_str(payload, "visual_index_root"),
    )


def _optional_method_ids(payload: JSONDict) -> list[int]:
    method_ids = payload.get("method_ids")
    if method_ids is None:
        return []
    if not isinstance(method_ids, list):
        raise LabAPIError(400, "invalid_field", "'method_ids' must be a list of integers")
    ids: list[int] = []
    for raw in method_ids:
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise LabAPIError(400, "invalid_field", "'method_ids' must contain only integers")
        if raw < 0:
            raise LabAPIError(400, "invalid_field", "'method_ids' must be non-negative")
        ids.append(raw)
    return ids


_METHOD_SOURCE_EVIDENCE_FIELDS = (
    "evidence_text",
    "evidence_start",
    "evidence_end",
    "evidence_strategy",
)

_PAPER_EXTRACTION_STATUS_CONFIDENCE_FIELDS = (
    "confidence_score",
    "confidence_reason",
)


def _table_column_names(conn: Any, table: str) -> set[str]:
    """Return column names for a SQLite table."""
    return {
        row["name"] if hasattr(row, "keys") else row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _method_source_evidence_select(conn: Any) -> str:
    """Return SELECT aliases for optional method_sources evidence columns."""
    columns = _table_column_names(conn, "method_sources")
    return ",\n                    ".join(
        (
            f"ms.{field}"
            if field in columns
            else f"NULL AS {field}"
        )
        for field in _METHOD_SOURCE_EVIDENCE_FIELDS
    )


def _paper_from_payload(payload: JSONDict) -> JSONDict:
    paper = payload.get("paper")
    if paper is None:
        paper = payload
    if not isinstance(paper, dict):
        raise LabAPIError(400, "invalid_field", "'paper' must be an object")
    meta = {
        "title": _required_str(paper, "title"),
        "authors": paper.get("authors"),
        "abstract": paper.get("abstract"),
        "url": paper.get("url"),
        "publish_date": paper.get("publish_date"),
        "external_id": paper.get("external_id") if "external_id" in paper else payload.get("external_id"),
        "content": paper.get("content", paper.get("abstract", "")),
    }
    source_type_supplied = False
    for key in (
        "source_type",
        "extractor_model",
        "extractor_prompt_version",
        "visual_source",
        "visual_source_path",
        "visual_source_url",
    ):
        if key in paper:
            meta[key] = paper[key]
            source_type_supplied = source_type_supplied or key == "source_type"
        elif key in payload:
            meta[key] = payload[key]
            source_type_supplied = source_type_supplied or key == "source_type"
    if source_type_supplied and meta.get("source_type"):
        _mark_source_type_explicit(meta)
    if not isinstance(meta["content"], str) or not meta["content"].strip():
        raise LabAPIError(400, "missing_field", "'content' or 'abstract' must be a non-empty string")
    return meta


def _methods_llm(methods: Any) -> LLMFn:
    if not isinstance(methods, list):
        raise LabAPIError(400, "invalid_field", "'methods' must be a list when provided")

    def llm_fn(_prompt: str) -> str:
        return "```json\n" + json.dumps({"methods": methods}) + "\n```"

    return llm_fn


def _select_llm(payload: JSONDict, default_llm: LLMFn | None) -> LLMFn | None:
    if "methods" in payload:
        return _methods_llm(payload["methods"])
    return default_llm


def _ingestion_metadata(result: JSONDict, llm_fn: LLMFn | None) -> JSONDict:
    extraction_record = result.get("extraction_status")
    extraction_error = (
        result.get("extraction_error")
        or result.get("planned", {}).get("extraction_error")
        or (
            extraction_record.get("error_text")
            if isinstance(extraction_record, dict)
            else None
        )
    )
    method_ids = result.get("method_ids") or []
    chunk_ids = result.get("chunk_ids") or []

    if isinstance(extraction_record, dict):
        extraction_status = extraction_record.get("status") or "unknown"
    else:
        extraction_status = extraction_record
    if extraction_status is None:
        if extraction_error:
            extraction_status = "error"
        elif llm_fn is None:
            extraction_status = "skipped"
        elif method_ids:
            extraction_status = "extracted"
        else:
            extraction_status = "empty"

    compression_status = result.get("compression_status")
    if compression_status is None:
        compression_status = "chunked" if chunk_ids else "unchanged"
    metadata = {
        "compression_status": compression_status,
        "extraction_status": extraction_status,
    }
    if isinstance(extraction_record, dict):
        metadata["extraction"] = extraction_record
        for key in _PAPER_EXTRACTION_STATUS_CONFIDENCE_FIELDS:
            if key in extraction_record:
                metadata[key] = extraction_record[key]
    if extraction_error:
        metadata["extraction_error"] = extraction_error
    for key in ("compression", "status"):
        if key in result:
            metadata[key] = result[key]
    return metadata


def _limit_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    suffix = "\n... truncated"
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def _row_to_dict(row: Any) -> JSONDict:
    return dict(row) if row is not None else {}


def _unique_ints(values: list[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _paper_record(row: Any) -> JSONDict:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "authors": row["authors"],
        "abstract": row["abstract"],
        "url": row["url"],
        "publish_date": row["publish_date"],
        "created_at": row["created_at"],
    }


def _extraction_status_record(row: Any | None) -> JSONDict | None:
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
        "extractor_model": row["extractor_model"],
        "extractor_prompt_version": row["extractor_prompt_version"],
        "error_text": row["error_text"],
        "confidence_score": row["confidence_score"],
        "confidence_reason": row["confidence_reason"],
        "external_id": row["external_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

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
            extractor_model,
            extractor_prompt_version,
            error_text,
            confidence_score,
            confidence_reason,
            external_id,
            created_at,
            updated_at
        FROM paper_extraction_status
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    return _extraction_status_record(row)


def _attach_current_extraction_status(conn: Any, result: JSONDict) -> JSONDict:
    paper_id = result.get("paper_id")
    if isinstance(paper_id, bool) or paper_id is None:
        return result
    try:
        parsed_paper_id = int(paper_id)
    except (TypeError, ValueError):
        return result
    extraction_status = _load_extraction_status(conn, parsed_paper_id)
    if extraction_status is None:
        return result
    return {**result, "extraction_status": extraction_status}


class ResearchFSLabAPI:
    """Small JSON dispatcher for ResearchFS lab and auto-research agent access."""

    def __init__(
        self,
        *,
        conn: Any | None = None,
        db_path: str | None = None,
        llm_fn: LLMFn | None = None,
        arxiv_fetcher: ArxivFetcher | None = None,
        repo_fetcher: RepoFetcher | None = None,
        code_repo_searcher: CodeRepoSearcher | None = None,
        visual_index_runner=None,
        reader_fn: RoutedAskReader | None = None,
        verifier_fn: RoutedAskVerifier | None = None,
    ) -> None:
        if conn is not None and db_path is not None:
            raise ValueError("Pass either conn or db_path, not both")
        self._conn = conn
        self._db_path = db_path
        self._llm_fn = llm_fn
        self._arxiv_fetcher = arxiv_fetcher or fetch_arxiv_source
        self._repo_fetcher = repo_fetcher or fetch_github_readme
        self._code_repo_searcher = code_repo_searcher
        self._visual_index_runner = visual_index_runner
        self._reader_fn = reader_fn
        self._verifier_fn = verifier_fn

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        if self._conn is not None:
            yield self._conn
            return
        conn = get_connection(self._db_path)
        try:
            initialize_db(conn)
            yield conn
        finally:
            conn.close()

    def dispatch(self, method: str, path: str, body: Any = None) -> LabAPIResponse:
        """Dispatch one in-process request and return a JSON response object."""
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        method = method.upper()
        try:
            if method == "GET" and route == "/":
                return self._handle_index()
            if method == "GET" and route == "/health":
                return self._handle_health()
            if method == "POST" and route == "/ingest":
                return self._handle_ingest(_parse_body(body))
            if method == "POST" and route == "/ingest/arxiv":
                return self._handle_ingest_arxiv(_parse_body(body))
            if method == "POST" and route == "/ingest/repo":
                return self._handle_ingest_repo(_parse_body(body))
            if method == "POST" and route == "/ingest/file":
                return self._handle_ingest_file(_parse_body(body))
            if method == "POST" and route.startswith("/paper/") and route.endswith("/code-repositories/enrich"):
                paper_id_text = route.removeprefix("/paper/").removesuffix("/code-repositories/enrich")
                paper_id = _url_positive_int(paper_id_text, "Paper", "invalid_paper_id")
                return self._handle_enrich_paper_code_repositories(paper_id)
            if method == "GET" and route == "/papers":
                return self._handle_papers()
            if method == "GET" and route == "/stats":
                return self._handle_stats()
            if method == "GET" and route == "/arxiv/search":
                query = _query_optional_str(parsed.query, "query")
                limit = _query_optional_int(parsed.query, "limit")
                max_chunk_chars = _query_optional_int(parsed.query, "max_chunk_chars")
                include_chunks = _query_optional_bool(parsed.query, "include_chunks")
                return self._handle_arxiv_search(query, limit, max_chunk_chars, include_chunks)
            if method == "GET" and route.startswith("/arxiv/papers/") and route.endswith("/chunks"):
                arxiv_id = unquote(route.removeprefix("/arxiv/papers/").removesuffix("/chunks"))
                version = _query_optional_int(parsed.query, "version")
                limit = _query_optional_int(parsed.query, "limit")
                offset = _query_optional_int(parsed.query, "offset")
                max_chars = _query_optional_int(parsed.query, "max_chars")
                return self._handle_arxiv_paper_chunks(arxiv_id, version, limit, offset, max_chars)
            if method == "GET" and route.startswith("/arxiv/papers/"):
                arxiv_id = unquote(route.removeprefix("/arxiv/papers/"))
                return self._handle_arxiv_paper(arxiv_id)
            if method == "GET" and route == "/arxiv/assets/missing":
                asset_type = _query_optional_str(parsed.query, "asset_type")
                category = _query_optional_str(parsed.query, "category")
                limit = _query_optional_int(parsed.query, "limit")
                return self._handle_arxiv_missing_assets(asset_type, category, limit)
            if method == "GET" and route == "/arxiv/jobs":
                status = _query_optional_str(parsed.query, "status")
                job_types = _query_optional_str_list(parsed.query, "job_types")
                limit = _query_optional_int(parsed.query, "limit")
                return self._handle_arxiv_jobs(status, job_types, limit)
            if method == "GET" and route == "/arxiv/sync/status":
                return self._handle_arxiv_sync_status()
            if method == "GET" and route == "/memory/export":
                paper_ids = _query_optional_int_list(parsed.query, "paper_ids")
                max_chars = _query_optional_int(parsed.query, "max_chars")
                return self._handle_corpus_memory_export(paper_ids, max_chars)
            if method == "GET" and route.startswith("/paper/") and route.endswith("/memory/export"):
                paper_id_text = route.removeprefix("/paper/").removesuffix("/memory/export")
                paper_id = _url_positive_int(paper_id_text, "Paper", "invalid_paper_id")
                max_chars = _query_optional_int(parsed.query, "max_chars")
                return self._handle_paper_memory_export(paper_id, max_chars)
            if method == "GET" and route.startswith("/paper/") and route.endswith("/memory"):
                paper_id_text = route.removeprefix("/paper/").removesuffix("/memory")
                paper_id = _url_positive_int(paper_id_text, "Paper", "invalid_paper_id")
                max_chars = _query_optional_int(parsed.query, "max_chars")
                return self._handle_paper_memory(paper_id, max_chars)
            if method == "GET" and route.startswith("/paper/"):
                paper_id_text = route.removeprefix("/paper/")
                paper_id = _url_positive_int(paper_id_text, "Paper", "invalid_paper_id")
                return self._handle_paper_detail(paper_id)
            if method == "POST" and route == "/evaluate/extraction":
                return self._handle_evaluate_extraction(_parse_body(body))
            if method == "POST" and route == "/memory/import":
                return self._handle_memory_import(_parse_body(body))
            if method == "POST" and route == "/retrieve":
                return self._handle_retrieve(_parse_body(body))
            if method == "POST" and route == "/brief":
                return self._handle_brief(_parse_body(body))
            if method == "POST" and route == "/ask":
                return self._handle_ask(_parse_body(body))
            if method == "GET" and route.startswith("/method/"):
                method_id_text = route.removeprefix("/method/")
                method_id = _url_positive_int(method_id_text, "Method", "invalid_method_id")
                return self._handle_method_detail(method_id)
            if method == "POST" and route == "/feedback":
                return self._handle_feedback(_parse_body(body))
            if method == "POST" and route == "/experiment":
                return self._handle_experiment(_parse_body(body))
            if method == "POST" and route == "/lesson":
                return self._handle_lesson(_parse_body(body))
            return _json_error(404, "not_found", f"No route for {method} {parsed.path}")
        except LabAPIError as exc:
            return _json_error(exc.status, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive API boundary
            return _json_error(500, "internal_error", str(exc))

    def _handle_index(self) -> LabAPIResponse:
        return LabAPIResponse(
            200,
            {
                "ok": True,
                "service": SERVICE_NAME,
                "local_only": True,
                "caveat": LOCAL_ONLY_CAVEAT,
                "routes": _implemented_routes(),
            },
        )

    def _handle_health(self) -> LabAPIResponse:
        with self._connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return LabAPIResponse(
            200,
            {
                "ok": True,
                "service": {"name": SERVICE_NAME, "status": "ok"},
                "db": {"status": "ok"},
                "local_only": True,
            },
        )

    def _handle_arxiv_search(
        self,
        query: str | None,
        limit: int | None,
        max_chunk_chars: int | None,
        include_chunks: bool | None,
    ) -> LabAPIResponse:
        if query is None:
            raise LabAPIError(400, "missing_field", "'query' must be provided")
        kwargs: JSONDict = {}
        if limit is not None:
            kwargs["limit"] = limit
        if max_chunk_chars is not None:
            kwargs["max_chunk_chars"] = max_chunk_chars
        if include_chunks is not None:
            kwargs["include_chunks"] = include_chunks
        with self._connection() as conn:
            try:
                payload = search_paperstream(conn, query, **kwargs)
            except (TypeError, ValueError) as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
        return LabAPIResponse(200, payload)

    def _handle_arxiv_paper_chunks(
        self,
        arxiv_id: str,
        version: int | None,
        limit: int | None,
        offset: int | None,
        max_chars: int | None,
    ) -> LabAPIResponse:
        kwargs: JSONDict = {}
        if version is not None:
            kwargs["version"] = version
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if max_chars is not None:
            kwargs["max_chars"] = max_chars
        with self._connection() as conn:
            try:
                payload = get_arxiv_paper_chunks(conn, arxiv_id, **kwargs)
            except (TypeError, ValueError) as exc:
                code = "invalid_arxiv_id" if "arxiv_id" in str(exc) else "invalid_field"
                raise LabAPIError(400, code, str(exc)) from exc
        if payload is None:
            raise LabAPIError(404, "not_found", f"No chunks found for arXiv paper id {arxiv_id}")
        return LabAPIResponse(200, payload)

    def _handle_arxiv_paper(self, arxiv_id: str) -> LabAPIResponse:
        with self._connection() as conn:
            try:
                paper = get_arxiv_paper_state(conn, arxiv_id)
            except (TypeError, ValueError) as exc:
                raise LabAPIError(400, "invalid_arxiv_id", str(exc)) from exc
        if paper is None:
            raise LabAPIError(404, "not_found", f"No arXiv paper found for id {arxiv_id}")
        return LabAPIResponse(200, {"ok": True, **paper})

    def _handle_arxiv_missing_assets(
        self,
        asset_type: str | None,
        category: str | None,
        limit: int | None,
    ) -> LabAPIResponse:
        if asset_type is None:
            raise LabAPIError(400, "invalid_field", "'asset_type' is required")
        kwargs: JSONDict = {"asset_type": asset_type}
        if category is not None:
            kwargs["category"] = category
        if limit is not None:
            kwargs["limit"] = limit
        with self._connection() as conn:
            try:
                assets = list_missing_arxiv_assets(conn, **kwargs)
            except (TypeError, ValueError) as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "assets": assets})

    def _handle_arxiv_jobs(
        self,
        status: str | None,
        job_types: list[str] | None,
        limit: int | None,
    ) -> LabAPIResponse:
        kwargs: JSONDict = {}
        if status is not None:
            kwargs["status"] = status
        if job_types is not None:
            kwargs["job_types"] = job_types
        if limit is not None:
            kwargs["limit"] = limit
        with self._connection() as conn:
            try:
                jobs = list_arxiv_jobs(conn, **kwargs)
            except (TypeError, ValueError) as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "jobs": jobs})

    def _handle_arxiv_sync_status(self) -> LabAPIResponse:
        with self._connection() as conn:
            status = get_paperstream_status(conn)
        return LabAPIResponse(200, {"ok": True, **status})

    def _handle_ingest(self, payload: JSONDict) -> LabAPIResponse:
        meta = _paper_from_payload(payload)
        llm_fn = _select_llm(payload, self._llm_fn)
        visual_index, visual_index_root = _visual_index_options(payload)
        enrich_code = _optional_bool(payload, "enrich_code", True)
        with self._connection() as conn:
            result = ingest_paper(
                meta,
                llm_fn=llm_fn,
                conn=conn,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=self._visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=self._code_repo_searcher,
            )
            result = _attach_current_extraction_status(conn, result)
        body = {"ok": True, **result, "ingestion": _ingestion_metadata(result, llm_fn)}
        return LabAPIResponse(200, body)

    def _handle_ingest_arxiv(self, payload: JSONDict) -> LabAPIResponse:
        arxiv_id = _required_str(payload, "id")
        try:
            meta = self._arxiv_fetcher(arxiv_id)
        except ValueError as exc:
            raise LabAPIError(400, "invalid_arxiv_id", str(exc)) from exc
        except Exception as exc:
            raise LabAPIError(502, "fetch_failed", str(exc)) from exc
        if not isinstance(meta, dict):
            raise LabAPIError(502, "fetch_failed", "arXiv fetcher did not return paper metadata")
        meta = dict(meta)
        if "source_type" in payload:
            raw_source_type = payload["source_type"]
            if not isinstance(raw_source_type, str):
                raise LabAPIError(400, "invalid_field", "'source_type' must be a string")
            try:
                meta["source_type"] = _source_type_label(raw_source_type)
            except Exception as exc:
                raise LabAPIError(400, "invalid_field", "'source_type' must be a non-empty string") from exc
            _mark_source_type_explicit(meta)
        for key in ("extractor_model", "extractor_prompt_version", "external_id"):
            if key in payload:
                meta[key] = payload[key]
        llm_fn = _select_llm(payload, self._llm_fn)
        visual_index, visual_index_root = _visual_index_options(payload)
        enrich_code = _optional_bool(payload, "enrich_code", True)
        with self._connection() as conn:
            result = ingest_paper(
                meta,
                llm_fn=llm_fn,
                conn=conn,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=self._visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=self._code_repo_searcher,
            )
            result = _attach_current_extraction_status(conn, result)
        body = {
            "ok": True,
            "arxiv_id": arxiv_id,
            **result,
            "ingestion": _ingestion_metadata(result, llm_fn),
        }
        return LabAPIResponse(200, body)

    def _handle_ingest_repo(self, payload: JSONDict) -> LabAPIResponse:
        repo = _required_str(payload, "repo")
        try:
            meta = self._repo_fetcher(repo)
        except ValueError as exc:
            raise LabAPIError(400, "invalid_repo", str(exc)) from exc
        except Exception as exc:
            raise LabAPIError(502, "fetch_failed", str(exc)) from exc
        if not isinstance(meta, dict):
            raise LabAPIError(502, "fetch_failed", "Repository fetcher did not return paper metadata")
        meta = dict(meta)
        raw_source_type = payload.get("source_type", "repo_readme")
        if not isinstance(raw_source_type, str):
            raise LabAPIError(400, "invalid_field", "'source_type' must be a string")
        try:
            meta["source_type"] = _source_type_label(raw_source_type)
        except Exception as exc:
            raise LabAPIError(400, "invalid_field", "'source_type' must be a non-empty string") from exc
        _mark_source_type_explicit(meta)
        for key in ("extractor_model", "extractor_prompt_version", "external_id"):
            if key in payload:
                meta[key] = payload[key]
        llm_fn = _select_llm(payload, self._llm_fn)
        visual_index, visual_index_root = _visual_index_options(payload)
        enrich_code = _optional_bool(payload, "enrich_code", True)
        with self._connection() as conn:
            result = ingest_paper(
                meta,
                llm_fn=llm_fn,
                conn=conn,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=self._visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=self._code_repo_searcher,
            )
            result = _attach_current_extraction_status(conn, result)
        body = {
            "ok": True,
            "repo": repo,
            **result,
            "ingestion": _ingestion_metadata(result, llm_fn),
        }
        return LabAPIResponse(200, body)

    def _handle_ingest_file(self, payload: JSONDict) -> LabAPIResponse:
        path = _required_str(payload, "path")
        override_keys = (
            "title",
            "authors",
            "abstract",
            "publish_date",
            "source_type",
            "extractor_model",
            "extractor_prompt_version",
            "external_id",
        )
        overrides = {key: payload[key] for key in override_keys if key in payload}
        try:
            meta = load_local_file_metadata(path, overrides)
        except FileNotFoundError as exc:
            raise LabAPIError(400, "file_not_found", str(exc)) from exc
        except LocalFileExtractionError as exc:
            raise LabAPIError(400, "extraction_failed", str(exc)) from exc
        except ValueError as exc:
            raise LabAPIError(400, "unsupported_file_type", str(exc)) from exc
        llm_fn = _select_llm(payload, self._llm_fn)
        visual_index, visual_index_root = _visual_index_options(payload)
        enrich_code = _optional_bool(payload, "enrich_code", True)
        with self._connection() as conn:
            result = ingest_paper(
                meta,
                llm_fn=llm_fn,
                conn=conn,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=self._visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=self._code_repo_searcher,
            )
            result = _attach_current_extraction_status(conn, result)
        body = {
            "ok": True,
            "path": path,
            **result,
            "ingestion": _ingestion_metadata(result, llm_fn),
        }
        return LabAPIResponse(200, body)

    def _handle_enrich_paper_code_repositories(self, paper_id: int) -> LabAPIResponse:
        with self._connection() as conn:
            try:
                result = enrich_paper_code_repositories(
                    conn,
                    paper_id,
                    search_fn=self._code_repo_searcher,
                )
            except LookupError as exc:
                raise LabAPIError(404, "paper_not_found", str(exc)) from exc
            except ValueError as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
            except Exception as exc:
                raise LabAPIError(502, "code_repo_search_failed", str(exc)) from exc
        return LabAPIResponse(
            200,
            {
                "ok": True,
                "code_repository_enrichment": result,
                "code_repositories": _code_repository_rows_from_enrichment(result),
            },
        )


    def _handle_papers(self) -> LabAPIResponse:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    p.paper_id,
                    p.title,
                    p.authors,
                    p.abstract,
                    p.url,
                    p.publish_date,
                    p.created_at,
                    s.paper_id AS status_paper_id,
                    s.status,
                    s.source_type,
                    s.chunk_count,
                    s.method_count,
                    s.source_char_count,
                    s.compressed_char_count,
                    s.compression_ratio,
                    s.extractor_model,
                    s.extractor_prompt_version,
                    s.error_text,
                    s.confidence_score,
                    s.confidence_reason,
                    s.created_at AS status_created_at,
                    s.updated_at AS status_updated_at
                FROM papers p
                LEFT JOIN paper_extraction_status s ON s.paper_id = p.paper_id
                ORDER BY p.paper_id
                """
            )
            papers: list[JSONDict] = []
            for row in cur.fetchall():
                status = None
                if row["status_paper_id"] is not None:
                    status = {
                        "paper_id": row["status_paper_id"],
                        "status": row["status"],
                        "source_type": row["source_type"],
                        "chunk_count": row["chunk_count"],
                        "method_count": row["method_count"],
                        "source_char_count": row["source_char_count"],
                        "compressed_char_count": row["compressed_char_count"],
                        "compression_ratio": row["compression_ratio"],
                        "extractor_model": row["extractor_model"],
                        "extractor_prompt_version": row["extractor_prompt_version"],
                        "error_text": row["error_text"],
                        "confidence_score": row["confidence_score"],
                        "confidence_reason": row["confidence_reason"],
                        "created_at": row["status_created_at"],
                        "updated_at": row["status_updated_at"],
                    }
                papers.append({**_paper_record(row), "extraction_status": status})
        return LabAPIResponse(200, {"ok": True, "papers": papers})

    def _handle_paper_detail(self, paper_id: int) -> LabAPIResponse:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT paper_id, title, authors, abstract, url, publish_date, created_at
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            )
            paper = cur.fetchone()
            if paper is None:
                raise LabAPIError(404, "paper_not_found", f"No paper found for id {paper_id}")

            extraction_status = _load_extraction_status(conn, paper_id)

            cur.execute(
                """
                SELECT chunk_id, paper_id, section, content, created_at
                FROM chunks
                WHERE paper_id = ?
                ORDER BY chunk_id
                """,
                (paper_id,),
            )
            chunks = [_row_to_dict(row) for row in cur.fetchall()]
            chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
            for chunk in chunks:
                chunk["method_ids"] = []

            cur.execute(
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
            )
            methods = [_row_to_dict(row) for row in cur.fetchall()]

            evidence_select = _method_source_evidence_select(conn)
            cur.execute(
                f"""
                SELECT
                    ms.method_id,
                    m.name AS method_name,
                    ms.chunk_id,
                    c.section,
                    c.content,
                    {evidence_select}
                FROM method_sources ms
                JOIN methods m ON m.method_id = ms.method_id
                JOIN chunks c ON c.chunk_id = ms.chunk_id
                WHERE c.paper_id = ?
                ORDER BY ms.method_id, ms.chunk_id
                """,
                (paper_id,),
            )
            provenance = [_row_to_dict(row) for row in cur.fetchall()]
            for link in provenance:
                chunk = chunks_by_id.get(link["chunk_id"])
                if chunk is not None:
                    chunk["method_ids"].append(link["method_id"])

        return LabAPIResponse(
            200,
            {
                "ok": True,
                "paper": _paper_record(paper),
                "extraction_status": extraction_status,
                "methods": methods,
                "chunks": chunks,
                "provenance": provenance,
            },
        )

    def _handle_paper_memory(self, paper_id: int, max_chars: int | None) -> LabAPIResponse:
        with self._connection() as conn:
            try:
                memory = rfs.get_paper_memory(paper_id, conn=conn, max_chars=max_chars)
            except LookupError as exc:
                raise LabAPIError(404, "paper_not_found", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, **memory})

    def _handle_paper_memory_export(self, paper_id: int, max_chars: int | None) -> LabAPIResponse:
        with self._connection() as conn:
            try:
                bundle = export_paper_memory_bundle(paper_id, conn=conn, max_chars=max_chars)
            except LookupError as exc:
                raise LabAPIError(404, "paper_not_found", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "bundle": bundle})

    def _handle_corpus_memory_export(
        self,
        paper_ids: list[int] | None,
        max_chars: int | None,
    ) -> LabAPIResponse:
        with self._connection() as conn:
            try:
                bundle = export_corpus_memory_bundle(
                    paper_ids=paper_ids,
                    conn=conn,
                    max_chars=max_chars,
                )
            except LookupError as exc:
                raise LabAPIError(404, "paper_not_found", str(exc)) from exc
            except ValueError as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "bundle": bundle})


    def _handle_memory_import(self, payload: JSONDict) -> LabAPIResponse:
        bundle = payload.get("bundle", payload)
        try:
            with self._connection() as conn:
                result = import_memory_bundle(bundle, conn=conn)
        except ValueError as exc:
            raise LabAPIError(400, "invalid_bundle", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, **result})

    def _handle_stats(self) -> LabAPIResponse:
        with self._connection() as conn:
            cur = conn.cursor()
            feedback_row = cur.execute(
                """
                SELECT COUNT(*) AS count, AVG(rating) AS average_rating
                FROM method_feedback
                """
            ).fetchone()
            feedback = {
                "count": feedback_row["count"],
                "average_rating": feedback_row["average_rating"],
            }
            counts = {
                "papers": cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
                "chunks": cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "methods": cur.execute("SELECT COUNT(*) FROM methods").fetchone()[0],
                "lessons": cur.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
                "retrieval_events": cur.execute(
                    "SELECT COUNT(*) FROM retrieval_events"
                ).fetchone()[0],
                "method_feedback": feedback["count"],
            }
            cur.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM paper_extraction_status
                GROUP BY status
                ORDER BY status
                """
            )
            extraction_statuses = {
                row["status"]: row["count"] for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT COALESCE(source_type, 'unknown') AS source_type, COUNT(*) AS count
                FROM paper_extraction_status
                GROUP BY COALESCE(source_type, 'unknown')
                ORDER BY source_type
                """
            )
            source_types = {
                row["source_type"]: row["count"] for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT
                    COUNT(*) AS paper_count,
                    COALESCE(SUM(source_char_count), 0) AS source_char_count,
                    COALESCE(SUM(compressed_char_count), 0) AS compressed_char_count,
                    AVG(compression_ratio) AS average_compression_ratio
                FROM paper_extraction_status
                """
            )
            compression_row = cur.fetchone()
            source_chars = compression_row["source_char_count"]
            compressed_chars = compression_row["compressed_char_count"]
            overall_ratio = (
                compressed_chars / source_chars
                if source_chars
                else None
            )
            compression = {
                "paper_count": compression_row["paper_count"],
                "source_char_count": source_chars,
                "compressed_char_count": compressed_chars,
                "average_compression_ratio": compression_row[
                    "average_compression_ratio"
                ],
                "overall_compression_ratio": overall_ratio,
            }
            confidence_row = cur.execute(
                """
                SELECT
                    COUNT(confidence_score) AS count,
                    AVG(confidence_score) AS average_score,
                    COALESCE(SUM(CASE WHEN confidence_score < 0.5 THEN 1 ELSE 0 END), 0) AS low_confidence_count,
                    COALESCE(SUM(CASE WHEN confidence_score >= 0.8 THEN 1 ELSE 0 END), 0) AS high_confidence_count
                FROM paper_extraction_status
                """
            ).fetchone()
            confidence = {
                "count": confidence_row["count"],
                "average_score": confidence_row["average_score"],
                "low_confidence_count": confidence_row["low_confidence_count"],
                "high_confidence_count": confidence_row["high_confidence_count"],
            }

        return LabAPIResponse(
            200,
            {
                "ok": True,
                "counts": counts,
                "extraction_statuses": extraction_statuses,
                "source_types": source_types,
                "compression": compression,
                "confidence": confidence,
                "feedback": feedback,
            },
        )

    def _handle_evaluate_extraction(self, payload: JSONDict) -> LabAPIResponse:
        paper_id = _required_int(payload, "paper_id")
        if paper_id <= 0:
            raise LabAPIError(400, "invalid_field", "'paper_id' must be positive")
        expected_methods = payload.get("expected_methods")
        if not isinstance(expected_methods, list) or not expected_methods:
            raise LabAPIError(400, "missing_field", "'expected_methods' must be a non-empty list")
        notes = _optional_str(payload, "notes")
        require_source_coverage = _optional_bool(payload, "require_source_coverage")
        thresholds = {
            "min_precision": _optional_metric_threshold(payload, "min_precision"),
            "min_recall": _optional_metric_threshold(payload, "min_recall"),
            "min_f1": _optional_metric_threshold(payload, "min_f1"),
            "min_source_coverage": _optional_metric_threshold(payload, "min_source_coverage"),
            "min_evidence_coverage": _optional_metric_threshold(payload, "min_evidence_coverage"),
        }
        thresholds = {key: value for key, value in thresholds.items() if value is not None}
        with self._connection() as conn:
            try:
                result = evaluate_extraction(
                    conn,
                    paper_id,
                    expected_methods,
                    notes=notes,
                    require_source_coverage=require_source_coverage,
                    **thresholds,
                )
            except LookupError as exc:
                raise LabAPIError(404, "paper_not_found", str(exc)) from exc
            except ValueError as exc:
                raise LabAPIError(400, "invalid_evaluation", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "evaluation": result})


    def _handle_retrieve(self, payload: JSONDict) -> LabAPIResponse:
        query = _required_str(payload, "query")
        max_chars = _optional_int(payload, "max_chars")
        agent_run_id = _optional_int(payload, "agent_run_id")
        experiment_id = _optional_int(payload, "experiment_id")
        mode = payload.get("mode", "method")
        if mode not in ("method", "graph", "memory"):
            raise LabAPIError(400, "invalid_field", "'mode' must be 'method', 'graph', or 'memory'")
        log_requested = payload.get("log") is True
        include_scores = payload.get("include_scores") is True
        event_type = payload.get("event_type", "lab_api_retrieve")
        if event_type is None:
            event_type = "lab_api_retrieve"
        if not isinstance(event_type, str) or not event_type.strip():
            raise LabAPIError(400, "invalid_field", "'event_type' must be a non-empty string")
        event_id = None
        extra_body: JSONDict = {}
        scores = None

        if mode == "memory":
            raw_max_papers = _optional_int(payload, "max_papers")
            max_papers = 3 if raw_max_papers is None else raw_max_papers
            if max_papers <= 0:
                raise LabAPIError(400, "invalid_field", "'max_papers' must be positive")
            source_types = _optional_source_types(payload)
            with self._connection() as conn:
                try:
                    brief = rfs.get_research_brief(
                        query,
                        conn=conn,
                        max_chars=max_chars,
                        max_papers=max_papers,
                        include_scores=include_scores,
                        agent_run_id=agent_run_id,
                        source_types=source_types,
                    )
                except ValueError as exc:
                    raise LabAPIError(400, "invalid_field", str(exc)) from exc
                context = brief["context"]
                method_ids = brief.get("method_ids") or []
                if include_scores:
                    scores = brief.get("scores")
                extra_body = {
                    "paper_ids": brief.get("paper_ids") or [],
                    "chunk_ids": brief.get("chunk_ids") or [],
                    "lesson_ids": brief.get("lesson_ids") or [],
                    "max_papers": brief.get("max_papers", max_papers),
                    "source_types": brief.get("source_types"),
                }
                if log_requested:
                    event_id = rfs.log_retrieval_event(
                        event_type.strip(),
                        query,
                        context,
                        agent_run_id=agent_run_id,
                        experiment_id=experiment_id,
                        conn=conn,
                        method_ids=_unique_ints(method_ids),
                    )
        else:
            with self._connection() as conn:
                if include_scores:
                    method_context, method_ids, scores = rfs.search_methods_with_scores(
                        query, conn=conn, max_chars=max_chars, agent_run_id=agent_run_id
                    )
                else:
                    method_context, method_ids = rfs.search_methods_and_ids(
                        query, conn=conn, max_chars=max_chars, agent_run_id=agent_run_id
                    )
                if mode == "graph":
                    context = rfs.get_graph_context(
                        query, conn=conn, max_chars=max_chars, agent_run_id=agent_run_id
                    )
                else:
                    context = method_context
                context = _limit_text(context, max_chars)
                if log_requested:
                    event_id = rfs.log_retrieval_event(
                        event_type.strip(),
                        query,
                        context,
                        agent_run_id=agent_run_id,
                        experiment_id=experiment_id,
                        conn=conn,
                        method_ids=_unique_ints(method_ids),
                    )
        body = {
            "ok": True,
            "query": query,
            "mode": mode,
            "context": context,
            "method_ids": method_ids,
            "max_chars": max_chars,
        }
        body.update(extra_body)
        if include_scores:
            body["scores"] = scores
        if event_id is not None:
            body["event_id"] = event_id
        return LabAPIResponse(200, body)

    def _handle_brief(self, payload: JSONDict) -> LabAPIResponse:
        query = _required_str(payload, "query")
        max_chars = _optional_int(payload, "max_chars")
        raw_max_papers = _optional_int(payload, "max_papers")
        max_papers = 3 if raw_max_papers is None else raw_max_papers
        if max_papers <= 0:
            raise LabAPIError(400, "invalid_field", "'max_papers' must be positive")
        include_scores = _optional_bool(payload, "include_scores")
        log_requested = _optional_bool(payload, "log")
        agent_run_id = _optional_int(payload, "agent_run_id")
        source_types = _optional_source_types(payload)
        experiment_id = _optional_int(payload, "experiment_id")
        event_type = payload.get("event_type", "research_brief")
        if event_type is None:
            event_type = "research_brief"
        if not isinstance(event_type, str) or not event_type.strip():
            raise LabAPIError(400, "invalid_field", "'event_type' must be a non-empty string")

        event_id = None
        with self._connection() as conn:
            try:
                body = rfs.get_research_brief(
                    query,
                    conn=conn,
                    max_chars=max_chars,
                    max_papers=max_papers,
                    include_scores=include_scores,
                    agent_run_id=agent_run_id,
                    source_types=source_types,
                )
            except ValueError as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
            if log_requested:
                event_id = rfs.log_retrieval_event(
                    event_type.strip(),
                    query,
                    body["context"],
                    agent_run_id=agent_run_id,
                    experiment_id=experiment_id,
                    conn=conn,
                    method_ids=_unique_ints(body.get("method_ids") or []),
                )
        if event_id is not None:
            body["event_id"] = event_id
        return LabAPIResponse(200, body)

    def _handle_ask(self, payload: JSONDict) -> LabAPIResponse:
        query = _required_str(payload, "query")
        max_chars = _optional_non_negative_int(payload, "max_chars")
        max_papers = _optional_positive_int(payload, "max_papers")
        if max_papers is None:
            max_papers = 3
        source_types = payload.get("source_types") if payload.get("source_types") is not None else None
        source_type = payload.get("source_type") if "source_type" in payload else None
        include_scores = _optional_bool(payload, "include_scores")
        log_requested = _optional_bool(payload, "log")
        agent_run_id = _optional_positive_int(payload, "agent_run_id")
        experiment_id = _optional_positive_int(payload, "experiment_id")
        event_type = payload.get("event_type", "routed_ask")
        if event_type is None:
            event_type = "routed_ask"
        if not isinstance(event_type, str) or not event_type.strip():
            raise LabAPIError(400, "invalid_field", "'event_type' must be a non-empty string")

        k = _optional_positive_int(payload, "k")
        if k is None:
            k = 2
        max_reader_tokens = _optional_positive_int(payload, "max_reader_tokens")
        if max_reader_tokens is None:
            max_reader_tokens = 1200
        min_score = _optional_finite_float(payload, "min_score")
        overflow_policy = payload.get("overflow_policy", "drop_lowest_score")
        if overflow_policy is None:
            overflow_policy = "drop_lowest_score"
        if overflow_policy not in {"truncate", "raise", "drop_lowest_score"}:
            raise LabAPIError(
                400,
                "invalid_field",
                "'overflow_policy' must be one of 'truncate', 'raise', or 'drop_lowest_score'",
            )
        max_retries = _optional_non_negative_int(payload, "max_retries")
        if max_retries is None:
            max_retries = 1

        from researchfs.ask import run_routed_ask

        with self._connection() as conn:
            try:
                body = run_routed_ask(
                    query,
                    conn=conn,
                    reader=self._reader_fn,
                    verifier=self._verifier_fn,
                    max_chars=max_chars,
                    max_papers=max_papers,
                    k=k,
                    max_reader_tokens=max_reader_tokens,
                    min_score=min_score,
                    overflow_policy=overflow_policy,
                    max_retries=max_retries,
                    include_scores=include_scores,
                    source_types=source_types,
                    source_type=source_type,
                    agent_run_id=agent_run_id,
                    experiment_id=experiment_id,
                    log=log_requested,
                    event_type=event_type,
                )
            except ValueError as exc:
                raise LabAPIError(400, "invalid_field", str(exc)) from exc
            except LookupError as exc:
                raise LabAPIError(404, "not_found", str(exc)) from exc
        return LabAPIResponse(200, body)

    def _handle_feedback(self, payload: JSONDict) -> LabAPIResponse:
        method_id = _required_int(payload, "method_id")
        if method_id <= 0:
            raise LabAPIError(400, "invalid_field", "'method_id' must be positive")
        event_id = _optional_int(payload, "event_id")
        rating = _required_feedback_rating(payload)
        label = _optional_str(payload, "label")
        note = _optional_str(payload, "note")
        source = _optional_str(payload, "source")
        with self._connection() as conn:
            try:
                feedback_id = rfs.log_method_feedback(
                    method_id=method_id,
                    rating=rating,
                    event_id=event_id,
                    label=label,
                    note=note,
                    source=source,
                    conn=conn,
                )
            except ValueError as exc:
                raise LabAPIError(400, "invalid_feedback", str(exc)) from exc
        return LabAPIResponse(200, {"ok": True, "feedback_id": feedback_id})

    def _handle_method_detail(self, method_id: int) -> LabAPIResponse:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT method_id, name, description, implementation_hint, risks, category, created_at
                FROM methods
                WHERE method_id = ?
                """,
                (method_id,),
            )
            method = cur.fetchone()
            if method is None:
                raise LabAPIError(404, "method_not_found", f"No method found for id {method_id}")
            evidence_select = _method_source_evidence_select(conn)
            cur.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.paper_id,
                    p.title,
                    p.authors,
                    p.url,
                    p.publish_date,
                    c.section,
                    c.content,
                    {evidence_select}
                FROM method_sources ms
                JOIN chunks c ON c.chunk_id = ms.chunk_id
                JOIN papers p ON p.paper_id = c.paper_id
                WHERE ms.method_id = ?
                ORDER BY c.chunk_id
                """,
                (method_id,),
            )
            chunks = [_row_to_dict(row) for row in cur.fetchall()]
            feedback_summary = rfs.get_method_feedback_summary(method_id, conn=conn)
        return LabAPIResponse(
            200,
            {
                "ok": True,
                "method": _row_to_dict(method),
                "provenance_chunks": chunks,
                "feedback_summary": feedback_summary,
            },
        )

    def _handle_experiment(self, payload: JSONDict) -> LabAPIResponse:
        commit_hash = _required_str(payload, "commit_hash")
        status = _required_str(payload, "status")
        description = payload.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise LabAPIError(400, "invalid_field", "'description' must be a string")
        val_bpb = _optional_float(payload, "val_bpb")
        memory_gb = _optional_float(payload, "memory_gb")
        agent_run_id = _optional_int(payload, "agent_run_id")
        method_ids = _optional_method_ids(payload)
        with self._connection() as conn:
            experiment_id = rfs.log_experiment(
                commit_hash,
                val_bpb,
                memory_gb,
                status,
                description,
                agent_run_id=agent_run_id,
                conn=conn,
            )
            if method_ids:
                rfs.link_experiment_methods(
                    experiment_id,
                    method_ids,
                    role=str(payload.get("method_role") or "tested"),
                    note=payload.get("method_note"),
                    conn=conn,
                )
        return LabAPIResponse(200, {"ok": True, "experiment_id": experiment_id})

    def _handle_lesson(self, payload: JSONDict) -> LabAPIResponse:
        label = _required_str(payload, "label")
        content = _required_str(payload, "content")
        outcome = payload.get("outcome")
        if outcome is not None and not isinstance(outcome, str):
            raise LabAPIError(400, "invalid_field", "'outcome' must be a string")
        method_id = _optional_int(payload, "method_id")
        experiment_id = _optional_int(payload, "experiment_id")
        with self._connection() as conn:
            lesson_id = rfs.log_lesson(
                label,
                content,
                outcome=outcome,
                method_id=method_id,
                experiment_id=experiment_id,
                conn=conn,
            )
        return LabAPIResponse(200, {"ok": True, "lesson_id": lesson_id})


def dispatch_request(
    method: str,
    path: str,
    body: Any = None,
    *,
    conn: Any | None = None,
    db_path: str | None = None,
    llm_fn: LLMFn | None = None,
    arxiv_fetcher: ArxivFetcher | None = None,
    repo_fetcher: RepoFetcher | None = None,
    code_repo_searcher: CodeRepoSearcher | None = None,
    visual_index_runner=None,
    reader_fn: RoutedAskReader | None = None,
    verifier_fn: RoutedAskVerifier | None = None,
) -> LabAPIResponse:
    """Convenience wrapper for a single in-process ResearchFS lab API request."""
    api = ResearchFSLabAPI(
        conn=conn,
        db_path=db_path,
        llm_fn=llm_fn,
        arxiv_fetcher=arxiv_fetcher,
        repo_fetcher=repo_fetcher,
        code_repo_searcher=code_repo_searcher,
        visual_index_runner=visual_index_runner,
        reader_fn=reader_fn,
        verifier_fn=verifier_fn,
    )
    return api.dispatch(method, path, body)


def _make_handler(api: ResearchFSLabAPI) -> type[BaseHTTPRequestHandler]:
    class ResearchFSHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self) -> None:
            for key, value in _CORS_HEADERS.items():
                self.send_header(key, value)

        def _send(self, response: LabAPIResponse) -> None:
            encoded = response.to_json_bytes()
            self.send_response(response.status)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _read_body(self) -> bytes:
            length_text = self.headers.get("Content-Length", "0")
            try:
                length = int(length_text)
            except ValueError:
                length = 0
            return self.rfile.read(max(0, length))

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self._send(api.dispatch("GET", self.path))

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            self._send(api.dispatch("POST", self.path, self._read_body()))

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

    return ResearchFSHandler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the no-dependency ResearchFS lab API")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to bind")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to ResearchFS project DB)")
    args = parser.parse_args(argv)

    api = ResearchFSLabAPI(db_path=args.db_path)
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(api))
    print(f"ResearchFS lab API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
