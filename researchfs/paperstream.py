"""PaperStream state helpers for arXiv ingest.

This module is deliberately synchronous and stdlib-only.  It includes a bounded
OAI-PMH metadata page source plus helpers that operate on the SQLite schema
defined for PaperStream and provide deterministic keys, paths, queue state
transitions, event logging, Bloom-filter persistence, and status summaries.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, TextIO

ASSET_TYPES = frozenset(
    {
        "metadata",
        "pdf",
        "source",
        "parsed_text",
        "chunks",
        "embedding",
        "bm25_index",
        "vector_index",
    }
)

ASSET_STATUSES = frozenset(
    {
        "seen",
        "queued",
        "running",
        "downloaded",
        "imported",
        "parsed",
        "indexed",
        "failed",
        "skipped",
    }
)

JOB_TYPES = frozenset(
    {
        "sync_metadata",
        "download_pdf",
        "download_source",
        "import_s3_bin",
        "extract_s3_bin",
        "parse_pdf",
        "parse_source",
        "extract_references",
        "chunk_text",
        "embed_chunks",
        "index_bm25",
        "index_vector",
    }
)
PAPERSTREAM_SAFE_WORK_JOB_TYPES = ("parse_pdf", "parse_source", "chunk_text", "index_bm25")


JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "skipped"})

S3_BIN_STATUSES = frozenset({"seen", "downloaded", "imported", "failed", "skipped"})
_S3_MANIFEST_PARSER_VERSION = 1

_MODERN_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_LEGACY_ARXIV_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.-]*/\d{7}$")
_FILTER_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PAPERSTREAM_SEARCH_TERM_RE = re.compile(r"[0-9A-Za-z]+")
_PAPERSTREAM_SEARCH_LIKE_SOURCES = ("title", "abstract", "authors", "categories", "primary_category", "abs_url")
_PAPERSTREAM_CHUNKS_MAX_LIMIT = 500
_PAPERSTREAM_CHUNKS_MAX_CHARS = 20_000
_PAPERSTREAM_SEARCH_MAX_LIMIT = 100
_PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS = _PAPERSTREAM_CHUNKS_MAX_CHARS
_PAPERSTREAM_REBUILD_MAX_LIMIT = 1_000
_PAPERSTREAM_REQUEUE_MAX_LIMIT = _PAPERSTREAM_REBUILD_MAX_LIMIT
_PAPERSTREAM_INSPECTION_MAX_LIMIT = 1_000

_ASSET_STORAGE_LAYOUT = {
    "metadata": ("metadata", ".json"),
    "pdf": ("pdf", ".pdf"),
    "source": ("source", ".tar.gz"),
    "parsed_text": ("text", ".json"),
    "chunks": ("chunks", ".json"),
    "embedding": ("embeddings", ".json"),
    "bm25_index": ("bm25", ".sqlite"),
    "vector_index": ("vector", ".sqlite"),
}

_OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
_ARXIV_RAW_NAMESPACE = "http://arxiv.org/OAI/arXivRaw/"
_OAI_IDENTIFIER_PREFIX = "oai:arXiv.org:"
_DEFAULT_OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"


def _xml_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = " ".join(element.text.split())
    return text or None


def _xml_full_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    text = " ".join(" ".join(element.itertext()).split())
    return text or None


def _xml_local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1]
    return tag


def _xml_child(element: ET.Element | None, local_name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in element:
        if _xml_local_name(child.tag) == local_name:
            return child
    return None


def _xml_children(element: ET.Element | None, local_name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in element if _xml_local_name(child.tag) == local_name]


def _parse_xml(xml_text: str | bytes, label: str) -> ET.Element:
    if not isinstance(xml_text, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be XML text or bytes")
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"malformed {label}: {exc}") from exc


def _serialize_xml(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return ET.tostring(element, encoding="unicode", short_empty_elements=False)


def _metadata_element(record_element: ET.Element | None) -> ET.Element | None:
    metadata = _xml_child(record_element, "metadata")
    if metadata is None:
        return None
    for child in metadata:
        if _xml_local_name(child.tag) in {"arXivRaw", "arXiv"}:
            return child
    return next(iter(metadata), None)


def _arxiv_id_from_oai_identifier(identifier: str | None) -> str | None:
    if not identifier:
        return None
    identifier = identifier.strip()
    if identifier.startswith(_OAI_IDENTIFIER_PREFIX):
        identifier = identifier[len(_OAI_IDENTIFIER_PREFIX) :]
    return identifier or None


def _split_arxiv_version(arxiv_id: str | None) -> tuple[str | None, int | None]:
    if not arxiv_id:
        return None, None
    arxiv_id = arxiv_id.strip()
    match = re.match(r"^(.+?)v(\d+)$", arxiv_id)
    if not match:
        return arxiv_id, None
    return match.group(1), int(match.group(2))


def _version_number(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"v?(\d+)", value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_author_list(arxiv_raw: ET.Element | None) -> list[str]:
    authors_element = _xml_child(arxiv_raw, "authors")
    if authors_element is None:
        return []
    child_authors = [
        _xml_full_text(child)
        for child in authors_element
        if _xml_local_name(child.tag) in {"author", "name"}
    ]
    authors = [author for author in child_authors if author]
    if authors:
        return authors
    text = _xml_full_text(authors_element)
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()]
    return parts or [text]


def _parse_categories(arxiv_raw: ET.Element | None) -> list[str]:
    categories_text = _xml_text(_xml_child(arxiv_raw, "categories"))
    if not categories_text:
        categories_text = _xml_text(_xml_child(arxiv_raw, "category"))
    if not categories_text:
        return []
    return [category for category in categories_text.split() if category]


def _parse_versions(arxiv_raw: ET.Element | None) -> list[dict[str, Any]]:
    version_entries = _xml_children(arxiv_raw, "versions")
    if not version_entries:
        version_entries = _xml_children(arxiv_raw, "version")
    versions = []
    for entry in version_entries:
        child_versions = _xml_children(entry, "version")
        entry_date = _xml_text(_xml_child(entry, "date"))
        if child_versions and entry_date is not None:
            version_text = _xml_text(child_versions[0])
            versions.append(
                {
                    "version": _version_number(version_text) or len(versions) + 1,
                    "submitted_at": entry_date,
                    "announced_at": entry_date,
                }
            )
            continue
        nested_entries = child_versions if child_versions else [entry]
        for node in nested_entries:
            version_text = _xml_text(_xml_child(node, "version")) or _xml_text(node)
            date_text = _xml_text(_xml_child(node, "date")) or entry_date
            version_number = _version_number(version_text) or len(versions) + 1
            versions.append(
                {
                    "version": version_number,
                    "submitted_at": date_text,
                    "announced_at": date_text,
                }
            )
    return versions


def parse_oai_list_records(xml_text: str | bytes) -> dict[str, Any]:
    """Parse one OAI-PMH ListRecords response into deterministic records."""
    root = _parse_xml(xml_text, "OAI-PMH XML")
    error_elements = _xml_children(root, "error")
    if error_elements:
        error_element = error_elements[0]
        error_code = _coerce_text(error_element.attrib.get("code"))
        error_message = _xml_full_text(error_element)
        details = []
        if error_code:
            details.append(f"code={error_code}")
        if error_message:
            details.append(f"message={error_message}")
        detail_text = ": " + ", ".join(details) if details else ""
        raise ValueError(f"OAI-PMH error response{detail_text}")
    response_date = _xml_text(_xml_child(root, "responseDate"))
    if response_date is None:
        raise ValueError("OAI-PMH ListRecords response missing responseDate")
    list_records = _xml_child(root, "ListRecords")
    if list_records is None:
        raise ValueError("OAI-PMH ListRecords response missing ListRecords")
    records = []
    for record_element in _xml_children(list_records, "record"):
        header = _xml_child(record_element, "header")
        identifier = _xml_text(_xml_child(header, "identifier"))
        datestamp = _xml_text(_xml_child(header, "datestamp"))
        set_specs = [
            set_spec
            for set_spec in (_xml_text(child) for child in _xml_children(header, "setSpec"))
            if set_spec
        ]
        deleted = header is not None and header.attrib.get("status") == "deleted"
        parsed_record = {
            "identifier": identifier,
            "datestamp": datestamp,
            "set_specs": set_specs,
            "raw_xml": _serialize_xml(record_element),
        }
        if deleted:
            parsed_record["deleted"] = True
        records.append(parsed_record)

    token_element = _xml_child(list_records, "resumptionToken")
    token_attributes = dict(token_element.attrib) if token_element is not None else {}
    return {
        "response_date": response_date,
        "resumption_token": _xml_text(token_element),
        "resumption_token_attributes": token_attributes,
        "records": records,
    }

def _coerce_oai_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, (bytes, bytearray)):
        try:
            return bytes(response).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"OAI-PMH response is not UTF-8 text: {exc}") from exc
    if hasattr(response, "read"):
        try:
            return _coerce_oai_response_text(response.read())
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"could not read OAI-PMH response body: {exc}") from exc
    for attribute_name in ("text", "content"):
        if hasattr(response, attribute_name):
            try:
                value = getattr(response, attribute_name)
            except Exception as exc:
                raise RuntimeError(f"could not read OAI-PMH response {attribute_name}: {exc}") from exc
            return _coerce_oai_response_text(value() if callable(value) else value)
    raise RuntimeError(
        "OAI-PMH request_fn must return XML text/bytes or an object with read(), text, or content"
    )


def _validate_oai_max_pages(max_pages: int | None) -> int | None:
    if max_pages is None:
        return None
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise ValueError("max_pages must be a non-negative integer or None")
    if max_pages < 0:
        raise ValueError("max_pages must be a non-negative integer or None")
    return max_pages


def _validate_oai_rate_limit_seconds(rate_limit_seconds: float) -> float:
    if isinstance(rate_limit_seconds, bool):
        raise ValueError("rate_limit_seconds must be a non-negative finite number")
    try:
        parsed = float(rate_limit_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("rate_limit_seconds must be a non-negative finite number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError("rate_limit_seconds must be a non-negative finite number")
    return parsed


def _oai_nonempty_param(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _build_oai_list_records_url(base_url: str, params: list[tuple[str, str]]) -> str:
    try:
        parts = urllib.parse.urlsplit(base_url)
    except ValueError as exc:
        raise ValueError(f"malformed OAI-PMH base_url: {exc}") from exc
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("OAI-PMH base_url must be an absolute http(s) URL")
    if parts.fragment:
        raise ValueError("OAI-PMH base_url must not include a fragment")
    query_items = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = urllib.parse.urlencode(query_items + params)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _default_oai_request(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OAI-PMH request failed for {url}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"malformed OAI-PMH request URL {url!r}: {exc}") from exc

def _request_rate_limited_arxiv_url(
    url: str,
    *,
    request_fn: Any,
    sleep_fn: Any = None,
    rate_limit_seconds: float = 0.0,
    request_label: str = "arXiv request",
) -> Any:
    """Run an external arXiv request behind the shared local rate-limit seam."""
    rate_limit_seconds = _validate_oai_rate_limit_seconds(rate_limit_seconds)
    if not callable(request_fn):
        raise TypeError("request_fn must be callable")
    sleeper = time.sleep if sleep_fn is None else sleep_fn
    if not callable(sleeper):
        raise TypeError("sleep_fn must be callable")
    if rate_limit_seconds > 0.0:
        try:
            sleeper(rate_limit_seconds)
        except Exception as exc:
            raise RuntimeError(
                f"{request_label} rate limiter failed before requesting {url}: {exc}"
            ) from exc
    try:
        return request_fn(url)
    except (RuntimeError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(f"{request_label} failed for {url}: {exc}") from exc




def iter_oai_metadata_pages(
    *,
    metadata_prefix: str = "arXivRaw",
    set_spec: str | None = None,
    from_date: str | None = None,
    until_date: str | None = None,
    base_url: str = _DEFAULT_OAI_BASE_URL,
    initial_resumption_token: str | None = None,
    max_pages: int | None = None,
    request_fn: Any = None,
    sleep_fn: Any = None,
    rate_limit_seconds: float = 3.0,
) -> Iterator[str]:
    """Yield OAI-PMH ListRecords XML pages, following resumption tokens."""
    max_pages = _validate_oai_max_pages(max_pages)
    rate_limit_seconds = _validate_oai_rate_limit_seconds(rate_limit_seconds)
    request = _default_oai_request if request_fn is None else request_fn

    token = _oai_nonempty_param("initial_resumption_token", initial_resumption_token)
    if token:
        metadata_prefix = set_spec = from_date = until_date = None
    else:
        metadata_prefix = _oai_nonempty_param("metadata_prefix", metadata_prefix)
        set_spec = _oai_nonempty_param("set_spec", set_spec)
        from_date = _oai_nonempty_param("from_date", from_date)
        until_date = _oai_nonempty_param("until_date", until_date)

    pages_seen = 0
    while max_pages is None or pages_seen < max_pages:
        if token:
            params = [("verb", "ListRecords"), ("resumptionToken", token)]
        else:
            params = [("verb", "ListRecords"), ("metadataPrefix", metadata_prefix)]
            if set_spec is not None:
                params.append(("set", set_spec))
            if from_date is not None:
                params.append(("from", from_date))
            if until_date is not None:
                params.append(("until", until_date))
        url = _build_oai_list_records_url(base_url, params)
        response = _request_rate_limited_arxiv_url(
            url,
            request_fn=request,
            sleep_fn=sleep_fn,
            rate_limit_seconds=rate_limit_seconds,
            request_label="OAI-PMH request",
        )
        xml_text = _coerce_oai_response_text(response)
        try:
            parsed_page = parse_oai_list_records(xml_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed OAI-PMH ListRecords response from {url}: {exc}") from exc
        next_token = parsed_page.get("resumption_token")
        token = next_token.strip() if isinstance(next_token, str) else None
        yield xml_text
        pages_seen += 1
        if not token:
            break


def parse_arxiv_raw_record(record: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """Parse one OAI record or arXivRaw metadata element into paper metadata."""
    raw_oai_xml = None
    identifier = None
    oai_datestamp = None
    deleted = False
    if isinstance(record, dict):
        raw_oai_xml = record.get("raw_xml")
        identifier = record.get("identifier")
        oai_datestamp = record.get("datestamp")
        deleted = bool(record.get("deleted"))
        root = _parse_xml(raw_oai_xml, "OAI record XML") if raw_oai_xml else None
    else:
        raw_oai_xml = record.decode("utf-8") if isinstance(record, (bytes, bytearray)) else record
        root = _parse_xml(record, "OAI record XML")

    header = _xml_child(root, "header")
    if header is not None:
        identifier = identifier or _xml_text(_xml_child(header, "identifier"))
        oai_datestamp = oai_datestamp or _xml_text(_xml_child(header, "datestamp"))
        deleted = deleted or header.attrib.get("status") == "deleted"

    arxiv_raw = _metadata_element(root)
    if arxiv_raw is None and root is not None and _xml_local_name(root.tag) in {"arXivRaw", "arXiv"}:
        arxiv_raw = root
    if raw_oai_xml is None:
        raw_oai_xml = _serialize_xml(root)

    raw_id = _xml_text(_xml_child(arxiv_raw, "id")) or _arxiv_id_from_oai_identifier(identifier)
    arxiv_id, id_version = _split_arxiv_version(raw_id)
    versions = _parse_versions(arxiv_raw)
    latest_version = id_version
    if versions:
        version_numbers = [version["version"] for version in versions if version.get("version")]
        if version_numbers:
            latest_version = max(version_numbers)

    if not arxiv_id and not deleted:
        raise ValueError("missing required arXiv id in non-deleted record")

    categories = _parse_categories(arxiv_raw)
    primary_category = _xml_text(_xml_child(arxiv_raw, "primary_category"))
    if primary_category is None:
        primary_element = _xml_child(arxiv_raw, "primaryCategory")
        primary_category = (
            primary_element.attrib.get("term")
            if primary_element is not None and primary_element.attrib.get("term")
            else None
        )
    if primary_category is None and categories:
        primary_category = categories[0]

    return {
        "arxiv_id": arxiv_id,
        "latest_version": latest_version,
        "title": _xml_full_text(_xml_child(arxiv_raw, "title")),
        "abstract": _xml_full_text(_xml_child(arxiv_raw, "abstract")),
        "authors": _parse_author_list(arxiv_raw),
        "categories": categories,
        "primary_category": primary_category,
        "license": _xml_text(_xml_child(arxiv_raw, "license")),
        "doi": _xml_text(_xml_child(arxiv_raw, "doi")),
        "journal_ref": _xml_text(_xml_child(arxiv_raw, "journal-ref")),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "oai_datestamp": oai_datestamp,
        "raw_oai_xml": raw_oai_xml,
        "versions": versions,
        "deleted": deleted,
    }


def _utc_timestamp(value: datetime | str | None = None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("timestamp must not be empty")
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc).replace(microsecond=0)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _row_to_dict(row: sqlite3.Row | tuple[Any, ...] | None, description: Any = None) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        return dict(row)
    if description is None:
        raise TypeError("cursor description is required for tuple rows")
    return {column[0]: row[index] for index, column in enumerate(description)}


def _validate_arxiv_id(arxiv_id: str) -> str:
    if not isinstance(arxiv_id, str):
        raise TypeError("arxiv_id must be a string")
    arxiv_id = arxiv_id.strip()
    if not arxiv_id:
        raise ValueError("arxiv_id must not be empty")
    if arxiv_id.endswith("/") or arxiv_id.startswith("/"):
        raise ValueError("arxiv_id must be relative")
    if ".." in arxiv_id or "\\" in arxiv_id or ":" in arxiv_id:
        raise ValueError("arxiv_id contains unsafe path/key characters")
    if re.search(r"v\d+$", arxiv_id):
        raise ValueError("arxiv_id must not include a version suffix")
    if not (_MODERN_ARXIV_ID_RE.fullmatch(arxiv_id) or _LEGACY_ARXIV_ID_RE.fullmatch(arxiv_id)):
        raise ValueError(f"invalid arxiv_id: {arxiv_id!r}")
    return arxiv_id


def _validate_version(version: int | str | None, *, required: bool = True) -> int | None:
    if version is None:
        if required:
            raise ValueError("version is required")
        return None
    if isinstance(version, bool):
        raise TypeError("version must be a positive integer")
    try:
        parsed = int(version)
    except (TypeError, ValueError) as exc:
        raise TypeError("version must be a positive integer") from exc
    if parsed < 1:
        raise ValueError("version must be a positive integer")
    return parsed


def _validate_asset_type(asset_type: str | None, *, required: bool = True) -> str | None:
    if asset_type is None:
        if required:
            raise ValueError("asset_type is required")
        return None
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"invalid asset_type: {asset_type!r}")
    return asset_type


def _validate_asset_status(status: str) -> str:
    if status not in ASSET_STATUSES:
        raise ValueError(f"invalid asset status: {status!r}")
    return status


def _validate_job_type(job_type: str) -> str:
    if job_type not in JOB_TYPES:
        raise ValueError(f"invalid job_type: {job_type!r}")
    return job_type


def _validate_filter_name(filter_name: str) -> str:
    if not isinstance(filter_name, str):
        raise TypeError("filter_name must be a string")
    filter_name = filter_name.strip()
    if not filter_name or not _FILTER_NAME_RE.fullmatch(filter_name):
        raise ValueError("filter_name must contain only letters, digits, '_', '.', ':', or '-'")
    return filter_name


def _validate_source_name(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value != "manual" and not _SQL_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be 'manual' or a SQLite identifier")
    return value


def _validate_worker_id(worker_id: str) -> str:
    if not isinstance(worker_id, str):
        raise TypeError("worker_id must be a string")
    worker_id = worker_id.strip()
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    return worker_id


def _key_bytes(key: str | bytes) -> bytes:
    if isinstance(key, bytes):
        if not key:
            raise ValueError("Bloom filter key must not be empty")
        return key
    if not isinstance(key, str):
        key = str(key)
    if not key:
        raise ValueError("Bloom filter key must not be empty")
    return key.encode("utf-8")


def arxiv_asset_key(arxiv_id: str, version: int | str, asset_type: str) -> str:
    """Return the canonical durable asset key for one arXiv version asset."""

    arxiv_id = _validate_arxiv_id(arxiv_id)
    version = _validate_version(version)
    asset_type = _validate_asset_type(asset_type)
    return f"{arxiv_id}:v{version}:{asset_type}"


def arxiv_job_key(job_type: str, arxiv_id: str | None = None, version: int | str | None = None) -> str:
    """Return the canonical durable job key.

    Global jobs such as ``sync_metadata`` have a key equal to the job type.  Jobs
    scoped to a paper version use ``{job_type}:{arxiv_id}:v{version}``.
    """

    job_type = _validate_job_type(job_type)
    if arxiv_id is None:
        if version is not None:
            raise ValueError("version requires arxiv_id")
        return job_type
    arxiv_id = _validate_arxiv_id(arxiv_id)
    version = _validate_version(version)
    return f"{job_type}:{arxiv_id}:v{version}"


def arxiv_storage_path(root: str | Path, arxiv_id: str, version: int | str, asset_type: str) -> str:
    """Return the deterministic internal storage path for an arXiv asset."""

    arxiv_id = _validate_arxiv_id(arxiv_id)
    version = _validate_version(version)
    asset_type = _validate_asset_type(asset_type)
    directory, suffix = _ASSET_STORAGE_LAYOUT[asset_type]
    if _MODERN_ARXIV_ID_RE.fullmatch(arxiv_id):
        prefix, leaf = arxiv_id.split(".", 1)
        shard = (prefix, leaf)
    else:
        archive, leaf = arxiv_id.split("/", 1)
        shard = (archive, leaf)
    return str(Path(root) / "arxiv" / directory / Path(*shard) / f"v{version}{suffix}")

def s3_bin_storage_path(root: str | Path, asset_type: str, filename: str) -> str:
    """Return the deterministic internal storage path for one downloaded S3 bin."""

    asset_type = _validate_s3_bin_asset_type(asset_type)
    filename = _validate_s3_bin_relative_path(filename, "filename", required=True)
    return str(Path(root) / "arxiv" / "s3_bins" / asset_type / Path(filename))


asset_key = arxiv_asset_key
job_key = arxiv_job_key
object_path = arxiv_storage_path


def upsert_arxiv_asset(
    conn: sqlite3.Connection,
    arxiv_id: str,
    version: int | str,
    asset_type: str,
    status: str = "seen",
    storage_path: str | Path | None = None,
    checksum: str | None = None,
    byte_size: int | None = None,
    source: str | None = None,
    error_message: str | None = None,
) -> str:
    """Insert or update one durable asset state row and return its asset key."""

    asset_key = arxiv_asset_key(arxiv_id, version, asset_type)
    status = _validate_asset_status(status)
    storage_path_text = None if storage_path is None else str(storage_path)
    if byte_size is not None:
        byte_size = int(byte_size)
        if byte_size < 0:
            raise ValueError("byte_size must be non-negative")
    conn.execute(
        """
        INSERT INTO arxiv_assets (
            asset_key, arxiv_id, version, asset_type, status,
            storage_path, checksum, byte_size, source, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_key) DO UPDATE SET
            status = excluded.status,
            storage_path = excluded.storage_path,
            checksum = excluded.checksum,
            byte_size = excluded.byte_size,
            source = excluded.source,
            error_message = excluded.error_message,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            asset_key,
            _validate_arxiv_id(arxiv_id),
            _validate_version(version),
            _validate_asset_type(asset_type),
            status,
            storage_path_text,
            checksum,
            byte_size,
            source,
            error_message,
        ),
    )
    return asset_key


def _validate_s3_bin_asset_type(asset_type: Any) -> str:
    if not isinstance(asset_type, str):
        raise TypeError("asset_type must be a string")
    asset_type = asset_type.strip()
    if asset_type not in {"pdf", "source"}:
        raise ValueError("asset_type must be 'pdf' or 'source'")
    return asset_type


def _validate_s3_bin_status(status: Any) -> str:
    status = _validate_s3_bin_text(status, "status", required=True)
    if status not in S3_BIN_STATUSES:
        raise ValueError(f"invalid S3 bin status: {status!r}")
    return status


def _validate_s3_bin_text(value: Any, label: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _validate_s3_bin_relative_path(value: Any, label: str, *, required: bool = False) -> str | None:
    text = _validate_s3_bin_text(value, label, required=required)
    if text is None:
        return None
    if text.startswith("/") or "\\" in text:
        raise ValueError(f"{label} must be a safe relative path")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a safe relative path")
    return text


def _optional_s3_bin_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _coerce_nonnegative_int(value, label)


def s3_bin_key(asset_type: str, filename: str) -> str:
    """Return the deterministic local key for one offline arXiv S3 bin."""

    asset_type = _validate_s3_bin_asset_type(asset_type)
    filename = _validate_s3_bin_relative_path(filename, "filename", required=True)
    return f"{asset_type}:{filename}"


def _normalize_s3_bin_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("S3 manifest records must be dictionaries")
    asset_type = _validate_s3_bin_asset_type(record.get("asset_type"))
    filename = _validate_s3_bin_relative_path(record.get("filename"), "filename", required=True)
    bin_key_value = _validate_s3_bin_text(record.get("bin_key"), "bin_key") or s3_bin_key(asset_type, filename)
    status_provided = "status" in record
    status = record.get("status")
    if status is None:
        status = "seen"
    provided_fields = {"bin_key", "asset_type", "filename"}
    if status_provided:
        provided_fields.add("status")
    normalized = {
        "bin_key": bin_key_value,
        "asset_type": asset_type,
        "filename": filename,
        "size_bytes": _optional_s3_bin_nonnegative_int(record.get("size_bytes"), "size_bytes"),
        "checksum": _validate_s3_bin_text(record.get("checksum"), "checksum"),
        "first_item": _validate_s3_bin_text(record.get("first_item"), "first_item"),
        "last_item": _validate_s3_bin_text(record.get("last_item"), "last_item"),
        "item_count": _optional_s3_bin_nonnegative_int(record.get("item_count"), "item_count"),
        "status": _validate_s3_bin_status(status),
        "storage_path": _validate_s3_bin_text(record.get("storage_path"), "storage_path"),
        "_provided_fields": provided_fields,
    }
    for optional_field in ("size_bytes", "checksum", "first_item", "last_item", "item_count", "storage_path"):
        if optional_field in record:
            normalized["_provided_fields"].add(optional_field)
    return normalized


_S3_BIN_COLUMNS = (
    "bin_key",
    "asset_type",
    "filename",
    "size_bytes",
    "checksum",
    "first_item",
    "last_item",
    "item_count",
    "status",
    "storage_path",
)


def _s3_bin_row(conn: sqlite3.Connection, bin_key_value: str) -> sqlite3.Row | tuple[Any, ...] | None:
    return conn.execute(
        """
        SELECT bin_key, asset_type, filename, size_bytes, checksum,
               first_item, last_item, item_count, status, storage_path
        FROM arxiv_s3_bins
        WHERE bin_key = ?
        """,
        (bin_key_value,),
    ).fetchone()


def _s3_bin_change_summary(
    conn: sqlite3.Connection,
    record: dict[str, Any],
) -> tuple[bool, bool, list[str], dict[str, Any]]:
    row = _s3_bin_row(conn, record["bin_key"])
    effective_record = dict(record)
    if row is None:
        return True, True, list(_S3_BIN_COLUMNS), effective_record
    provided_fields = set(record.get("_provided_fields", _S3_BIN_COLUMNS))
    for index, column in enumerate(_S3_BIN_COLUMNS):
        if column not in provided_fields:
            effective_record[column] = _row_value(row, column, index)
    changed_fields: list[str] = []
    for index, column in enumerate(_S3_BIN_COLUMNS):
        if _row_value(row, column, index) != effective_record[column]:
            changed_fields.append(column)
    return False, bool(changed_fields), changed_fields, effective_record


def _s3_bin_result(record: dict[str, Any], *, inserted: bool, changed: bool, changed_fields: list[str]) -> dict[str, Any]:
    action = "inserted" if inserted else "updated" if changed else "unchanged"
    return {
        "bin_key": record["bin_key"],
        "asset_type": record["asset_type"],
        "filename": record["filename"],
        "status": record["status"],
        "action": action,
        "inserted": inserted,
        "updated": bool(changed and not inserted),
        "changed": changed,
        "changed_fields": changed_fields,
    }


def upsert_arxiv_s3_bin(conn: sqlite3.Connection, record: dict[str, Any]) -> dict[str, Any]:
    """Insert or update one arXiv S3 bin metadata row without touching bin contents."""

    normalized = dict(record) if isinstance(record, dict) and "_provided_fields" in record else _normalize_s3_bin_record(record)
    savepoint_name = "paperstream_s3_bin_upsert"
    _metadata_savepoint(conn, savepoint_name)
    try:
        inserted, changed, changed_fields, effective_record = _s3_bin_change_summary(conn, normalized)
        if inserted:
            conn.execute(
                """
                INSERT INTO arxiv_s3_bins (
                    bin_key, asset_type, filename, size_bytes, checksum,
                    first_item, last_item, item_count, status, storage_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(effective_record[column] for column in _S3_BIN_COLUMNS),
            )
            append_arxiv_event(
                conn,
                "s3_bin_seen",
                asset_type=effective_record["asset_type"],
                payload={
                    "bin_key": effective_record["bin_key"],
                    "filename": effective_record["filename"],
                    "status": effective_record["status"],
                },
            )
        elif changed:
            conn.execute(
                """
                UPDATE arxiv_s3_bins
                SET asset_type = ?,
                    filename = ?,
                    size_bytes = ?,
                    checksum = ?,
                    first_item = ?,
                    last_item = ?,
                    item_count = ?,
                    status = ?,
                    storage_path = ?
                WHERE bin_key = ?
                """,
                (
                    effective_record["asset_type"],
                    effective_record["filename"],
                    effective_record["size_bytes"],
                    effective_record["checksum"],
                    effective_record["first_item"],
                    effective_record["last_item"],
                    effective_record["item_count"],
                    effective_record["status"],
                    effective_record["storage_path"],
                    effective_record["bin_key"],
                ),
            )
            append_arxiv_event(
                conn,
                "s3_bin_updated",
                asset_type=effective_record["asset_type"],
                payload={
                    "bin_key": effective_record["bin_key"],
                    "filename": effective_record["filename"],
                    "changed_fields": changed_fields,
                    "status": effective_record["status"],
                },
            )
        _metadata_release(conn, savepoint_name)
        return _s3_bin_result(
            effective_record,
            inserted=inserted,
            changed=changed,
            changed_fields=changed_fields,
        )
    except Exception:
        _metadata_rollback(conn, savepoint_name)
        raise


def _normalize_s3_manifest_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    seen_bin_keys: set[str] = set()
    for index, record in enumerate(records, start=1):
        try:
            normalized = _normalize_s3_bin_record(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid S3 manifest record {index}: {exc}") from exc
        if normalized["bin_key"] in seen_bin_keys:
            raise ValueError(f"duplicate S3 manifest bin_key at record {index}: {normalized['bin_key']!r}")
        seen_bin_keys.add(normalized["bin_key"])
        normalized_records.append(normalized)
    return normalized_records


def _s3_manifest_import_failure_payload(
    exc: BaseException,
    *,
    input_path: str | None = None,
) -> dict[str, Any]:
    error = str(exc)
    if input_path:
        for path_text in {input_path, str(Path(input_path).expanduser())}:
            if path_text:
                error = error.replace(path_text, "<input>")
    payload: dict[str, Any] = {"error": error, "parser_version": _S3_MANIFEST_PARSER_VERSION}
    match = re.search(r"record (\d+)", error)
    if match:
        payload["record_index"] = int(match.group(1))
    return payload


def import_arxiv_s3_manifest(
    conn: sqlite3.Connection,
    records: Iterable[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import offline arXiv S3 bin manifest metadata without downloading bins."""

    try:
        normalized_records = _normalize_s3_manifest_records(records)
    except (TypeError, ValueError) as exc:
        if not dry_run:
            append_arxiv_event(conn, "s3_manifest_import_failed", payload=_s3_manifest_import_failure_payload(exc))
        raise
    result: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "parser_version": _S3_MANIFEST_PARSER_VERSION,
        "records_seen": len(normalized_records),
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "changed": 0,
        "rows": [],
    }
    if dry_run:
        for normalized in normalized_records:
            inserted, changed, changed_fields, effective_record = _s3_bin_change_summary(conn, normalized)
            row_summary = _s3_bin_result(
                effective_record,
                inserted=inserted,
                changed=changed,
                changed_fields=changed_fields,
            )
            result["rows"].append(row_summary)
            if inserted:
                result["inserted"] += 1
            elif changed:
                result["updated"] += 1
                result["changed"] += 1
            else:
                result["unchanged"] += 1
        return result

    savepoint_name = "paperstream_s3_manifest_import"
    _metadata_savepoint(conn, savepoint_name)
    try:
        for normalized in normalized_records:
            row_summary = upsert_arxiv_s3_bin(conn, normalized)
            result["rows"].append(row_summary)
            if row_summary["inserted"]:
                result["inserted"] += 1
            elif row_summary["changed"]:
                result["updated"] += 1
                result["changed"] += 1
            else:
                result["unchanged"] += 1
        _metadata_release(conn, savepoint_name)
        return result
    except Exception:
        _metadata_rollback(conn, savepoint_name)
        raise


def _validate_arxiv_bin_member_storage_path(value: Any) -> str | None:
    text = _validate_s3_bin_text(value, "storage_path")
    if text is None:
        return None
    if "\x00" in text or "\n" in text or "\r" in text:
        raise ValueError("storage_path must be a single-line local path")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text) or text.startswith("//"):
        raise ValueError("storage_path must be a local path, not a URL")
    return text


def _normalize_arxiv_bin_member_record(
    record: dict[str, Any],
    *,
    bin_key: str | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("S3 bin member records must be dictionaries")
    bin_key_value = record.get("bin_key")
    if bin_key_value is None:
        bin_key_value = bin_key
    provided_fields = {"bin_key", "arxiv_id", "version", "asset_type"}
    for optional_field in ("storage_path", "checksum"):
        if optional_field in record:
            provided_fields.add(optional_field)
    normalized = {
        "bin_key": _validate_s3_bin_text(bin_key_value, "bin_key", required=True),
        "arxiv_id": _validate_arxiv_id(record.get("arxiv_id", "")),
        "version": _validate_version(record.get("version")),
        "asset_type": _validate_s3_bin_asset_type(record.get("asset_type")),
        "storage_path": _validate_arxiv_bin_member_storage_path(record.get("storage_path")),
        "checksum": _validate_s3_bin_text(record.get("checksum"), "checksum"),
        "_provided_fields": provided_fields,
    }
    return normalized


_ARXIV_BIN_MEMBER_COLUMNS = (
    "bin_key",
    "arxiv_id",
    "version",
    "asset_type",
    "storage_path",
    "checksum",
)


def _arxiv_bin_member_bin_row(conn: sqlite3.Connection, bin_key_value: str) -> sqlite3.Row | tuple[Any, ...] | None:
    return conn.execute(
        """
        SELECT bin_key, asset_type
        FROM arxiv_s3_bins
        WHERE bin_key = ?
        """,
        (bin_key_value,),
    ).fetchone()


def _validate_arxiv_bin_member_bins(
    conn: sqlite3.Connection,
    records: Iterable[dict[str, Any]],
) -> None:
    bin_asset_types: dict[str, str] = {}
    for record in records:
        bin_key_value = record["bin_key"]
        if bin_key_value not in bin_asset_types:
            row = _arxiv_bin_member_bin_row(conn, bin_key_value)
            if row is None:
                raise ValueError(f"arxiv_s3_bins row does not exist for bin_key {bin_key_value!r}")
            bin_asset_types[bin_key_value] = str(_row_value(row, "asset_type", 1))
        if bin_asset_types[bin_key_value] != record["asset_type"]:
            raise ValueError(
                f"asset_type {record['asset_type']!r} does not match bin {bin_key_value!r} "
                f"asset_type {bin_asset_types[bin_key_value]!r}"
            )


def _arxiv_bin_member_row(
    conn: sqlite3.Connection,
    record: dict[str, Any],
) -> sqlite3.Row | tuple[Any, ...] | None:
    return conn.execute(
        """
        SELECT bin_key, arxiv_id, version, asset_type, storage_path, checksum
        FROM arxiv_bin_members
        WHERE bin_key = ? AND arxiv_id = ? AND version = ? AND asset_type = ?
        """,
        (record["bin_key"], record["arxiv_id"], record["version"], record["asset_type"]),
    ).fetchone()


def _arxiv_bin_member_change_summary(
    conn: sqlite3.Connection,
    record: dict[str, Any],
) -> tuple[bool, bool, list[str], dict[str, Any]]:
    row = _arxiv_bin_member_row(conn, record)
    effective_record = dict(record)
    if row is None:
        return True, True, list(_ARXIV_BIN_MEMBER_COLUMNS), effective_record
    provided_fields = set(record.get("_provided_fields", _ARXIV_BIN_MEMBER_COLUMNS))
    for index, column in enumerate(_ARXIV_BIN_MEMBER_COLUMNS):
        if column not in provided_fields:
            effective_record[column] = _row_value(row, column, index)
    changed_fields: list[str] = []
    for index, column in enumerate(_ARXIV_BIN_MEMBER_COLUMNS):
        if _row_value(row, column, index) != effective_record[column]:
            changed_fields.append(column)
    return False, bool(changed_fields), changed_fields, effective_record


def _arxiv_bin_member_asset_would_change(
    conn: sqlite3.Connection,
    record: dict[str, Any],
) -> bool:
    if record["storage_path"] is None:
        return False
    asset_key_value = arxiv_asset_key(record["arxiv_id"], record["version"], record["asset_type"])
    row = conn.execute(
        """
        SELECT status, storage_path, checksum, source
        FROM arxiv_assets
        WHERE asset_key = ?
        """,
        (asset_key_value,),
    ).fetchone()
    if row is None:
        return True
    return (
        _row_value(row, "status", 0) != "imported"
        or _row_value(row, "storage_path", 1) != record["storage_path"]
        or _row_value(row, "checksum", 2) != record["checksum"]
        or _row_value(row, "source", 3) != "s3_bin_member"
    )


def _arxiv_bin_member_result(
    record: dict[str, Any],
    *,
    inserted: bool,
    changed: bool,
    changed_fields: list[str],
    asset_imported: bool,
) -> dict[str, Any]:
    action = "inserted" if inserted else "updated" if changed else "unchanged"
    asset_key_value = arxiv_asset_key(record["arxiv_id"], record["version"], record["asset_type"])
    return {
        "bin_key": record["bin_key"],
        "arxiv_id": record["arxiv_id"],
        "version": record["version"],
        "asset_type": record["asset_type"],
        "asset_key": asset_key_value,
        "storage_path": record["storage_path"],
        "checksum": record["checksum"],
        "action": action,
        "inserted": inserted,
        "updated": bool(changed and not inserted),
        "changed": changed,
        "changed_fields": changed_fields,
        "asset_imported": asset_imported,
    }


def upsert_arxiv_bin_member(
    conn: sqlite3.Connection,
    record: dict[str, Any],
    *,
    upsert_asset: bool = True,
) -> dict[str, Any]:
    """Insert or update one offline arXiv S3 bin member reconciliation row."""

    normalized = dict(record) if isinstance(record, dict) and "_provided_fields" in record else _normalize_arxiv_bin_member_record(record)
    _validate_arxiv_bin_member_bins(conn, (normalized,))
    savepoint_name = "paperstream_bin_member_upsert"
    _metadata_savepoint(conn, savepoint_name)
    try:
        inserted, changed, changed_fields, effective_record = _arxiv_bin_member_change_summary(conn, normalized)
        if inserted:
            conn.execute(
                """
                INSERT INTO arxiv_bin_members (
                    bin_key, arxiv_id, version, asset_type, storage_path, checksum
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                tuple(effective_record[column] for column in _ARXIV_BIN_MEMBER_COLUMNS),
            )
            append_arxiv_event(
                conn,
                "s3_bin_member_seen",
                arxiv_id=effective_record["arxiv_id"],
                version=effective_record["version"],
                asset_type=effective_record["asset_type"],
                payload={
                    "bin_key": effective_record["bin_key"],
                    "storage_path": effective_record["storage_path"],
                    "checksum": effective_record["checksum"],
                },
            )
        elif changed:
            conn.execute(
                """
                UPDATE arxiv_bin_members
                SET storage_path = ?,
                    checksum = ?
                WHERE bin_key = ? AND arxiv_id = ? AND version = ? AND asset_type = ?
                """,
                (
                    effective_record["storage_path"],
                    effective_record["checksum"],
                    effective_record["bin_key"],
                    effective_record["arxiv_id"],
                    effective_record["version"],
                    effective_record["asset_type"],
                ),
            )
            append_arxiv_event(
                conn,
                "s3_bin_member_updated",
                arxiv_id=effective_record["arxiv_id"],
                version=effective_record["version"],
                asset_type=effective_record["asset_type"],
                payload={
                    "bin_key": effective_record["bin_key"],
                    "changed_fields": changed_fields,
                    "storage_path": effective_record["storage_path"],
                    "checksum": effective_record["checksum"],
                },
            )
        asset_imported = False
        asset_key_value = arxiv_asset_key(
            effective_record["arxiv_id"],
            effective_record["version"],
            effective_record["asset_type"],
        )
        if upsert_asset and effective_record["storage_path"] is not None:
            asset_imported = _arxiv_bin_member_asset_would_change(conn, effective_record)
            if asset_imported:
                upsert_arxiv_asset(
                    conn,
                    effective_record["arxiv_id"],
                    effective_record["version"],
                    effective_record["asset_type"],
                    status="imported",
                    storage_path=effective_record["storage_path"],
                    checksum=effective_record["checksum"],
                    source="s3_bin_member",
                )
                append_arxiv_event(
                    conn,
                    "asset_imported",
                    arxiv_id=effective_record["arxiv_id"],
                    version=effective_record["version"],
                    asset_type=effective_record["asset_type"],
                    payload={
                        "asset_key": asset_key_value,
                        "bin_key": effective_record["bin_key"],
                        "storage_path": effective_record["storage_path"],
                        "checksum": effective_record["checksum"],
                        "source": "s3_bin_member",
                    },
                )
        _metadata_release(conn, savepoint_name)
        return _arxiv_bin_member_result(
            effective_record,
            inserted=inserted,
            changed=changed,
            changed_fields=changed_fields,
            asset_imported=asset_imported,
        )
    except Exception:
        _metadata_rollback(conn, savepoint_name)
        raise


def _normalize_arxiv_bin_member_records(
    records: Iterable[dict[str, Any]],
    *,
    bin_key: str | None = None,
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    seen_members: set[tuple[str, str, int, str]] = set()
    for index, record in enumerate(records, start=1):
        try:
            normalized = _normalize_arxiv_bin_member_record(record, bin_key=bin_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid S3 bin member record {index}: {exc}") from exc
        member_key = (
            normalized["bin_key"],
            normalized["arxiv_id"],
            normalized["version"],
            normalized["asset_type"],
        )
        if member_key in seen_members:
            raise ValueError(f"duplicate S3 bin member record {index}: {member_key!r}")
        seen_members.add(member_key)
        normalized_records.append(normalized)
    return normalized_records


def import_arxiv_bin_members(
    conn: sqlite3.Connection,
    records: Iterable[dict[str, Any]],
    *,
    bin_key: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import offline S3 bin member rows and reconcile local imported assets."""

    normalized_records = _normalize_arxiv_bin_member_records(records, bin_key=bin_key)
    _validate_arxiv_bin_member_bins(conn, normalized_records)
    result: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "bin_key": bin_key,
        "records_seen": len(normalized_records),
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "changed": 0,
        "assets_imported": 0,
        "rows": [],
    }
    if dry_run:
        for normalized in normalized_records:
            inserted, changed, changed_fields, effective_record = _arxiv_bin_member_change_summary(conn, normalized)
            asset_imported = _arxiv_bin_member_asset_would_change(conn, effective_record)
            row_summary = _arxiv_bin_member_result(
                effective_record,
                inserted=inserted,
                changed=changed,
                changed_fields=changed_fields,
                asset_imported=asset_imported,
            )
            result["rows"].append(row_summary)
            if inserted:
                result["inserted"] += 1
            elif changed:
                result["updated"] += 1
                result["changed"] += 1
            else:
                result["unchanged"] += 1
            if asset_imported:
                result["assets_imported"] += 1
        return result

    savepoint_name = "paperstream_bin_member_import"
    _metadata_savepoint(conn, savepoint_name)
    try:
        for normalized in normalized_records:
            row_summary = upsert_arxiv_bin_member(conn, normalized)
            result["rows"].append(row_summary)
            if row_summary["inserted"]:
                result["inserted"] += 1
            elif row_summary["changed"]:
                result["updated"] += 1
                result["changed"] += 1
            else:
                result["unchanged"] += 1
            if row_summary["asset_imported"]:
                result["assets_imported"] += 1
        _metadata_release(conn, savepoint_name)
        return result
    except Exception:
        _metadata_rollback(conn, savepoint_name)
        raise


def _s3_bin_job_key(job_type: str, bin_key: str) -> str:
    job_type = _validate_job_type(job_type)
    bin_key = _validate_s3_bin_text(bin_key, "bin_key", required=True)
    return f"{job_type}:{bin_key}"


def _s3_bin_row_dict(conn: sqlite3.Connection, bin_key: str) -> dict[str, Any]:
    cursor = conn.execute("SELECT * FROM arxiv_s3_bins WHERE bin_key = ?", (bin_key,))
    row = _row_to_dict(cursor.fetchone(), cursor.description)
    if row is None:
        raise ValueError(f"arxiv_s3_bins row does not exist for bin_key {bin_key!r}")
    return row


def enqueue_arxiv_s3_bin_import_job(
    conn: sqlite3.Connection,
    bin_key: str,
    *,
    source: str,
    priority: int = 100,
    max_attempts: int = 5,
    run_after: datetime | str | None = None,
) -> str:
    """Idempotently queue one offline S3 bin import job and return its bin-scoped key."""

    bin_key = _validate_s3_bin_text(bin_key, "bin_key", required=True)
    source = _validate_s3_bin_text(source, "source", required=True)
    bin_row = _s3_bin_row_dict(conn, bin_key)
    asset_type = _validate_s3_bin_asset_type(bin_row["asset_type"])
    job_key_value = _s3_bin_job_key("import_s3_bin", bin_key)
    priority = int(priority)
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    payload_json = _json_text({"source": source, "bin_key": bin_key})
    if run_after is None:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO arxiv_jobs (
                job_key, job_type, asset_type, priority, max_attempts, payload_json
            )
            VALUES (?, 'import_s3_bin', ?, ?, ?, ?)
            """,
            (job_key_value, asset_type, priority, max_attempts, payload_json),
        )
    else:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO arxiv_jobs (
                job_key, job_type, asset_type, priority, max_attempts, run_after, payload_json
            )
            VALUES (?, 'import_s3_bin', ?, ?, ?, ?, ?)
            """,
            (job_key_value, asset_type, priority, max_attempts, _utc_timestamp(run_after), payload_json),
        )
    if cursor.rowcount:
        append_arxiv_event(
            conn,
            "job_queued",
            asset_type=asset_type,
            job_key=job_key_value,
            payload={"source": "s3_bin_import_queue", "bin_key": bin_key},
        )
    return job_key_value


def append_arxiv_event(
    conn: sqlite3.Connection,
    event_type: str,
    arxiv_id: str | None = None,
    version: int | str | None = None,
    asset_type: str | None = None,
    job_key: str | None = None,
    payload: Any = None,
    now: datetime | str | None = None,
) -> int:
    """Append an audit event and return its event id."""

    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("event_type must be a non-empty string")
    arxiv_id = None if arxiv_id is None else _validate_arxiv_id(arxiv_id)
    version = _validate_version(version, required=False)
    if version is not None and arxiv_id is None:
        raise ValueError("version requires arxiv_id")
    asset_type = _validate_asset_type(asset_type, required=False)
    payload_json = _json_text(payload)
    if now is None:
        cursor = conn.execute(
            """
            INSERT INTO arxiv_events (
                event_type, arxiv_id, version, asset_type, job_key, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_type.strip(), arxiv_id, version, asset_type, job_key, payload_json),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO arxiv_events (
                event_type, arxiv_id, version, asset_type, job_key, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_type.strip(), arxiv_id, version, asset_type, job_key, payload_json, _utc_timestamp(now)),
        )
    return int(cursor.lastrowid)


def enqueue_arxiv_job(
    conn: sqlite3.Connection,
    job_type: str,
    arxiv_id: str | None = None,
    version: int | str | None = None,
    asset_type: str | None = None,
    priority: int = 100,
    max_attempts: int = 5,
    run_after: datetime | str | None = None,
    payload: Any = None,
) -> str:
    """Idempotently enqueue one durable job and return its job key."""

    job_type = _validate_job_type(job_type)
    job_key = arxiv_job_key(job_type, arxiv_id, version)
    arxiv_id = None if arxiv_id is None else _validate_arxiv_id(arxiv_id)
    version = _validate_version(version, required=False)
    if version is not None and arxiv_id is None:
        raise ValueError("version requires arxiv_id")
    asset_type = _validate_asset_type(asset_type, required=False)
    priority = int(priority)
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    payload_json = _json_text(payload)
    if run_after is None:
        conn.execute(
            """
            INSERT OR IGNORE INTO arxiv_jobs (
                job_key, job_type, arxiv_id, version, asset_type,
                priority, max_attempts, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_key, job_type, arxiv_id, version, asset_type, priority, max_attempts, payload_json),
        )
    else:
        conn.execute(
            """
            INSERT OR IGNORE INTO arxiv_jobs (
                job_key, job_type, arxiv_id, version, asset_type,
                priority, max_attempts, run_after, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_key,
                job_type,
                arxiv_id,
                version,
                asset_type,
                priority,
                max_attempts,
                _utc_timestamp(run_after),
                payload_json,
            ),
        )
    return job_key


def _row_value(row: sqlite3.Row | tuple[Any, ...], name: str, index: int) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[name]
    return row[index]


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError("metadata list field must be iterable") from exc
    items: list[str] = []
    for item in iterator:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _metadata_savepoint(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f"SAVEPOINT {name}")


def _metadata_release(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f"RELEASE SAVEPOINT {name}")


def _metadata_rollback(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
    conn.execute(f"RELEASE SAVEPOINT {name}")


def _normalize_metadata_record(parsed_record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed_record, dict):
        raise TypeError("parsed_record must be a dictionary")
    arxiv_id = _validate_arxiv_id(str(parsed_record.get("arxiv_id", "")))
    versions = parsed_record.get("versions") or []
    if versions and not isinstance(versions, list):
        versions = list(versions)
    version_numbers: list[int] = []
    for version in versions:
        if isinstance(version, dict) and version.get("version") is not None:
            version_number = _validate_version(version.get("version"))
            if version_number is not None:
                version_numbers.append(version_number)
    latest_version_value = parsed_record.get("latest_version")
    if latest_version_value is None:
        latest_version_value = max(version_numbers, default=1)
    latest_version = _validate_version(latest_version_value)
    title = _coerce_text(parsed_record.get("title"))
    abstract = _coerce_text(parsed_record.get("abstract"))
    authors = _coerce_text_list(parsed_record.get("authors"))
    categories = _coerce_text_list(parsed_record.get("categories"))
    primary_category = _coerce_text(parsed_record.get("primary_category"))
    if primary_category is None and categories:
        primary_category = categories[0]
    raw_oai_xml = parsed_record.get("raw_oai_xml", parsed_record.get("raw_xml"))
    if raw_oai_xml is not None and not isinstance(raw_oai_xml, str):
        raw_oai_xml = raw_oai_xml.decode("utf-8") if isinstance(raw_oai_xml, bytes) else str(raw_oai_xml)
    normalized_versions: list[dict[str, Any]] = []
    if not versions:
        versions = [{"version": latest_version}]
    for version in versions:
        if not isinstance(version, dict):
            raise TypeError("version metadata entries must be dictionaries")
        version_number = _validate_version(version.get("version", latest_version))
        version_authors = _coerce_text_list(version.get("authors", authors))
        version_categories = _coerce_text_list(version.get("categories", categories))
        normalized_versions.append(
            {
                "version": version_number,
                "submitted_at": _coerce_text(version.get("submitted_at")),
                "announced_at": _coerce_text(version.get("announced_at")),
                "title": _coerce_text(version.get("title", title)),
                "abstract": _coerce_text(version.get("abstract", abstract)),
                "authors_json": _json_text(version_authors),
                "categories_json": _json_text(version_categories),
                "raw_version_metadata_json": _json_text(version),
            }
        )
    return {
        "arxiv_id": arxiv_id,
        "latest_version": latest_version,
        "title": title,
        "abstract": abstract,
        "authors_json": _json_text(authors),
        "primary_category": primary_category,
        "categories_json": _json_text(categories),
        "license": _coerce_text(parsed_record.get("license")),
        "doi": _coerce_text(parsed_record.get("doi")),
        "journal_ref": _coerce_text(parsed_record.get("journal_ref")),
        "abs_url": _coerce_text(parsed_record.get("abs_url")),
        "oai_datestamp": _coerce_text(parsed_record.get("oai_datestamp")),
        "raw_oai_xml": raw_oai_xml,
        "versions": normalized_versions,
    }


def _metadata_change_summary(conn: sqlite3.Connection, metadata: dict[str, Any]) -> dict[str, Any]:
    paper_columns = (
        "latest_version",
        "title",
        "abstract",
        "authors_json",
        "primary_category",
        "categories_json",
        "license",
        "doi",
        "journal_ref",
        "abs_url",
        "oai_datestamp",
        "raw_oai_xml",
    )
    if not _table_exists(conn, "arxiv_papers"):
        return {
            "inserted": True,
            "paper_changed": True,
            "changed_fields": list(paper_columns),
            "versions_inserted": len(metadata["versions"]),
            "versions_changed": 0,
            "changed": True,
        }
    row = conn.execute(
        """
        SELECT latest_version, title, abstract, authors_json, primary_category,
               categories_json, license, doi, journal_ref, abs_url,
               oai_datestamp, raw_oai_xml
        FROM arxiv_papers
        WHERE arxiv_id = ?
        """,
        (metadata["arxiv_id"],),
    ).fetchone()
    inserted = row is None
    if row is None:
        changed_fields = list(paper_columns)
    else:
        changed_fields = [
            column
            for index, column in enumerate(paper_columns)
            if _row_value(row, column, index) != metadata[column]
        ]
    paper_changed = bool(changed_fields)
    if not _table_exists(conn, "arxiv_versions"):
        versions_inserted = len(metadata["versions"])
        return {
            "inserted": inserted,
            "paper_changed": paper_changed,
            "changed_fields": changed_fields,
            "versions_inserted": versions_inserted,
            "versions_changed": 0,
            "changed": paper_changed or versions_inserted > 0,
        }
    versions_inserted = 0
    versions_changed = 0
    for version in metadata["versions"]:
        version_row = conn.execute(
            """
            SELECT submitted_at, announced_at, title, abstract, authors_json,
                   categories_json, raw_version_metadata_json
            FROM arxiv_versions
            WHERE arxiv_id = ? AND version = ?
            """,
            (metadata["arxiv_id"], version["version"]),
        ).fetchone()
        if version_row is None:
            versions_inserted += 1
            continue
        version_columns = (
            "submitted_at",
            "announced_at",
            "title",
            "abstract",
            "authors_json",
            "categories_json",
            "raw_version_metadata_json",
        )
        for index, column in enumerate(version_columns):
            if _row_value(version_row, column, index) != version[column]:
                versions_changed += 1
                break
    return {
        "inserted": inserted,
        "paper_changed": paper_changed,
        "changed_fields": changed_fields,
        "versions_inserted": versions_inserted,
        "versions_changed": versions_changed,
        "changed": paper_changed or versions_inserted > 0 or versions_changed > 0,
    }


def _metadata_job_already_exists(
    conn: sqlite3.Connection,
    job_type: str,
    arxiv_id: str,
    version: int,
) -> bool:
    if not _table_exists(conn, "arxiv_jobs"):
        return False
    row = conn.execute(
        "SELECT 1 FROM arxiv_jobs WHERE job_key = ?",
        (arxiv_job_key(job_type, arxiv_id, version),),
    ).fetchone()
    return row is not None

_READY_DOWNLOAD_ASSET_STATUSES = frozenset({"downloaded", "imported", "parsed", "indexed"})


def _metadata_asset_status(
    conn: sqlite3.Connection,
    arxiv_id: str,
    version: int,
    asset_type: str,
) -> str | None:
    if not _table_exists(conn, "arxiv_assets"):
        return None
    row = conn.execute(
        "SELECT status FROM arxiv_assets WHERE asset_key = ?",
        (arxiv_asset_key(arxiv_id, version, asset_type),),
    ).fetchone()
    return _row_value(row, "status", 0) if row is not None else None

def _validate_enqueue_asset_types(asset_types: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(asset_types, str):
        raw_asset_types = asset_types.split(",")
    else:
        try:
            raw_asset_types = list(asset_types)
        except TypeError as exc:
            raise TypeError("asset_types must be an iterable of asset type strings") from exc
    parsed: list[str] = []
    seen: set[str] = set()
    for asset_type in raw_asset_types:
        if not isinstance(asset_type, str):
            raise TypeError("asset_types must contain only strings")
        asset_type = asset_type.strip()
        if asset_type not in {"pdf", "source"}:
            raise ValueError("asset_types must contain only 'pdf' and 'source'")
        if asset_type not in seen:
            parsed.append(asset_type)
            seen.add(asset_type)
    if not parsed:
        raise ValueError("asset_types must include at least one of 'pdf' or 'source'")
    return tuple(parsed)

def _coerce_sync_enqueue_asset_types(enqueue_assets: Iterable[str] | str | bool | None) -> tuple[str, ...]:
    if enqueue_assets is None or enqueue_assets is False:
        return ()
    if enqueue_assets is True:
        return ("pdf", "source")
    return _validate_enqueue_asset_types(enqueue_assets)

def _download_job_payload(
    arxiv_id: str,
    version: int | str,
    asset_type: str,
    queue_source: str,
    asset_key: str,
) -> dict[str, str]:
    asset_type = _validate_asset_type(asset_type)
    if asset_type not in {"pdf", "source"}:
        raise ValueError("download job payload asset_type must be 'pdf' or 'source'")
    return {
        "source": queue_source,
        "asset_key": asset_key,
        "intended_source": "arxiv_pdf" if asset_type == "pdf" else "arxiv_source",
        "download_url": _arxiv_download_url(arxiv_id, asset_type),
    }




def _validate_download_job_types(job_types: Iterable[str] | str) -> tuple[str, ...]:
    return _validate_job_type_subset(
        job_types,
        allowed={"download_pdf", "download_source"},
        label="job_types",
        description="'download_pdf' and 'download_source'",
    )


def _validate_parse_job_types(job_types: Iterable[str] | str) -> tuple[str, ...]:
    return _validate_job_type_subset(
        job_types,
        allowed={"parse_pdf", "parse_source"},
        label="job_types",
        description="'parse_pdf' and 'parse_source'",
    )


def _validate_work_job_types(job_types: Iterable[str] | str) -> tuple[str, ...]:
    return _validate_job_type_subset(
        job_types,
        allowed={
            "download_pdf",
            "download_source",
            "import_s3_bin",
            "extract_s3_bin",
            "parse_pdf",
            "parse_source",
            "chunk_text",
            "index_bm25",
        },
        label="job_types",
        description="'download_pdf', 'download_source', 'import_s3_bin', 'extract_s3_bin', 'parse_pdf', 'parse_source', 'chunk_text', and 'index_bm25'",
    )


def _validate_job_type_subset(
    job_types: Iterable[str] | str,
    *,
    allowed: set[str],
    label: str,
    description: str,
) -> tuple[str, ...]:
    if isinstance(job_types, str):
        raw_job_types = job_types.split(",")
    else:
        try:
            raw_job_types = list(job_types)
        except TypeError as exc:
            raise TypeError(f"{label} must be an iterable of job type strings") from exc
    parsed: list[str] = []
    seen: set[str] = set()
    for job_type in raw_job_types:
        if not isinstance(job_type, str):
            raise TypeError(f"{label} must contain only strings")
        job_type = job_type.strip()
        if job_type not in allowed:
            raise ValueError(f"{label} must contain only {description}")
        if job_type not in seen:
            parsed.append(job_type)
            seen.add(job_type)
    if not parsed:
        raise ValueError(f"{label} must include at least one supported job type")
    return tuple(parsed)


def _coerce_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _coerce_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be positive")
    return parsed

def _validate_chunk_params(chunk_size: Any, overlap: Any) -> tuple[int, int]:
    chunk_size = _coerce_positive_int(chunk_size, "chunk_size")
    overlap = _coerce_nonnegative_int(overlap, "overlap")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    return chunk_size, overlap




def enqueue_missing_arxiv_assets(
    conn: sqlite3.Connection,
    *,
    asset_types: Iterable[str] | str = ("pdf", "source"),
    category: str | None = None,
    limit: int = 100,
    priority: int = 100,
    max_attempts: int = 5,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Queue missing PDF/source asset downloads from already-synced metadata."""

    parsed_asset_types = _validate_enqueue_asset_types(asset_types)
    limit = _coerce_nonnegative_int(limit, "limit")
    priority = _coerce_nonnegative_int(priority, "priority")
    max_attempts = _coerce_positive_int(max_attempts, "max_attempts")
    if category is not None:
        if not isinstance(category, str):
            raise TypeError("category must be a string")
        category = category.strip()
        if not category:
            raise ValueError("category must not be empty")
    result = {
        "ok": True,
        "asset_types": list(parsed_asset_types),
        "category": category,
        "limit": limit,
        "papers_scanned": 0,
        "assets_selected": 0,
        "assets_created": 0,
        "assets_queued": 0,
        "jobs_queued": 0,
        "asset_keys": [],
        "job_keys": [],
    }
    if limit == 0:
        return result

    params: list[Any] = []
    category_clause = ""
    if category is not None:
        category_clause = "WHERE p.primary_category = ? OR p.categories_json LIKE ?"
        params.extend([category, f"%{json.dumps(category, separators=(',', ':'))}%"])
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT p.arxiv_id, COALESCE(MAX(v.version), p.latest_version) AS latest_available_version
        FROM arxiv_papers AS p
        LEFT JOIN arxiv_versions AS v ON v.arxiv_id = p.arxiv_id
        {category_clause}
        GROUP BY p.arxiv_id
        HAVING latest_available_version IS NOT NULL
        ORDER BY p.arxiv_id
        LIMIT ?
        """,
        params,
    ).fetchall()

    for row in rows:
        arxiv_id = _row_value(row, "arxiv_id", 0)
        version = _validate_version(_row_value(row, "latest_available_version", 1))
        result["papers_scanned"] += 1
        for asset_type in parsed_asset_types:
            result["assets_selected"] += 1
            asset_key_value = arxiv_asset_key(arxiv_id, version, asset_type)
            asset_row = conn.execute(
                "SELECT status FROM arxiv_assets WHERE asset_key = ?",
                (asset_key_value,),
            ).fetchone()
            asset_exists = asset_row is not None
            asset_status = _row_value(asset_row, "status", 0) if asset_row is not None else None
            if not asset_exists:
                upsert_arxiv_asset(
                    conn,
                    arxiv_id,
                    version,
                    asset_type,
                    status="queued",
                    source="metadata_queue",
                )
                result["assets_created"] += 1
                result["assets_queued"] += 1
                result["asset_keys"].append(asset_key_value)
                append_arxiv_event(
                    conn,
                    "asset_seen",
                    arxiv_id=arxiv_id,
                    version=version,
                    asset_type=asset_type,
                    payload={"asset_key": asset_key_value, "source": "metadata_queue"},
                    now=now,
                )
            if asset_status in {"downloaded", "imported", "parsed", "indexed"}:
                continue
            job_type = "download_pdf" if asset_type == "pdf" else "download_source"
            if _metadata_job_already_exists(conn, job_type, arxiv_id, version):
                continue
            job_payload = _download_job_payload(
                arxiv_id,
                version,
                asset_type,
                "metadata_queue",
                asset_key_value,
            )
            job_key_value = enqueue_arxiv_job(
                conn,
                job_type,
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                priority=priority,
                max_attempts=max_attempts,
                run_after=now,
                payload=job_payload,
            )
            result["jobs_queued"] += 1
            result["job_keys"].append(job_key_value)
            append_arxiv_event(
                conn,
                "job_queued",
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                job_key=job_key_value,
                payload=job_payload,
                now=now,
            )
    return result


def _metadata_existing_paper_state(conn: sqlite3.Connection, arxiv_id: str) -> dict[str, Any] | None:
    if not _table_exists(conn, "arxiv_papers"):
        return None
    row = conn.execute(
        "SELECT latest_version, oai_datestamp FROM arxiv_papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "latest_version": _row_value(row, "latest_version", 0),
        "oai_datestamp": _row_value(row, "oai_datestamp", 1),
    }


def upsert_paperstream_metadata_record(
    conn: sqlite3.Connection,
    parsed_record: dict[str, Any],
    *,
    enqueue_assets: Iterable[str] | str | bool | None = False,
) -> dict[str, Any]:
    """Upsert one parsed arXiv metadata record and append audit events."""

    metadata = _normalize_metadata_record(parsed_record)
    enqueue_asset_types = _coerce_sync_enqueue_asset_types(enqueue_assets)
    existing_paper = _metadata_existing_paper_state(conn, metadata["arxiv_id"])
    versionless_existing_update = (
        existing_paper is not None
        and not parsed_record.get("versions")
        and parsed_record.get("latest_version") is None
    )
    if existing_paper is not None:
        existing_datestamp = str(existing_paper.get("oai_datestamp") or "").strip()
        incoming_datestamp = str(metadata.get("oai_datestamp") or "").strip()
        if existing_datestamp and (not incoming_datestamp or existing_datestamp > incoming_datestamp):
            latest_version = _validate_version(existing_paper["latest_version"])
            return {
                "arxiv_id": metadata["arxiv_id"],
                "latest_version": latest_version,
                "inserted": False,
                "changed": False,
                "changed_fields": [],
                "versions_seen": 0 if versionless_existing_update else len(metadata["versions"]),
                "versions_changed": 0,
                "versions_inserted": 0,
                "asset_key": arxiv_asset_key(metadata["arxiv_id"], latest_version, "metadata"),
                "jobs_queued": 0,
                "job_keys": [],
            }
        if versionless_existing_update:
            metadata["latest_version"] = _validate_version(existing_paper["latest_version"])
            metadata["versions"] = []
    savepoint_name = "paperstream_metadata_upsert"
    _metadata_savepoint(conn, savepoint_name)
    try:
        change_summary = _metadata_change_summary(conn, metadata)
        conn.execute(
            """
            INSERT INTO arxiv_papers (
                arxiv_id, latest_version, title, abstract, authors_json,
                primary_category, categories_json, license, doi, journal_ref,
                abs_url, oai_datestamp, raw_oai_xml
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                latest_version = excluded.latest_version,
                title = excluded.title,
                abstract = excluded.abstract,
                authors_json = excluded.authors_json,
                primary_category = excluded.primary_category,
                categories_json = excluded.categories_json,
                license = excluded.license,
                doi = excluded.doi,
                journal_ref = excluded.journal_ref,
                abs_url = excluded.abs_url,
                last_seen_at = CURRENT_TIMESTAMP,
                oai_datestamp = excluded.oai_datestamp,
                raw_oai_xml = excluded.raw_oai_xml
            """,
            (
                metadata["arxiv_id"],
                metadata["latest_version"],
                metadata["title"],
                metadata["abstract"],
                metadata["authors_json"],
                metadata["primary_category"],
                metadata["categories_json"],
                metadata["license"],
                metadata["doi"],
                metadata["journal_ref"],
                metadata["abs_url"],
                metadata["oai_datestamp"],
                metadata["raw_oai_xml"],
            ),
        )
        for version in metadata["versions"]:
            conn.execute(
                """
                INSERT INTO arxiv_versions (
                    arxiv_id, version, submitted_at, announced_at, title,
                    abstract, authors_json, categories_json, raw_version_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id, version) DO UPDATE SET
                    submitted_at = excluded.submitted_at,
                    announced_at = excluded.announced_at,
                    title = excluded.title,
                    abstract = excluded.abstract,
                    authors_json = excluded.authors_json,
                    categories_json = excluded.categories_json,
                    raw_version_metadata_json = excluded.raw_version_metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    metadata["arxiv_id"],
                    version["version"],
                    version["submitted_at"],
                    version["announced_at"],
                    version["title"],
                    version["abstract"],
                    version["authors_json"],
                    version["categories_json"],
                    version["raw_version_metadata_json"],
                ),
            )
        paper_event = "paper_updated" if change_summary["changed"] and not change_summary["inserted"] else "paper_seen"
        paper_payload = {
            "latest_version": metadata["latest_version"],
            "changed": change_summary["changed"],
        }
        if paper_event == "paper_updated":
            paper_payload["changed_fields"] = change_summary["changed_fields"]
        append_arxiv_event(
            conn,
            paper_event,
            arxiv_id=metadata["arxiv_id"],
            version=metadata["latest_version"],
            payload=paper_payload,
        )
        for version in metadata["versions"]:
            append_arxiv_event(
                conn,
                "version_seen",
                arxiv_id=metadata["arxiv_id"],
                version=version["version"],
                payload={"changed": bool(change_summary["versions_inserted"] or change_summary["versions_changed"])},
            )
        metadata_asset_key = upsert_arxiv_asset(
            conn,
            metadata["arxiv_id"],
            metadata["latest_version"],
            "metadata",
            status="seen",
            source="oai",
        )
        append_arxiv_event(
            conn,
            "asset_seen",
            arxiv_id=metadata["arxiv_id"],
            version=metadata["latest_version"],
            asset_type="metadata",
            payload={"asset_key": metadata_asset_key},
        )
        job_keys: list[str] = []
        for asset_type in enqueue_asset_types:
            job_type = f"download_{asset_type}"
            asset_key_value = arxiv_asset_key(
                metadata["arxiv_id"],
                metadata["latest_version"],
                asset_type,
            )
            asset_status = _metadata_asset_status(
                conn,
                metadata["arxiv_id"],
                metadata["latest_version"],
                asset_type,
            )
            if asset_status in _READY_DOWNLOAD_ASSET_STATUSES:
                continue
            if asset_status is None:
                upsert_arxiv_asset(
                    conn,
                    metadata["arxiv_id"],
                    metadata["latest_version"],
                    asset_type,
                    status="queued",
                    source="metadata_sync",
                )
                append_arxiv_event(
                    conn,
                    "asset_seen",
                    arxiv_id=metadata["arxiv_id"],
                    version=metadata["latest_version"],
                    asset_type=asset_type,
                    payload={"asset_key": asset_key_value, "source": "metadata_sync"},
                )
            job_preexisted = _metadata_job_already_exists(
                conn,
                job_type,
                metadata["arxiv_id"],
                metadata["latest_version"],
            )
            job_payload = _download_job_payload(
                metadata["arxiv_id"],
                metadata["latest_version"],
                asset_type,
                "metadata_sync",
                asset_key_value,
            )
            job_key_value = enqueue_arxiv_job(
                conn,
                job_type,
                arxiv_id=metadata["arxiv_id"],
                version=metadata["latest_version"],
                asset_type=asset_type,
                payload=job_payload,
            )
            if not job_preexisted:
                job_keys.append(job_key_value)
                append_arxiv_event(
                    conn,
                    "job_queued",
                    arxiv_id=metadata["arxiv_id"],
                    version=metadata["latest_version"],
                    asset_type=asset_type,
                    job_key=job_key_value,
                    payload=job_payload,
                )
        _metadata_release(conn, savepoint_name)
        return {
            "arxiv_id": metadata["arxiv_id"],
            "latest_version": metadata["latest_version"],
            "inserted": change_summary["inserted"],
            "changed": change_summary["changed"],
            "changed_fields": change_summary["changed_fields"],
            "versions_seen": len(metadata["versions"]),
            "versions_changed": change_summary["versions_changed"],
            "versions_inserted": change_summary["versions_inserted"],
            "asset_key": metadata_asset_key,
            "jobs_queued": len(job_keys),
            "job_keys": job_keys,
        }
    except Exception:
        _metadata_rollback(conn, savepoint_name)
        raise


def _coerce_metadata_page(page: Any) -> dict[str, Any]:
    if callable(page):
        page = page()
    if isinstance(page, dict):
        if "records" in page:
            return page
        if "arxiv_id" in page or "identifier" in page or "raw_xml" in page:
            return {"records": [page], "response_date": page.get("response_date")}
    if isinstance(page, (str, bytes)):
        return parse_oai_list_records(page)
    raise TypeError("metadata sync pages must be XML text/bytes, parsed page dictionaries, or callables")


def _deleted_oai_record_audit_payload(record: Any) -> dict[str, Any] | None:
    """Return a JSON-safe tombstone audit payload for deleted OAI records."""

    identifier = None
    oai_datestamp = None
    set_specs: list[str] = []
    deleted = False
    root = None
    if isinstance(record, dict):
        identifier = _coerce_text(record.get("identifier"))
        oai_datestamp = _coerce_text(record.get("datestamp") or record.get("oai_datestamp"))
        set_specs = _coerce_text_list(record.get("set_specs"))
        deleted = bool(record.get("deleted"))
        raw_oai_xml = record.get("raw_xml")
        has_oai_header_fields = (
            "identifier" in record and "datestamp" in record and "set_specs" in record
        )
        needs_header_parse = not has_oai_header_fields or (
            deleted and (identifier is None or oai_datestamp is None or not set_specs)
        )
        if raw_oai_xml and needs_header_parse:
            root = _parse_xml(raw_oai_xml, "OAI record XML")
    else:
        root = _parse_xml(record, "OAI record XML")

    if root is not None:
        header = _xml_child(root, "header")
        deleted = deleted or (header is not None and header.attrib.get("status") == "deleted")
        identifier = identifier or _xml_text(_xml_child(header, "identifier"))
        oai_datestamp = oai_datestamp or _xml_text(_xml_child(header, "datestamp"))
        if not set_specs:
            set_specs = [
                set_spec
                for set_spec in (_xml_text(child) for child in _xml_children(header, "setSpec"))
                if set_spec
            ]

    if not deleted:
        return None

    arxiv_id, _version = _split_arxiv_version(_arxiv_id_from_oai_identifier(identifier))
    return {
        "identifier": identifier,
        "arxiv_id": arxiv_id,
        "oai_datestamp": oai_datestamp,
        "set_specs": set_specs,
    }


def _event_arxiv_id_from_deleted_payload(payload: dict[str, Any]) -> str | None:
    arxiv_id = payload.get("arxiv_id")
    if arxiv_id is None:
        return None
    try:
        return _validate_arxiv_id(str(arxiv_id))
    except (TypeError, ValueError):
        return None


def _coerce_metadata_record(record: Any) -> dict[str, Any] | None:
    if isinstance(record, dict) and record.get("deleted"):
        return None
    if isinstance(record, dict) and "arxiv_id" in record:
        return record
    parsed = parse_arxiv_raw_record(record)
    if parsed.get("deleted"):
        return None
    return parsed


def _planned_metadata_jobs(
    conn: sqlite3.Connection,
    metadata: dict[str, Any],
    enqueue_asset_types: tuple[str, ...],
) -> int:
    planned = 0
    for asset_type in enqueue_asset_types:
        asset_status = _metadata_asset_status(conn, metadata["arxiv_id"], metadata["latest_version"], asset_type)
        if asset_status in _READY_DOWNLOAD_ASSET_STATUSES:
            continue
        job_type = f"download_{asset_type}"
        if not _metadata_job_already_exists(conn, job_type, metadata["arxiv_id"], metadata["latest_version"]):
            planned += 1
    return planned


def _sync_state_row(conn: sqlite3.Connection, sync_name: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT committed_from, last_response_date, metadata_prefix, set_spec
        FROM arxiv_sync_state
        WHERE sync_name = ?
        """,
        (sync_name,),
    ).fetchone()
    if row is None:
        return {
            "committed_from": None,
            "last_response_date": None,
            "metadata_prefix": "arXivRaw",
            "set_spec": None,
        }
    return {
        "committed_from": _row_value(row, "committed_from", 0),
        "last_response_date": _row_value(row, "last_response_date", 1),
        "metadata_prefix": _row_value(row, "metadata_prefix", 2) or "arXivRaw",
        "set_spec": _row_value(row, "set_spec", 3),
    }


def _start_metadata_sync(
    conn: sqlite3.Connection,
    sync_name: str,
    *,
    metadata_prefix: str,
    set_spec: str | None,
) -> dict[str, Any]:
    if not isinstance(sync_name, str) or not sync_name.strip():
        raise ValueError("sync_name must be a non-empty string")
    sync_name = sync_name.strip()
    previous_state = _sync_state_row(conn, sync_name)
    now_text = _utc_timestamp()
    conn.execute(
        """
        INSERT INTO arxiv_sync_state (
            sync_name, metadata_prefix, set_spec, status,
            records_seen_last_run, records_updated_last_run,
            records_inserted_last_run, versions_inserted_last_run,
            jobs_queued_last_run, error_message, started_at, finished_at, updated_at
        )
        VALUES (?, ?, ?, 'running', 0, 0, 0, 0, 0, NULL, ?, NULL, ?)
        ON CONFLICT(sync_name) DO UPDATE SET
            metadata_prefix = excluded.metadata_prefix,
            set_spec = excluded.set_spec,
            status = 'running',
            records_seen_last_run = 0,
            records_updated_last_run = 0,
            records_inserted_last_run = 0,
            versions_inserted_last_run = 0,
            jobs_queued_last_run = 0,
            error_message = NULL,
            pending_response_date = NULL,
            started_at = excluded.started_at,
            finished_at = NULL,
            updated_at = excluded.updated_at
        """,
        (sync_name, metadata_prefix, set_spec, now_text, now_text),
    )
    append_arxiv_event(
        conn,
        "sync_started",
        payload={
            "sync_name": sync_name,
            "committed_from": previous_state["committed_from"],
            "metadata_prefix": metadata_prefix,
            "set_spec": set_spec,
        },
    )
    return previous_state


def _finish_metadata_sync(
    conn: sqlite3.Connection,
    sync_name: str,
    status: str,
    counts: dict[str, int],
    *,
    committed_from: str | None,
    last_response_date: str | None,
    pending_response_date: str | None,
    last_resumption_token: str | None,
    error_message: str | None = None,
) -> None:
    now_text = _utc_timestamp()
    if status == "succeeded":
        conn.execute(
            """
            UPDATE arxiv_sync_state
            SET committed_from = ?,
                last_response_date = ?,
                pending_response_date = NULL,
                last_resumption_token = ?,
                status = 'succeeded',
                records_seen_last_run = ?,
                records_updated_last_run = ?,
                records_inserted_last_run = ?,
                versions_inserted_last_run = ?,
                jobs_queued_last_run = ?,
                error_message = NULL,
                finished_at = ?,
                updated_at = ?
            WHERE sync_name = ?
            """,
            (
                committed_from,
                last_response_date,
                last_resumption_token,
                counts["records_seen"],
                counts["records_updated"],
                counts["records_inserted"],
                counts["versions_inserted"],
                counts["jobs_queued"],
                now_text,
                now_text,
                sync_name,
            ),
        )
        append_arxiv_event(
            conn,
            "sync_succeeded",
            payload={"sync_name": sync_name, **counts, "committed_from": committed_from},
        )
        append_arxiv_event(
            conn,
            "cursor_committed",
            payload={
                "sync_name": sync_name,
                "committed_from": committed_from,
                "last_response_date": last_response_date,
                "last_resumption_token": last_resumption_token,
            },
        )
        return
    conn.execute(
        """
        UPDATE arxiv_sync_state
        SET pending_response_date = ?,
            last_resumption_token = ?,
            status = 'failed',
            records_seen_last_run = ?,
            records_updated_last_run = ?,
            records_inserted_last_run = ?,
            versions_inserted_last_run = ?,
            jobs_queued_last_run = ?,
            error_message = ?,
            finished_at = ?,
            updated_at = ?
        WHERE sync_name = ?
        """,
        (
            pending_response_date,
            last_resumption_token,
            counts["records_seen"],
            counts["records_updated"],
            counts["records_inserted"],
            counts["versions_inserted"],
            counts["jobs_queued"],
            error_message,
            now_text,
            now_text,
            sync_name,
        ),
    )
    append_arxiv_event(
        conn,
        "sync_failed",
        payload={"sync_name": sync_name, **counts, "error": error_message},
    )



def _metadata_version_key(arxiv_id: str, version: int) -> str:
    """Return the canonical persisted metadata-version Bloom key."""

    return f"{_validate_arxiv_id(arxiv_id)}:v{_validate_version(version)}"


def _metadata_version_exists(conn: sqlite3.Connection, arxiv_id: str, version: int) -> bool:
    if not _table_exists(conn, "arxiv_versions"):
        return False
    row = conn.execute(
        """
        SELECT 1
        FROM arxiv_versions
        WHERE arxiv_id = ? AND version = ?
        LIMIT 1
        """,
        (_validate_arxiv_id(arxiv_id), _validate_version(version)),
    ).fetchone()
    return row is not None


def _rebuild_metadata_versions_bloom_filter(
    conn: sqlite3.Connection,
    *,
    false_positive_rate: float = 0.01,
) -> BloomFilter:
    return rebuild_bloom_filter(
        conn,
        "seen_metadata_versions",
        _metadata_version_keys_for_bloom(conn),
        false_positive_rate=false_positive_rate,
        source_table="arxiv_versions",
        source_column="version",
    )


def _load_or_rebuild_metadata_versions_bloom_filter(
    conn: sqlite3.Connection,
) -> tuple[BloomFilter, str]:
    bloom_filter = load_bloom_filter(conn, "seen_metadata_versions")
    if bloom_filter is not None:
        return bloom_filter, "loaded"
    return _rebuild_metadata_versions_bloom_filter(conn), "rebuilt"


def _metadata_sync_bloom_precheck(
    conn: sqlite3.Connection,
    bloom_filter: BloomFilter,
    metadata: dict[str, Any],
) -> dict[str, int]:
    counts = {
        "metadata_bloom_prechecks": 0,
        "metadata_bloom_hits": 0,
        "metadata_bloom_misses": 0,
        "metadata_bloom_sqlite_checks": 0,
        "metadata_bloom_confirmed": 0,
        "metadata_bloom_false_positives": 0,
    }
    arxiv_id = metadata["arxiv_id"]
    for version in metadata["versions"]:
        version_number = version["version"]
        version_key = _metadata_version_key(arxiv_id, version_number)
        counts["metadata_bloom_prechecks"] += 1
        if version_key not in bloom_filter:
            counts["metadata_bloom_misses"] += 1
            continue
        counts["metadata_bloom_hits"] += 1
        counts["metadata_bloom_sqlite_checks"] += 1
        if _metadata_version_exists(conn, arxiv_id, version_number):
            counts["metadata_bloom_confirmed"] += 1
        else:
            counts["metadata_bloom_false_positives"] += 1
    return counts


def sync_paperstream_metadata(
    conn: sqlite3.Connection,
    pages: Iterable[Any],
    *,
    sync_name: str = "default",
    dry_run: bool = False,
    max_pages: int | None = None,
    max_records: int | None = None,
    enqueue_assets: Iterable[str] | str | bool | None = False,
    metadata_prefix: str = "arXivRaw",
    set_spec: str | None = None,
) -> dict[str, Any]:
    """Consume OAI metadata pages and update local PaperStream metadata state."""

    if max_pages is not None and int(max_pages) < 0:
        raise ValueError("max_pages must be non-negative")
    if max_records is not None and int(max_records) < 0:
        raise ValueError("max_records must be non-negative")
    max_pages = None if max_pages is None else int(max_pages)
    max_records = None if max_records is None else int(max_records)
    enqueue_asset_types = _coerce_sync_enqueue_asset_types(enqueue_assets)
    metadata_prefix = _oai_nonempty_param("metadata_prefix", metadata_prefix)
    if metadata_prefix is None:
        raise ValueError("metadata_prefix must not be empty")
    set_spec = _oai_nonempty_param("set_spec", set_spec)
    counts = {
        "pages_seen": 0,
        "pages_committed": 0,
        "records_seen": 0,
        "records_updated": 0,
        "records_inserted": 0,
        "records_deleted": 0,
        "versions_inserted": 0,
        "versions_changed": 0,
        "assets_seen": 0,
        "jobs_queued": 0,
        "metadata_bloom_prechecks": 0,
        "metadata_bloom_hits": 0,
        "metadata_bloom_misses": 0,
        "metadata_bloom_sqlite_checks": 0,
        "metadata_bloom_confirmed": 0,
        "metadata_bloom_false_positives": 0,
        "metadata_bloom_filter_loaded": 0,
        "metadata_bloom_filter_rebuilt": 0,
        "metadata_bloom_filter_refreshed": 0,
    }
    if dry_run:
        for page_ref in pages:
            if max_pages is not None and counts["pages_seen"] >= max_pages:
                break
            if max_records is not None and counts["records_seen"] >= max_records:
                break
            page = _coerce_metadata_page(page_ref)
            counts["pages_seen"] += 1
            for record in page.get("records", []):
                if max_records is not None and counts["records_seen"] >= max_records:
                    break
                counts["records_seen"] += 1
                deleted_payload = _deleted_oai_record_audit_payload(record)
                if deleted_payload is not None:
                    counts["records_deleted"] += 1
                    continue
                parsed = _coerce_metadata_record(record)
                metadata = _normalize_metadata_record(parsed)
                summary = _metadata_change_summary(conn, metadata)
                if summary["inserted"]:
                    counts["records_inserted"] += 1
                if summary["changed"] and not summary["inserted"]:
                    counts["records_updated"] += 1
                counts["versions_inserted"] += int(summary["versions_inserted"])
                counts["assets_seen"] += 1
                counts["jobs_queued"] += _planned_metadata_jobs(conn, metadata, enqueue_asset_types)
        return {"ok": True, "dry_run": True, "metadata_prefix": metadata_prefix, "set_spec": set_spec, **counts}

    previous_state: dict[str, Any] = {}
    last_committed_response_date: str | None = None
    last_seen_response_date: str | None = None
    last_resumption_token: str | None = None
    last_processed_page_complete: bool | None = None
    start_savepoint = "paperstream_sync_start"
    _metadata_savepoint(conn, start_savepoint)
    try:
        sync_name = sync_name.strip()
        previous_state = _start_metadata_sync(
            conn,
            sync_name,
            metadata_prefix=metadata_prefix,
            set_spec=set_spec,
        )
        metadata_bloom_filter, metadata_bloom_filter_state = _load_or_rebuild_metadata_versions_bloom_filter(conn)
        if metadata_bloom_filter_state == "loaded":
            counts["metadata_bloom_filter_loaded"] = 1
        else:
            counts["metadata_bloom_filter_rebuilt"] = 1
        _metadata_release(conn, start_savepoint)
    except Exception:
        _metadata_rollback(conn, start_savepoint)
        raise
    try:
        for page_ref in pages:
            if max_pages is not None and counts["pages_seen"] >= max_pages:
                break
            if max_records is not None and counts["records_seen"] >= max_records:
                break
            page = _coerce_metadata_page(page_ref)
            page_response_date = _coerce_text(page.get("response_date"))
            page_resumption_token = _coerce_text(page.get("resumption_token"))
            last_seen_response_date = page_response_date or last_seen_response_date
            last_resumption_token = page_resumption_token
            records = list(page.get("records", []))
            counts["pages_seen"] += 1
            page_savepoint = "paperstream_sync_page"
            page_counts = {
                "records_seen": 0,
                "records_updated": 0,
                "records_inserted": 0,
                "records_deleted": 0,
                "versions_inserted": 0,
                "assets_seen": 0,
                "jobs_queued": 0,
            }
            page_bloom_counts = {
                "metadata_bloom_prechecks": 0,
                "metadata_bloom_hits": 0,
                "metadata_bloom_misses": 0,
                "metadata_bloom_sqlite_checks": 0,
                "metadata_bloom_confirmed": 0,
                "metadata_bloom_false_positives": 0,
            }
            page_versions_changed = 0
            page_complete = True
            _metadata_savepoint(conn, page_savepoint)
            try:
                for record in records:
                    if max_records is not None and counts["records_seen"] + page_counts["records_seen"] >= max_records:
                        page_complete = False
                        break
                    page_counts["records_seen"] += 1
                    deleted_payload = _deleted_oai_record_audit_payload(record)
                    if deleted_payload is not None:
                        page_counts["records_deleted"] += 1
                        append_arxiv_event(
                            conn,
                            "paper_deleted",
                            arxiv_id=_event_arxiv_id_from_deleted_payload(deleted_payload),
                            payload=deleted_payload,
                        )
                        continue
                    parsed = _coerce_metadata_record(record)
                    if parsed.get("versions"):
                        metadata_for_bloom = _normalize_metadata_record(parsed)
                        bloom_counts = _metadata_sync_bloom_precheck(
                            conn,
                            metadata_bloom_filter,
                            metadata_for_bloom,
                        )
                        for key, value in bloom_counts.items():
                            page_bloom_counts[key] += value
                    result = upsert_paperstream_metadata_record(
                        conn,
                        parsed,
                        enqueue_assets=enqueue_asset_types or False,
                    )
                    if result["inserted"]:
                        page_counts["records_inserted"] += 1
                    if result["changed"] and not result["inserted"]:
                        page_counts["records_updated"] += 1
                    page_counts["versions_inserted"] += int(result["versions_inserted"])
                    page_versions_changed += int(result["versions_changed"])
                    page_counts["assets_seen"] += 1
                    page_counts["jobs_queued"] += int(result["jobs_queued"])
                for key, value in page_counts.items():
                    counts[key] += value
                counts["versions_changed"] += page_versions_changed
                for key, value in page_bloom_counts.items():
                    counts[key] += value
                last_seen_response_date = page_response_date or last_seen_response_date
                last_resumption_token = page_resumption_token
                if page_complete:
                    counts["pages_committed"] += 1
                    last_committed_response_date = page_response_date or last_committed_response_date
                    conn.execute(
                        """
                        UPDATE arxiv_sync_state
                        SET pending_response_date = ?,
                            last_resumption_token = ?,
                            records_seen_last_run = ?,
                            records_updated_last_run = ?,
                            records_inserted_last_run = ?,
                            versions_inserted_last_run = ?,
                            jobs_queued_last_run = ?,
                            updated_at = ?
                        WHERE sync_name = ?
                        """,
                        (
                            last_committed_response_date,
                            last_resumption_token,
                            counts["records_seen"],
                            counts["records_updated"],
                            counts["records_inserted"],
                            counts["versions_inserted"],
                            counts["jobs_queued"],
                            _utc_timestamp(),
                            sync_name,
                        ),
                    )
                    append_arxiv_event(
                        conn,
                        "sync_page_committed",
                        payload={
                            "sync_name": sync_name,
                            "page": counts["pages_seen"],
                            "response_date": page_response_date,
                            "resumption_token": page_resumption_token,
                            **page_counts,
                        },
                    )
                _metadata_release(conn, page_savepoint)
            except Exception:
                _metadata_rollback(conn, page_savepoint)
                raise
            last_processed_page_complete = page_complete
            if not page_complete:
                break
        previous_committed_from = previous_state.get("committed_from")
        previous_last_response_date = previous_state.get("last_response_date")
        partial_error: str | None = None
        if last_processed_page_complete is None:
            partial_error = "metadata sync incomplete: no OAI pages were processed"
        elif not last_processed_page_complete:
            partial_error = "metadata sync incomplete: max_records cut a page before it was fully processed"
        elif last_resumption_token is not None:
            partial_error = "metadata sync incomplete: resumption token remains after last committed page"

        if partial_error is not None:
            finish_savepoint = "paperstream_sync_finish"
            _metadata_savepoint(conn, finish_savepoint)
            try:
                pending_response_date = last_seen_response_date or last_committed_response_date
                _finish_metadata_sync(
                    conn,
                    sync_name,
                    "failed",
                    counts,
                    committed_from=previous_committed_from,
                    last_response_date=previous_last_response_date,
                    pending_response_date=pending_response_date,
                    last_resumption_token=last_resumption_token,
                    error_message=partial_error,
                )
                _metadata_release(conn, finish_savepoint)
            except Exception:
                _metadata_rollback(conn, finish_savepoint)
                raise
            return {
                "ok": False,
                "partial": True,
                "dry_run": False,
                "sync_name": sync_name,
                "metadata_prefix": metadata_prefix,
                "set_spec": set_spec,
                "committed_from": previous_committed_from,
                "last_response_date": previous_last_response_date,
                "pending_response_date": pending_response_date,
                "last_resumption_token": last_resumption_token,
                "last_page_complete": last_processed_page_complete,
                "resumption_token_exhausted": last_resumption_token is None,
                "error": partial_error,
                **counts,
            }

        finish_savepoint = "paperstream_sync_finish"
        _metadata_savepoint(conn, finish_savepoint)
        try:
            committed_cursor = last_committed_response_date or previous_committed_from
            last_response_date = last_committed_response_date or previous_last_response_date
            if counts["versions_inserted"] + counts["versions_changed"] > 0:
                metadata_bloom_filter = _rebuild_metadata_versions_bloom_filter(
                    conn,
                    false_positive_rate=metadata_bloom_filter.false_positive_rate,
                )
                counts["metadata_bloom_filter_refreshed"] = 1
            _finish_metadata_sync(
                conn,
                sync_name,
                "succeeded",
                counts,
                committed_from=committed_cursor,
                last_response_date=last_response_date,
                pending_response_date=None,
                last_resumption_token=last_resumption_token,
            )
            _metadata_release(conn, finish_savepoint)
        except Exception:
            _metadata_rollback(conn, finish_savepoint)
            raise
        return {
            "ok": True,
            "dry_run": False,
            "sync_name": sync_name,
            "metadata_prefix": metadata_prefix,
            "set_spec": set_spec,
            "committed_from": last_committed_response_date or previous_committed_from,
            "last_response_date": last_committed_response_date or previous_last_response_date,
            "last_resumption_token": last_resumption_token,
            "last_page_complete": last_processed_page_complete,
            "resumption_token_exhausted": True,
            **counts,
        }
    except Exception as exc:
        fail_savepoint = "paperstream_sync_failure"
        _metadata_savepoint(conn, fail_savepoint)
        try:
            _finish_metadata_sync(
                conn,
                sync_name,
                "failed",
                counts,
                committed_from=previous_state.get("committed_from"),
                last_response_date=previous_state.get("last_response_date"),
                pending_response_date=last_seen_response_date,
                last_resumption_token=last_resumption_token,
                error_message=str(exc),
            )
            _metadata_release(conn, fail_savepoint)
        except Exception:
            _metadata_rollback(conn, fail_savepoint)
        raise


def claim_next_arxiv_job(
    conn: sqlite3.Connection,
    worker_id: str,
    job_types: Iterable[str] | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any] | None:
    """Atomically claim the next eligible queued job for ``worker_id``."""

    worker_id = _validate_worker_id(worker_id)
    now_text = _utc_timestamp(now)
    params: list[Any] = [worker_id, now_text, now_text, now_text]
    job_type_clause = ""
    if job_types is not None:
        job_type_values = [_validate_job_type(job_type) for job_type in job_types]
        if not job_type_values:
            return None
        placeholders = ",".join("?" for _ in job_type_values)
        job_type_clause = f"AND job_type IN ({placeholders})"
        params.extend(job_type_values)
    cursor = conn.execute(
        f"""
        UPDATE arxiv_jobs
        SET status = 'running',
            locked_by = ?,
            locked_at = ?,
            updated_at = ?
        WHERE job_key = (
            SELECT job_key
            FROM arxiv_jobs
            WHERE status = 'queued'
              AND (run_after IS NULL OR run_after <= ?)
              AND attempts < max_attempts
              {job_type_clause}
            ORDER BY priority ASC, run_after ASC, created_at ASC, job_key ASC
            LIMIT 1
        )
        RETURNING *
        """,
        params,
    )
    return _row_to_dict(cursor.fetchone(), cursor.description)


def mark_arxiv_job_succeeded(
    conn: sqlite3.Connection,
    job_key: str,
    now: datetime | str | None = None,
) -> dict[str, Any] | None:
    """Mark a job succeeded, clear lock/error fields, and append an audit event."""

    if not isinstance(job_key, str) or not job_key.strip():
        raise ValueError("job_key must be a non-empty string")
    now_text = _utc_timestamp(now)
    cursor = conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = 'succeeded',
            locked_by = NULL,
            locked_at = NULL,
            error_message = NULL,
            updated_at = ?
        WHERE job_key = ? AND status IN ('running', 'queued')
        RETURNING *
        """,
        (now_text, job_key.strip()),
    )
    job = _row_to_dict(cursor.fetchone(), cursor.description)
    if job is None:
        return None
    append_arxiv_event(
        conn,
        "job_succeeded",
        arxiv_id=job["arxiv_id"],
        version=job["version"],
        asset_type=job["asset_type"],
        job_key=job["job_key"],
        payload={
            "job_type": job["job_type"],
            "status": job["status"],
            "attempts": job["attempts"],
            "max_attempts": job["max_attempts"],
            "run_after": job["run_after"],
            "error_message": job["error_message"],
            "succeeded_at": now_text,
        },
        now=now_text,
    )
    return job


def mark_arxiv_job_failed(
    conn: sqlite3.Connection,
    job_key: str,
    error_message: str,
    now: datetime | str | None = None,
    retry_delay_seconds: int = 60,
    backoff_seconds: int | None = None,
) -> dict[str, Any] | None:
    """Record a failed attempt, requeueing until ``max_attempts`` is reached."""

    if backoff_seconds is not None:
        retry_delay_seconds = backoff_seconds
    if not isinstance(job_key, str) or not job_key.strip():
        raise ValueError("job_key must be a non-empty string")
    if not isinstance(error_message, str) or not error_message.strip():
        raise ValueError("error_message must be a non-empty string")
    retry_delay_seconds = int(retry_delay_seconds)
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    now_dt = datetime.now(timezone.utc) if now is None else now
    now_text = _utc_timestamp(now_dt)
    job_key_value = job_key.strip()
    error_message_value = error_message.strip()
    row = conn.execute(
        "SELECT attempts, max_attempts FROM arxiv_jobs WHERE job_key = ?",
        (job_key_value,),
    ).fetchone()
    if row is None:
        return None
    attempts = int(row["attempts"] if isinstance(row, sqlite3.Row) else row[0])
    max_attempts = int(row["max_attempts"] if isinstance(row, sqlite3.Row) else row[1])
    next_attempts = attempts + 1
    terminal = next_attempts >= max_attempts
    if isinstance(now_dt, str):
        try:
            base_dt = datetime.fromisoformat(now_text.replace("Z", "+00:00"))
        except ValueError:
            base_dt = datetime.strptime(now_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if base_dt.tzinfo is None:
            base_dt = base_dt.replace(tzinfo=timezone.utc)
        else:
            base_dt = base_dt.astimezone(timezone.utc)
    elif now_dt.tzinfo is None:
        base_dt = now_dt.replace(tzinfo=timezone.utc)
    else:
        base_dt = now_dt.astimezone(timezone.utc)
    delay = retry_delay_seconds * (2 ** max(next_attempts - 1, 0))
    run_after = now_text if terminal else _utc_timestamp(base_dt + timedelta(seconds=delay))
    cursor = conn.execute(
        """
        UPDATE arxiv_jobs
        SET status = ?,
            attempts = ?,
            run_after = ?,
            locked_by = NULL,
            locked_at = NULL,
            error_message = ?,
            updated_at = ?
        WHERE job_key = ? AND status IN ('running', 'queued')
        RETURNING *
        """,
        (
            "failed" if terminal else "queued",
            next_attempts,
            run_after,
            error_message_value,
            now_text,
            job_key_value,
        ),
    )
    job = _row_to_dict(cursor.fetchone(), cursor.description)
    if job is None:
        return None
    append_arxiv_event(
        conn,
        "job_failed",
        arxiv_id=job["arxiv_id"],
        version=job["version"],
        asset_type=job["asset_type"],
        job_key=job["job_key"],
        payload={
            "job_type": job["job_type"],
            "status": job["status"],
            "attempts": job["attempts"],
            "max_attempts": job["max_attempts"],
            "run_after": job["run_after"],
            "error_message": _paperstream_path_safe_error_message(job["error_message"]),
            "failed_at": now_text,
        },
        now=now_text,
    )
    return job


def _paperstream_utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        text = _utc_timestamp(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError("now must be a datetime or parseable timestamp string") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("now must be a datetime, timestamp string, or None")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_stale_after_seconds(stale_after_seconds: Any) -> int | float:
    if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, (int, float)):
        raise ValueError("stale_after_seconds must be a positive number of seconds")
    if not math.isfinite(float(stale_after_seconds)) or stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be a positive number of seconds")
    return stale_after_seconds


def _validate_paperstream_requeue_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer")
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit > _PAPERSTREAM_REQUEUE_MAX_LIMIT:
        raise ValueError(f"limit must be <= {_PAPERSTREAM_REQUEUE_MAX_LIMIT}")
    return limit


def _paperstream_path_safe_error_message(error_message: Any) -> str | None:
    if error_message is None:
        return None
    text = str(error_message)
    if not text:
        return None
    if "/" in text or "\\" in text:
        return "[redacted]"
    return text


def retry_failed_arxiv_jobs(
    conn: sqlite3.Connection,
    *,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Explicitly reset bounded terminal failed jobs so they can be claimed again."""

    limit = _validate_paperstream_requeue_limit(limit)
    job_type_values = _validate_job_type_filter(job_types)
    now_dt = _paperstream_utc_datetime(now)
    now_text = _utc_timestamp(now_dt)

    job_type_clause = ""
    params: list[Any] = []
    if job_type_values is not None:
        placeholders = ",".join("?" for _ in job_type_values)
        job_type_clause = f"AND job_type IN ({placeholders})"
        params.extend(job_type_values)
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT job_key, job_type, arxiv_id, version, asset_type,
               status, attempts, max_attempts, error_message
        FROM arxiv_jobs
        WHERE status = 'failed'
          {job_type_clause}
        ORDER BY priority ASC, run_after ASC, updated_at ASC, created_at ASC, job_key ASC
        LIMIT ?
        """,
        params,
    ).fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        updated = conn.execute(
            """
            UPDATE arxiv_jobs
            SET status = 'queued',
                attempts = 0,
                run_after = ?,
                locked_by = NULL,
                locked_at = NULL,
                error_message = NULL,
                updated_at = ?
            WHERE job_key = ?
              AND status = 'failed'
            RETURNING job_key
            """,
            (now_text, now_text, row["job_key"]),
        ).fetchone()
        if updated is None:
            continue
        previous_error_message = _paperstream_path_safe_error_message(row["error_message"])
        append_arxiv_event(
            conn,
            "job_retry_queued",
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            asset_type=row["asset_type"],
            job_key=row["job_key"],
            payload={
                "reason": "manual_retry",
                "previous_status": row["status"],
                "previous_attempts": row["attempts"],
                "previous_error_message": previous_error_message,
                "queued_at": now_text,
            },
            now=now_dt,
        )
        jobs.append(
            {
                "job_key": row["job_key"],
                "job_type": row["job_type"],
                "arxiv_id": row["arxiv_id"],
                "version": row["version"],
                "asset_type": row["asset_type"],
                "previous_attempts": row["attempts"],
                "max_attempts": row["max_attempts"],
                "queued_at": now_text,
            }
        )

    return {
        "ok": True,
        "limit": limit,
        "retried": len(jobs),
        "queued_at": now_text,
        "jobs": jobs,
    }


def requeue_stale_arxiv_jobs(
    conn: sqlite3.Connection,
    *,
    stale_after_seconds: int | float,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Explicitly requeue bounded running jobs whose locks are stale."""

    stale_after_seconds = _validate_stale_after_seconds(stale_after_seconds)
    limit = _validate_paperstream_requeue_limit(limit)
    job_type_values = _validate_job_type_filter(job_types)
    now_dt = _paperstream_utc_datetime(now)
    now_text = _utc_timestamp(now_dt)
    stale_before = _utc_timestamp(now_dt - timedelta(seconds=float(stale_after_seconds)))

    job_type_clause = ""
    params: list[Any] = [stale_before]
    if job_type_values is not None:
        placeholders = ",".join("?" for _ in job_type_values)
        job_type_clause = f"AND job_type IN ({placeholders})"
        params.extend(job_type_values)
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT job_key, job_type, arxiv_id, version, asset_type,
               attempts, max_attempts, locked_by, locked_at
        FROM arxiv_jobs
        WHERE status = 'running'
          AND locked_at IS NOT NULL
          AND datetime(locked_at) <= datetime(?)
          AND attempts < max_attempts
          {job_type_clause}
        ORDER BY datetime(locked_at) ASC, locked_at ASC, priority ASC, created_at ASC, job_key ASC
        LIMIT ?
        """,
        params,
    ).fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        updated = conn.execute(
            """
            UPDATE arxiv_jobs
            SET status = 'queued',
                locked_by = NULL,
                locked_at = NULL,
                run_after = ?,
                updated_at = ?
            WHERE job_key = ?
              AND status = 'running'
              AND locked_at = ?
              AND attempts < max_attempts
            RETURNING job_key
            """,
            (now_text, now_text, row["job_key"], row["locked_at"]),
        ).fetchone()
        if updated is None:
            continue
        append_arxiv_event(
            conn,
            "job_requeued",
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            asset_type=row["asset_type"],
            job_key=row["job_key"],
            payload={
                "reason": "stale_lock",
                "stale_before": stale_before,
                "previous_locked_by": row["locked_by"],
                "previous_locked_at": row["locked_at"],
            },
            now=now_dt,
        )
        jobs.append(
            {
                "job_key": row["job_key"],
                "job_type": row["job_type"],
                "arxiv_id": row["arxiv_id"],
                "version": row["version"],
                "asset_type": row["asset_type"],
                "attempts": row["attempts"],
                "max_attempts": row["max_attempts"],
                "previous_locked_by": row["locked_by"],
                "previous_locked_at": row["locked_at"],
            }
        )

    return {
        "ok": True,
        "stale_after_seconds": stale_after_seconds,
        "stale_before": stale_before,
        "limit": limit,
        "requeued": len(jobs),
        "jobs": jobs,
    }

def _arxiv_download_url(arxiv_id: str, asset_type: str) -> str:
    arxiv_id = _validate_arxiv_id(arxiv_id)
    asset_type = _validate_asset_type(asset_type)
    if asset_type == "pdf":
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if asset_type == "source":
        return f"https://arxiv.org/src/{arxiv_id}"
    raise ValueError(f"unsupported download asset_type: {asset_type!r}")


def _fetch_url_bytes(url: str) -> bytes:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if not isinstance(data, bytes):
        data = bytes(data)
    return data


def _ensure_download_bytes(data: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("fetch_fn must return bytes")
    return data if isinstance(data, bytes) else bytes(data)


def _write_bytes_atomic(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _download_failure_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


_PARSE_INPUT_ASSET_BY_JOB_TYPE = {
    "parse_pdf": "pdf",
    "parse_source": "source",
}
_PARSE_INPUT_READY_STATUSES = frozenset({"downloaded", "imported", "parsed", "indexed"})
_PARSED_TEXT_READY_STATUSES = frozenset({"parsed", "indexed"})


def _parse_failure_text(exc: BaseException) -> str:
    return _download_failure_text(exc)


def _parse_job_failure_text(
    exc: BaseException,
    *,
    job_type: str | None = None,
    input_asset_type: str | None = None,
) -> str:
    """Return a stable, path-safe parse job failure message."""

    asset_type = input_asset_type
    if asset_type is None:
        if job_type == "parse_pdf":
            asset_type = "pdf"
        elif job_type == "parse_source":
            asset_type = "source"
        else:
            asset_type = "input"
    if isinstance(exc, FileNotFoundError):
        return f"missing {asset_type} asset file"
    if isinstance(exc, PermissionError):
        return f"unreadable {asset_type} asset file"
    if isinstance(exc, ValueError) and str(exc).startswith("missing usable "):
        return f"missing {asset_type} asset"
    if job_type == "parse_pdf":
        return "pdf parser failed"
    if job_type == "parse_source":
        return "source parser failed"
    return "parse job failed"


def _ensure_parse_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("parser must return text")
    text = text.strip()
    if not text:
        raise ValueError("parser returned empty text")
    return text


def _decode_source_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


_SOURCE_MEMBER_TEXT_SUFFIXES = frozenset({".tex", ".bbl", ".bib", ".sty", ".cls"})
_SOURCE_MEMBER_GENERATED_SUFFIXES = frozenset({".aux", ".log", ".out", ".toc"})


def _normalize_source_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _source_archive_member_allowed(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return False
    basename = parts[-1]
    if basename in {"", ".", ".."}:
        return False
    if any(part == ".." or part.startswith(".") or part in {"__MACOSX"} for part in parts):
        return False
    if basename.startswith(("~", "#")) or basename.endswith(("~", "#")):
        return False
    suffix = PurePosixPath(basename).suffix.lower()
    if suffix in _SOURCE_MEMBER_GENERATED_SUFFIXES:
        return False
    return suffix in _SOURCE_MEMBER_TEXT_SUFFIXES


def _extract_source_text_from_archive(data: bytes) -> str:
    chunks: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not _source_archive_member_allowed(member.name):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text = _normalize_source_text(_decode_source_text(extracted.read()))
            if text:
                chunks.append(text)
    if not chunks:
        raise ValueError("source archive does not contain supported source files")
    return "\n\n".join(chunks)


def _extract_source_text_from_bytes(data: bytes | bytearray | memoryview) -> str:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("source parser input must be bytes")
    data = data if isinstance(data, bytes) else bytes(data)
    if not data:
        raise ValueError("source asset is empty")
    try:
        return _ensure_parse_text(_extract_source_text_from_archive(data))
    except tarfile.ReadError:
        pass
    try:
        decompressed = gzip.decompress(data)
    except (EOFError, OSError):
        decompressed = None
    if decompressed is not None:
        return _ensure_parse_text(_normalize_source_text(_decode_source_text(decompressed)))
    return _ensure_parse_text(_normalize_source_text(_decode_source_text(data)))


def _parsed_text_json_bytes(
    *,
    arxiv_id: str,
    version: int,
    source_asset_type: str,
    text: str,
) -> bytes:
    payload = {
        "arxiv_id": arxiv_id,
        "asset_type": "parsed_text",
        "source_asset_type": source_asset_type,
        "text": text,
        "version": version,
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_parsed_text_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid parsed_text JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("parsed_text JSON must be an object")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("parsed_text JSON text must be a string")
    if not text.strip():
        raise ValueError("parsed_text JSON text is empty")
    return payload


def _chunks_json_bytes(
    *,
    arxiv_id: str,
    version: int,
    source_asset_type: str | None,
    parsed_asset_key: str,
    chunks: list[str],
    chunk_size: int,
    overlap: int,
) -> bytes:
    payload = {
        "arxiv_id": arxiv_id,
        "asset_type": "chunks",
        "chunk_size": chunk_size,
        "chunks": chunks,
        "overlap": overlap,
        "source_asset_key": parsed_asset_key,
        "source_asset_type": source_asset_type,
        "version": version,
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_chunks_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid chunks JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("chunks JSON must be an object")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("chunks JSON chunks must be a list")
    normalized_chunks: list[str] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, str):
            raise ValueError(f"chunks JSON chunk {index} must be a string")
        text = chunk.strip()
        if not text:
            raise ValueError(f"chunks JSON chunk {index} is empty")
        normalized_chunks.append(text)
    if not normalized_chunks:
        raise ValueError("chunks JSON chunks is empty")
    payload = dict(payload)
    payload["chunks"] = normalized_chunks
    return payload


def _cleanup_sqlite_temp_files(path: Path) -> None:
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-wal"),
        Path(f"{path}-shm"),
    ):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _write_bm25_index_atomic(
    path: str | Path,
    *,
    arxiv_id: str,
    version: int,
    chunks_payload: dict[str, Any],
    chunks_asset: dict[str, Any],
) -> tuple[str, int]:
    chunks = chunks_payload["chunks"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_text = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    temp_path = Path(temp_path_text)
    try:
        index_conn = sqlite3.connect(str(temp_path))
        try:
            index_conn.execute("PRAGMA journal_mode=DELETE")
            index_conn.execute("PRAGMA synchronous=FULL")
            index_conn.execute(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            index_conn.execute(
                """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    chunk_index INTEGER NOT NULL UNIQUE,
                    text TEXT NOT NULL
                )
                """
            )
            index_conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='id',
                    tokenize='unicode61'
                )
                """
            )
            metadata = {
                "arxiv_id": arxiv_id,
                "asset_type": "bm25_index",
                "chunk_count": str(len(chunks)),
                "chunks_asset_key": str(chunks_asset["asset_key"]),
                "chunks_checksum": str(chunks_asset.get("checksum") or ""),
                "chunks_storage_path": str(chunks_asset.get("storage_path") or ""),
                "index_schema": "researchfs.paperstream.bm25.v1",
                "source": "index_bm25",
                "source_asset_type": str(chunks_payload.get("source_asset_type") or ""),
                "version": str(version),
            }
            index_conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            index_conn.executemany(
                "INSERT INTO chunks(id, chunk_index, text) VALUES (?, ?, ?)",
                ((index + 1, index, text) for index, text in enumerate(chunks)),
            )
            index_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            index_conn.commit()
            index_conn.execute("PRAGMA optimize")
            index_conn.commit()
        finally:
            index_conn.close()
        data = temp_path.read_bytes()
        checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
        byte_size = len(data)
        os.replace(temp_path, target)
        return checksum, byte_size
    except Exception:
        _cleanup_sqlite_temp_files(temp_path)
        raise



def _validate_paperstream_rebuild_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer")
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit > _PAPERSTREAM_REBUILD_MAX_LIMIT:
        raise ValueError(f"limit must be <= {_PAPERSTREAM_REBUILD_MAX_LIMIT}")
    return limit


def _validate_paperstream_rebuild_filters(
    arxiv_id: str | None,
    version: int | str | None,
) -> tuple[str | None, int | None]:
    if arxiv_id is not None:
        arxiv_id = _validate_arxiv_id(arxiv_id)
    version = _validate_version(version, required=False)
    return arxiv_id, version


def _paperstream_chunks_asset_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    arxiv_id: str | None,
    version: int | None,
) -> list[dict[str, Any]]:
    clauses = [
        "asset_type = 'chunks'",
        "status IN ('parsed', 'indexed')",
        "storage_path IS NOT NULL",
        "storage_path <> ''",
    ]
    params: list[Any] = []
    if arxiv_id is not None:
        clauses.append("arxiv_id = ?")
        params.append(arxiv_id)
    if version is not None:
        clauses.append("version = ?")
        params.append(version)
    params.append(limit)
    cursor = conn.execute(
        f"""
        SELECT asset_key, arxiv_id, version, asset_type, status, storage_path,
               checksum, byte_size, source
        FROM arxiv_assets
        WHERE {" AND ".join(clauses)}
        ORDER BY arxiv_id ASC, version ASC, asset_key ASC
        LIMIT ?
        """,
        tuple(params),
    )
    return [row_dict for row in cursor.fetchall() if (row_dict := _row_to_dict(row, cursor.description)) is not None]


def _paperstream_chunks_error(asset: dict[str, Any], exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        error = "missing_chunks_asset"
        message = "chunks asset file is missing"
    else:
        error = "unreadable_chunks_asset"
        message = "chunks asset file is unreadable"
    return {
        "asset_key": asset.get("asset_key"),
        "arxiv_id": asset.get("arxiv_id"),
        "version": asset.get("version"),
        "error": error,
        "message": message,
    }


def _materialized_chunks_current(
    conn: sqlite3.Connection,
    *,
    asset: dict[str, Any],
    chunk_count: int,
) -> bool:
    if not asset.get("checksum"):
        return False
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM arxiv_chunks
        WHERE arxiv_id = ?
          AND version = ?
          AND chunks_asset_key = ?
          AND COALESCE(chunks_checksum, '') = COALESCE(?, '')
        """,
        (
            asset["arxiv_id"],
            asset["version"],
            asset["asset_key"],
            asset.get("checksum"),
        ),
    ).fetchone()
    count = int((row["count"] if isinstance(row, sqlite3.Row) else row[0]) or 0)
    return count == chunk_count


def _replace_materialized_chunks(
    conn: sqlite3.Connection,
    *,
    asset: dict[str, Any],
    chunks_payload: dict[str, Any],
) -> int:
    arxiv_id = _validate_arxiv_id(str(asset["arxiv_id"]))
    version = _validate_version(asset["version"])
    chunks = chunks_payload["chunks"]
    source_asset_key = chunks_payload.get("source_asset_key")
    if source_asset_key is not None:
        source_asset_key = str(source_asset_key)
    source_asset_type = chunks_payload.get("source_asset_type")
    if source_asset_type is not None:
        source_asset_type = _validate_asset_type(str(source_asset_type), required=False)
    savepoint = "paperstream_chunks_materialize"
    _metadata_savepoint(conn, savepoint)
    try:
        conn.execute(
            "DELETE FROM arxiv_chunks WHERE arxiv_id = ? AND version = ?",
            (arxiv_id, version),
        )
        conn.executemany(
            """
            INSERT INTO arxiv_chunks (
                arxiv_id, version, chunk_index, text, chunks_asset_key,
                source_asset_key, source_asset_type, chunks_checksum
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    arxiv_id,
                    version,
                    index,
                    text,
                    asset["asset_key"],
                    source_asset_key,
                    source_asset_type,
                    asset.get("checksum"),
                )
                for index, text in enumerate(chunks)
            ),
        )
        _metadata_release(conn, savepoint)
    except Exception:
        _metadata_rollback(conn, savepoint)
        raise
    return len(chunks)


def _materialize_chunks_asset(
    conn: sqlite3.Connection,
    asset: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    chunks_payload = _read_chunks_payload(Path(str(asset["storage_path"])))
    chunk_count = len(chunks_payload["chunks"])
    if not force and _materialized_chunks_current(conn, asset=asset, chunk_count=chunk_count):
        return {
            "asset_key": asset["asset_key"],
            "arxiv_id": asset["arxiv_id"],
            "version": asset["version"],
            "status": "skipped",
            "checksum": asset.get("checksum"),
            "chunk_count": chunk_count,
        }
    materialized_count = _replace_materialized_chunks(
        conn,
        asset=asset,
        chunks_payload=chunks_payload,
    )
    return {
        "asset_key": asset["asset_key"],
        "arxiv_id": asset["arxiv_id"],
        "version": asset["version"],
        "status": "materialized",
        "checksum": asset.get("checksum"),
        "chunk_count": materialized_count,
    }


def materialize_paperstream_chunks(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    arxiv_id: str | None = None,
    version: int | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize ready chunks JSON assets into SQLite rows and FTS."""

    limit = _validate_paperstream_rebuild_limit(limit)
    arxiv_id, version = _validate_paperstream_rebuild_filters(arxiv_id, version)
    if not isinstance(force, bool):
        raise TypeError("force must be a boolean")
    assets = _paperstream_chunks_asset_rows(
        conn,
        limit=limit,
        arxiv_id=arxiv_id,
        version=version,
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for asset in assets:
        try:
            results.append(_materialize_chunks_asset(conn, asset, force=force))
        except (OSError, ValueError, sqlite3.Error) as exc:
            errors.append(_paperstream_chunks_error(asset, exc))
    return {
        "ok": not errors,
        "limit": limit,
        "processed": len(assets),
        "materialized": sum(1 for item in results if item["status"] == "materialized"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "errors": errors,
        "results": results,
    }


def _paperstream_bm25_asset_current(
    asset: dict[str, Any] | None,
    chunks_asset: dict[str, Any],
) -> bool:
    if asset is None or asset.get("status") != "indexed" or not asset.get("storage_path"):
        return False
    path = Path(str(asset["storage_path"]))
    if not path.is_file():
        return False
    checksum = asset.get("checksum")
    if isinstance(checksum, str) and checksum.startswith("sha256:"):
        file_checksum = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        if checksum != file_checksum:
            return False
    try:
        index_conn = _paperstream_open_bm25_index(path)
        try:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in index_conn.execute("SELECT key, value FROM metadata")
            }
        finally:
            index_conn.close()
    except (OSError, sqlite3.Error):
        return False
    return (
        metadata.get("arxiv_id") == str(chunks_asset["arxiv_id"])
        and metadata.get("version") == str(chunks_asset["version"])
        and metadata.get("chunks_asset_key") == str(chunks_asset["asset_key"])
        and metadata.get("chunks_checksum") == str(chunks_asset.get("checksum") or "")
    )


def _paperstream_public_bm25_asset_summary(
    *,
    asset_key: str,
    arxiv_id: str,
    version: int,
    status: str,
    checksum: Any,
    byte_size: Any,
    chunk_count: int,
    chunks_asset_key: str,
) -> dict[str, Any]:
    return {
        "asset_key": asset_key,
        "arxiv_id": arxiv_id,
        "version": version,
        "status": status,
        "checksum": checksum,
        "byte_size": byte_size,
        "chunk_count": chunk_count,
        "chunks_asset_key": chunks_asset_key,
    }


def rebuild_paperstream_bm25_indexes(
    conn: sqlite3.Connection,
    object_root: str | Path,
    *,
    limit: int = 100,
    arxiv_id: str | None = None,
    version: int | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize chunks, then rebuild missing or stale local BM25 indexes."""

    limit = _validate_paperstream_rebuild_limit(limit)
    arxiv_id, version = _validate_paperstream_rebuild_filters(arxiv_id, version)
    if not isinstance(force, bool):
        raise TypeError("force must be a boolean")
    materialization = materialize_paperstream_chunks(
        conn,
        limit=limit,
        arxiv_id=arxiv_id,
        version=version,
        force=force,
    )
    materialization_errors = {item.get("asset_key") for item in materialization["errors"]}
    assets = _paperstream_chunks_asset_rows(
        conn,
        limit=limit,
        arxiv_id=arxiv_id,
        version=version,
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(materialization["errors"])
    for chunks_asset in assets:
        if chunks_asset["asset_key"] in materialization_errors:
            continue
        try:
            paper_arxiv_id = _validate_arxiv_id(str(chunks_asset["arxiv_id"]))
            paper_version = _validate_version(chunks_asset["version"])
            bm25_asset_key_value = arxiv_asset_key(paper_arxiv_id, paper_version, "bm25_index")
            existing_cursor = conn.execute(
                "SELECT * FROM arxiv_assets WHERE asset_key = ?",
                (bm25_asset_key_value,),
            )
            existing_bm25_asset = _row_to_dict(existing_cursor.fetchone(), existing_cursor.description)
            chunks_payload = _read_chunks_payload(Path(str(chunks_asset["storage_path"])))
            chunk_count = len(chunks_payload["chunks"])
            if not force and _paperstream_bm25_asset_current(existing_bm25_asset, chunks_asset):
                results.append(
                    _paperstream_public_bm25_asset_summary(
                        asset_key=bm25_asset_key_value,
                        arxiv_id=paper_arxiv_id,
                        version=paper_version,
                        status="skipped",
                        checksum=existing_bm25_asset.get("checksum") if existing_bm25_asset else None,
                        byte_size=existing_bm25_asset.get("byte_size") if existing_bm25_asset else None,
                        chunk_count=chunk_count,
                        chunks_asset_key=str(chunks_asset["asset_key"]),
                    )
                )
                continue
            storage_path = arxiv_storage_path(object_root, paper_arxiv_id, paper_version, "bm25_index")
            checksum, byte_size = _write_bm25_index_atomic(
                storage_path,
                arxiv_id=paper_arxiv_id,
                version=paper_version,
                chunks_payload=chunks_payload,
                chunks_asset=chunks_asset,
            )
            savepoint = "paperstream_bm25_rebuild"
            _metadata_savepoint(conn, savepoint)
            try:
                bm25_asset_key_value = upsert_arxiv_asset(
                    conn,
                    paper_arxiv_id,
                    paper_version,
                    "bm25_index",
                    status="indexed",
                    storage_path=storage_path,
                    checksum=checksum,
                    byte_size=byte_size,
                    source="rebuild_indexes",
                )
                append_arxiv_event(
                    conn,
                    "asset_bm25_rebuilt",
                    arxiv_id=paper_arxiv_id,
                    version=paper_version,
                    asset_type="bm25_index",
                    payload={
                        "asset_key": bm25_asset_key_value,
                        "source_asset_key": chunks_asset["asset_key"],
                        "checksum": checksum,
                        "byte_size": byte_size,
                        "chunk_count": chunk_count,
                    },
                )
                _metadata_release(conn, savepoint)
            except Exception:
                _metadata_rollback(conn, savepoint)
                raise
            results.append(
                _paperstream_public_bm25_asset_summary(
                    asset_key=bm25_asset_key_value,
                    arxiv_id=paper_arxiv_id,
                    version=paper_version,
                    status="indexed",
                    checksum=checksum,
                    byte_size=byte_size,
                    chunk_count=chunk_count,
                    chunks_asset_key=str(chunks_asset["asset_key"]),
                )
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            errors.append(_paperstream_chunks_error(chunks_asset, exc))
    return {
        "ok": not errors,
        "limit": limit,
        "processed": len(assets),
        "materialized": int(materialization["materialized"]),
        "indexed": sum(1 for item in results if item["status"] == "indexed"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "errors": errors,
        "materialization": materialization,
        "results": results,
    }

def work_next_arxiv_parse_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    pdf_parser_fn: Any = None,
    source_parser_fn: Any = None,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str = ("parse_pdf", "parse_source"),
) -> dict[str, Any]:
    """Claim and run one queued arXiv parse job over an existing local asset."""

    job_type_values = _validate_parse_job_types(job_types)
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    job_type: str | None = None
    arxiv_id: str | None = None
    version: int | None = None
    input_asset_type: str | None = None
    input_asset_key: str | None = None
    parsed_asset_key_value: str | None = None
    storage_path: str | None = None
    chunk_job_key: str | None = None
    try:
        job_type = _validate_job_type(str(job["job_type"]))
        input_asset_type = _PARSE_INPUT_ASSET_BY_JOB_TYPE[job_type]
        arxiv_id = _validate_arxiv_id(str(job["arxiv_id"]))
        version = _validate_version(job["version"])
        job_asset_type = job.get("asset_type")
        if job_asset_type is not None:
            job_asset_type = _validate_asset_type(job_asset_type, required=False)
            if job_asset_type not in {input_asset_type, "parsed_text"}:
                raise ValueError(f"{job_type} job cannot parse {job_asset_type!r} asset")

        parsed_asset_key_value = arxiv_asset_key(arxiv_id, version, "parsed_text")
        existing_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (parsed_asset_key_value,),
        )
        existing_parsed_asset = _row_to_dict(existing_cursor.fetchone(), existing_cursor.description)
        if existing_parsed_asset is not None and existing_parsed_asset["status"] in _PARSED_TEXT_READY_STATUSES:
            append_arxiv_event(
                conn,
                "asset_parse_skipped",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="parsed_text",
                job_key=job_key_value,
                payload={
                    "asset_key": parsed_asset_key_value,
                    "reason": "parsed_text_already_available",
                    "status": existing_parsed_asset["status"],
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            return {
                "worked": True,
                "ok": True,
                "skipped": True,
                "reason": "parsed_text_already_available",
                "job": succeeded_job or job,
                "asset": existing_parsed_asset,
            }

        input_asset_key = arxiv_asset_key(arxiv_id, version, input_asset_type)
        input_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (input_asset_key,),
        )
        input_asset = _row_to_dict(input_cursor.fetchone(), input_cursor.description)
        if (
            input_asset is None
            or input_asset["status"] not in _PARSE_INPUT_READY_STATUSES
            or not input_asset.get("storage_path")
        ):
            raise ValueError(f"missing usable {input_asset_type} asset")
        input_path = Path(str(input_asset["storage_path"]))
        if not input_path.is_file():
            raise FileNotFoundError(f"missing {input_asset_type} asset file: {input_path}")

        if job_type == "parse_pdf":
            if pdf_parser_fn is None:
                from researchfs.pdf import extract_pdf_text

                pdf_parser = extract_pdf_text
            else:
                pdf_parser = pdf_parser_fn
            if not callable(pdf_parser):
                raise TypeError("pdf_parser_fn must be callable")
            text = _ensure_parse_text(pdf_parser(input_path))
        else:
            source_parser = _extract_source_text_from_bytes if source_parser_fn is None else source_parser_fn
            if not callable(source_parser):
                raise TypeError("source_parser_fn must be callable")
            text = _ensure_parse_text(source_parser(input_path.read_bytes()))

        storage_path = arxiv_storage_path(object_root, arxiv_id, version, "parsed_text")
        data = _parsed_text_json_bytes(
            arxiv_id=arxiv_id,
            version=version,
            source_asset_type=input_asset_type,
            text=text,
        )
        checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
        byte_size = len(data)
        success_savepoint = "paperstream_parse_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            _write_bytes_atomic(storage_path, data)
            parsed_asset_key_value = upsert_arxiv_asset(
                conn,
                arxiv_id,
                version,
                "parsed_text",
                status="parsed",
                storage_path=storage_path,
                checksum=checksum,
                byte_size=byte_size,
                source=job_type,
            )
            chunk_job_key = enqueue_arxiv_job(
                conn,
                "chunk_text",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="parsed_text",
                payload={
                    "source": "paperstream_parse",
                    "asset_key": parsed_asset_key_value,
                    "source_asset_key": input_asset_key,
                    "source_asset_type": input_asset_type,
                },
            )
            append_arxiv_event(
                conn,
                "asset_parsed",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="parsed_text",
                job_key=job_key_value,
                payload={
                    "asset_key": parsed_asset_key_value,
                    "chunk_job_key": chunk_job_key,
                    "source_asset_key": input_asset_key,
                    "source_asset_type": input_asset_type,
                    "storage_path": storage_path,
                    "checksum": checksum,
                    "byte_size": byte_size,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "asset": {
                "asset_key": parsed_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "parsed_text",
                "status": "parsed",
                "storage_path": storage_path,
                "checksum": checksum,
                "byte_size": byte_size,
                "source": job_type,
            },
            "chunk_job_key": chunk_job_key,
            "source_asset": input_asset,
        }
    except Exception as exc:
        error_message = _parse_job_failure_text(exc, job_type=job_type, input_asset_type=input_asset_type)
        failed_asset: dict[str, Any] | None = None
        failed_job: dict[str, Any] | None = None
        if arxiv_id is not None and version is not None:
            failure_savepoint = "paperstream_parse_failure"
            _metadata_savepoint(conn, failure_savepoint)
            try:
                parsed_asset_key_value = upsert_arxiv_asset(
                    conn,
                    arxiv_id,
                    version,
                    "parsed_text",
                    status="failed",
                    source=job_type,
                    error_message=error_message,
                )
                append_arxiv_event(
                    conn,
                    "asset_parse_failed",
                    arxiv_id=arxiv_id,
                    version=version,
                    asset_type="parsed_text",
                    job_key=job_key_value,
                    payload={
                        "asset_key": parsed_asset_key_value,
                        "error": error_message,
                        "source_asset_key": input_asset_key,
                        "source_asset_type": input_asset_type,
                    },
                    now=now,
                )
                failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
                _metadata_release(conn, failure_savepoint)
            except Exception:
                _metadata_rollback(conn, failure_savepoint)
                raise
            failed_asset = {
                "asset_key": parsed_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "parsed_text",
                "status": "failed",
                "source": job_type,
                "error_message": error_message,
            }
        else:
            failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "asset": failed_asset,
            "error": error_message,
        }


def work_next_arxiv_chunk_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    chunk_size: int = 2000,
    overlap: int = 200,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str = ("chunk_text",),
) -> dict[str, Any]:
    """Claim and run one queued arXiv chunk job over an existing parsed_text asset."""

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    job_type_values = _validate_job_type_subset(
        job_types,
        allowed={"chunk_text"},
        label="job_types",
        description="'chunk_text'",
    )
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    job_type = "chunk_text"
    arxiv_id: str | None = None
    version: int | None = None
    parsed_asset_key_value: str | None = None
    chunks_asset_key_value: str | None = None
    embed_job_key: str | None = None
    bm25_job_key: str | None = None
    materialized_chunks: int | None = None
    storage_path: str | None = None
    try:
        job_type = _validate_job_type(str(job["job_type"]))
        if job_type != "chunk_text":
            raise ValueError(f"unsupported chunk job_type: {job_type!r}")
        arxiv_id = _validate_arxiv_id(str(job["arxiv_id"]))
        version = _validate_version(job["version"])
        job_asset_type = job.get("asset_type")
        if job_asset_type is not None:
            job_asset_type = _validate_asset_type(job_asset_type, required=False)
            if job_asset_type not in {"parsed_text", "chunks"}:
                raise ValueError(f"chunk_text job cannot chunk {job_asset_type!r} asset")

        chunks_asset_key_value = arxiv_asset_key(arxiv_id, version, "chunks")
        existing_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (chunks_asset_key_value,),
        )
        existing_chunks_asset = _row_to_dict(existing_cursor.fetchone(), existing_cursor.description)
        if existing_chunks_asset is not None and existing_chunks_asset["status"] in _PARSED_TEXT_READY_STATUSES:
            append_arxiv_event(
                conn,
                "asset_chunk_skipped",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="chunks",
                job_key=job_key_value,
                payload={
                    "asset_key": chunks_asset_key_value,
                    "reason": "chunks_already_available",
                    "status": existing_chunks_asset["status"],
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            return {
                "worked": True,
                "ok": True,
                "skipped": True,
                "reason": "chunks_already_available",
                "job": succeeded_job or job,
                "asset": existing_chunks_asset,
            }

        parsed_asset_key_value = arxiv_asset_key(arxiv_id, version, "parsed_text")
        parsed_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (parsed_asset_key_value,),
        )
        parsed_asset = _row_to_dict(parsed_cursor.fetchone(), parsed_cursor.description)
        if (
            parsed_asset is None
            or parsed_asset["status"] not in _PARSED_TEXT_READY_STATUSES
            or not parsed_asset.get("storage_path")
        ):
            raise ValueError("missing usable parsed_text asset")
        parsed_path = Path(str(parsed_asset["storage_path"]))
        if not parsed_path.is_file():
            raise FileNotFoundError(f"missing parsed_text asset file: {parsed_path}")

        parsed_payload = _read_parsed_text_payload(parsed_path)
        parsed_text = str(parsed_payload["text"]).strip()
        source_asset_type = parsed_payload.get("source_asset_type")
        if source_asset_type is not None:
            source_asset_type = _validate_asset_type(str(source_asset_type), required=False)

        from researchfs.extractor import chunk_text

        chunks = chunk_text(parsed_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("chunker returned no chunks")
        storage_path = arxiv_storage_path(object_root, arxiv_id, version, "chunks")
        data = _chunks_json_bytes(
            arxiv_id=arxiv_id,
            version=version,
            source_asset_type=source_asset_type,
            parsed_asset_key=parsed_asset_key_value,
            chunks=chunks,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
        byte_size = len(data)
        success_savepoint = "paperstream_chunk_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            _write_bytes_atomic(storage_path, data)
            chunks_asset_key_value = upsert_arxiv_asset(
                conn,
                arxiv_id,
                version,
                "chunks",
                status="parsed",
                storage_path=storage_path,
                checksum=checksum,
                byte_size=byte_size,
                source=job_type,
            )
            materialized_chunks = _replace_materialized_chunks(
                conn,
                asset={
                    "asset_key": chunks_asset_key_value,
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "checksum": checksum,
                },
                chunks_payload=json.loads(data.decode("utf-8")),
            )
            embed_job_key = enqueue_arxiv_job(
                conn,
                "embed_chunks",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="chunks",
                payload={
                    "source": "paperstream_chunk",
                    "asset_key": chunks_asset_key_value,
                    "source_asset_key": parsed_asset_key_value,
                    "chunk_count": len(chunks),
                },
            )
            bm25_job_key = enqueue_arxiv_job(
                conn,
                "index_bm25",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="chunks",
                payload={
                    "source": "paperstream_chunk",
                    "asset_key": chunks_asset_key_value,
                    "source_asset_key": parsed_asset_key_value,
                    "chunk_count": len(chunks),
                },
            )
            append_arxiv_event(
                conn,
                "asset_chunked",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="chunks",
                job_key=job_key_value,
                payload={
                    "asset_key": chunks_asset_key_value,
                    "embed_job_key": embed_job_key,
                    "bm25_job_key": bm25_job_key,
                    "source_asset_key": parsed_asset_key_value,
                    "storage_path": storage_path,
                    "checksum": checksum,
                    "byte_size": byte_size,
                    "chunk_count": len(chunks),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "asset": {
                "asset_key": chunks_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "chunks",
                "status": "parsed",
                "storage_path": storage_path,
                "checksum": checksum,
                "byte_size": byte_size,
                "source": job_type,
            },
            "embed_job_key": embed_job_key,
            "bm25_job_key": bm25_job_key,
            "source_asset": parsed_asset,
            "chunk_count": len(chunks),
            "materialized_chunks": materialized_chunks,
        }
    except Exception as exc:
        error_message = _parse_failure_text(exc)
        failed_asset: dict[str, Any] | None = None
        failed_job: dict[str, Any] | None = None
        if arxiv_id is not None and version is not None:
            failure_savepoint = "paperstream_chunk_failure"
            _metadata_savepoint(conn, failure_savepoint)
            try:
                chunks_asset_key_value = upsert_arxiv_asset(
                    conn,
                    arxiv_id,
                    version,
                    "chunks",
                    status="failed",
                    source=job_type,
                    error_message=error_message,
                )
                append_arxiv_event(
                    conn,
                    "asset_chunk_failed",
                    arxiv_id=arxiv_id,
                    version=version,
                    asset_type="chunks",
                    job_key=job_key_value,
                    payload={
                        "asset_key": chunks_asset_key_value,
                        "error": error_message,
                        "source_asset_key": parsed_asset_key_value,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                    },
                    now=now,
                )
                failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
                _metadata_release(conn, failure_savepoint)
            except Exception:
                _metadata_rollback(conn, failure_savepoint)
                raise
            failed_asset = {
                "asset_key": chunks_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "chunks",
                "status": "failed",
                "source": job_type,
                "error_message": error_message,
            }
        else:
            failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "asset": failed_asset,
            "error": error_message,
        }


def work_next_arxiv_bm25_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str = ("index_bm25",),
) -> dict[str, Any]:
    """Claim and run one queued local BM25 index job over a chunks asset."""

    job_type_values = _validate_job_type_subset(
        job_types,
        allowed={"index_bm25"},
        label="job_types",
        description="'index_bm25'",
    )
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    job_type = "index_bm25"
    arxiv_id: str | None = None
    version: int | None = None
    chunks_asset_key_value: str | None = None
    bm25_asset_key_value: str | None = None
    storage_path: str | None = None
    try:
        job_type = _validate_job_type(str(job["job_type"]))
        if job_type != "index_bm25":
            raise ValueError(f"unsupported BM25 job_type: {job_type!r}")
        arxiv_id = _validate_arxiv_id(str(job["arxiv_id"]))
        version = _validate_version(job["version"])
        job_asset_type = job.get("asset_type")
        if job_asset_type is not None:
            job_asset_type = _validate_asset_type(job_asset_type, required=False)
            if job_asset_type not in {"chunks", "bm25_index"}:
                raise ValueError(f"index_bm25 job cannot index {job_asset_type!r} asset")

        bm25_asset_key_value = arxiv_asset_key(arxiv_id, version, "bm25_index")
        existing_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (bm25_asset_key_value,),
        )
        existing_bm25_asset = _row_to_dict(existing_cursor.fetchone(), existing_cursor.description)
        existing_storage_path = existing_bm25_asset.get("storage_path") if existing_bm25_asset else None
        existing_index_path = Path(str(existing_storage_path)) if existing_storage_path else None
        existing_checksum = existing_bm25_asset.get("checksum") if existing_bm25_asset else None
        existing_index_ready = (
            existing_bm25_asset is not None
            and existing_bm25_asset["status"] == "indexed"
            and existing_index_path is not None
            and existing_index_path.is_file()
            and (
                not isinstance(existing_checksum, str)
                or not existing_checksum.startswith("sha256:")
                or existing_checksum == f"sha256:{hashlib.sha256(existing_index_path.read_bytes()).hexdigest()}"
            )
        )
        if existing_index_ready:
            append_arxiv_event(
                conn,
                "asset_bm25_skipped",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="bm25_index",
                job_key=job_key_value,
                payload={
                    "asset_key": bm25_asset_key_value,
                    "reason": "bm25_index_already_available",
                    "status": existing_bm25_asset["status"],
                    "storage_path": existing_bm25_asset["storage_path"],
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            return {
                "worked": True,
                "ok": True,
                "skipped": True,
                "reason": "bm25_index_already_available",
                "job": succeeded_job or job,
                "asset": existing_bm25_asset,
            }

        chunks_asset_key_value = arxiv_asset_key(arxiv_id, version, "chunks")
        chunks_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (chunks_asset_key_value,),
        )
        chunks_asset = _row_to_dict(chunks_cursor.fetchone(), chunks_cursor.description)
        if (
            chunks_asset is None
            or chunks_asset["status"] not in _PARSED_TEXT_READY_STATUSES
            or not chunks_asset.get("storage_path")
        ):
            raise ValueError("missing usable chunks asset")
        chunks_path = Path(str(chunks_asset["storage_path"]))
        if not chunks_path.is_file():
            raise FileNotFoundError(f"missing chunks asset file: {chunks_path}")

        chunks_payload = _read_chunks_payload(chunks_path)
        storage_path = arxiv_storage_path(object_root, arxiv_id, version, "bm25_index")
        checksum, byte_size = _write_bm25_index_atomic(
            storage_path,
            arxiv_id=arxiv_id,
            version=version,
            chunks_payload=chunks_payload,
            chunks_asset=chunks_asset,
        )
        chunk_count = len(chunks_payload["chunks"])
        success_savepoint = "paperstream_bm25_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            bm25_asset_key_value = upsert_arxiv_asset(
                conn,
                arxiv_id,
                version,
                "bm25_index",
                status="indexed",
                storage_path=storage_path,
                checksum=checksum,
                byte_size=byte_size,
                source=job_type,
            )
            append_arxiv_event(
                conn,
                "asset_bm25_indexed",
                arxiv_id=arxiv_id,
                version=version,
                asset_type="bm25_index",
                job_key=job_key_value,
                payload={
                    "asset_key": bm25_asset_key_value,
                    "source_asset_key": chunks_asset_key_value,
                    "storage_path": storage_path,
                    "checksum": checksum,
                    "byte_size": byte_size,
                    "chunk_count": chunk_count,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "asset": {
                "asset_key": bm25_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "bm25_index",
                "status": "indexed",
                "storage_path": storage_path,
                "checksum": checksum,
                "byte_size": byte_size,
                "source": job_type,
            },
            "source_asset": chunks_asset,
            "chunk_count": chunk_count,
        }
    except Exception as exc:
        error_message = _parse_failure_text(exc)
        failed_asset: dict[str, Any] | None = None
        failed_job: dict[str, Any] | None = None
        if arxiv_id is not None and version is not None:
            failure_savepoint = "paperstream_bm25_failure"
            _metadata_savepoint(conn, failure_savepoint)
            try:
                bm25_asset_key_value = upsert_arxiv_asset(
                    conn,
                    arxiv_id,
                    version,
                    "bm25_index",
                    status="failed",
                    source=job_type,
                    error_message=error_message,
                )
                append_arxiv_event(
                    conn,
                    "asset_bm25_failed",
                    arxiv_id=arxiv_id,
                    version=version,
                    asset_type="bm25_index",
                    job_key=job_key_value,
                    payload={
                        "asset_key": bm25_asset_key_value,
                        "error": error_message,
                        "source_asset_key": chunks_asset_key_value,
                    },
                    now=now,
                )
                failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
                _metadata_release(conn, failure_savepoint)
            except Exception:
                _metadata_rollback(conn, failure_savepoint)
                raise
            failed_asset = {
                "asset_key": bm25_asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": "bm25_index",
                "status": "failed",
                "source": job_type,
                "error_message": error_message,
            }
        else:
            failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "asset": failed_asset,
            "error": error_message,
        }


def _s3_bin_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload_text = job.get("payload_json")
    if not payload_text:
        raise ValueError("import_s3_bin job payload_json is required")
    try:
        payload = json.loads(str(payload_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid import_s3_bin payload JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("import_s3_bin payload must be an object")
    payload["bin_key"] = _validate_s3_bin_text(payload.get("bin_key"), "bin_key", required=True)
    payload["source"] = _validate_s3_bin_text(payload.get("source"), "source", required=True)
    return payload


def _checksum_matches(checksum: Any, data: bytes) -> bool:
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        return True
    return checksum == f"sha256:{hashlib.sha256(data).hexdigest()}"


def _file_checksum_matches(checksum: Any, path: Path) -> bool:
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        return True
    return checksum == f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

_S3_BIN_PDF_MEMBER_RE = re.compile(r"^(?P<arxiv_id>\d{4}\.\d{4,5})(?:v(?P<version>[1-9]\d*))?\.pdf$")
_S3_BIN_SOURCE_MEMBER_RE = re.compile(
    r"^(?P<arxiv_id>\d{4}\.\d{4,5})(?:v(?P<version>[1-9]\d*))?(?:\.tar\.gz|\.tgz)?$"
)


def _safe_s3_bin_member_parts(member_name: Any) -> tuple[str, ...] | None:
    if not isinstance(member_name, str):
        return None
    member_name = member_name.strip()
    if not member_name or "\x00" in member_name or "\\" in member_name:
        return None
    path = PurePosixPath(member_name)
    if path.is_absolute():
        return None
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    return tuple(parts)


def _parse_s3_bin_member_name(member_name: Any, asset_type: str) -> dict[str, Any] | None:
    asset_type = _validate_s3_bin_asset_type(asset_type)
    parts = _safe_s3_bin_member_parts(member_name)
    if parts is None or len(parts) > 2:
        return None
    if len(parts) == 2:
        prefix = parts[0].lower()
        if asset_type == "pdf" and prefix != "pdf":
            return None
        if asset_type == "source" and prefix not in {"src", "source"}:
            return None
    leaf = parts[-1]
    pattern = _S3_BIN_PDF_MEMBER_RE if asset_type == "pdf" else _S3_BIN_SOURCE_MEMBER_RE
    match = pattern.fullmatch(leaf)
    if match is None:
        return None
    version_text = match.group("version")
    return {
        "member_name": "/".join(parts),
        "arxiv_id": match.group("arxiv_id"),
        "version": int(version_text) if version_text is not None else None,
        "asset_type": asset_type,
    }


def _paper_latest_version(conn: sqlite3.Connection, arxiv_id: str) -> int | None:
    row = conn.execute(
        "SELECT latest_version FROM arxiv_papers WHERE arxiv_id = ?",
        (_validate_arxiv_id(arxiv_id),),
    ).fetchone()
    if row is None:
        return None
    return _validate_version(row["latest_version"] if isinstance(row, sqlite3.Row) else row[0])


def _paper_version_exists(conn: sqlite3.Connection, arxiv_id: str, version: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM arxiv_versions WHERE arxiv_id = ? AND version = ?",
        (_validate_arxiv_id(arxiv_id), _validate_version(version)),
    ).fetchone()
    return row is not None


def _s3_bin_extract_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload_text = job.get("payload_json")
    if not payload_text:
        raise ValueError("extract_s3_bin job payload_json is required")
    try:
        payload = json.loads(str(payload_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid extract_s3_bin payload JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("extract_s3_bin payload must be an object")
    payload["bin_key"] = _validate_s3_bin_text(payload.get("bin_key"), "bin_key", required=True)
    if payload.get("source") is not None:
        payload["source"] = _validate_s3_bin_text(payload.get("source"), "source", required=True)
    return payload


def _safe_arxiv_asset_storage_path(
    object_root: str | Path,
    arxiv_id: str,
    version: int,
    asset_type: str,
) -> str:
    storage_path = arxiv_storage_path(object_root, arxiv_id, version, asset_type)
    root = (Path(object_root) / "arxiv").resolve(strict=False)
    target = Path(storage_path).resolve(strict=False)
    try:
        common = os.path.commonpath((str(root), str(target)))
    except ValueError as exc:
        raise ValueError("asset storage path escapes object root") from exc
    if common != str(root):
        raise ValueError("asset storage path escapes object root")
    return storage_path


def _parse_job_type_for_asset(asset_type: str) -> str:
    asset_type = _validate_s3_bin_asset_type(asset_type)
    return "parse_pdf" if asset_type == "pdf" else "parse_source"


def work_next_arxiv_s3_bin_extract_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    now: datetime | str | None = None,
    job_types: Iterable[str] | str = ("extract_s3_bin",),
) -> dict[str, Any]:
    """Claim and run one queued offline arXiv S3 bin extraction job."""

    job_type_values = _validate_job_type_subset(
        job_types,
        allowed={"extract_s3_bin"},
        label="job_types",
        description="'extract_s3_bin'",
    )
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    bin_key_value: str | None = None
    asset_type: str | None = None
    source: str | None = None
    try:
        if _validate_job_type(str(job["job_type"])) != "extract_s3_bin":
            raise ValueError(f"unsupported S3 bin extract job_type: {job['job_type']!r}")
        payload = _s3_bin_extract_payload(job)
        bin_key_value = payload["bin_key"]
        source = payload.get("source")
        bin_row = _s3_bin_row_dict(conn, bin_key_value)
        asset_type = _validate_s3_bin_asset_type(bin_row["asset_type"])
        if bin_row["status"] not in {"downloaded", "imported"}:
            raise ValueError(f"S3 bin {bin_key_value!r} is not downloaded")
        if not bin_row.get("storage_path"):
            raise ValueError(f"S3 bin {bin_key_value!r} has no local storage_path")
        bin_path = Path(str(bin_row["storage_path"]))
        if not bin_path.is_file():
            raise FileNotFoundError(f"missing S3 bin file: {bin_path}")

        members_seen = 0
        extracted_members: list[dict[str, Any]] = []
        with tarfile.open(bin_path, mode="r:*") as tar:
            for member in tar:
                members_seen += 1
                if member.isdir() or not member.isfile():
                    continue
                parsed_member = _parse_s3_bin_member_name(member.name, asset_type)
                if parsed_member is None:
                    continue
                version = parsed_member["version"]
                if version is None:
                    version = _paper_latest_version(conn, parsed_member["arxiv_id"])
                    if version is None:
                        continue
                storage_path = _safe_arxiv_asset_storage_path(
                    object_root,
                    parsed_member["arxiv_id"],
                    version,
                    asset_type,
                )
                member_file = tar.extractfile(member)
                if member_file is None:
                    raise ValueError(f"recognized S3 bin member is not readable: {parsed_member['member_name']}")
                data = member_file.read()
                checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
                byte_size = len(data)
                _write_bytes_atomic(storage_path, data)
                extracted_members.append(
                    {
                        "bin_key": bin_key_value,
                        "arxiv_id": parsed_member["arxiv_id"],
                        "version": version,
                        "asset_type": asset_type,
                        "storage_path": storage_path,
                        "checksum": checksum,
                        "byte_size": byte_size,
                        "member_name": parsed_member["member_name"],
                    }
                )
        if not extracted_members:
            raise ValueError(f"S3 bin {bin_key_value!r} contains no recognized {asset_type} members")

        success_savepoint = "paperstream_s3_bin_extract_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            assets_imported = 0
            parse_jobs_queued = 0
            rows: list[dict[str, Any]] = []
            for extracted in extracted_members:
                member_record = {
                    "bin_key": extracted["bin_key"],
                    "arxiv_id": extracted["arxiv_id"],
                    "version": extracted["version"],
                    "asset_type": extracted["asset_type"],
                    "storage_path": extracted["storage_path"],
                    "checksum": extracted["checksum"],
                }
                member_result = upsert_arxiv_bin_member(conn, member_record, upsert_asset=False)
                asset_key_value: str | None = None
                parse_job_key: str | None = None
                if _paper_version_exists(conn, extracted["arxiv_id"], extracted["version"]):
                    asset_key_value = upsert_arxiv_asset(
                        conn,
                        extracted["arxiv_id"],
                        extracted["version"],
                        extracted["asset_type"],
                        status="imported",
                        storage_path=extracted["storage_path"],
                        checksum=extracted["checksum"],
                        byte_size=extracted["byte_size"],
                        source="s3_bin_member",
                    )
                    assets_imported += 1
                    append_arxiv_event(
                        conn,
                        "asset_imported",
                        arxiv_id=extracted["arxiv_id"],
                        version=extracted["version"],
                        asset_type=extracted["asset_type"],
                        payload={
                            "asset_key": asset_key_value,
                            "bin_key": bin_key_value,
                            "member_name": extracted["member_name"],
                            "storage_path": extracted["storage_path"],
                            "checksum": extracted["checksum"],
                            "byte_size": extracted["byte_size"],
                            "source": "s3_bin_member",
                        },
                        now=now,
                    )
                    parse_job_type = _parse_job_type_for_asset(extracted["asset_type"])
                    parse_job_key = arxiv_job_key(parse_job_type, extracted["arxiv_id"], extracted["version"])
                    parse_job_existed = conn.execute(
                        "SELECT 1 FROM arxiv_jobs WHERE job_key = ?",
                        (parse_job_key,),
                    ).fetchone() is not None
                    parse_job_key = enqueue_arxiv_job(
                        conn,
                        parse_job_type,
                        arxiv_id=extracted["arxiv_id"],
                        version=extracted["version"],
                        asset_type=extracted["asset_type"],
                        payload={
                            "source": "s3_bin_extract",
                            "asset_key": asset_key_value,
                            "bin_key": bin_key_value,
                            "member_name": extracted["member_name"],
                        },
                    )
                    if not parse_job_existed:
                        parse_jobs_queued += 1
                rows.append(
                    {
                        **member_result,
                        "byte_size": extracted["byte_size"],
                        "member_name": extracted["member_name"],
                        "asset_imported": asset_key_value is not None,
                        "parse_job_key": parse_job_key,
                    }
                )
            counts = {
                "members_seen": members_seen,
                "members_extracted": len(extracted_members),
                "assets_imported": assets_imported,
                "parse_jobs_queued": parse_jobs_queued,
            }
            conn.execute(
                """
                UPDATE arxiv_s3_bins
                SET status = 'imported',
                    imported_at = ?
                WHERE bin_key = ?
                """,
                (_utc_timestamp(now), bin_key_value),
            )
            append_arxiv_event(
                conn,
                "s3_bin_extracted",
                asset_type=asset_type,
                job_key=job_key_value,
                payload={
                    "bin_key": bin_key_value,
                    "source": source,
                    **counts,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "bin": {
                "bin_key": bin_key_value,
                "asset_type": asset_type,
                "status": "imported",
                "storage_path": bin_row["storage_path"],
                "imported_at": _utc_timestamp(now),
            },
            **counts,
            "rows": rows,
        }
    except Exception as exc:
        error_message = _download_failure_text(exc)
        failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        failed_job_cursor = conn.execute(
            """
            UPDATE arxiv_jobs
            SET status = 'failed',
                run_after = ?,
                locked_by = NULL,
                locked_at = NULL,
                error_message = ?,
                updated_at = ?
            WHERE job_key = ?
            RETURNING *
            """,
            (_utc_timestamp(now), error_message, _utc_timestamp(now), job_key_value),
        )
        failed_job = _row_to_dict(failed_job_cursor.fetchone(), failed_job_cursor.description) or failed_job
        if bin_key_value is not None:
            conn.execute(
                """
                UPDATE arxiv_s3_bins
                SET status = 'failed'
                WHERE bin_key = ?
                """,
                (bin_key_value,),
            )
            append_arxiv_event(
                conn,
                "s3_bin_extract_failed",
                asset_type=asset_type,
                job_key=job_key_value,
                payload={"bin_key": bin_key_value, "error": error_message, "source": source},
                now=now,
            )
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "bin_key": bin_key_value,
            "error": error_message,
        }


def work_next_arxiv_s3_bin_import_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    fetch_fn: Any = None,
    now: datetime | str | None = None,
    sleep_fn: Any = None,
    rate_limit_seconds: float = 0.0,
    job_types: Iterable[str] | str = ("import_s3_bin",),
) -> dict[str, Any]:
    """Claim and run one queued offline arXiv S3 bin import job."""

    rate_limit_seconds = float(rate_limit_seconds)
    if rate_limit_seconds < 0.0:
        raise ValueError("rate_limit_seconds must be non-negative")
    job_type_values = _validate_job_type_subset(
        job_types,
        allowed={"import_s3_bin"},
        label="job_types",
        description="'import_s3_bin'",
    )
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    bin_key_value: str | None = None
    asset_type: str | None = None
    storage_path: str | None = None
    source: str | None = None
    try:
        if _validate_job_type(str(job["job_type"])) != "import_s3_bin":
            raise ValueError(f"unsupported S3 bin import job_type: {job['job_type']!r}")
        payload = _s3_bin_payload(job)
        bin_key_value = payload["bin_key"]
        source = payload["source"]
        bin_row = _s3_bin_row_dict(conn, bin_key_value)
        asset_type = _validate_s3_bin_asset_type(bin_row["asset_type"])
        filename = _validate_s3_bin_relative_path(bin_row["filename"], "filename", required=True)
        storage_path = s3_bin_storage_path(object_root, asset_type, filename)
        existing_path = Path(str(bin_row["storage_path"])) if bin_row.get("storage_path") else None
        existing_ready = (
            bin_row["status"] in {"downloaded", "imported"}
            and existing_path is not None
            and existing_path.is_file()
            and (bin_row.get("size_bytes") is None or existing_path.stat().st_size == int(bin_row["size_bytes"]))
            and _file_checksum_matches(bin_row.get("checksum"), existing_path)
        )
        if existing_ready:
            append_arxiv_event(
                conn,
                "s3_bin_import_skipped",
                asset_type=asset_type,
                job_key=job_key_value,
                payload={
                    "bin_key": bin_key_value,
                    "reason": "s3_bin_already_downloaded",
                    "storage_path": str(existing_path),
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            return {
                "worked": True,
                "ok": True,
                "skipped": True,
                "reason": "s3_bin_already_downloaded",
                "job": succeeded_job or job,
                "bin": bin_row,
            }

        using_default_fetch = fetch_fn is None
        if using_default_fetch:
            if rate_limit_seconds:
                sleeper = time.sleep if sleep_fn is None else sleep_fn
                sleeper(rate_limit_seconds)
            fetch = _fetch_url_bytes
        else:
            fetch = fetch_fn
            if not callable(fetch):
                raise TypeError("fetch_fn must be callable")
        data = _ensure_download_bytes(fetch(source))
        byte_size = len(data)
        if bin_row.get("size_bytes") is not None and byte_size != int(bin_row["size_bytes"]):
            raise ValueError(f"S3 bin size mismatch for {bin_key_value}: expected {bin_row['size_bytes']}, got {byte_size}")
        checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if not _checksum_matches(bin_row.get("checksum"), data):
            raise ValueError(f"S3 bin checksum mismatch for {bin_key_value}")
        _write_bytes_atomic(storage_path, data)

        success_savepoint = "paperstream_s3_bin_import_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            conn.execute(
                """
                UPDATE arxiv_s3_bins
                SET status = 'downloaded',
                    storage_path = ?,
                    checksum = ?,
                    size_bytes = ?,
                    downloaded_at = ?
                WHERE bin_key = ?
                """,
                (storage_path, checksum, byte_size, _utc_timestamp(now), bin_key_value),
            )
            extract_job_key = _s3_bin_job_key("extract_s3_bin", bin_key_value)
            conn.execute(
                """
                INSERT OR IGNORE INTO arxiv_jobs (
                    job_key, job_type, asset_type, priority, max_attempts, payload_json
                )
                VALUES (?, 'extract_s3_bin', ?, ?, ?, ?)
                """,
                (
                    extract_job_key,
                    asset_type,
                    int(job.get("priority") or 100),
                    int(job.get("max_attempts") or 5),
                    _json_text({"source": "s3_bin_import", "bin_key": bin_key_value}),
                ),
            )
            append_arxiv_event(
                conn,
                "s3_bin_imported",
                asset_type=asset_type,
                job_key=job_key_value,
                payload={
                    "bin_key": bin_key_value,
                    "extract_job_key": extract_job_key,
                    "storage_path": storage_path,
                    "checksum": checksum,
                    "byte_size": byte_size,
                    "source": source,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "bin": {
                "bin_key": bin_key_value,
                "asset_type": asset_type,
                "status": "downloaded",
                "storage_path": storage_path,
                "checksum": checksum,
                "size_bytes": byte_size,
            },
            "extract_job_key": extract_job_key,
        }
    except Exception as exc:
        error_message = _download_failure_text(exc)
        failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        if bin_key_value is not None and asset_type is not None:
            conn.execute(
                """
                UPDATE arxiv_s3_bins
                SET status = 'failed',
                    storage_path = NULL
                WHERE bin_key = ?
                """,
                (bin_key_value,),
            )
            append_arxiv_event(
                conn,
                "s3_bin_import_failed",
                asset_type=asset_type,
                job_key=job_key_value,
                payload={"bin_key": bin_key_value, "error": error_message, "source": source},
                now=now,
            )
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "bin_key": bin_key_value,
            "error": error_message,
        }

def work_next_arxiv_download_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    fetch_fn: Any = None,
    now: datetime | str | None = None,
    sleep_fn: Any = None,
    rate_limit_seconds: float = 0.0,
    job_types: Iterable[str] | str = ("download_pdf", "download_source"),
) -> dict[str, Any]:
    """Claim and run one queued arXiv asset download job."""
    rate_limit_seconds = _validate_oai_rate_limit_seconds(rate_limit_seconds)
    if fetch_fn is None and rate_limit_seconds > 0.0:
        sleeper = time.sleep if sleep_fn is None else sleep_fn
        if not callable(sleeper):
            raise TypeError("sleep_fn must be callable")

    job_type_values = _validate_download_job_types(job_types)
    job = claim_next_arxiv_job(conn, worker_id, job_types=job_type_values, now=now)
    if job is None:
        return {"worked": False}

    job_key_value = str(job["job_key"])
    arxiv_id: str | None = None
    version: int | None = None
    asset_type: str | None = None
    parse_job_key: str | None = None
    storage_path: str | None = None
    url: str | None = None
    try:
        job_type = _validate_job_type(str(job["job_type"]))
        arxiv_id = _validate_arxiv_id(str(job["arxiv_id"]))
        version = _validate_version(job["version"])
        expected_asset_type = "pdf" if job_type == "download_pdf" else "source"
        asset_type = _validate_asset_type(job.get("asset_type") or expected_asset_type)
        if asset_type != expected_asset_type:
            raise ValueError(f"{job_type} job cannot download {asset_type!r} asset")

        asset_key_value = arxiv_asset_key(arxiv_id, version, asset_type)
        existing_cursor = conn.execute(
            "SELECT * FROM arxiv_assets WHERE asset_key = ?",
            (asset_key_value,),
        )
        existing_asset = _row_to_dict(existing_cursor.fetchone(), existing_cursor.description)
        if existing_asset is not None and existing_asset["status"] in {
            "downloaded",
            "imported",
            "parsed",
            "indexed",
        }:
            append_arxiv_event(
                conn,
                "asset_download_skipped",
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                job_key=job_key_value,
                payload={
                    "asset_key": asset_key_value,
                    "reason": "asset_already_available",
                    "status": existing_asset["status"],
                },
                now=now,
            )

            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            return {
                "worked": True,
                "ok": True,
                "skipped": True,
                "reason": "asset_already_available",
                "job": succeeded_job or job,
                "asset": existing_asset,
            }

        url = _arxiv_download_url(arxiv_id, asset_type)
        using_default_fetch = fetch_fn is None
        if using_default_fetch:
            fetch = _fetch_url_bytes
        else:
            fetch = fetch_fn
            if not callable(fetch):
                raise TypeError("fetch_fn must be callable")
        append_arxiv_event(
            conn,
            "asset_download_attempted",
            arxiv_id=arxiv_id,
            version=version,
            asset_type=asset_type,
            job_key=job_key_value,
            payload={"asset_key": asset_key_value, "url": url},
            now=now,
        )

        if using_default_fetch:
            response_data = _request_rate_limited_arxiv_url(
                url,
                request_fn=fetch,
                sleep_fn=sleep_fn,
                rate_limit_seconds=rate_limit_seconds,
                request_label="arXiv download request",
            )
        else:
            response_data = fetch(url)
        data = _ensure_download_bytes(response_data)
        storage_path = arxiv_storage_path(object_root, arxiv_id, version, asset_type)
        _write_bytes_atomic(storage_path, data)
        checksum = f"sha256:{hashlib.sha256(data).hexdigest()}"
        byte_size = len(data)
        success_savepoint = "paperstream_download_success"
        _metadata_savepoint(conn, success_savepoint)
        try:
            asset_key_value = upsert_arxiv_asset(
                conn,
                arxiv_id,
                version,
                asset_type,
                status="downloaded",
                storage_path=storage_path,
                checksum=checksum,
                byte_size=byte_size,
                source="arxiv",
            )
            parse_job_type = "parse_pdf" if asset_type == "pdf" else "parse_source"
            parse_job_key = enqueue_arxiv_job(
                conn,
                parse_job_type,
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                payload={"source": "arxiv_download", "asset_key": asset_key_value},
            )
            append_arxiv_event(
                conn,
                "asset_downloaded",
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                job_key=job_key_value,
                payload={
                    "asset_key": asset_key_value,
                    "parse_job_key": parse_job_key,
                    "storage_path": storage_path,
                    "checksum": checksum,
                    "byte_size": byte_size,
                    "url": url,
                },
                now=now,
            )
            succeeded_job = mark_arxiv_job_succeeded(conn, job_key_value, now=now)
            _metadata_release(conn, success_savepoint)
        except Exception:
            _metadata_rollback(conn, success_savepoint)
            raise
        return {
            "worked": True,
            "ok": True,
            "job": succeeded_job or job,
            "asset": {
                "asset_key": asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": asset_type,
                "status": "downloaded",
                "storage_path": storage_path,
                "checksum": checksum,
                "byte_size": byte_size,
                "source": "arxiv",
            },
            "parse_job_key": parse_job_key,
        }
    except Exception as exc:
        error_message = _download_failure_text(exc)
        failed_asset: dict[str, Any] | None = None
        if arxiv_id is not None and version is not None and asset_type is not None:
            asset_key_value = upsert_arxiv_asset(
                conn,
                arxiv_id,
                version,
                asset_type,
                status="failed",
                source="arxiv",
                error_message=error_message,
            )
            append_arxiv_event(
                conn,
                "asset_failed",
                arxiv_id=arxiv_id,
                version=version,
                asset_type=asset_type,
                job_key=job_key_value,
                payload={"asset_key": asset_key_value, "error": error_message, "url": url},
                now=now,
            )
            failed_asset = {
                "asset_key": asset_key_value,
                "arxiv_id": arxiv_id,
                "version": version,
                "asset_type": asset_type,
                "status": "failed",
                "source": "arxiv",
                "error_message": error_message,
            }
        failed_job = mark_arxiv_job_failed(conn, job_key_value, error_message, now=now)
        return {
            "worked": True,
            "ok": False,
            "job": failed_job or job,
            "asset": failed_asset,
            "error": error_message,
        }


class BloomFilter:
    """Small deterministic pure-Python Bloom filter.

    The filter is an accelerator only; callers must still use SQLite for final
    truth.  Hashing uses SHA-256 double hashing so results are stable across
    Python processes.
    """

    def __init__(
        self,
        capacity: int | None = None,
        false_positive_rate: float = 0.01,
        *,
        bit_count: int | None = None,
        hash_count: int | None = None,
        bits: bytes | bytearray | memoryview | None = None,
        item_count: int = 0,
    ) -> None:
        false_positive_rate = float(false_positive_rate)
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be between 0 and 1")
        if bit_count is None or hash_count is None:
            if capacity is None:
                raise ValueError("capacity is required when bit_count/hash_count are omitted")
            capacity = max(1, int(capacity))
            bit_count, hash_count = self.parameters_for_capacity(capacity, false_positive_rate)
        bit_count = int(bit_count)
        hash_count = int(hash_count)
        item_count = int(item_count)
        if bit_count < 1:
            raise ValueError("bit_count must be positive")
        if hash_count < 1:
            raise ValueError("hash_count must be positive")
        if item_count < 0:
            raise ValueError("item_count must be non-negative")
        byte_count = (bit_count + 7) // 8
        if bits is None:
            self._bits = bytearray(byte_count)
        else:
            self._bits = bytearray(bytes(bits))
            if len(self._bits) != byte_count:
                raise ValueError("bits length does not match bit_count")
        self.false_positive_rate = false_positive_rate
        self.bit_count = bit_count
        self.hash_count = hash_count
        self.item_count = item_count

    @staticmethod
    def parameters_for_capacity(capacity: int, false_positive_rate: float = 0.01) -> tuple[int, int]:
        capacity = max(1, int(capacity))
        false_positive_rate = float(false_positive_rate)
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be between 0 and 1")
        bit_count = max(8, math.ceil(-(capacity * math.log(false_positive_rate)) / (math.log(2) ** 2)))
        hash_count = max(1, math.ceil((bit_count / capacity) * math.log(2)))
        return bit_count, hash_count

    @classmethod
    def from_bytes(
        cls,
        bits: bytes | bytearray | memoryview,
        bit_count: int,
        hash_count: int,
        item_count: int = 0,
        false_positive_rate: float = 0.01,
    ) -> "BloomFilter":
        return cls(
            false_positive_rate=false_positive_rate,
            bit_count=bit_count,
            hash_count=hash_count,
            bits=bits,
            item_count=item_count,
        )

    def _hashes(self, key: str | bytes) -> Iterator[int]:
        data = _key_bytes(key)
        digest = hashlib.sha256(data).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big") or 1
        for index in range(self.hash_count):
            yield (h1 + index * h2 + index * index) % self.bit_count

    def add(self, key: str | bytes) -> bool:
        """Add ``key``; return True if at least one bit was newly set."""

        indexes = tuple(self._hashes(key))
        changed = False
        for index in indexes:
            byte_index, bit_index = divmod(index, 8)
            mask = 1 << bit_index
            if not self._bits[byte_index] & mask:
                changed = True
                self._bits[byte_index] |= mask
        if changed:
            self.item_count += 1
        return changed

    def contains(self, key: str | bytes) -> bool:
        return all(self._bits[index // 8] & (1 << (index % 8)) for index in self._hashes(key))

    def __contains__(self, key: object) -> bool:
        try:
            return self.contains(key if isinstance(key, bytes) else str(key))
        except ValueError:
            return False

    def to_bytes(self) -> bytes:
        return bytes(self._bits)

    def copy(self) -> "BloomFilter":
        return BloomFilter.from_bytes(
            self.to_bytes(),
            self.bit_count,
            self.hash_count,
            item_count=self.item_count,
            false_positive_rate=self.false_positive_rate,
        )


def save_bloom_filter(
    conn: sqlite3.Connection,
    filter_name: str,
    bloom_filter: BloomFilter,
    source_table: str = "manual",
    source_column: str = "manual",
) -> str:
    """Persist a Bloom filter snapshot by name."""

    if not isinstance(bloom_filter, BloomFilter):
        raise TypeError("bloom_filter must be a BloomFilter")
    filter_name = _validate_filter_name(filter_name)
    source_table = _validate_source_name(source_table, "source_table")
    source_column = _validate_source_name(source_column, "source_column")
    conn.execute(
        """
        INSERT INTO arxiv_bloom_filters (
            filter_name, false_positive_rate, bit_count, hash_count, item_count,
            bits, source_table, source_column, rebuilt_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(filter_name) DO UPDATE SET
            false_positive_rate = excluded.false_positive_rate,
            bit_count = excluded.bit_count,
            hash_count = excluded.hash_count,
            item_count = excluded.item_count,
            bits = excluded.bits,
            source_table = excluded.source_table,
            source_column = excluded.source_column,
            rebuilt_at = CURRENT_TIMESTAMP
        """,
        (
            filter_name,
            bloom_filter.false_positive_rate,
            bloom_filter.bit_count,
            bloom_filter.hash_count,
            bloom_filter.item_count,
            sqlite3.Binary(bloom_filter.to_bytes()),
            source_table,
            source_column,
        ),
    )
    return filter_name


def load_bloom_filter(conn: sqlite3.Connection, filter_name: str) -> BloomFilter | None:
    """Load a persisted Bloom filter, or ``None`` when absent."""

    filter_name = _validate_filter_name(filter_name)
    row = conn.execute(
        """
        SELECT false_positive_rate, bit_count, hash_count, item_count, bits
        FROM arxiv_bloom_filters
        WHERE filter_name = ?
        """,
        (filter_name,),
    ).fetchone()
    if row is None:
        return None
    false_positive_rate = row["false_positive_rate"] if isinstance(row, sqlite3.Row) else row[0]
    bit_count = row["bit_count"] if isinstance(row, sqlite3.Row) else row[1]
    hash_count = row["hash_count"] if isinstance(row, sqlite3.Row) else row[2]
    item_count = row["item_count"] if isinstance(row, sqlite3.Row) else row[3]
    bits = row["bits"] if isinstance(row, sqlite3.Row) else row[4]
    return BloomFilter.from_bytes(
        bits,
        bit_count=int(bit_count),
        hash_count=int(hash_count),
        item_count=int(item_count),
        false_positive_rate=float(false_positive_rate),
    )


def rebuild_bloom_filter(
    conn: sqlite3.Connection,
    filter_name: str,
    keys: Iterable[str | bytes],
    false_positive_rate: float = 0.01,
    source_table: str = "manual",
    source_column: str = "manual",
) -> BloomFilter:
    """Rebuild a Bloom filter from explicit keys and persist it."""

    filter_name = _validate_filter_name(filter_name)
    key_list = list(keys)
    bloom_filter = BloomFilter(capacity=max(1, len(key_list)), false_positive_rate=false_positive_rate)
    for key in key_list:
        bloom_filter.add(key)
    save_bloom_filter(conn, filter_name, bloom_filter, source_table, source_column)
    return bloom_filter

def bloom_db_contains(
    conn: sqlite3.Connection,
    bloom_filter: BloomFilter,
    table_name: str,
    column_name: str,
    key: str | bytes,
) -> bool:
    """Return True only when Bloom membership is confirmed by SQLite."""

    if not isinstance(bloom_filter, BloomFilter):
        raise TypeError("bloom_filter must be a BloomFilter")
    table_name = _validate_source_name(table_name, "table_name")
    column_name = _validate_source_name(column_name, "column_name")
    if not isinstance(key, (str, bytes)):
        key = str(key)
    _key_bytes(key)
    if not bloom_filter.contains(key):
        return False
    row = conn.execute(
        f"SELECT 1 FROM {table_name} WHERE {column_name} = ? LIMIT 1",
        (key,),
    ).fetchone()
    return row is not None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = (), default: Any = 0) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return default
    value = row[0] if not isinstance(row, sqlite3.Row) else row[0]
    return default if value is None else value


def get_paperstream_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return status/metric fields for PaperStream schema tables."""

    sync_status: dict[str, Any] = {
        "last_successful_oai_sync": None,
        "last_response_date": None,
        "last_committed_cursor": None,
        "pending_cursor": None,
        "metadata_prefix": "arXivRaw",
        "set_spec": None,
        "records_seen_last_run": 0,
        "records_updated_last_run": 0,
        "records_inserted_last_run": 0,
        "versions_inserted_last_run": 0,
        "jobs_queued_last_run": 0,
    }
    if _table_exists(conn, "arxiv_sync_state"):
        sync_columns = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in conn.execute("PRAGMA table_info(arxiv_sync_state)")
        }
        records_inserted_expr = (
            "records_inserted_last_run"
            if "records_inserted_last_run" in sync_columns
            else "0 AS records_inserted_last_run"
        )
        versions_inserted_expr = (
            "versions_inserted_last_run"
            if "versions_inserted_last_run" in sync_columns
            else "0 AS versions_inserted_last_run"
        )
        metadata_prefix_expr = (
            "metadata_prefix"
            if "metadata_prefix" in sync_columns
            else "'arXivRaw' AS metadata_prefix"
        )
        set_spec_expr = (
            "set_spec"
            if "set_spec" in sync_columns
            else "NULL AS set_spec"
        )
        latest_row = conn.execute(
            f"""
            SELECT committed_from, last_response_date, pending_response_date,
                   {metadata_prefix_expr}, {set_spec_expr},
                   records_seen_last_run, records_updated_last_run,
                   {records_inserted_expr}, {versions_inserted_expr},
                   jobs_queued_last_run
            FROM arxiv_sync_state
            ORDER BY COALESCE(updated_at, finished_at) DESC, sync_name ASC
            LIMIT 1
            """
        ).fetchone()
        if latest_row is not None:
            sync_status.update(
                {
                    "last_response_date": latest_row["last_response_date"] if isinstance(latest_row, sqlite3.Row) else latest_row[1],
                    "last_committed_cursor": latest_row["committed_from"] if isinstance(latest_row, sqlite3.Row) else latest_row[0],
                    "pending_cursor": latest_row["pending_response_date"] if isinstance(latest_row, sqlite3.Row) else latest_row[2],
                    "metadata_prefix": (latest_row["metadata_prefix"] if isinstance(latest_row, sqlite3.Row) else latest_row[3]) or "arXivRaw",
                    "set_spec": latest_row["set_spec"] if isinstance(latest_row, sqlite3.Row) else latest_row[4],
                    "records_seen_last_run": int((latest_row["records_seen_last_run"] if isinstance(latest_row, sqlite3.Row) else latest_row[5]) or 0),
                    "records_updated_last_run": int((latest_row["records_updated_last_run"] if isinstance(latest_row, sqlite3.Row) else latest_row[6]) or 0),
                    "records_inserted_last_run": int((latest_row["records_inserted_last_run"] if isinstance(latest_row, sqlite3.Row) else latest_row[7]) or 0),
                    "versions_inserted_last_run": int((latest_row["versions_inserted_last_run"] if isinstance(latest_row, sqlite3.Row) else latest_row[8]) or 0),
                    "jobs_queued_last_run": int((latest_row["jobs_queued_last_run"] if isinstance(latest_row, sqlite3.Row) else latest_row[9]) or 0),
                }
            )

        successful_row = conn.execute(
            """
            SELECT finished_at, last_response_date, committed_from
            FROM arxiv_sync_state
            WHERE status = 'succeeded'
            ORDER BY COALESCE(finished_at, updated_at) DESC, sync_name ASC
            LIMIT 1
            """
        ).fetchone()
        if successful_row is not None:
            sync_status.update(
                {
                    "last_successful_oai_sync": successful_row["finished_at"] if isinstance(successful_row, sqlite3.Row) else successful_row[0],
                    "last_response_date": successful_row["last_response_date"] if isinstance(successful_row, sqlite3.Row) else successful_row[1],
                    "last_committed_cursor": successful_row["committed_from"] if isinstance(successful_row, sqlite3.Row) else successful_row[2],
                }
            )


    papers = {"total": 0}
    if _table_exists(conn, "arxiv_papers"):
        papers["total"] = int(_scalar(conn, "SELECT COUNT(*) FROM arxiv_papers"))

    assets = {"total": 0, "by_type": {}, "by_status": {}}
    if _table_exists(conn, "arxiv_assets"):
        assets["total"] = int(_scalar(conn, "SELECT COUNT(*) FROM arxiv_assets"))
        for row in conn.execute(
            "SELECT asset_type, COUNT(*) AS count FROM arxiv_assets GROUP BY asset_type ORDER BY asset_type"
        ):
            asset_type = row["asset_type"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            assets["by_type"][asset_type] = int(count)
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM arxiv_assets GROUP BY status ORDER BY status"
        ):
            status_name = row["status"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            assets["by_status"][status_name] = int(count)
    jobs = {"total": 0, "by_type": {}, "by_status": {}, "oldest_queued_job": None}
    queued_jobs = running_jobs = failed_jobs = 0
    oldest_queued_job = None
    if _table_exists(conn, "arxiv_jobs"):
        jobs["total"] = int(_scalar(conn, "SELECT COUNT(*) FROM arxiv_jobs"))
        for row in conn.execute(
            "SELECT job_type, COUNT(*) AS count FROM arxiv_jobs GROUP BY job_type ORDER BY job_type"
        ):
            job_type = row["job_type"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            jobs["by_type"][job_type] = int(count)
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM arxiv_jobs GROUP BY status ORDER BY status"
        ):
            status_name = row["status"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            jobs["by_status"][status_name] = int(count)
        queued_jobs = int(jobs["by_status"].get("queued", 0))
        running_jobs = int(jobs["by_status"].get("running", 0))
        failed_jobs = int(jobs["by_status"].get("failed", 0))
        oldest_queued_job = _scalar(
            conn,
            "SELECT MIN(run_after) FROM arxiv_jobs WHERE status = 'queued'",
            default=None,
        )
        jobs["oldest_queued_job"] = oldest_queued_job


    s3_bins = {"total": 0, "by_type": {}, "by_status": {}}
    if _table_exists(conn, "arxiv_s3_bins"):
        s3_bins["total"] = int(_scalar(conn, "SELECT COUNT(*) FROM arxiv_s3_bins"))
        for row in conn.execute(
            "SELECT asset_type, COUNT(*) AS count FROM arxiv_s3_bins GROUP BY asset_type ORDER BY asset_type"
        ):
            asset_type = row["asset_type"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            s3_bins["by_type"][asset_type] = int(count)
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM arxiv_s3_bins GROUP BY status ORDER BY status"
        ):
            status_name = row["status"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            s3_bins["by_status"][status_name] = int(count)


    bin_members = {"total": 0, "by_type": {}, "with_storage_path": 0, "without_storage_path": 0}
    if _table_exists(conn, "arxiv_bin_members"):
        bin_members["total"] = int(_scalar(conn, "SELECT COUNT(*) FROM arxiv_bin_members"))
        bin_members["with_storage_path"] = int(
            _scalar(
                conn,
                "SELECT COUNT(*) FROM arxiv_bin_members WHERE storage_path IS NOT NULL AND storage_path <> ''",
            )
        )
        bin_members["without_storage_path"] = int(bin_members["total"]) - int(bin_members["with_storage_path"])
        for row in conn.execute(
            "SELECT asset_type, COUNT(*) AS count FROM arxiv_bin_members GROUP BY asset_type ORDER BY asset_type"
        ):
            asset_type = row["asset_type"] if isinstance(row, sqlite3.Row) else row[0]
            count = row["count"] if isinstance(row, sqlite3.Row) else row[1]
            bin_members["by_type"][asset_type] = int(count)
    assets_failed = int(assets["by_status"].get("failed", 0))

    storage_reconciliation_errors = 0
    if _table_exists(conn, "arxiv_events"):
        storage_reconciliation_errors = int(
            _scalar(
                conn,
                "SELECT COUNT(*) FROM arxiv_events WHERE event_type = 'storage_reconciliation_error'",
            )
        )

    bloom_filters: dict[str, dict[str, Any]] = {}
    if _table_exists(conn, "arxiv_bloom_filters"):
        rows = conn.execute(
            """
            SELECT filter_name, item_count, false_positive_rate, bit_count,
                   hash_count, source_table, source_column, rebuilt_at
            FROM arxiv_bloom_filters
            ORDER BY filter_name
            """
        ).fetchall()
        for row in rows:
            name = row["filter_name"] if isinstance(row, sqlite3.Row) else row[0]
            bloom_filters[name] = {
                "item_count": int((row["item_count"] if isinstance(row, sqlite3.Row) else row[1]) or 0),
                "false_positive_rate": float(row["false_positive_rate"] if isinstance(row, sqlite3.Row) else row[2]),
                "bit_count": int(row["bit_count"] if isinstance(row, sqlite3.Row) else row[3]),
                "hash_count": int(row["hash_count"] if isinstance(row, sqlite3.Row) else row[4]),
                "source_table": row["source_table"] if isinstance(row, sqlite3.Row) else row[5],
                "source_column": row["source_column"] if isinstance(row, sqlite3.Row) else row[6],
                "rebuilt_at": row["rebuilt_at"] if isinstance(row, sqlite3.Row) else row[7],
            }

    sync = {
        "last_successful_oai_sync": sync_status["last_successful_oai_sync"],
        "last_response_date": sync_status["last_response_date"],
        "last_committed_cursor": sync_status["last_committed_cursor"],
        "pending_cursor": sync_status["pending_cursor"],
        "metadata_prefix": sync_status["metadata_prefix"],
        "set_spec": sync_status["set_spec"],
        "records_seen_last_run": sync_status["records_seen_last_run"],
        "records_updated_last_run": sync_status["records_updated_last_run"],
        "records_inserted_last_run": sync_status["records_inserted_last_run"],
        "versions_inserted_last_run": sync_status["versions_inserted_last_run"],
        "jobs_queued_last_run": sync_status["jobs_queued_last_run"],
    }

    status = {
        **sync_status,
        "queued_jobs": queued_jobs,
        "running_jobs": running_jobs,
        "failed_jobs": failed_jobs,
        "oldest_queued_job": oldest_queued_job,
        "assets_failed": assets_failed,
        "storage_reconciliation_errors": storage_reconciliation_errors,
        "papers": papers,
        "assets": assets,
        "jobs": jobs,
        "sync": sync,
        "bloom_filters": bloom_filters,
        "bloom_filter_item_count": sum(item["item_count"] for item in bloom_filters.values()),
        "s3_bins": s3_bins,
        "arxiv_s3_bins": s3_bins,
        "bin_members": bin_members,
        "arxiv_bin_members": bin_members,
        "bloom_filter_false_positive_target": max(
            (item["false_positive_rate"] for item in bloom_filters.values()),
            default=None,
        ),
    }
    return status


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item for item in value.split() if item]
    else:
        parsed = value
    if isinstance(parsed, list):
        return parsed
    return []

def _validate_paperstream_search_query(query: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    terms: list[str] = []
    seen: set[str] = set()
    for match in _PAPERSTREAM_SEARCH_TERM_RE.finditer(normalized):
        term = match.group(0).lower()
        if term not in seen:
            terms.append(term)
            seen.add(term)
    if not terms:
        raise ValueError("query must include at least one searchable term")
    return normalized, tuple(terms)


def _validate_paperstream_search_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer")
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit > _PAPERSTREAM_SEARCH_MAX_LIMIT:
        raise ValueError(f"limit must be <= {_PAPERSTREAM_SEARCH_MAX_LIMIT}")
    return limit


def _validate_paperstream_max_chunk_chars(max_chunk_chars: Any) -> int:
    if isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int):
        raise ValueError("max_chunk_chars must be a non-negative integer")
    if max_chunk_chars < 0:
        raise ValueError("max_chunk_chars must be non-negative")
    if max_chunk_chars > _PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS:
        raise ValueError(f"max_chunk_chars must be <= {_PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS}")
    return max_chunk_chars


def _paperstream_like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _paperstream_fts_query(terms: Iterable[str]) -> str:
    quoted_terms = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms]
    if not quoted_terms:
        raise ValueError("query must include at least one searchable term")
    return " OR ".join(quoted_terms)


def _paperstream_public_categories(value: Any) -> list[str]:
    return [str(item) for item in _json_list(value) if item is not None]

def _paperstream_public_authors(value: Any) -> list[str]:
    return [str(item) for item in _json_list(value) if item is not None]


def _paperstream_source_texts(row: dict[str, Any]) -> dict[str, str]:
    categories = " ".join(_paperstream_public_categories(row.get("categories_json")))
    return {
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "authors": str(row.get("authors_json") or ""),
        "categories": categories,
        "primary_category": str(row.get("primary_category") or ""),
        "abs_url": str(row.get("abs_url") or ""),
    }


def _paperstream_match_sources(row: dict[str, Any], terms: Iterable[str]) -> list[str]:
    term_values = tuple(terms)
    matches: list[str] = []
    source_texts = _paperstream_source_texts(row)
    for source in _PAPERSTREAM_SEARCH_LIKE_SOURCES:
        source_text = source_texts.get(source, "").lower()
        if any(term in source_text for term in term_values):
            matches.append(source)
    return matches


def _paperstream_public_paper(row: dict[str, Any], *, metadata_score: int | None = None, match_sources: list[str] | None = None) -> dict[str, Any]:
    paper = {
        "arxiv_id": row["arxiv_id"],
        "latest_version": int(row["latest_version"]),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "authors": _paperstream_public_authors(row.get("authors_json")),
        "primary_category": row.get("primary_category"),
        "categories": _paperstream_public_categories(row.get("categories_json")),
        "abs_url": row.get("abs_url"),
    }
    if metadata_score is not None:
        paper["metadata_score"] = int(metadata_score)
    if match_sources is not None:
        paper["match_sources"] = match_sources
    return paper


def _paperstream_metadata_rows_for_search(conn: sqlite3.Connection, terms: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "arxiv_papers"):
        return []
    has_versions = _table_exists(conn, "arxiv_versions")
    version_join = (
        """
        LEFT JOIN arxiv_versions AS v
          ON v.arxiv_id = p.arxiv_id
         AND v.version = COALESCE((SELECT MAX(v2.version) FROM arxiv_versions AS v2 WHERE v2.arxiv_id = p.arxiv_id), p.latest_version)
        """
        if has_versions
        else ""
    )
    latest_version_expr = (
        "COALESCE((SELECT MAX(v3.version) FROM arxiv_versions AS v3 WHERE v3.arxiv_id = p.arxiv_id), p.latest_version)"
        if has_versions
        else "p.latest_version"
    )
    select_exprs = {
        "title": "COALESCE(v.title, p.title)" if has_versions else "p.title",
        "abstract": "COALESCE(v.abstract, p.abstract)" if has_versions else "p.abstract",
        "authors_json": "COALESCE(v.authors_json, p.authors_json)" if has_versions else "p.authors_json",
        "categories_json": "COALESCE(v.categories_json, p.categories_json)" if has_versions else "p.categories_json",
    }
    source_exprs = {
        "title": f"LOWER(COALESCE({select_exprs['title']}, ''))",
        "abstract": f"LOWER(COALESCE({select_exprs['abstract']}, ''))",
        "authors": f"LOWER(COALESCE({select_exprs['authors_json']}, ''))",
        "categories": f"LOWER(COALESCE({select_exprs['categories_json']}, ''))",
        "primary_category": "LOWER(COALESCE(p.primary_category, ''))",
        "abs_url": "LOWER(COALESCE(p.abs_url, ''))",
    }
    checks: list[str] = []
    score_parts: list[str] = []
    score_params: list[Any] = []
    where_params: list[Any] = []
    for term in terms:
        pattern = _paperstream_like_pattern(term)
        for source in _PAPERSTREAM_SEARCH_LIKE_SOURCES:
            expression = source_exprs[source]
            check = f"{expression} LIKE ? ESCAPE '\\'"
            checks.append(check)
            where_params.append(pattern)
            score_parts.append(f"CASE WHEN {check} THEN 1 ELSE 0 END")
            score_params.append(pattern)
    score_sql = " + ".join(score_parts)
    where_sql = " OR ".join(checks)
    cursor = conn.execute(
        f"""
        SELECT p.arxiv_id,
               {latest_version_expr} AS latest_version,
               {select_exprs['title']} AS title,
               {select_exprs['abstract']} AS abstract,
               {select_exprs['authors_json']} AS authors_json,
               p.primary_category AS primary_category,
               {select_exprs['categories_json']} AS categories_json,
               p.abs_url AS abs_url,
               ({score_sql}) AS metadata_score
        FROM arxiv_papers AS p
        {version_join}
        WHERE {where_sql}
        ORDER BY metadata_score DESC, p.arxiv_id ASC
        LIMIT ?
        """,
        [*score_params, *where_params, limit],
    )
    rows: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        row_dict = _row_to_dict(row, cursor.description)
        if row_dict is not None:
            rows.append(row_dict)
    return rows


def _paperstream_public_paper_for_version(conn: sqlite3.Connection, arxiv_id: str, version: int) -> dict[str, Any] | None:
    if not _table_exists(conn, "arxiv_papers"):
        return None
    has_versions = _table_exists(conn, "arxiv_versions")
    if has_versions:
        cursor = conn.execute(
            """
            SELECT p.arxiv_id,
                   p.latest_version,
                   COALESCE(v.title, p.title) AS title,
                   COALESCE(v.abstract, p.abstract) AS abstract,
                   COALESCE(v.authors_json, p.authors_json) AS authors_json,
                   p.primary_category AS primary_category,
                   COALESCE(v.categories_json, p.categories_json) AS categories_json,
                   p.abs_url AS abs_url
            FROM arxiv_papers AS p
            LEFT JOIN arxiv_versions AS v
              ON v.arxiv_id = p.arxiv_id AND v.version = ?
            WHERE p.arxiv_id = ?
            """,
            (version, arxiv_id),
        )
    else:
        cursor = conn.execute(
            """
            SELECT arxiv_id, latest_version, title, abstract, authors_json, primary_category, categories_json, abs_url
            FROM arxiv_papers
            WHERE arxiv_id = ?
            """,
            (arxiv_id,),
        )
    row = _row_to_dict(cursor.fetchone(), cursor.description)
    if row is None:
        return None
    return _paperstream_public_paper(row)


def _paperstream_bm25_asset_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not _table_exists(conn, "arxiv_assets"):
        return []
    cursor = conn.execute(
        """
        SELECT asset_key, arxiv_id, version, storage_path
        FROM arxiv_assets
        WHERE asset_type = 'bm25_index' AND status = 'indexed'
        ORDER BY arxiv_id ASC, version ASC, asset_key ASC
        """
    )
    return [row_dict for row in cursor.fetchall() if (row_dict := _row_to_dict(row, cursor.description)) is not None]


def _paperstream_open_bm25_index(storage_path: Any) -> sqlite3.Connection:
    if not storage_path:
        raise FileNotFoundError("BM25 index storage path is empty")
    path = Path(str(storage_path)).expanduser()
    if not path.exists():
        raise FileNotFoundError("BM25 index file is missing")
    uri = path.resolve().as_uri() + "?mode=ro"
    index_conn = sqlite3.connect(uri, uri=True)
    index_conn.row_factory = sqlite3.Row
    return index_conn


def _paperstream_bm25_error(asset: dict[str, Any], exc: Exception) -> dict[str, Any]:
    error = "missing_bm25_index" if isinstance(exc, FileNotFoundError) else "unreadable_bm25_index"
    return {
        "asset_key": asset.get("asset_key"),
        "arxiv_id": asset.get("arxiv_id"),
        "version": asset.get("version"),
        "error": error,
        "message": "BM25 index file is missing" if error == "missing_bm25_index" else "BM25 index file is unreadable",
    }


def _paperstream_search_bm25_asset(
    conn: sqlite3.Connection,
    asset: dict[str, Any],
    fts_query: str,
    *,
    limit: int,
    max_chunk_chars: int,
    paper_cache: dict[tuple[str, int], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    arxiv_id = str(asset["arxiv_id"])
    version = _validate_version(asset.get("version"))
    cache_key = (arxiv_id, version)
    if cache_key not in paper_cache:
        paper_cache[cache_key] = _paperstream_public_paper_for_version(conn, arxiv_id, version)
    paper = paper_cache[cache_key] or {
        "arxiv_id": arxiv_id,
        "latest_version": version,
        "title": None,
        "abstract": None,
        "authors": [],
        "primary_category": None,
        "categories": [],
        "abs_url": None,
    }
    index_conn = _paperstream_open_bm25_index(asset.get("storage_path"))
    try:
        cursor = index_conn.execute(
            """
            SELECT c.chunk_index AS chunk_index,
                   c.text AS text,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks AS c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC, c.chunk_index ASC
            LIMIT ?
            """,
            (fts_query, limit),
        )
        chunks: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            text = str(row["text"] or "")
            chunk = {
                "arxiv_id": arxiv_id,
                "version": version,
                "chunk_index": int(row["chunk_index"]),
                "text": text[:max_chunk_chars],
                "score": float(row["score"]),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": paper.get("authors", []),
                "primary_category": paper.get("primary_category"),
                "categories": paper.get("categories", []),
                "abs_url": paper.get("abs_url"),
            }
            chunks.append(chunk)
        return chunks
    finally:
        index_conn.close()



def _paperstream_search_materialized_chunks(
    conn: sqlite3.Connection,
    fts_query: str,
    *,
    limit: int,
    max_chunk_chars: int,
    paper_cache: dict[tuple[str, int], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "arxiv_chunks") or not _table_exists(conn, "arxiv_chunks_fts"):
        return []
    cursor = conn.execute(
        """
        SELECT c.arxiv_id AS arxiv_id,
               c.version AS version,
               c.chunk_index AS chunk_index,
               c.text AS text,
               bm25(arxiv_chunks_fts) AS score
        FROM arxiv_chunks_fts
        JOIN arxiv_chunks AS c ON c.rowid = arxiv_chunks_fts.rowid
        WHERE arxiv_chunks_fts MATCH ?
        ORDER BY score ASC, c.arxiv_id ASC, c.version ASC, c.chunk_index ASC
        LIMIT ?
        """,
        (fts_query, limit),
    )
    chunks: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        arxiv_id = str(row["arxiv_id"])
        version = _validate_version(row["version"])
        cache_key = (arxiv_id, version)
        if cache_key not in paper_cache:
            paper_cache[cache_key] = _paperstream_public_paper_for_version(conn, arxiv_id, version)
        paper = paper_cache[cache_key] or {
            "arxiv_id": arxiv_id,
            "latest_version": version,
            "title": None,
            "abstract": None,
            "authors": [],
            "primary_category": None,
            "categories": [],
            "abs_url": None,
        }
        text = str(row["text"] or "")
        chunks.append(
            {
                "arxiv_id": arxiv_id,
                "version": version,
                "chunk_index": int(row["chunk_index"]),
                "text": text[:max_chunk_chars],
                "score": float(row["score"]),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": paper.get("authors", []),
                "primary_category": paper.get("primary_category"),
                "categories": paper.get("categories", []),
                "abs_url": paper.get("abs_url"),
            }
        )
    return chunks

def search_paperstream(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    max_chunk_chars: int = 500,
    include_chunks: bool = True,
) -> dict[str, Any]:
    """Search PaperStream metadata and materialized/local BM25 chunk indexes with bounded public results."""

    normalized_query, terms = _validate_paperstream_search_query(query)
    limit = _validate_paperstream_search_limit(limit)
    max_chunk_chars = _validate_paperstream_max_chunk_chars(max_chunk_chars)
    if not isinstance(include_chunks, bool):
        raise TypeError("include_chunks must be a boolean")

    metadata_rows = _paperstream_metadata_rows_for_search(conn, terms, limit)
    papers = [
        _paperstream_public_paper(
            row,
            metadata_score=int(row.get("metadata_score") or 0),
            match_sources=_paperstream_match_sources(row, terms),
        )
        for row in metadata_rows
    ]
    errors: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    if include_chunks:
        fts_query = _paperstream_fts_query(terms)
        paper_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
        seen_chunks: set[tuple[str, int, int]] = set()
        for chunk in _paperstream_search_materialized_chunks(
            conn,
            fts_query,
            limit=limit,
            max_chunk_chars=max_chunk_chars,
            paper_cache=paper_cache,
        ):
            key = (chunk["arxiv_id"], chunk["version"], chunk["chunk_index"])
            seen_chunks.add(key)
            chunks.append(chunk)
        for asset in _paperstream_bm25_asset_rows(conn):
            try:
                for chunk in _paperstream_search_bm25_asset(
                    conn,
                    asset,
                    fts_query,
                    limit=limit,
                    max_chunk_chars=max_chunk_chars,
                    paper_cache=paper_cache,
                ):
                    key = (chunk["arxiv_id"], chunk["version"], chunk["chunk_index"])
                    if key in seen_chunks:
                        continue
                    seen_chunks.add(key)
                    chunks.append(chunk)
            except (OSError, sqlite3.Error) as exc:
                errors.append(_paperstream_bm25_error(asset, exc))
        chunks.sort(key=lambda item: (item["score"], item["arxiv_id"], item["version"], item["chunk_index"]))
        del chunks[limit:]
    return {
        "ok": True,
        "attribution": _arxiv_attribution(),
        "query": normalized_query,
        "limit": limit,
        "papers": papers,
        "chunks": chunks,
        "errors": errors,
    }


def _validate_category_filter(category: str | None) -> str | None:
    if category is None:
        return None
    if not isinstance(category, str):
        raise TypeError("category must be a string")
    category = category.strip()
    if not category:
        raise ValueError("category must not be empty")
    return category


def _category_filter_clause(category: str | None, params: list[Any]) -> str:
    if category is None:
        return ""
    params.extend([category, f"%{json.dumps(category, separators=(',', ':'))}%"])
    return "AND (p.primary_category = ? OR p.categories_json LIKE ?)"


def _validate_optional_job_status(status: str | None) -> str | None:
    if status is None:
        return None
    if not isinstance(status, str):
        raise TypeError("status must be a string")
    status = status.strip()
    if status not in JOB_STATUSES:
        raise ValueError(f"invalid job status: {status!r}")
    return status


def _validate_job_type_filter(job_types: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if job_types is None:
        return None
    return _validate_job_type_subset(
        job_types,
        allowed=set(JOB_TYPES),
        label="job_types",
        description=", ".join(repr(job_type) for job_type in sorted(JOB_TYPES)),
    )


_ARXIV_ATTRIBUTION_NOTICE = (
    "Data from arXiv.org.",
    "This project is not endorsed by arXiv.",
    "Users should retrieve e-print content from arXiv unless redistribution is licensed.",
)


def _arxiv_attribution() -> list[str]:
    return list(_ARXIV_ATTRIBUTION_NOTICE)


def _paper_links(arxiv_id: str) -> dict[str, str]:
    return {
        "arxiv_abs": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "arxiv_source": f"https://arxiv.org/e-print/{arxiv_id}",
    }

def _resolve_arxiv_latest_version(conn: sqlite3.Connection, arxiv_id: str) -> int | None:
    row = conn.execute(
        """
        SELECT latest_version
        FROM arxiv_papers
        WHERE arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    latest_row = conn.execute(
        """
        SELECT COALESCE(MAX(version), ?) AS latest_version
        FROM arxiv_versions
        WHERE arxiv_id = ?
        """,
        (row["latest_version"], arxiv_id),
    ).fetchone()
    return _validate_version(
        latest_row["latest_version"] if isinstance(latest_row, sqlite3.Row) else latest_row[0]
    )


def _arxiv_version_exists(conn: sqlite3.Connection, arxiv_id: str, version: int) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM arxiv_versions
        WHERE arxiv_id = ? AND version = ?
        LIMIT 1
        """,
        (arxiv_id, version),
    ).fetchone()
    if row is not None:
        return True
    paper_row = conn.execute(
        """
        SELECT latest_version
        FROM arxiv_papers
        WHERE arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()
    if paper_row is None:
        return False
    return _validate_version(paper_row["latest_version"]) == version


def _arxiv_chunk_counts(conn: sqlite3.Connection, arxiv_id: str) -> dict[str, Any]:
    if not _table_exists(conn, "arxiv_chunks"):
        return {
            "total": 0,
            "materialized_versions": [],
            "by_version": {},
        }
    rows = conn.execute(
        """
        SELECT version, COUNT(*) AS count
        FROM arxiv_chunks
        WHERE arxiv_id = ?
        GROUP BY version
        ORDER BY version ASC
        """,
        (arxiv_id,),
    ).fetchall()
    by_version = {str(_validate_version(row["version"])): int(row["count"] or 0) for row in rows}
    return {
        "total": sum(by_version.values()),
        "materialized_versions": [int(version) for version in by_version],
        "by_version": by_version,
    }


def _validate_paper_chunks_limit(limit: Any) -> int:
    limit = _coerce_positive_int(limit, "limit")
    if limit > _PAPERSTREAM_CHUNKS_MAX_LIMIT:
        raise ValueError(f"limit must be <= {_PAPERSTREAM_CHUNKS_MAX_LIMIT}")
    return limit


def _validate_paper_chunks_max_chars(max_chars: Any) -> int:
    max_chars = _coerce_nonnegative_int(max_chars, "max_chars")
    if max_chars > _PAPERSTREAM_CHUNKS_MAX_CHARS:
        raise ValueError(f"max_chars must be <= {_PAPERSTREAM_CHUNKS_MAX_CHARS}")
    return max_chars


def get_arxiv_paper_chunks(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    version: int | str | None = None,
    limit: int = 50,
    offset: int = 0,
    max_chars: int = 1000,
) -> dict[str, Any] | None:
    """Return bounded materialized chunks for one arXiv paper without local paths."""

    arxiv_id = _validate_arxiv_id(arxiv_id)
    version = _validate_version(version, required=False)
    limit = _validate_paper_chunks_limit(limit)
    offset = _coerce_nonnegative_int(offset, "offset")
    max_chars = _validate_paper_chunks_max_chars(max_chars)
    if version is None:
        version = _resolve_arxiv_latest_version(conn, arxiv_id)
    if version is None or not _arxiv_version_exists(conn, arxiv_id, version):
        return None

    total = 0
    chunks: list[dict[str, Any]] = []
    if version is not None and _table_exists(conn, "arxiv_chunks"):
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM arxiv_chunks
            WHERE arxiv_id = ? AND version = ?
            """,
            (arxiv_id, version),
        ).fetchone()
        total = int((total_row["count"] if isinstance(total_row, sqlite3.Row) else total_row[0]) or 0)
        cursor = conn.execute(
            """
            SELECT chunk_index, text, source_asset_type, chunks_asset_key, chunks_checksum
            FROM arxiv_chunks
            WHERE arxiv_id = ? AND version = ?
            ORDER BY chunk_index ASC
            LIMIT ? OFFSET ?
            """,
            (arxiv_id, version, limit, offset),
        )
        chunks = [
            {
                "chunk_index": int(row["chunk_index"]),
                "text": str(row["text"] or "")[:max_chars],
                "source_asset_type": row["source_asset_type"],
                "chunks_asset_key": row["chunks_asset_key"],
                "chunks_checksum": row["chunks_checksum"],
            }
            for row in cursor.fetchall()
        ]

    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "total": total,
        "limit": limit,
        "offset": offset,
        "chunks": chunks,
        "links": _paper_links(arxiv_id),
        "attribution": _arxiv_attribution(),
    }


def _validate_inspection_asset_type(asset_type: str) -> str:
    if not isinstance(asset_type, str):
        raise TypeError("asset_type must be a string")
    asset_type = asset_type.strip()
    if asset_type not in {"pdf", "source"}:
        raise ValueError("asset_type must be 'pdf' or 'source'")
    return asset_type


def get_arxiv_paper_state(conn: sqlite3.Connection, arxiv_id: str) -> dict[str, Any] | None:
    """Return JSON-ready PaperStream state for one arXiv paper without local paths."""

    arxiv_id = _validate_arxiv_id(arxiv_id)
    row = conn.execute(
        """
        SELECT arxiv_id, latest_version, title, abstract, categories_json, primary_category
        FROM arxiv_papers
        WHERE arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    latest_row = conn.execute(
        """
        SELECT COALESCE(MAX(version), ?) AS latest_version
        FROM arxiv_versions
        WHERE arxiv_id = ?
        """,
        (row["latest_version"], arxiv_id),
    ).fetchone()
    latest_version = _validate_version(
        latest_row["latest_version"] if isinstance(latest_row, sqlite3.Row) else latest_row[0]
    )
    version_row = conn.execute(
        """
        SELECT title, abstract, categories_json
        FROM arxiv_versions
        WHERE arxiv_id = ? AND version = ?
        """,
        (arxiv_id, latest_version),
    ).fetchone()
    latest_title = row["title"]
    latest_abstract = row["abstract"]
    categories_json = row["categories_json"]
    if version_row is not None:
        latest_title = version_row["title"] or latest_title
        latest_abstract = version_row["abstract"] or latest_abstract
        categories_json = version_row["categories_json"] or categories_json
    categories = [str(item) for item in _json_list(categories_json) if item is not None]
    assets: dict[str, dict[str, Any]] = {}
    for asset_row in conn.execute(
        """
        SELECT asset_type, asset_key, status, byte_size, source, error_message,
               first_seen_at, updated_at
        FROM arxiv_assets
        WHERE arxiv_id = ? AND version = ?
        ORDER BY asset_type, asset_key
        """,
        (arxiv_id, latest_version),
    ):
        assets[str(asset_row["asset_type"])] = {
            "asset_key": asset_row["asset_key"],
            "status": asset_row["status"],
            "byte_size": asset_row["byte_size"],
            "source": asset_row["source"],
            "error_message": asset_row["error_message"],
            "first_seen_at": asset_row["first_seen_at"],
            "updated_at": asset_row["updated_at"],
        }

    jobs = {"total": 0, "by_status": {}, "by_type": {}}
    job_statuses: dict[str, dict[str, int]] = {}
    for job_row in conn.execute(
        """
        SELECT job_type, status, COUNT(*) AS count
        FROM arxiv_jobs
        WHERE arxiv_id = ? AND version = ?
        GROUP BY job_type, status
        ORDER BY job_type, status
        """,
        (arxiv_id, latest_version),
    ):
        job_type = str(job_row["job_type"])
        status = str(job_row["status"])
        count = int(job_row["count"] or 0)
        jobs["total"] += count
        jobs["by_status"][status] = int(jobs["by_status"].get(status, 0)) + count
        jobs["by_type"][job_type] = int(jobs["by_type"].get(job_type, 0)) + count
        job_statuses.setdefault(job_type, {})[status] = count

    chunk_counts = _arxiv_chunk_counts(conn, arxiv_id)
    latest_version_chunk_count = int(chunk_counts["by_version"].get(str(latest_version), 0))
    bm25_asset = assets.get("bm25_index")

    return {
        "arxiv_id": arxiv_id,
        "latest_version": latest_version,
        "title": latest_title,
        "abstract": latest_abstract,
        "categories": categories,
        "primary_category": row["primary_category"],
        "assets": assets,
        "asset_statuses": {
            asset_type: asset_state["status"]
            for asset_type, asset_state in sorted(assets.items())
        },
        "chunks": {
            "total": chunk_counts["total"],
            "materialized_versions": chunk_counts["materialized_versions"],
            "latest_version_count": latest_version_chunk_count,
            "by_version": chunk_counts["by_version"],
        },
        "indexes": {
            "bm25_indexed": bm25_asset is not None and bm25_asset["status"] == "indexed",
            "materialized_chunks": latest_version_chunk_count > 0,
        },
        "jobs": jobs,
        "job_statuses": job_statuses,
        "links": _paper_links(arxiv_id),
        "attribution": _arxiv_attribution(),
    }


def _validate_paperstream_inspection_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer")
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    if limit > _PAPERSTREAM_INSPECTION_MAX_LIMIT:
        raise ValueError(f"limit must be <= {_PAPERSTREAM_INSPECTION_MAX_LIMIT}")
    return limit


def list_missing_arxiv_assets(
    conn: sqlite3.Connection,
    *,
    asset_type: str,
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List latest-version papers without a ready or in-progress asset row."""

    asset_type = _validate_inspection_asset_type(asset_type)
    category = _validate_category_filter(category)
    limit = _validate_paperstream_inspection_limit(limit)
    params: list[Any] = [asset_type]
    category_clause = _category_filter_clause(category, params)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT p.arxiv_id,
               COALESCE(MAX(v.version), p.latest_version) AS latest_version,
               COALESCE(v_latest.title, p.title) AS title,
               p.primary_category,
               COALESCE(v_latest.categories_json, p.categories_json) AS categories_json
        FROM arxiv_papers AS p
        LEFT JOIN arxiv_versions AS v ON v.arxiv_id = p.arxiv_id
        LEFT JOIN arxiv_versions AS v_latest
          ON v_latest.arxiv_id = p.arxiv_id
         AND v_latest.version = COALESCE((SELECT MAX(v3.version) FROM arxiv_versions AS v3 WHERE v3.arxiv_id = p.arxiv_id), p.latest_version)
        WHERE NOT EXISTS (
            SELECT 1
            FROM arxiv_assets AS a
            WHERE a.arxiv_id = p.arxiv_id
              AND a.version = COALESCE((SELECT MAX(v2.version) FROM arxiv_versions AS v2 WHERE v2.arxiv_id = p.arxiv_id), p.latest_version)
              AND a.asset_type = ?
              AND a.status IN ('queued', 'running', 'downloaded', 'imported', 'parsed', 'indexed')
        )
        {category_clause}
        GROUP BY p.arxiv_id
        HAVING latest_version IS NOT NULL
        ORDER BY p.arxiv_id
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "arxiv_id": row["arxiv_id"],
            "latest_version": _validate_version(row["latest_version"]),
            "title": row["title"],
            "primary_category": row["primary_category"],
            "categories": [str(item) for item in _json_list(row["categories_json"]) if item is not None],
            "asset_type": asset_type,
            "links": _paper_links(row["arxiv_id"]),
            "attribution": _arxiv_attribution(),
        }
        for row in rows
    ]


def list_arxiv_jobs(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    job_types: Iterable[str] | str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return bounded PaperStream queue rows without payload_json."""

    status = _validate_optional_job_status(status)
    job_type_values = _validate_job_type_filter(job_types)
    limit = _validate_paperstream_inspection_limit(limit)
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if job_type_values is not None:
        placeholders = ",".join("?" for _ in job_type_values)
        clauses.append(f"job_type IN ({placeholders})")
        params.extend(job_type_values)
    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT job_key, job_type, arxiv_id, version, asset_type, status,
               priority, attempts, max_attempts, run_after, locked_by, locked_at,
               error_message, created_at, updated_at
        FROM arxiv_jobs
        {where_clause}
        ORDER BY priority ASC, run_after ASC, created_at ASC, job_key ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "job_key": row["job_key"],
            "job_type": row["job_type"],
            "arxiv_id": row["arxiv_id"],
            "version": row["version"],
            "asset_type": row["asset_type"],
            "status": row["status"],
            "priority": row["priority"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "run_after": row["run_after"],
            "locked_by": row["locked_by"],
            "locked_at": row["locked_at"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def _paperstream_json_dump(payload: Any, stdout: TextIO) -> None:
    stdout.write(json.dumps(payload, sort_keys=True, indent=2))
    stdout.write("\n")


def _paperstream_positive_rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed >= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return parsed

def _paperstream_nonnegative_int(value: str) -> int:
    try:
        return _coerce_nonnegative_int(value, "value")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc


def _paperstream_positive_int(value: str) -> int:
    try:
        return _coerce_positive_int(value, "value")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc


def _paperstream_asset_type_list(value: str) -> tuple[str, ...]:
    try:
        return _validate_enqueue_asset_types(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list containing only pdf and source") from exc


def _paperstream_work_job_type_list(value: str) -> tuple[str, ...]:
    try:
        return _validate_work_job_types(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list containing only download_pdf, download_source, import_s3_bin, extract_s3_bin, parse_pdf, parse_source, chunk_text, and index_bm25"
        ) from exc


def _paperstream_asset_type(value: str) -> str:
    try:
        return _validate_inspection_asset_type(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be pdf or source") from exc


def _paperstream_job_status(value: str) -> str:
    try:
        status = _validate_optional_job_status(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must be one of queued, running, succeeded, failed, or skipped"
        ) from exc
    if status is None:
        raise argparse.ArgumentTypeError("must not be empty")
    return status


def _paperstream_job_type_filter(value: str) -> tuple[str, ...]:
    try:
        parsed = _validate_job_type_filter(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated list of supported PaperStream job types") from exc
    if parsed is None:
        raise argparse.ArgumentTypeError("must not be empty")
    return parsed


def _paperstream_positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from exc
    try:
        _validate_stale_after_seconds(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from exc
    return parsed

def _open_researchfs_connection(db_path: str | None) -> sqlite3.Connection:
    from researchfs.db import get_connection

    return get_connection(db_path)


def _search_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(
            search_paperstream(
                conn,
                args.query,
                limit=args.limit,
                max_chunk_chars=args.max_chunk_chars,
                include_chunks=not args.no_chunks,
            ),
            stdout,
        )
    finally:
        conn.close()
    return 0


def _status_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(get_paperstream_status(conn), stdout)
    finally:
        conn.close()
    return 0

def _paper_state_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(get_arxiv_paper_state(conn, args.arxiv_id), stdout)
    finally:
        conn.close()
    return 0


def _paper_chunks_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(
            get_arxiv_paper_chunks(
                conn,
                args.arxiv_id,
                version=args.version,
                limit=args.limit,
                offset=args.offset,
                max_chars=args.max_chars,
            ),
            stdout,
        )
    finally:
        conn.close()
    return 0


def _missing_assets_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(
            list_missing_arxiv_assets(
                conn,
                asset_type=args.asset_type,
                category=args.category,
                limit=args.limit,
            ),
            stdout,
        )
    finally:
        conn.close()
    return 0


def _jobs_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        _paperstream_json_dump(
            list_arxiv_jobs(
                conn,
                status=args.status,
                job_types=args.job_types,
                limit=args.limit,
            ),
            stdout,
        )
    finally:
        conn.close()
    return 0


def _requeue_stale_jobs_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        result = requeue_stale_arxiv_jobs(
            conn,
            stale_after_seconds=args.stale_after_seconds,
            job_types=args.job_types,
            limit=args.limit,
        )
        conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


def _retry_failed_jobs_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        result = retry_failed_arxiv_jobs(
            conn,
            job_types=args.job_types,
            limit=args.limit,
        )
        conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


def _rebuild_indexes_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        result = rebuild_paperstream_bm25_indexes(
            conn,
            args.object_root,
            limit=args.limit,
            arxiv_id=args.arxiv_id,
            version=args.version,
            force=args.force,
        )
        conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0



def _read_arxiv_s3_manifest_records(input_path: str) -> list[dict[str, Any]]:
    text = Path(input_path).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("S3 manifest input must not be empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            records.append(item)
        return records
    if isinstance(parsed, dict):
        if "bins" not in parsed:
            return [parsed]
        parsed = parsed["bins"]
    if not isinstance(parsed, list):
        raise ValueError("S3 manifest input must be a JSON array, an object with a bins array, or JSONL")
    return parsed


def _import_s3_manifest_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        records = _read_arxiv_s3_manifest_records(args.input)
        result = import_arxiv_s3_manifest(conn, records, dry_run=args.dry_run)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception as exc:
        conn.rollback()
        if not args.dry_run:
            append_arxiv_event(
                conn,
                "s3_manifest_import_failed",
                payload=_s3_manifest_import_failure_payload(exc, input_path=args.input),
            )
            conn.commit()
        raise
    finally:
        conn.close()
    return 0


def _read_arxiv_bin_member_records(input_path: str) -> tuple[list[dict[str, Any]], str | None]:
    text = Path(input_path).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("S3 bin member input must not be empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            records.append(item)
        return records, None
    if isinstance(parsed, dict):
        if "members" in parsed:
            members = parsed["members"]
            if not isinstance(members, list):
                raise ValueError("S3 bin member input members field must be a JSON array")
            default_bin_key = parsed.get("bin_key")
            return members, default_bin_key
        return [parsed], None
    if not isinstance(parsed, list):
        raise ValueError("S3 bin member input must be a JSON array, an object with a members array, or JSONL")
    return parsed, None


def _import_s3_members_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        records, input_bin_key = _read_arxiv_bin_member_records(args.input)
        result = import_arxiv_bin_members(
            conn,
            records,
            bin_key=args.bin_key if args.bin_key is not None else input_bin_key,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0



def _asset_keys_for_bloom(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "arxiv_assets"):
        return []
    rows = conn.execute(
        """
        SELECT asset_key
        FROM arxiv_assets
        WHERE asset_key IS NOT NULL AND asset_key <> ''
        ORDER BY asset_key
        """
    ).fetchall()
    return [str(row["asset_key"] if isinstance(row, sqlite3.Row) else row[0]) for row in rows]

def _metadata_version_keys_for_bloom(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "arxiv_versions"):
        return []
    rows = conn.execute(
        """
        SELECT arxiv_id, version
        FROM arxiv_versions
        WHERE arxiv_id IS NOT NULL
          AND arxiv_id <> ''
          AND version IS NOT NULL
        ORDER BY arxiv_id, version
        """
    ).fetchall()
    return [
        _metadata_version_key(
            row["arxiv_id"] if isinstance(row, sqlite3.Row) else row[0],
            row["version"] if isinstance(row, sqlite3.Row) else row[1],
        )
        for row in rows
    ]


def _rebuild_bloom_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        if args.filter_name == "seen_metadata_versions":
            bloom_filter = _rebuild_metadata_versions_bloom_filter(
                conn,
                false_positive_rate=args.false_positive_rate,
            )
            source_table = "arxiv_versions"
            source_column = "version"
        else:
            keys = _asset_keys_for_bloom(conn)
            source_table = "arxiv_assets"
            source_column = "asset_key"
            bloom_filter = rebuild_bloom_filter(
                conn,
                args.filter_name,
                keys,
                false_positive_rate=args.false_positive_rate,
                source_table=source_table,
                source_column=source_column,
            )
        conn.commit()
        _paperstream_json_dump(
            {
                "filter_name": args.filter_name,
                "source_table": source_table,
                "source_column": source_column,
                "item_count": bloom_filter.item_count,
                "false_positive_rate": bloom_filter.false_positive_rate,
                "bit_count": bloom_filter.bit_count,
                "hash_count": bloom_filter.hash_count,
            },
            stdout,
        )
    finally:
        conn.close()
    return 0


def _read_metadata_sync_pages(input_path: str | None) -> list[str]:
    if input_path is None:
        raise ValueError("sync-metadata requires --input for fixture-driven sync")
    return [Path(input_path).expanduser().read_text(encoding="utf-8")]


def _metadata_sync_pages(args: argparse.Namespace, conn: sqlite3.Connection) -> Iterable[str]:
    if args.input:
        return _read_metadata_sync_pages(args.input)

    iter_pages = globals().get("iter_oai_metadata_pages")
    if not callable(iter_pages):
        raise RuntimeError("iter_oai_metadata_pages is not available in researchfs.paperstream")
    from_date = args.from_date
    if from_date is None:
        from_date = _sync_state_row(conn, args.sync_name).get("committed_from")
    if from_date is None and getattr(args, "no_catch_up", False):
        from_date = datetime.now(timezone.utc).date().isoformat()
    return iter_pages(
        metadata_prefix=args.metadata_prefix,
        set_spec=args.set_spec,
        from_date=from_date,
        until_date=args.until_date,
        max_pages=args.max_pages,
        rate_limit_seconds=args.rate_limit_seconds,
    )


def _sync_metadata_command(args: argparse.Namespace, stdout: TextIO) -> int:
    enqueue_asset_types = _coerce_sync_enqueue_asset_types(getattr(args, "enqueue_assets", False))
    if getattr(args, "metadata_only", False) and enqueue_asset_types:
        raise ValueError("sync-metadata --metadata-only cannot be combined with --enqueue-assets")
    if not args.dry_run and args.max_pages is None:
        raise ValueError("sync-metadata non-dry-run requires --max-pages to avoid an unbounded harvest")
    sync_metadata = globals().get("sync_paperstream_metadata")
    if not callable(sync_metadata):
        raise RuntimeError("sync_paperstream_metadata is not available in researchfs.paperstream")
    conn = _open_researchfs_connection(args.db_path)
    try:
        pages = _metadata_sync_pages(args, conn)
        result = sync_metadata(
            conn,
            pages,
            sync_name=args.sync_name,
            dry_run=args.dry_run,
            max_pages=args.max_pages,
            max_records=args.max_records,
            enqueue_assets=getattr(args, "enqueue_assets", False),
            metadata_prefix=args.metadata_prefix,
            set_spec=args.set_spec,
        )
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0

def _enqueue_assets_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    try:
        result = enqueue_missing_arxiv_assets(
            conn,
            asset_types=args.asset_types,
            category=args.category,
            limit=args.limit,
            priority=args.priority,
            max_attempts=args.max_attempts,
        )
        conn.commit()
        _paperstream_json_dump(result, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0



def _next_queued_arxiv_job_type(conn: sqlite3.Connection, job_types: Iterable[str] | str) -> str | None:
    job_type_values = _validate_work_job_types(job_types)
    placeholders = ",".join("?" for _ in job_type_values)
    row = conn.execute(
        f"""
        SELECT job_type
        FROM arxiv_jobs
        WHERE status = 'queued'
          AND (run_after IS NULL OR run_after <= CURRENT_TIMESTAMP)
          AND attempts < max_attempts
          AND job_type IN ({placeholders})
        ORDER BY priority ASC, run_after ASC, created_at ASC, job_key ASC
        LIMIT 1
        """,
        job_type_values,
    ).fetchone()
    if row is None:
        return None
    return str(row["job_type"] if isinstance(row, sqlite3.Row) else row[0])

def _utc_day_bounds(now: datetime | str | None = None) -> tuple[str, str]:
    now_dt = _paperstream_utc_datetime(now)
    start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return _utc_timestamp(start), _utc_timestamp(end)


def count_arxiv_download_attempts_today(
    conn: sqlite3.Connection,
    *,
    now: datetime | str | None = None,
) -> int:
    """Count durable arXiv download requests attempted during the current UTC day."""

    start, end = _utc_day_bounds(now)
    row = conn.execute(
        """
        SELECT
            (
                SELECT COUNT(*)
                FROM arxiv_events
                WHERE event_type = 'asset_download_attempted'
                  AND datetime(created_at) >= datetime(?)
                  AND datetime(created_at) < datetime(?)
            )
            +
            (
                SELECT COUNT(*)
                FROM arxiv_events legacy
                WHERE legacy.event_type IN ('asset_downloaded', 'asset_failed')
                  AND datetime(legacy.created_at) >= datetime(?)
                  AND datetime(legacy.created_at) < datetime(?)
                  AND (
                      legacy.event_type != 'asset_failed'
                      OR substr(legacy.job_key, 1, 13) = 'download_pdf:'
                      OR substr(legacy.job_key, 1, 16) = 'download_source:'
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM arxiv_events attempted
                      WHERE attempted.event_type = 'asset_download_attempted'
                        AND attempted.job_key = legacy.job_key
                  )
            )
        """,
        (start, end, start, end),
    ).fetchone()
    return int(row[0] if row is not None else 0)


def _daily_download_cap_stop_payload(count: int, cap: int) -> dict[str, Any]:
    return {
        "worked": False,
        "stop_reason": "daily_download_cap",
        "daily_download_count": count,
        "daily_download_cap": cap,
    }



def _work_next_paperstream_job(
    conn: sqlite3.Connection,
    *,
    worker_id: str,
    object_root: str | Path,
    job_types: Iterable[str] | str,
    rate_limit_seconds: float,
    chunk_size: int,
    overlap: int,
    daily_download_cap: int | None = None,
) -> dict[str, Any]:
    job_type_values = _validate_work_job_types(job_types)
    cap = (
        None
        if daily_download_cap is None
        else _coerce_nonnegative_int(daily_download_cap, "daily_download_cap")
    )
    if any(job_type in {"download_pdf", "download_source"} for job_type in job_type_values):
        if cap is None:
            raise ValueError(
                "--daily-download-cap (daily_download_cap) is required when work job_types include download_pdf or download_source"
            )
        rate_limit_seconds = _validate_oai_rate_limit_seconds(rate_limit_seconds)
    if "chunk_text" in job_type_values:
        chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    job_type = _next_queued_arxiv_job_type(conn, job_type_values)
    if job_type is None:
        return {"worked": False}
    if job_type in {"download_pdf", "download_source"} and cap is not None:
        download_count = count_arxiv_download_attempts_today(conn)
        if download_count >= cap:
            return _daily_download_cap_stop_payload(download_count, cap)
    if job_type in {"download_pdf", "download_source"}:
        return work_next_arxiv_download_job(
            conn,
            worker_id=worker_id,
            object_root=object_root,
            job_types=(job_type,),
            rate_limit_seconds=rate_limit_seconds,
        )
    if job_type == "import_s3_bin":
        return work_next_arxiv_s3_bin_import_job(
            conn,
            worker_id=worker_id,
            object_root=object_root,
            job_types=(job_type,),
            rate_limit_seconds=rate_limit_seconds,
        )
    if job_type == "extract_s3_bin":
        return work_next_arxiv_s3_bin_extract_job(
            conn,
            worker_id=worker_id,
            object_root=object_root,
            job_types=(job_type,),
        )
    if job_type == "chunk_text":
        return work_next_arxiv_chunk_job(
            conn,
            worker_id=worker_id,
            object_root=object_root,
            chunk_size=chunk_size,
            overlap=overlap,
            job_types=(job_type,),
        )
    if job_type == "index_bm25":
        return work_next_arxiv_bm25_job(
            conn,
            worker_id=worker_id,
            object_root=object_root,
            job_types=(job_type,),
        )
    return work_next_arxiv_parse_job(
        conn,
        worker_id=worker_id,
        object_root=object_root,
        job_types=(job_type,),
    )

def _work_command(args: argparse.Namespace, stdout: TextIO) -> int:
    conn = _open_researchfs_connection(args.db_path)
    results: list[dict[str, Any]] = []
    jobs_attempted = 0
    stopped: dict[str, Any] | None = None
    try:
        for _ in range(args.max_jobs):
            result = _work_next_paperstream_job(
                conn,
                worker_id=args.worker_id,
                object_root=args.object_root,
                job_types=args.job_types,
                rate_limit_seconds=args.rate_limit_seconds,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                daily_download_cap=getattr(args, "daily_download_cap", None),
            )
            if not result.get("worked"):
                stopped = result if result.get("stop_reason") else None
                break
            jobs_attempted += 1
            results.append(result)
            conn.commit()
        payload = {
            "ok": all(result.get("ok") is not False for result in results),
            "jobs_attempted": jobs_attempted,
            "jobs_succeeded": sum(1 for result in results if result.get("ok") is True),
            "jobs_failed": sum(1 for result in results if result.get("ok") is False),
            "jobs_skipped": sum(1 for result in results if result.get("skipped") is True),
            "results": results,
        }
        if stopped is not None:
            payload.update(
                {
                    "stop_reason": stopped["stop_reason"],
                    "daily_download_count": stopped["daily_download_count"],
                    "daily_download_cap": stopped["daily_download_cap"],
                }
            )
        _paperstream_json_dump(payload, stdout)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchfs.paperstream",
        description="Inspect and maintain local PaperStream metadata state.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite DB path (defaults to RESEARCHFS_DB_PATH or the project researchfs.db)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser(
        "search",
        help="Search PaperStream metadata and materialized/local BM25 chunk indexes as JSON",
    )
    search.add_argument("--query", required=True, help="Search query text")
    search.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=10,
        help=f"Maximum paper and chunk rows to return, capped at {_PAPERSTREAM_SEARCH_MAX_LIMIT} (default: 10)",
    )
    search.add_argument(
        "--max-chunk-chars",
        type=_paperstream_nonnegative_int,
        default=500,
        help=f"Maximum characters to include per chunk hit, capped at {_PAPERSTREAM_SEARCH_MAX_CHUNK_CHARS} (default: 500)",
    )
    search.add_argument(
        "--no-chunks",
        action="store_true",
        help="Search metadata only; do not open local BM25 indexes",
    )
    search.set_defaults(func=_search_command)


    status = subparsers.add_parser("status", help="Print PaperStream status as JSON")
    status.set_defaults(func=_status_command)

    paper_state = subparsers.add_parser(
        "paper-state",
        help="Print one arXiv paper's PaperStream state as JSON",
    )
    paper_state.add_argument("arxiv_id", help="arXiv id without a version suffix")
    paper_state.set_defaults(func=_paper_state_command)

    paper_chunks = subparsers.add_parser(
        "paper-chunks",
        help="Print bounded materialized chunks for one arXiv paper as JSON",
    )
    paper_chunks.add_argument("arxiv_id", help="arXiv id without a version suffix")
    paper_chunks.add_argument(
        "--version",
        type=_paperstream_positive_int,
        default=None,
        help="Optional arXiv version (defaults to latest version)",
    )
    paper_chunks.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=50,
        help="Maximum chunks to return (default: 50)",
    )
    paper_chunks.add_argument(
        "--offset",
        type=_paperstream_nonnegative_int,
        default=0,
        help="Number of chunks to skip (default: 0)",
    )
    paper_chunks.add_argument(
        "--max-chars",
        type=_paperstream_nonnegative_int,
        default=1000,
        help="Maximum characters to include per chunk (default: 1000)",
    )
    paper_chunks.set_defaults(func=_paper_chunks_command)

    missing_assets = subparsers.add_parser(
        "missing-assets",
        help="List latest-version arXiv papers missing a ready or queued asset",
    )
    missing_assets.add_argument(
        "--asset-type",
        required=True,
        type=_paperstream_asset_type,
        help="Asset type to inspect (for example: pdf or source)",
    )
    missing_assets.add_argument(
        "--category",
        default=None,
        help="Optional arXiv category filter matched against primary/category metadata",
    )
    missing_assets.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=100,
        help=f"Maximum missing asset rows to return, capped at {_PAPERSTREAM_INSPECTION_MAX_LIMIT} (default: 100)",
    )
    missing_assets.set_defaults(func=_missing_assets_command)

    jobs = subparsers.add_parser(
        "jobs",
        help="List bounded PaperStream queue rows as JSON",
    )
    jobs.add_argument(
        "--status",
        type=_paperstream_job_status,
        default=None,
        help="Optional job status filter: queued, running, succeeded, failed, or skipped",
    )
    jobs.add_argument(
        "--job-types",
        type=_paperstream_job_type_filter,
        default=None,
        help="Optional comma-separated PaperStream job type filter",
    )
    jobs.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=100,
        help=f"Maximum job rows to return, capped at {_PAPERSTREAM_INSPECTION_MAX_LIMIT} (default: 100)",
    )
    jobs.set_defaults(func=_jobs_command)

    requeue_stale_jobs = subparsers.add_parser(
        "requeue-stale-jobs",
        help="Explicitly requeue running PaperStream jobs with stale locks",
    )
    requeue_stale_jobs.add_argument(
        "--stale-after-seconds",
        required=True,
        type=_paperstream_positive_seconds,
        help="Requeue running locks older than this many seconds",
    )
    requeue_stale_jobs.add_argument(
        "--job-types",
        type=_paperstream_job_type_filter,
        default=None,
        help="Optional comma-separated PaperStream job type filter",
    )
    requeue_stale_jobs.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=100,
        help=f"Maximum stale running jobs to requeue, capped at {_PAPERSTREAM_REQUEUE_MAX_LIMIT} (default: 100)",
    )
    requeue_stale_jobs.set_defaults(func=_requeue_stale_jobs_command)

    retry_failed_jobs = subparsers.add_parser(
        "retry-failed-jobs",
        help="Explicitly retry terminal failed PaperStream jobs",
    )
    retry_failed_jobs.add_argument(
        "--job-types",
        type=_paperstream_job_type_filter,
        default=None,
        help="Optional comma-separated PaperStream job type filter",
    )
    retry_failed_jobs.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=100,
        help=f"Maximum failed jobs to retry, capped at {_PAPERSTREAM_REQUEUE_MAX_LIMIT} (default: 100)",
    )
    retry_failed_jobs.set_defaults(func=_retry_failed_jobs_command)

    import_s3_manifest = subparsers.add_parser(
        "import-s3-manifest",
        help="Import offline arXiv S3 bin manifest metadata as JSON",
    )
    import_s3_manifest.add_argument(
        "--input",
        required=True,
        help="Path to a JSON array, object with bins array, or JSONL S3 bin manifest",
    )
    import_s3_manifest.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned S3 bin metadata changes without mutating the DB",
    )
    import_s3_manifest.set_defaults(func=_import_s3_manifest_command)

    import_s3_members = subparsers.add_parser(
        "import-s3-members",
        help="Import offline arXiv S3 bin member rows as JSON",
    )
    import_s3_members.add_argument(
        "--input",
        required=True,
        help="Path to a JSON array, object with members array, or JSONL S3 bin members fixture",
    )
    import_s3_members.add_argument(
        "--bin-key",
        default=None,
        help="Default S3 bin key for member records that omit bin_key",
    )
    import_s3_members.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned S3 bin member changes without mutating the DB",
    )
    import_s3_members.set_defaults(func=_import_s3_members_command)

    rebuild_bloom = subparsers.add_parser(
        "rebuild-bloom",
        help="Rebuild a Bloom filter from arxiv_assets.asset_key",
    )
    rebuild_bloom.add_argument(
        "--filter-name",
        "--filter",
        dest="filter_name",
        default="asset_key",
        help="Persisted Bloom filter name (default: asset_key)",
    )
    rebuild_bloom.add_argument(
        "--false-positive-rate",
        type=_paperstream_positive_rate,
        default=0.01,
        help="Target false-positive rate between 0 and 1 (default: 0.01)",
    )
    rebuild_bloom.set_defaults(func=_rebuild_bloom_command)

    rebuild_indexes = subparsers.add_parser(
        "rebuild-indexes",
        help="Materialize chunks and rebuild missing or stale PaperStream BM25 indexes",
    )
    rebuild_indexes.add_argument(
        "--limit",
        type=_paperstream_positive_int,
        default=100,
        help=f"Maximum chunks assets to scan, capped at {_PAPERSTREAM_REBUILD_MAX_LIMIT} (default: 100)",
    )
    rebuild_indexes.add_argument(
        "--arxiv-id",
        default=None,
        help="Optional arXiv id without a version suffix",
    )
    rebuild_indexes.add_argument(
        "--version",
        type=_paperstream_positive_int,
        default=None,
        help="Optional arXiv version filter",
    )
    rebuild_indexes.add_argument(
        "--force",
        action="store_true",
        help="Rebuild materialized rows and BM25 files even when current artifacts exist",
    )
    rebuild_indexes.add_argument(
        "--object-root",
        default=os.environ.get("RESEARCHFS_OBJECT_ROOT", "object-store"),
        help="Local object root for rebuilt BM25 assets (default: RESEARCHFS_OBJECT_ROOT or ./object-store)",
    )
    rebuild_indexes.set_defaults(func=_rebuild_indexes_command)

    sync_metadata = subparsers.add_parser(
        "sync-metadata",
        help="Sync PaperStream metadata from a fixture or bounded OAI-PMH harvest",
    )
    sync_metadata.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report metadata changes without committing them",
    )
    sync_metadata.add_argument(
        "--input",
        help="Path to an OAI ListRecords XML fixture; when supplied no network requests are made",
    )
    sync_metadata.add_argument(
        "--sync-name",
        default="default",
        help="Sync state name passed to the metadata sync API (default: default)",
    )
    sync_metadata.add_argument(
        "--metadata-prefix",
        default="arXivRaw",
        help="OAI metadataPrefix for live requests (default: arXivRaw)",
    )
    sync_metadata.add_argument(
        "--set",
        dest="set_spec",
        default=None,
        help="Optional OAI set filter for live requests",
    )
    sync_metadata.add_argument(
        "--from",
        dest="from_date",
        default=None,
        help="Optional OAI from date for live requests",
    )
    sync_metadata.add_argument(
        "--until",
        dest="until_date",
        default=None,
        help="Optional OAI until date for live requests",
    )
    sync_metadata.add_argument("--max-pages", type=int, default=None, help="Maximum pages to process")
    sync_metadata.add_argument("--max-records", type=int, default=None, help="Maximum records to process")
    sync_metadata.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait before each live OAI request (default: 3.0)",
    )
    sync_metadata.add_argument(
        "--metadata-only",
        action="store_true",
        help="Explicitly keep metadata sync from enqueuing follow-up asset jobs (default)",
    )
    sync_metadata.add_argument(
        "--no-catch-up",
        action="store_true",
        help="When no --from or committed cursor exists, start live OAI sync from the current UTC date",
    )
    sync_metadata.add_argument(
        "--enqueue-assets",
        nargs="?",
        const="pdf,source",
        default=False,
        type=_paperstream_asset_type_list,
        help="Allow the sync API to enqueue follow-up asset jobs for pdf, source, or pdf,source",
    )
    sync_metadata.set_defaults(func=_sync_metadata_command)

    enqueue_assets = subparsers.add_parser(
        "enqueue-assets",
        help="Queue missing PDF/source asset download jobs from synced metadata",
    )
    enqueue_assets.add_argument(
        "--asset-types",
        type=_paperstream_asset_type_list,
        default=("pdf", "source"),
        help="Comma-separated asset types to queue: pdf, source, or pdf,source (default: pdf,source)",
    )
    enqueue_assets.add_argument(
        "--category",
        default=None,
        help="Optional arXiv category filter matched against primary/category metadata",
    )
    enqueue_assets.add_argument(
        "--limit",
        type=_paperstream_nonnegative_int,
        default=100,
        help="Maximum metadata papers to scan (default: 100)",
    )
    enqueue_assets.add_argument(
        "--priority",
        type=_paperstream_nonnegative_int,
        default=100,
        help="Queued job priority, lower runs first (default: 100)",
    )
    enqueue_assets.add_argument(
        "--max-attempts",
        type=_paperstream_positive_int,
        default=5,
        help="Maximum download job attempts (default: 5)",
    )
    enqueue_assets.set_defaults(func=_enqueue_assets_command)

    work = subparsers.add_parser(
        "work",
        help="Run queued PaperStream download, S3 bin import/extract, parse, chunk, or BM25 index jobs",
    )
    work.add_argument(
        "--job-types",
        type=_paperstream_work_job_type_list,
        default=PAPERSTREAM_SAFE_WORK_JOB_TYPES,
        help="Comma-separated job types to run: download_pdf, download_source, import_s3_bin, extract_s3_bin, parse_pdf, parse_source, chunk_text, index_bm25 (default: parse_pdf,parse_source,chunk_text,index_bm25)",
    )
    work.add_argument(
        "--max-jobs",
        type=_paperstream_positive_int,
        default=10,
        help="Maximum jobs to attempt (default: 10)",
    )
    work.add_argument(
        "--worker-id",
        default="paperstream-worker",
        help="Worker id recorded on claimed jobs (default: paperstream-worker)",
    )
    work.add_argument(
        "--object-root",
        default=os.environ.get("RESEARCHFS_OBJECT_ROOT", "object-store"),
        help="Local object root for downloaded assets (default: RESEARCHFS_OBJECT_ROOT or ./object-store)",
    )
    work.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait before each live arXiv download request (default: 3.0)",
    )
    work.add_argument(
        "--daily-download-cap",
        type=_paperstream_nonnegative_int,
        default=None,
        help="Stop before claiming a download job after this many UTC-day download attempts (default: unlimited)",
    )
    work.add_argument(
        "--chunk-size",
        type=_paperstream_positive_int,
        default=2000,
        help="Maximum text characters per chunk_text chunk (default: 2000)",
    )
    work.add_argument(
        "--overlap",
        type=_paperstream_nonnegative_int,
        default=200,
        help="Overlapping text characters between chunk_text chunks (default: 200)",
    )
    work.set_defaults(func=_work_command)

    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.func(args, stdout) or 0)
    except (OSError, sqlite3.Error, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSET_STATUSES",
    "ASSET_TYPES",
    "BloomFilter",
    "bloom_db_contains",
    "JOB_STATUSES",
    "S3_BIN_STATUSES",
    "JOB_TYPES",
    "PAPERSTREAM_SAFE_WORK_JOB_TYPES",
    "asset_key",
    "append_arxiv_event",
    "arxiv_asset_key",
    "arxiv_job_key",
    "job_key",
    "arxiv_storage_path",
    "claim_next_arxiv_job",
    "enqueue_arxiv_job",
    "enqueue_arxiv_s3_bin_import_job",
    "enqueue_missing_arxiv_assets",
    "get_paperstream_status",
    "import_arxiv_bin_members",
    "import_arxiv_s3_manifest",
    "get_arxiv_paper_chunks",
    "get_arxiv_paper_state",
    "list_arxiv_jobs",
    "list_missing_arxiv_assets",
    "load_bloom_filter",
    "mark_arxiv_job_failed",
    "mark_arxiv_job_succeeded",
    "object_path",
    "materialize_paperstream_chunks",
    "rebuild_bloom_filter",
    "rebuild_paperstream_bm25_indexes",
    "requeue_stale_arxiv_jobs",
    "retry_failed_arxiv_jobs",
    "parse_arxiv_raw_record",
    "s3_bin_key",
    "s3_bin_storage_path",
    "parse_oai_list_records",
    "search_paperstream",
    "sync_paperstream_metadata",
    "save_bloom_filter",
    "upsert_arxiv_asset",
    "upsert_arxiv_s3_bin",
    "upsert_arxiv_bin_member",
    "work_next_arxiv_chunk_job",
    "work_next_arxiv_bm25_job",
    "work_next_arxiv_download_job",
    "work_next_arxiv_s3_bin_import_job",
    "work_next_arxiv_s3_bin_extract_job",
    "work_next_arxiv_parse_job",
    "upsert_paperstream_metadata_record",
]
