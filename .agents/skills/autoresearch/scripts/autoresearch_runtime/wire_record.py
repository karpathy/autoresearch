"""Append-only log of HTTP requests as seen by process hooks (post-adapter body)."""

from __future__ import annotations

import json
import os
import re
import hashlib
import uuid
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_ALLOWLIST = frozenset(
    name.lower()
    for name in (
        "content-type",
        "x-openrouter-metadata",
        "http-referer",
        "x-title",
    )
)
_SECRET_QUERY_KEYS = frozenset({
    "api_key", "apikey", "x_api_key", "key", "token", "access_token", "refresh_token",
    "auth", "authorization", "secret", "client_secret", "password", "signature", "sig",
})
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[-_]?key|access[-_]?token|refresh[-_]?token|password|secret|token)\b[\"']?\s*[:=]\s*[\"']?)([^\s,;&\"']+)"
)
_AUTHORIZATION_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:proxy-authorization|authorization)\b[\"']?\s*[:=]\s*[\"']?)[^\"'?,;&]+"
)
_TOKEN_PATH_SEGMENT = re.compile(r"(?i)^(?:sk[-_]|rk[-_]|gh[pousr]_|github_pat_|hf_|xox[baprs]-)[A-Za-z0-9_-]{8,}$")


def _redact_text(value: str) -> str:
    value = _AUTHORIZATION_ASSIGNMENT.sub(r"\1***", value)
    return _SECRET_ASSIGNMENT.sub(r"\1***", value)


def redact_url(url: str) -> str:
    """Remove URL credentials, secret query values, fragments, and known token path segments."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = parsed.port
        netloc = hostname + (f":{port}" if port is not None else "")
        query = [
            (key, "***" if key.casefold().replace("-", "_") in _SECRET_QUERY_KEYS else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        segments = parsed.path.split("/")
        for index, segment in enumerate(segments):
            if segment.casefold().replace("-", "_") in _SECRET_QUERY_KEYS and index + 1 < len(segments):
                segments[index + 1] = "***"
            elif _TOKEN_PATH_SEGMENT.fullmatch(segment):
                segments[index] = "***"
        path = "/".join(segments)
        return urlunsplit((parsed.scheme, netloc, path, urlencode(query, doseq=True), ""))
    except (TypeError, ValueError):
        return "<invalid-url>"


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in {"authorization", "proxy-authorization"}:
            redacted[key] = "***"
        elif lowered in _ALLOWLIST:
            redacted[key] = redact_url(value) if lowered == "http-referer" else _redact_text(str(value))
    return redacted


def _body_text(body: bytes | str | None) -> str | None:
    if body is None:
        return None
    if isinstance(body, str):
        return body
    return body.decode("utf-8", errors="replace")


def _body_bytes(body: bytes | str | None) -> bytes | None:
    if body is None:
        return None
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    return None


def record_sent_request(
    *,
    url: str,
    method: str,
    body: bytes | str | None,
    headers: Mapping[str, str],
    status: int | None,
    response_body: bytes | str | None,
    attempt: int,
) -> None:
    """Write safe metadata by default; body capture requires an explicit environment opt-in."""
    raw = os.environ.get("AUTORESEARCH_WIRE_LOG", "").strip()
    if not raw:
        return
    path = Path(raw)
    request_bytes = _body_bytes(body)
    response_bytes = _body_bytes(response_body)
    capture_body = os.environ.get("AUTORESEARCH_WIRE_LOG_BODY") == "1"
    row = {
        "capture_id": uuid.uuid4().hex,
        "url": redact_url(url),
        "method": str(method),
        "headers": redact_headers(headers),
        "status": status,
        "attempt": attempt,
        "body_bytes": len(request_bytes) if request_bytes is not None else None,
        "response_body_bytes": len(response_bytes) if response_bytes is not None else None,
        "response_body_sha256": hashlib.sha256(response_bytes).hexdigest() if response_bytes is not None else None,
        "body_captured": bool(capture_body and request_bytes is not None),
        "response_body_captured": bool(capture_body and response_bytes is not None),
    }
    if capture_body:
        request_text = _body_text(body)
        response_text = _body_text(response_body)
        if request_text is not None:
            row["body"] = request_text
        if response_text is not None:
            row["response_body"] = response_text
    line = json.dumps(row, ensure_ascii=False)
    # Evaluators may read this journal themselves to reconcile a response with
    # its case. The parent runtime imports it into SQLite after evaluation.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
