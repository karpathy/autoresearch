"""Record provider HTTP traffic and durable attempt outcomes in evaluator processes.

The runtime journals an attempt before transport and completes it after a
response. It never retries provider requests because a transport failure may
have an unknown outcome.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from urllib.error import HTTPError


_OWN = Path(__file__).resolve()
_RATE_LIMIT_STATUSES = {429, 503}
_STATE = None  # Kept for callers of the old hook installer signature.


def _chain_next() -> None:
    """Load a later sitecustomize so this hook does not hide the environment."""
    for entry in list(sys.path):
        candidate = Path(entry) / "sitecustomize.py"
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not candidate.is_file() or resolved == _OWN:
            continue
        spec = importlib.util.spec_from_file_location("_autoresearch_next_sitecustomize", resolved)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


def _wire_enabled() -> bool:
    return bool(os.environ.get("AUTORESEARCH_WIRE_LOG", "").strip())


def _attempts_enabled() -> bool:
    return bool(os.environ.get("AUTORESEARCH_ATTEMPTS_LOG", "").strip())


def _stats_path() -> Path | None:
    raw = os.environ.get("AUTORESEARCH_NET_STATS", "").strip()
    return Path(raw) if raw else None


def _case_scope_required() -> bool:
    return os.environ.get("AUTORESEARCH_CASE_SCOPE_REQUIRED") == "1"


def _network_guard_enabled() -> bool:
    return _case_scope_required() or os.environ.get("AUTORESEARCH_RESUME_OFFLINE") == "1"


def _begin_attempt():
    """Durably record a scoped attempt before sending it; unknown outcomes stay open."""
    if os.environ.get("AUTORESEARCH_RESUME_OFFLINE") == "1":
        raise RuntimeError("HTTP is disabled while resuming offline")
    case_scope_required = _case_scope_required()
    raw = os.environ.get("AUTORESEARCH_ATTEMPTS_LOG", "").strip()
    if not raw:
        if case_scope_required:
            raise RuntimeError("Case-scoped HTTP attempt journal is unavailable")
        return None
    scripts = Path(__file__).resolve().parents[1]
    if str(scripts) not in sys.path:
        sys.path.append(str(scripts))
    from autoresearch_runtime.case_checkpoint import append_attempt, completed_ids, current_case_id

    case_id = current_case_id()
    if case_scope_required and (not isinstance(case_id, str) or not case_id.strip()):
        raise RuntimeError("HTTP request requires an active case scope")
    attempt_id = uuid.uuid4().hex
    record = {
        "event": "started",
        "attempt_id": attempt_id,
        "case_id": case_id,
    }
    resume_id = os.environ.get("AUTORESEARCH_RESUME_ID", "").strip()
    if resume_id:
        if not isinstance(case_id, str) or not case_id.strip():
            raise RuntimeError("HTTP resume requires an active case scope")
        case_log = os.environ.get("AUTORESEARCH_CASE_LOG", "").strip()
        if not case_log:
            raise RuntimeError("Case checkpoint journal is unavailable while resuming")
        if case_id in completed_ids(Path(case_log)):
            raise RuntimeError(f"HTTP request refused for completed case {case_id}")
        record["resume_id"] = resume_id
    path = Path(raw)
    append_attempt(path, record)
    return path, append_attempt, attempt_id, case_id, resume_id


def _finish_attempt(context, status: int | None) -> None:
    if context is None:
        return
    path, append_attempt, attempt_id, case_id, resume_id = context
    record = {
        "event": "finished",
        "attempt_id": attempt_id,
        "case_id": case_id,
        "http_status": status,
        "cost_usd": None,
    }
    if resume_id:
        record["resume_id"] = resume_id
    append_attempt(path, record)


def _read_stats() -> dict:
    path = _stats_path()
    if path is None:
        return {"rate_limited_responses": 0, "retry_wait_seconds": 0.0, "attempts": 0}
    try:
        from autoresearch_runtime.store import _sqlite_document
    except ImportError:
        # Standalone hook tests and external evaluators may expose only hooks/.
        _sqlite_document = None
    try:
        target = _sqlite_document(path) if _sqlite_document is not None else None
        if target is not None and target[2] == "net_stats":
            loaded = target[0].get_document(target[1], path, {})
        else:
            loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    loaded.setdefault("rate_limited_responses", 0)
    loaded.setdefault("retry_wait_seconds", 0.0)
    loaded.setdefault("attempts", 0)
    return loaded


def _write_stats(stats: dict) -> None:
    path = _stats_path()
    if path is None:
        return
    try:
        from autoresearch_runtime.store import _sqlite_document
        target = _sqlite_document(path)
        if target is not None and target[2] == "net_stats":
            target[0].set_document(target[1], path, stats)
            return
    except ImportError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(stats, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _note_attempt() -> None:
    stats = _read_stats()
    stats["attempts"] = int(stats.get("attempts") or 0) + 1
    _write_stats(stats)


def _note_limited(delay: float) -> None:
    stats = _read_stats()
    stats["rate_limited_responses"] = int(stats.get("rate_limited_responses") or 0) + 1
    stats["retry_wait_seconds"] = round(float(stats.get("retry_wait_seconds") or 0) + delay, 3)
    _write_stats(stats)


def _wire_tools():
    scripts = Path(__file__).resolve().parents[1]
    if str(scripts) not in sys.path:
        sys.path.append(str(scripts))
    from autoresearch_runtime.wire_record import record_sent_request, redact_headers

    return record_sent_request, redact_headers


def _urllib_header_map(request) -> dict[str, str]:
    headers: dict[str, str] = {}
    source = getattr(request, "headers", None)
    if source is not None and hasattr(source, "items"):
        for key, value in source.items():
            headers[str(key)] = str(value)
        return headers
    if hasattr(request, "header_items"):
        for key, value in request.header_items():
            headers[str(key)] = str(value)
    elif hasattr(request, "unredirected_hdrs"):
        for key, value in request.unredirected_hdrs.items():
            headers[str(key)] = str(value)
    return headers


def _read_http_error_body(error: HTTPError) -> bytes | None:
    try:
        raw = error.read()
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return None


def _read_urlopen_response_body(response) -> bytes | None:
    import io

    read = getattr(response, "read", None)
    if read is None:
        return None
    try:
        raw = read()
    except Exception:
        return None
    if isinstance(raw, bytes):
        if raw:
            response.fp = io.BytesIO(raw)
            response.length = len(raw)
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return None


def _record_urllib(request, *, status: int | None, response_body: bytes | str | None, attempt: int) -> None:
    if not _wire_enabled():
        return
    record_sent_request, redact_headers = _wire_tools()
    record_sent_request(
        url=request.get_full_url(),
        method=request.get_method() or "GET",
        body=request.data,
        headers=redact_headers(_urllib_header_map(request)),
        status=status,
        response_body=response_body,
        attempt=attempt,
    )


def _record_requests(request, *, status: int | None, response_body: bytes | str | None, attempt: int) -> None:
    if not _wire_enabled():
        return
    record_sent_request, redact_headers = _wire_tools()
    headers = {str(key): str(value) for key, value in request.headers.items()}
    record_sent_request(
        url=str(request.url),
        method=str(request.method or "GET"),
        body=request.body,
        headers=redact_headers(headers),
        status=status,
        response_body=response_body,
        attempt=attempt,
    )


def _record_httpx(request, *, status: int | None, response_body: bytes | str | None, attempt: int) -> None:
    if not _wire_enabled():
        return
    record_sent_request, redact_headers = _wire_tools()
    headers = {str(key): str(value) for key, value in request.headers.items()}
    body = request.content if hasattr(request, "content") else getattr(request, "stream", None)
    record_sent_request(
        url=str(request.url),
        method=str(request.method),
        body=body,
        headers=redact_headers(headers),
        status=status,
        response_body=response_body,
        attempt=attempt,
    )


def _install_urllib(_state=None) -> None:
    import urllib.request

    original = urllib.request.urlopen
    if getattr(original, "_autoresearch_hook", False):
        return

    def urlopen(request, *args, **kwargs):
        context = _begin_attempt()
        _note_attempt()
        try:
            response = original(request, *args, **kwargs)
        except HTTPError as error:
            _finish_attempt(context, error.code)
            response_body = _read_http_error_body(error) if _wire_enabled() else None
            _record_urllib(request, status=error.code, response_body=response_body, attempt=1)
            if error.code in _RATE_LIMIT_STATUSES:
                _note_limited(0)
            raise
        except BaseException:
            # Keep the start record open: the transport outcome may be unknown.
            raise
        status = getattr(response, "status", getattr(response, "code", 200))
        _finish_attempt(context, status)
        response_body = _read_urlopen_response_body(response) if _wire_enabled() else None
        _record_urllib(request, status=status, response_body=response_body, attempt=1)
        if status in _RATE_LIMIT_STATUSES:
            _note_limited(0)
        return response

    urlopen._autoresearch_hook = True
    urllib.request.urlopen = urlopen


def _install_requests(_state=None) -> None:
    try:
        import requests
    except Exception:
        return
    session = getattr(requests.sessions, "Session", None)
    if session is None or getattr(session.send, "_autoresearch_hook", False):
        return
    original = session.send

    def send(self, request, **kwargs):
        context = _begin_attempt()
        _note_attempt()
        response = original(self, request, **kwargs)
        _finish_attempt(context, response.status_code)
        content = None
        if _wire_enabled() and getattr(response, "_content_consumed", True):
            try:
                content = response.content
            except Exception:
                content = None
        _record_requests(request, status=response.status_code, response_body=content, attempt=1)
        if response.status_code in _RATE_LIMIT_STATUSES:
            _note_limited(0)
        return response

    send._autoresearch_hook = True
    session.send = send


def _install_httpx(_state=None) -> None:
    try:
        import httpx
    except Exception:
        return

    client = getattr(httpx, "Client", None)
    if client is not None and not getattr(client.send, "_autoresearch_hook", False):
        original = client.send

        def send(self, request, **kwargs):
            context = _begin_attempt()
            _note_attempt()
            response = original(self, request, **kwargs)
            _finish_attempt(context, response.status_code)
            can_read = (
                getattr(response, "is_stream_consumed", False)
                or hasattr(response, "_content")
                or getattr(response, "is_closed", False)
                or not hasattr(response, "is_stream_consumed")
            )
            content = None
            if _wire_enabled() and can_read:
                try:
                    content = response.content
                except Exception:
                    content = None
            _record_httpx(request, status=response.status_code, response_body=content, attempt=1)
            if response.status_code in _RATE_LIMIT_STATUSES:
                _note_limited(0)
            return response

        send._autoresearch_hook = True
        client.send = send

    async_client = getattr(httpx, "AsyncClient", None)
    if async_client is not None and not getattr(async_client.send, "_autoresearch_hook", False):
        original_async = async_client.send

        async def async_send(self, request, **kwargs):
            context = _begin_attempt()
            _note_attempt()
            response = await original_async(self, request, **kwargs)
            _finish_attempt(context, response.status_code)
            can_read = (
                getattr(response, "is_stream_consumed", False)
                or hasattr(response, "_content")
                or getattr(response, "is_closed", False)
                or not hasattr(response, "is_stream_consumed")
            )
            if _wire_enabled() and can_read:
                content = None
                try:
                    content = response.content
                except Exception:
                    content = None
                _record_httpx(request, status=response.status_code, response_body=content, attempt=1)
            if response.status_code in _RATE_LIMIT_STATUSES:
                _note_limited(0)
            return response

        async_send._autoresearch_hook = True
        async_client.send = async_send


def _install_aiohttp(_state=None) -> None:
    try:
        import aiohttp
    except Exception:
        return
    session = getattr(aiohttp, "ClientSession", None)
    if session is None or getattr(session._request, "_autoresearch_hook", False):
        return
    original = session._request

    async def _request(self, *args, **kwargs):
        context = _begin_attempt()
        _note_attempt()
        response = await original(self, *args, **kwargs)
        status = getattr(response, "status", 200)
        _finish_attempt(context, status)
        if status in _RATE_LIMIT_STATUSES:
            _note_limited(0)
        return response

    _request._autoresearch_hook = True
    session._request = _request


def _patch_imported_clients() -> None:
    _install_requests()
    _install_httpx()
    _install_aiohttp()
    real_import = __import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        if name == "requests" or name.startswith("requests."):
            _install_requests()
        if name == "httpx" or name.startswith("httpx."):
            _install_httpx()
        if name == "aiohttp" or name.startswith("aiohttp."):
            _install_aiohttp()
        return module

    builtins = sys.modules.get("builtins")
    if builtins is not None and not getattr(builtins.__import__, "_autoresearch_provider_hook", False):
        _import._autoresearch_provider_hook = True
        builtins.__import__ = _import


_chain_next()
if _wire_enabled() or _attempts_enabled() or _stats_path() is not None or _network_guard_enabled():
    _install_urllib()
    _patch_imported_clients()
