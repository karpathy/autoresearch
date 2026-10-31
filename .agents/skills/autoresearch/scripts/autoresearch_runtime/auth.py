"""Loopback access checks shared by the dashboard's HTTP routes."""

from __future__ import annotations

import ipaddress
import secrets
from urllib.parse import urlsplit
from typing import Any


class AuthError(PermissionError):
    def __init__(self, message: str, *, status: int = 401, code: str = "unauthorized") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _peer_is_loopback(handler: Any) -> bool:
    try:
        return ipaddress.ip_address(str(handler.client_address[0])).is_loopback
    except (ValueError, IndexError, TypeError):
        return False


def _expected_host(handler: Any) -> str:
    host, port = handler.server.server_address[:2]
    host = str(host).strip("[]").casefold()
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{int(port)}"


def _origin_is_exact(handler: Any, host: str) -> bool:
    origin = handler.headers.get("Origin")
    if not origin:
        return False
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.netloc.casefold() == host.casefold()
            and parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def authenticate(handler: Any, *, allow_stop_header: bool = False) -> str:
    """Allow local requests with an exact Host; only server shutdown needs a token."""
    if not _peer_is_loopback(handler):
        raise AuthError("Loopback access is required", status=403, code="loopback_required")

    expected_host = _expected_host(handler)
    host = handler.headers.get("Host", "").strip()
    if not host or host.casefold() != expected_host.casefold():
        raise AuthError("Host does not match the loopback service", status=403, code="invalid_host")

    origin = handler.headers.get("Origin")
    if origin is not None and not _origin_is_exact(handler, host):
        raise AuthError("Origin does not match the loopback service", status=403, code="invalid_origin")

    fetch_site = handler.headers.get("Sec-Fetch-Site", "").casefold()
    if fetch_site == "cross-site":
        raise AuthError("Cross-site requests are not allowed", status=403, code="cross_site_request")

    is_mutation = str(getattr(handler, "command", "GET")).upper() not in {"GET", "HEAD", "OPTIONS"}
    has_fetch_metadata = any(
        handler.headers.get(name) is not None
        for name in ("Sec-Fetch-Site", "Sec-Fetch-Mode", "Sec-Fetch-Dest", "Sec-Fetch-User")
    )
    if is_mutation and has_fetch_metadata and not _origin_is_exact(handler, host):
        raise AuthError("A same-origin request is required", status=403, code="origin_required")

    if allow_stop_header:
        token = str(getattr(handler.server, "stop_token", "") or "")
        if not token:
            raise AuthError("Local service stop is unavailable", status=503, code="stop_unavailable")
        supplied = handler.headers.get("X-AutoResearch-Stop", "")
        if not secrets.compare_digest(supplied, token):
            raise AuthError("Invalid stop token")
        return "stop-header"

    return "local"
