"""Loopback-only, read-only dashboard server. Lab data is served by the GET routes under /api/v1."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from typing import Any

from .auth import AuthError, authenticate
from .dashboard import ASSET_TYPES, read_asset, render_unified_dashboard

from .api_v1 import dispatch as dispatch_api_v1
from .paths import resolve_home
from .sqlite_store import SQLiteStore


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], root: Path, stop_token: str):
        self.root = root.resolve()
        self.stop_token = stop_token
        self.root_id = hashlib.sha256(str(self.root).casefold().encode("utf-8")).hexdigest()[:16]
        service_home = resolve_home().resolve()
        self.home_id = hashlib.sha256(str(service_home).casefold().encode("utf-8")).hexdigest()[:16]
        self.pid = os.getpid()
        self.service_id = secrets.token_urlsafe(16)
        self.closed = threading.Event()
        self.heartbeat_interval = 15.0
        self._revision_cache_lock = threading.Lock()
        self._cached_revision_payload: dict[str, Any] | None = None
        self._cached_revision_at: float = 0.0
        super().__init__(address, DashboardHandler)

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def revision_payload(self) -> dict[str, Any]:
        store = SQLiteStore(resolve_home())
        revision = str(store.get_revision()) if store.is_initialized() else "0"
        return {"revision": revision, "timestamp": time.time()}

    def cached_revision_payload(self, max_age: float = 0.5) -> dict[str, Any]:
        now = time.monotonic()
        with self._revision_cache_lock:
            if self._cached_revision_payload is not None and (now - self._cached_revision_at) < max_age:
                return self._cached_revision_payload
            payload = self.revision_payload()
            self._cached_revision_payload = payload
            self._cached_revision_at = now
            return payload

    def server_close(self) -> None:
        self.closed.set()
        super().server_close()

    def shutdown(self) -> None:
        self.closed.set()
        super().shutdown()


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DashboardHTTPServer

    def do_GET(self) -> None:
        if dispatch_api_v1(self):
            return
        try:
            authenticate(self)
        except AuthError as error:
            self._legacy_auth_error(error)
            return
        if urlsplit(self.path).path == "/api/health":
            store_state = "initialized" if SQLiteStore(resolve_home()).is_initialized() else "absent"
            health = {"ok": True, "root_id": self.server.root_id, "store": store_state}
            if parse_qs(urlsplit(self.path).query).get("identity") == ["1"]:
                health.update({"home_id": self.server.home_id, "service_id": self.server.service_id,
                               "pid": self.server.pid})
            body = json.dumps(health).encode("utf-8")
            self._send(200, body,
                       "application/json; charset=utf-8")
            return
        if self.path == "/" or self.path.startswith("/assets/"):
            if self.path.startswith("/assets/"):
                name = unquote(urlsplit(self.path).path.removeprefix("/assets/")).strip("/")
                body = read_asset(name)
                if body is None:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")
                else:
                    self._send(200, body, ASSET_TYPES[name])
            else:
                # The page is public, so it must contain no project or experiment snapshot.
                page = render_unified_dashboard(root_id=self.server.root_id)
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/events" or self.path.startswith("/api/events?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()

            interval = getattr(self.server, "heartbeat_interval", 15.0)
            last_revision = str(self.server.revision_payload()["revision"])
            last_ping = time.monotonic()
            try:
                while not self.server.closed.wait(min(interval, 1.0)):
                    now = time.monotonic()
                    payload = self.server.cached_revision_payload(0.5)
                    revision = str(payload["revision"])
                    if revision != last_revision:
                        message = f"event: update\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        self.wfile.write(message.encode("utf-8"))
                        self.wfile.flush()
                        last_revision = revision
                        last_ping = now
                    if now - last_ping >= interval:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping = now
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                self.close_connection = True
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if dispatch_api_v1(self):
            return
        try:
            authenticate(self, allow_stop_header=self.path == "/__stop")
        except AuthError as error:
            self._legacy_auth_error(error)
            return
        if self.path != "/__stop":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(200, b'{"stopping":true}', "application/json; charset=utf-8")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _refuse_method(self) -> None:
        # Only /api/v1 answers other methods, with 405 after authentication; other paths keep the default 501.
        if not dispatch_api_v1(self):
            self.send_error(501, f"Unsupported method ({self.command!r})")

    do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _refuse_method

    def _legacy_auth_error(self, error: AuthError) -> None:
        body = json.dumps({"ok": False, "error": {"code": error.code, "message": str(error)}}).encode("utf-8")
        self._send(error.status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def create_server(root: Path, *, port: int = 8765, stop_token: str = "") -> DashboardHTTPServer:
    return DashboardHTTPServer(("127.0.0.1", port), root, stop_token)

