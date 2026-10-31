"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
from typing import Any

from .auth import AuthError, authenticate
from .catalog import _human_goal, lab_lifecycle, lab_status, refresh_catalog, resolve_lab_path, save_catalog
from .dashboard import (
    ASSET_TYPES,
    STATUS_LABELS,
    assign_short_labels,
    build_problems,
    build_verdict,
    chart_series,
    metric_as_percent,
    metric_label,
    presentation_status,
    read_asset,
    reason_label,
    render_unified_dashboard,
)

from .dash_api import (
    AmbiguousLab, _open_lab, _safe_diff, archive_lab, dashboard_details, dashboard_status,
    delete_lab, render_live_page, restore_lab,
)
from .api_v1 import dispatch as dispatch_api_v1
from .paths import resolve_home
from .sqlite_store import SQLiteStore


_SERVERS_LOCK = threading.Lock()
_ACTIVE_SERVERS: dict[str, DashboardHTTPServer] = {}


def get_active_server(root: Path) -> DashboardHTTPServer | None:
    with _SERVERS_LOCK:
        return _ACTIVE_SERVERS.get(str(root.resolve()))


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
        self.clients: set[queue.Queue[str | None]] = set()
        self._clients_lock = threading.Lock()
        self._closed = False
        self.heartbeat_interval = 15.0
        self._revision_cache_lock = threading.Lock()
        self._cached_revision_payload: dict[str, Any] | None = None
        self._cached_revision_at: float = 0.0
        super().__init__(address, DashboardHandler)
        with _SERVERS_LOCK:
            _ACTIVE_SERVERS[str(self.root)] = self

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def register_client(self, q: queue.Queue[str | None]) -> None:
        with self._clients_lock:
            self.clients.add(q)

    def unregister_client(self, q: queue.Queue[str | None]) -> None:
        with self._clients_lock:
            self.clients.discard(q)

    def broadcast(self, event_type: str = "update", data: dict[str, Any] | None = None) -> None:
        # SQLite-backed pages observe the authoritative DB revision directly.
        # Avoid sending a second, potentially duplicate event through this queue.
        if SQLiteStore(resolve_home()).is_initialized():
            return
        if data is None:
            data = self.revision_payload()
        msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        with self._clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass

    def revision_payload(self) -> dict[str, Any]:
        store = SQLiteStore(resolve_home())
        if store.is_initialized():
            return {"revision": str(store.get_revision()), "timestamp": time.time()}
        status = dashboard_status(self.root)
        return {"revision": str(status.get("revision") or "0"),
                "timestamp": status.get("timestamp") or status.get("updated_at", time.time())}

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
        self._closed = True
        with _SERVERS_LOCK:
            if _ACTIVE_SERVERS.get(str(self.root)) is self:
                _ACTIVE_SERVERS.pop(str(self.root), None)
        with self._clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self.clients.clear()
        super().server_close()

    def shutdown(self) -> None:
        self._closed = True
        with self._clients_lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
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
                page = render_unified_dashboard({}, {}, live=True, root_id=self.server.root_id)
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            payload = dict(dashboard_status(self.server.root))
            if "revision" not in payload or not payload["revision"]:
                payload["revision"] = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()[:16]
            if "timestamp" not in payload:
                payload["timestamp"] = payload.get("updated_at", time.time())
            revision = str(payload.get("revision") or "")
            if revision and self.headers.get("If-None-Match") == revision:
                self._send(304, b"", "application/json; charset=utf-8", {"ETag": revision})
                return
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8", {"ETag": revision} if revision else None)
        elif self.path == "/api/events" or self.path.startswith("/api/events?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.flush()

            client_queue: queue.Queue[str | None] = queue.Queue()
            self.server.register_client(client_queue)
            interval = getattr(self.server, "heartbeat_interval", 15.0)
            sqlite_mode = SQLiteStore(resolve_home()).is_initialized()
            last_revision = (str(self.server.revision_payload().get("revision") or "0")
                             if sqlite_mode else None)
            last_ping = time.monotonic()
            try:
                while not getattr(self.server, "_closed", False):
                    try:
                        message = client_queue.get(timeout=min(interval, 1.0))
                        if message is None:
                            break
                        self.wfile.write(message.encode("utf-8"))
                        self.wfile.flush()
                        last_ping = time.monotonic()
                    except queue.Empty:
                        now = time.monotonic()
                        if sqlite_mode:
                            revision_payload = self.server.cached_revision_payload(0.5)
                            revision = str(revision_payload.get("revision") or "0")
                            if revision != last_revision:
                                message = f"event: update\ndata: {json.dumps(revision_payload, ensure_ascii=False)}\n\n"
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
                self.server.unregister_client(client_queue)
                self.close_connection = True
        elif self.path.startswith("/api/labs/"):
            parsed = urlsplit(self.path)
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            root_query = parse_qs(parsed.query).get("root", [None])[0]
            if len(parts) == 4 and parts[:2] == ["api", "labs"]:
                payload = dashboard_details(self.server.root, parts[2], parts[3], root_id=root_query)
                if isinstance(payload, dict) and payload.get("error") == "ambiguous":
                    self._send(409, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
                elif payload is not None:
                    self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                else:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")
            elif len(parts) == 7 and parts[:2] == ["api", "labs"] and parts[4] == "runs" and parts[6] == "diff":
                try:
                    opened = _open_lab(self.server.root, parts[2], parts[3], root_id=root_query)
                except AmbiguousLab:
                    opened = None
                diff = _safe_diff(opened[2], parts[5]) if opened else None
                if diff is not None:
                    self._send(200, diff.encode("utf-8"), "text/plain; charset=utf-8")
                else:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
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
        parsed = urlsplit(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        root_query = parse_qs(parsed.query).get("root", [None])[0]
        actions = {"delete": delete_lab, "archive": archive_lab, "restore": restore_lab}
        if len(parts) == 5 and parts[:2] == ["api", "labs"] and parts[4] in actions:
            confirm = self.headers.get("X-AutoResearch-Confirm", "")
            if not secrets.compare_digest(confirm, parts[3]):
                self._send(403, b'{"error":"confirm"}', "application/json; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                length = 0
            if length > 0:
                self.rfile.read(min(length, 4096))
            status, payload = actions[parts[4]](self.server.root, parts[2], parts[3], root_id=root_query)
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return
        if self.path == "/api/notify":
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(min(length, 65536)).decode("utf-8") if length > 0 else "{}"
                req_data = json.loads(body) if body else {}
            except Exception:
                req_data = {}
            event_type = req_data.get("event") or "update"
            event_data = req_data.get("data")
            if not isinstance(event_data, dict) or not event_data:
                status = dashboard_status(self.server.root)
                event_data = {
                    "revision": str(status.get("revision") or ""),
                    "timestamp": status.get("timestamp") or status.get("updated_at", time.time()),
                }
            self.server.broadcast(event_type, event_data)
            self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            return
        if self.path != "/__stop":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        self._send(200, b'{"stopping":true}', "application/json; charset=utf-8")
        threading.Thread(target=self.server.shutdown, daemon=True).start()

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


def notify_dashboard(root: Path, data: dict[str, Any] | None = None) -> bool:
    root = root.resolve()
    if data is None:
        status = dashboard_status(root)
        data = {
            "revision": str(status.get("revision") or ""),
            "timestamp": status.get("timestamp") or status.get("updated_at", time.time()),
        }
    server = get_active_server(root)
    if server is not None:
        server.broadcast("update", data)
        return True

    server_file = root / ".dashboard-server.json"
    if server_file.is_file():
        try:
            from .dash_process import _server_info

            info = _server_info(root)
            if isinstance(info, dict) and info.get("url") and info.get("token"):
                req = Request(
                    urljoin(info["url"], "/api/notify"),
                    data=json.dumps({"event": "update", "data": data}).encode("utf-8"),
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                with urlopen(req, timeout=1.0) as resp:
                    return resp.status == 200
        except Exception:
            pass
    return False


def create_server(root: Path, *, port: int = 8765, stop_token: str = "") -> DashboardHTTPServer:
    return DashboardHTTPServer(("127.0.0.1", port), root, stop_token)

