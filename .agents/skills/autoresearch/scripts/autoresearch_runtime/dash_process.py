"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from typing import Any


from .dash_http import DashboardHTTPServer, create_server


def _read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
from .paths import resolve_home
from .runtime_env import hidden_process_kwargs


_SERVER_PROCESSES: dict[str, subprocess.Popen[Any]] = {}

_START_LOCKS: dict[str, threading.Lock] = {}

_START_LOCKS_GUARD = threading.Lock()
_STOP_TOKEN_ENV = "AUTORESEARCH_DASHBOARD_STOP_TOKEN"

def _restrict_server_info_permissions(path: Path) -> bool:
    if os.name == "nt":
        user = os.environ.get("USERNAME")
        domain = os.environ.get("USERDOMAIN")
        if not user:
            return False
        account = f"{domain}\\{user}" if domain else user
        options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                                   "stderr": subprocess.DEVNULL, "check": False}
        options.update(hidden_process_kwargs())
        try:
            result = subprocess.run(
                ["icacls", os.fspath(path), "/inheritance:r", "/grant:r",
                 f"{account}:(F)", "*S-1-5-18:(F)"], **options,
            )
            return result.returncode == 0
        except OSError:
            return False
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return True
    except OSError:
        return False


def _add_warning(result: dict[str, Any], message: str) -> None:
    existing = result.get("warning")
    result["warning"] = f"{existing} {message}" if existing else message


def _write_server_info(root: Path, server: DashboardHTTPServer, token: str) -> None:
    data = {"host": "127.0.0.1", "port": server.server_port, "url": f"http://127.0.0.1:{server.server_port}/",
            "token": token, "pid": server.pid, "root_id": server.root_id,
            "home_id": server.home_id, "service_id": server.service_id}
    path = root / ".dashboard-server.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.touch()
    protected = _restrict_server_info_permissions(temporary)
    data["token_protection"] = "restricted" if protected else "best_effort"
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    _restrict_server_info_permissions(path)

def _server_info(root: Path, timeout: float = 0.5) -> dict[str, Any] | None:
    value = _read_json(root / ".dashboard-server.json")
    if (not isinstance(value, dict) or value.get("host") != "127.0.0.1"
            or not isinstance(value.get("url"), str) or not value.get("url")
            or not value.get("token") or isinstance(value.get("port"), bool)
            or not isinstance(value.get("port"), int) or not 1 <= value["port"] <= 65535
            or isinstance(value.get("pid"), bool) or not isinstance(value.get("pid"), int)
            or value.get("pid", 0) < 1 or not isinstance(value.get("home_id"), str)
            or not value.get("home_id") or not isinstance(value.get("service_id"), str)
            or not value.get("service_id")):
        return None
    try:
        parsed = urlsplit(value["url"])
        parsed_port = parsed.port
    except (TypeError, ValueError):
        return None
    expected_root_id = hashlib.sha256(str(root.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    expected_home_id = hashlib.sha256(str(resolve_home().resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed_port != value["port"]
            or parsed.username is not None or parsed.password is not None or parsed.path != "/"
            or parsed.query or parsed.fragment):
        return None
    try:
        with urlopen(urljoin(value["url"], "/api/health?identity=1"), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if (response.status == 200 and isinstance(payload, dict)
                    and payload.get("root_id") == expected_root_id == value.get("root_id")
                    and payload.get("home_id") == expected_home_id == value.get("home_id")
                    and payload.get("service_id") == value.get("service_id")
                    and payload.get("pid") == value.get("pid")):
                return value
    except (OSError, URLError, ValueError):
        return None
    return None

@contextlib.contextmanager
def _dashboard_start_lock(root: Path):
    root_key = str(root.resolve())
    with _START_LOCKS_GUARD:
        thread_lock = _START_LOCKS.setdefault(root_key, threading.Lock())
    thread_lock.acquire()
    lock_path = root / ".dashboard-start.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            thread_lock.release()

def start_dashboard(root: Path, *, open_browser: bool = True, timeout: float = 5,
                    port_start: int = 8765, force_open: bool = False) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _dashboard_start_lock(root):
        return _start_dashboard_locked(root, open_browser=open_browser, timeout=timeout,
                                       port_start=port_start, force_open=force_open)

def ensure_dashboard_server(root: Path, *, open_browser: bool = False, force_open: bool = False) -> dict[str, Any]:
    if os.environ.get("AUTORESEARCH_NO_DASHBOARD") == "1":
        return {"running": False, "url": None, "browser_opened": False, "pid": None}
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _dashboard_start_lock(root):
        info = _server_info(root, timeout=1.5)
        if info is None:
            result = _start_dashboard_locked(root, open_browser=False, timeout=5.0,
                                             port_start=8765, force_open=False)
            info = _server_info(root, timeout=1.5)
            if info is None:
                return result
        result = {"running": True, "url": info["url"], "browser_opened": False, "pid": info.get("pid")}
        if info.get("token_protection") != "restricted":
            _add_warning(result, "Dashboard stop-token file permissions could not be restricted; keep AUTORESEARCH_HOME private.")
        if open_browser:
            _maybe_open_browser(root, info, result, force_open=force_open)
        return result


def _start_dashboard_locked(root: Path, *, open_browser: bool, timeout: float,
                            port_start: int, force_open: bool) -> dict[str, Any]:
    info = _server_info(root)
    if info is None:
        token = secrets.token_urlsafe(32)
        scripts = Path(__file__).resolve().parents[1]
        log_file = root / ".dashboard-server.log"
        log_handle = open(log_file, "a", encoding="utf-8")
        options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log_handle,
                                   "stderr": log_handle, "cwd": str(scripts)}
        if os.name == "nt":
            options.update(hidden_process_kwargs())
        else:
            options["start_new_session"] = True
        options["env"] = {**os.environ, _STOP_TOKEN_ENV: token, "PYTHONPATH": os.pathsep.join(
            item for item in (str(scripts), os.environ.get("PYTHONPATH", "")) if item
        )}
        try:
            child = subprocess.Popen([sys.executable, "-m", "autoresearch_runtime.dashboard_server", "--serve",
                                      "--root", os.fspath(root), "--port-start", str(port_start)], **options)
        finally:
            log_handle.close()
        _SERVER_PROCESSES[str(root)] = child
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = _server_info(root)
            if info is not None:
                break
            time.sleep(0.1)
    if info is None:
        raise RuntimeError("Dashboard server did not start; it will be retried by the next CLI command")
    result = {"running": True, "url": info["url"], "browser_opened": False, "pid": info.get("pid")}
    if info.get("token_protection") != "restricted":
        _add_warning(result, "Dashboard stop-token file permissions could not be restricted; keep AUTORESEARCH_HOME private.")
    if open_browser:
        _maybe_open_browser(root, info, result, force_open=force_open)
    return result


def _maybe_open_browser(root: Path, info: dict[str, Any], result: dict[str, Any], *, force_open: bool) -> None:
    marker = root / ".dashboard-browser-opened"
    root_id = hashlib.sha256(str(root.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
    already_opened = _read_json(marker, {})
    if not force_open and isinstance(already_opened, dict) and already_opened.get("root_id") == root_id:
        return
    try:
        from . import dashboard_server
        result["browser_opened"] = bool(dashboard_server.webbrowser.open(info["url"], new=2))
    except Exception:
        result["browser_opened"] = False
    if not result["browser_opened"]:
        _add_warning(result, "Dashboard browser could not be opened; retry with `autoresearch dashboard ensure --open-browser`.")
        return
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"root_id": root_id, "opened_at": time.time()}), encoding="utf-8")
    os.replace(temporary, marker)

def dashboard_server_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    info = _server_info(root)
    child = _SERVER_PROCESSES.get(str(root))
    if info is None and child is not None and child.poll() is not None:
        _SERVER_PROCESSES.pop(str(root), None)
    return {"running": bool(info), "url": info.get("url") if info else None,
            "pid": info.get("pid") if info else None}

def _wait_process_exit(pid: int, timeout: float = 3.0) -> None:
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if handle:
                try:
                    kernel32.WaitForSingleObject(handle, int(timeout * 1000))
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:
            pass
    else:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.05)


def stop_dashboard(root: Path) -> dict[str, Any]:
    root = root.resolve()
    info = _server_info(root)
    if info is None:
        child = _SERVER_PROCESSES.get(str(root))
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                return {"running": True, "stopped": False, "url": None, "error": "Server did not stop"}
            _SERVER_PROCESSES.pop(str(root), None)
            return {"running": False, "stopped": True, "url": None}
        return {"running": False, "stopped": False, "url": None}
    request = Request(urljoin(info["url"], "/__stop"), data=b"", method="POST",
                      headers={"X-AutoResearch-Stop": info["token"]})
    try:
        with urlopen(request, timeout=2):
            pass
    except (OSError, URLError):
        return {"running": True, "stopped": False, "url": info["url"], "error": "Stop request failed"}
    pid = info.get("pid") if isinstance(info, dict) else None
    child = _SERVER_PROCESSES.get(str(root))
    if child is not None:
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            return {"running": True, "stopped": False, "url": info["url"], "error": "Server did not stop"}
        _SERVER_PROCESSES.pop(str(root), None)
    else:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and _server_info(root) is not None:
            time.sleep(0.05)
        if pid and pid != os.getpid():
            _wait_process_exit(pid, timeout=3.0)
        if _server_info(root) is not None:
            return {"running": True, "stopped": False, "url": info["url"], "error": "Server did not stop"}
    return {"running": False, "stopped": True, "url": info["url"]}

def serve(root: Path, *, port_start: int, token: str) -> int:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = None
    for port in range(port_start, port_start + 30):
        try:
            server = create_server(root, port=port, stop_token=token)
            break
        except OSError:
            continue
    if server is None:
        try:
            server = create_server(root, port=0, stop_token=token)
        except OSError:
            return 2
    _write_server_info(root, server, token)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        (root / ".dashboard-server.json").unlink(missing_ok=True)
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoResearch dashboard server")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port-start", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.error("internal server mode requires --serve")
    token = os.environ.get(_STOP_TOKEN_ENV, "")
    if not token:
        parser.error("internal server mode requires a user-context stop token")
    return serve(args.root, port_start=args.port_start, token=token)
