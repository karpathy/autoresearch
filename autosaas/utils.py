from __future__ import annotations

from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPSConnection
import subprocess
from pathlib import Path
from time import perf_counter, sleep
from typing import Tuple
from urllib.parse import urlparse
from uuid import uuid4


def utc_now_iso() -> str:
    """UTC timestamp suitable for TSV logging (no external calls, no telemetry)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Generate a local run id for correlating logs/reports."""
    return uuid4().hex


def run_command(command: str, cwd: Path | str, timeout: float = 300.0) -> Tuple[bool, str, float]:
    """Execute a shell command and report whether it passed plus a summary."""
    start = perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = perf_counter() - start
        summary = completed.stderr.strip() or completed.stdout.strip()
        if not summary:
            summary = f"command exited {completed.returncode}"
        return completed.returncode == 0, summary, duration
    except subprocess.TimeoutExpired:
        duration = perf_counter() - start
        return False, f"timeout after {timeout}s", duration
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected errors
        duration = perf_counter() - start
        return False, f"command failed: {exc}", duration


def wait_for_app_ready(
    url: str | None, timeout_s: float = 30.0, interval_s: float = 1.0
) -> Tuple[bool, str, float]:
    """Poll an HTTP endpoint until a successful response or timeout."""
    if not url:
        return False, "no url provided for app boot", 0.0

    normalized = url if "://" in url else f"http://{url}"
    start = perf_counter()
    parsed = urlparse(normalized)
    scheme = parsed.scheme or "http"
    host = parsed.hostname
    if not host:
        return False, f"invalid app boot url: {url}", 0.0

    target = parsed.path or "/"
    if parsed.query:
        target += f"?{parsed.query}"

    port = parsed.port or (443 if scheme == "https" else 80)
    connection_cls = HTTPSConnection if scheme == "https" else HTTPConnection
    last_error = None
    last_status: int | None = None

    while perf_counter() - start < timeout_s:
        conn = None
        try:
            conn = connection_cls(host, port, timeout=interval_s)
            conn.request("GET", target)
            response = conn.getresponse()
            response.read()
            duration = perf_counter() - start
            last_status = response.status
            if response.status < 400:
                return True, f"status {response.status}", duration
        except Exception as exc:  # pragma: no cover - best-effort polling
            last_error = exc
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
        sleep(interval_s)

    duration = perf_counter() - start
    message = f"timed out waiting for {url}"
    if last_status is not None:
        message += f" (last status {last_status})"
    if last_error:
        message += f": {last_error}"
    return False, message, duration
