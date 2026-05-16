"""
Autoresearch HTTP API — lightweight REST interface for the orchestrator.

Runs inside crsai-pytorch container on port 8300, exposes:
  GET  /status         — orchestrator state (idle, running, stopped) + current experiment
  GET  /results        — full results.tsv as JSON
  GET  /best           — current best val_bpb
  POST /start          — start experiment loop (tag, max_experiments)
  POST /stop           — graceful stop of current loop
  POST /baseline       — run baseline only

Usage:
    Inside crsai-pytorch container:
        AUTORESEARCH_PROFILE=rtx5060 python3 api.py
"""

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# State shared between HTTP handler and orchestrator thread
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state = {
    "status": "idle",  # idle | running | stopping | stopped
    "tag": None,
    "experiment_num": 0,
    "max_experiments": 0,
    "current_description": None,
    "best_val_bpb": None,
    "error": None,
    "started_at": None,
}
_stop_event = threading.Event()
_thread: threading.Thread | None = None

RESULTS_FILE = "results.tsv"
API_PORT = int(os.getenv("AUTORESEARCH_API_PORT", "8300"))
RAG_RESERVATION_SCRIPT = os.getenv(
    "AUTORESEARCH_RAG_RESERVATION_SCRIPT",
    "/workspace/scripts/rag_gpu_reservation.sh",
)


def _reserve_gpu_for_ml() -> tuple[bool, str]:
    """Force MCP-RAG to CPU before launching a GPU-bound ML run."""
    try:
        proc = subprocess.run(
            ["bash", RAG_RESERVATION_SCRIPT, "cpu"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, output.strip()
    return True, output.strip()


def _read_results_json() -> list[dict]:
    """Parse results.tsv into list of dicts."""
    path = Path(RESULTS_FILE)
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(headers, parts, strict=False))
        # Convert numeric fields
        for key in ("val_bpb", "memory_gb"):
            if key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError) as exc:
                    import logging

                    logging.debug("results_parse_float_failed", key=key, error=str(exc))
        rows.append(row)
    return rows


def _get_best_bpb() -> float | None:
    """Get best val_bpb from results (keeps only)."""
    results = _read_results_json()
    best = None
    for r in results:
        if r.get("status") == "keep":
            bpb = r.get("val_bpb")
            if isinstance(bpb, (int, float)) and (best is None or bpb < best):
                best = bpb
    return best


def _run_orchestrator(tag: str, batch_size: int, baseline_only: bool) -> None:
    """Run orchestrator as a bounded batch via subprocess, updating shared state."""
    with _lock:
        _state["status"] = "running"
        _state["tag"] = tag
        _state["max_experiments"] = batch_size
        _state["experiment_num"] = 0
        _state["error"] = None
        _state["started_at"] = time.time()

    try:
        cmd = ["python3", "-u", "orchestrator.py", "--tag", tag]
        if baseline_only:
            cmd.append("--baseline-only")
        else:
            cmd.extend(["--max-experiments", str(batch_size or 12)])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        # Stream output and watch for stop signal
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                print(f"[orch] {line}", flush=True)
                # Update experiment counter from log lines
                if "Experiment" in line and "/" in line:
                    try:
                        # Parse "[HH:MM:SS] --- Experiment N [M/B] ..."
                        parts = line.split("[")
                        for p in parts:
                            if "/" in p and "]" in p:
                                batch_part = p.split("]")[0]
                                cur = int(batch_part.split("/")[0])
                                with _lock:
                                    _state["experiment_num"] = cur
                                break
                    except (ValueError, IndexError):
                        pass
                if "Proposal:" in line:
                    desc = line.split("Proposal:", 1)[1].strip()
                    with _lock:
                        _state["current_description"] = desc

            if _stop_event.is_set():
                proc.terminate()
                proc.wait(timeout=10)
                break

        proc.wait()

    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state["status"] = "stopped"
            _state["current_description"] = None
            _state["best_val_bpb"] = _get_best_bpb()


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class AutoresearchHandler(BaseHTTPRequestHandler):
    """Minimal REST handler for autoresearch orchestrator."""

    def _json_response(self, data: dict | list, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/status":
            with _lock:
                data = dict(_state)
            data["best_val_bpb"] = _get_best_bpb()
            self._json_response(data)

        elif path == "/results":
            self._json_response(_read_results_json())

        elif path == "/best":
            best = _get_best_bpb()
            self._json_response({"best_val_bpb": best})

        elif path == "/health":
            self._json_response({"ok": True})

        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:
        global _thread
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/start":
            with _lock:
                if _state["status"] == "running":
                    self._json_response({"error": "already running"}, 409)
                    return

            body = self._read_body()
            tag = body.get("tag", time.strftime("%b%d").lower())
            batch_size = int(body.get("batch_size", body.get("max_experiments", 12)))
            baseline_only = bool(body.get("baseline_only", False))

            reserved, reservation_output = _reserve_gpu_for_ml()
            if not reserved:
                self._json_response(
                    {
                        "error": "gpu reservation failed",
                        "detail": reservation_output[-2000:],
                    },
                    503,
                )
                return

            _stop_event.clear()
            _thread = threading.Thread(
                target=_run_orchestrator,
                args=(tag, batch_size, baseline_only),
                daemon=True,
            )
            _thread.start()
            self._json_response(
                {
                    "started": True,
                    "tag": tag,
                    "batch_size": batch_size,
                    "reservation": reservation_output[-500:],
                }
            )

        elif path == "/stop":
            _stop_event.set()
            with _lock:
                if _state["status"] == "running":
                    _state["status"] = "stopping"
            self._json_response({"stopping": True})

        elif path == "/baseline":
            with _lock:
                if _state["status"] == "running":
                    self._json_response({"error": "already running"}, 409)
                    return

            body = self._read_body()
            tag = body.get("tag", time.strftime("%b%d").lower())

            reserved, reservation_output = _reserve_gpu_for_ml()
            if not reserved:
                self._json_response(
                    {
                        "error": "gpu reservation failed",
                        "detail": reservation_output[-2000:],
                    },
                    503,
                )
                return

            _stop_event.clear()
            _thread = threading.Thread(
                target=_run_orchestrator,
                args=(tag, 0, True),
                daemon=True,
            )
            _thread.start()
            self._json_response(
                {
                    "started": True,
                    "tag": tag,
                    "baseline_only": True,
                    "reservation": reservation_output[-500:],
                }
            )

        else:
            self._json_response({"error": "not found"}, 404)

    def log_message(self, format, *args) -> None:
        """Suppress default access logging."""
        pass


def main() -> None:
    server = HTTPServer(("0.0.0.0", API_PORT), AutoresearchHandler)  # noqa: S104
    print(f"Autoresearch API listening on :{API_PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _stop_event.set()
        server.shutdown()


if __name__ == "__main__":
    main()
