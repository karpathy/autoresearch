"""Server agent for the client-server topology (fixed; the agent does not edit it).

Runs on the server machine, from a checkout of the same commit as the client. It only starts and stops
application instances (harness.instance.Instance) and returns what the judge needs. It never receives code:
each request carries parameters only (arm label, app directory inside apps/, sampling interval, slot).

Start (on the server):
    .cache/venv/bin/python -m harness.server_agent --listen 0.0.0.0:8100 --app-ports 8101-8104 \
        --token-file ~/.aging-hidden/server_token

Every request must carry the header "X-Token: <token>". Endpoints:
    GET  /time                  -> {"t": server epoch}                 (clock offset and RTT)
    GET  /manifest              -> {"manifest": {...}, "ok": bool}     (same judge on both machines)
    POST /arms/<label>/start    {"app_dir", "sample_s", "slot"} -> {"port", "t0"}
    GET  /arms/<label>/status   -> {"alive", "pids_seen", "max_children"}
    POST /arms/<label>/stop     -> status + {"t0", "db_state", "psutil_csv", "instrumentation_jsonl"}
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .instance import Instance

ROOT = Path(__file__).resolve().parent.parent
APPS_ROOT = ROOT / "apps"
MANIFEST = ROOT / ".judge" / "manifest.json"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def manifest_status():
    if not MANIFEST.exists():
        return {"manifest": None, "ok": False}
    m = json.loads(MANIFEST.read_text())
    bad = [k for k, v in m.items() if not k.startswith("frozen/")
           and (not (ROOT / k).exists() or hashlib.sha256((ROOT / k).read_bytes()).hexdigest() != v)]
    return {"manifest": m, "ok": not bad, "bad": bad}


def server_pins(slot):
    """App i -> CPU i on the server, when it has at least 4 cores and taskset."""
    if (os.cpu_count() or 0) >= 4 and shutil.which("taskset"):
        return str(slot)
    return None


class Agent:
    def __init__(self, token, ports, workroot, instance_cls=Instance):
        self.token = token
        self.free_ports = list(ports)
        self.workroot = Path(workroot)
        self.instance_cls = instance_cls
        self.arms = {}
        self.lock = threading.Lock()

    def authorized(self, header):
        return bool(header) and hmac.compare_digest(header, self.token)

    def start(self, label, body):
        app_dir = (ROOT / str(body.get("app_dir", ""))).resolve()
        if APPS_ROOT.resolve() not in app_dir.parents or not app_dir.is_dir():
            raise ValueError("app_dir must be a directory inside apps/")
        sample_s = float(body.get("sample_s", 5.0))
        if not 0.5 <= sample_s <= 60:
            raise ValueError("sample_s out of range")
        slot = int(body.get("slot", 0))
        with self.lock:
            if label in self.arms:
                raise ValueError(f"arm {label} already running")
            if not self.free_ports:
                raise ValueError("no free application port")
            port = self.free_ports.pop(0)
            self.arms[label] = None
        try:
            inst = self.instance_cls(app_dir, self.workroot / label, sample_s=sample_s, cpus=server_pins(slot),
                                     host="0.0.0.0", port=port)
            inst.prepare()
            inst.start()
            inst.start_sampler()
        except Exception:
            with self.lock:
                self.arms.pop(label, None)
                self.free_ports.append(port)
            raise
        with self.lock:
            self.arms[label] = (inst, port)
        return {"port": port, "t0": inst.t0}

    def _get(self, label):
        with self.lock:
            entry = self.arms.get(label)
        if not entry:
            raise KeyError(label)
        return entry

    def status(self, label):
        inst, _ = self._get(label)
        return {"alive": inst.alive(), "pids_seen": sorted(inst.pids_seen), "max_children": inst.max_children}

    def stop(self, label):
        inst, port = self._get(label)
        out = self.status(label)
        inst.stop()
        out.update({"t0": inst.t0, "db_state": inst.db_state(),
                    "psutil_csv": inst.psutil_out.read_text() if inst.psutil_out.exists() else "",
                    "instrumentation_jsonl": inst.instr_out.read_text() if inst.instr_out.exists() else ""})
        with self.lock:
            self.arms.pop(label, None)
            self.free_ports.append(port)
        return out


def make_handler(agent):
    import time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method):
            if not agent.authorized(self.headers.get("X-Token")):
                return self._send(401, {"error": "unauthorized"})
            parts = self.path.strip("/").split("/")
            try:
                if method == "GET" and parts == ["time"]:
                    return self._send(200, {"t": time.time()})
                if method == "GET" and parts == ["manifest"]:
                    return self._send(200, manifest_status())
                if len(parts) == 3 and parts[0] == "arms" and LABEL_RE.match(parts[1]):
                    label, action = parts[1], parts[2]
                    if method == "POST" and action == "start":
                        n = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(n) or b"{}")
                        return self._send(200, agent.start(label, body))
                    if method == "GET" and action == "status":
                        return self._send(200, agent.status(label))
                    if method == "POST" and action == "stop":
                        return self._send(200, agent.stop(label))
                return self._send(404, {"error": "not found"})
            except KeyError as e:
                return self._send(404, {"error": f"unknown arm {e}"})
            except Exception as e:
                return self._send(400, {"error": str(e)})

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

    return Handler


def parse_ports(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8100")
    ap.add_argument("--app-ports", default="8101-8104")
    ap.add_argument("--token-file", default=str(Path.home() / ".aging-hidden" / "server_token"))
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    token = Path(a.token_file).expanduser().read_text().strip()
    if len(token) < 16:
        raise SystemExit("token too short (use at least 16 characters)")
    st = manifest_status()
    if not st["ok"]:
        raise SystemExit(f"judge files differ from the manifest: {st.get('bad')}")
    workroot = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="aging-server-"))
    host, _, port = a.listen.rpartition(":")
    agent = Agent(token, parse_ports(a.app_ports), workroot)
    print(f"server agent on {a.listen}, app ports {a.app_ports}, workdir {workroot}", flush=True)
    ThreadingHTTPServer((host, int(port)), make_handler(agent)).serve_forever()


if __name__ == "__main__":
    main()
