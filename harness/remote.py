"""Client side of the client-server topology (fixed; the agent does not edit it).

RemoteInstance has the same interface as harness.instance.Instance, but the application runs on the server
machine (harness/server_agent.py). The load generator stays on the client and measures latency there.

Configuration (judge only; the agent cannot read it):
    AGING_SERVER=host:port                       server agent address; unset = everything local, as before
    ~/.aging-hidden/server_token                 shared token (or AGING_SERVER_TOKEN_FILE)

Clocks: the server sample times are relative to the server t0 (server clock). The client measures the clock
offset (server - client) and the RTT before and after each arm, NTP style (best of several probes), and
shifts the server samples so that they are relative to the client t0, like the local mode.
"""

import json
import os
import time
import urllib.request
from pathlib import Path

from .instance import Instance

CLOCK_PROBES = 7
MAX_RTT_S = 0.2


def server_config():
    addr = os.environ.get("AGING_SERVER", "").strip()
    if not addr:
        return None
    tok_file = Path(os.environ.get("AGING_SERVER_TOKEN_FILE", "~/.aging-hidden/server_token")).expanduser()
    return {"addr": addr, "host": addr.rpartition(":")[0], "token": tok_file.read_text().strip()}


class Client:
    def __init__(self, addr, token, timeout=60):
        self.base = f"http://{addr}"
        self.token = token
        self.timeout = timeout

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-Token": self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server agent {method} {path}: {e.code} {e.read().decode(errors='replace')}")

    def clock(self, probes=CLOCK_PROBES):
        """Returns {"offset_s": server - client, "rtt_s"} from the probe with the smallest RTT."""
        best = None
        for _ in range(probes):
            t1 = time.time()
            ts = self.call("GET", "/time")["t"]
            t2 = time.time()
            rtt = t2 - t1
            if best is None or rtt < best["rtt_s"]:
                best = {"offset_s": ts - (t1 + t2) / 2, "rtt_s": rtt}
        return best


def check_server_manifest(client, local_manifest):
    """The frozen workload exists only on the client (it runs there), so it is left out of the comparison."""
    st = client.call("GET", "/manifest")
    strip = lambda m: {k: v for k, v in (m or {}).items() if not k.startswith("frozen/")}
    if not st.get("ok") or strip(st.get("manifest")) != strip(local_manifest):
        raise RuntimeError("server judge differs from the client judge (manifest mismatch)")


class RemoteInstance:
    def __init__(self, client, host, app_dir, workdir, root, sample_s=5.0, slot=0):
        self.client = client
        self.host = host
        self.app_dir = Path(app_dir)
        self.workdir = Path(workdir)
        self.root = Path(root)
        self.sample_s = sample_s
        self.slot = slot
        self.label = self.workdir.name
        self.port = None
        self.t0 = None
        self.pids_seen = set()
        self.max_children = 0
        self.clock = {}
        self._final = None
        self._local = Instance(app_dir, workdir)  # reuses the local readers on the returned files
        self.psutil_out = self._local.psutil_out
        self.instr_out = self._local.instr_out

    def prepare(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

    def start(self):
        before = self.client.clock()
        if before["rtt_s"] > MAX_RTT_S:
            raise RuntimeError(f"RTT to server too high ({before['rtt_s']:.3f} s)")
        rel = str(self.app_dir.resolve().relative_to(self.root.resolve()))
        r = self.client.call("POST", f"/arms/{self.label}/start",
                             {"app_dir": rel, "sample_s": self.sample_s, "slot": self.slot})
        self.t0 = time.time()
        self.port = r["port"]
        self.clock = {"before": before, "server_t0": r["t0"]}
        return self.port

    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def start_sampler(self):
        pass  # the server starts sampling together with the instance

    def alive(self):
        st = self.client.call("GET", f"/arms/{self.label}/status")
        self.pids_seen = set(st["pids_seen"])
        self.max_children = st["max_children"]
        return st["alive"]

    def stop(self):
        if self._final is not None or self.port is None:
            return
        r = self.client.call("POST", f"/arms/{self.label}/stop")
        self._final = r
        self.clock["after"] = self.client.clock()
        self.pids_seen |= set(r["pids_seen"])
        self.max_children = max(self.max_children, r["max_children"])
        self.psutil_out.write_text(r["psutil_csv"])
        self.instr_out.write_text(r["instrumentation_jsonl"])
        (self.workdir / "clock.json").write_text(json.dumps(self.clock, indent=1))

    def shift_s(self):
        """Seconds to add to a server-relative time to make it relative to the client t0."""
        offset = self.clock["before"]["offset_s"]
        if "after" in self.clock:
            offset = (offset + self.clock["after"]["offset_s"]) / 2
        return (self.clock["server_t0"] - offset) - self.t0

    def db_state(self, full=False):
        return (self._final or {}).get("db_state")

    def probe_rows(self):
        d = self.shift_s()
        return [dict(r, t=r["t"] + d) for r in self._local.probe_rows()]

    def instrumentation(self):
        return self._local.instrumentation()
