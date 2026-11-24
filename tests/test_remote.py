"""Tests of the client-server topology (harness/server_agent.py + harness/remote.py) over real HTTP on
localhost, with a simulated application instance on the server side.

Run:  .cache/venv/bin/python tests/test_remote.py
"""

import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import server_agent as S  # noqa: E402
from harness.remote import Client, RemoteInstance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "t" * 32
SKEW = 1000.0  # server clock ahead of the client by 1000 s


class FakeInstance:
    def __init__(self, app_src, workdir, sample_s=5.0, cpus=None, host=None, port=None):
        self.workdir = Path(workdir)
        self.psutil_out = self.workdir / "psutil.csv"
        self.instr_out = self.workdir / "instrumentation.jsonl"
        self.port, self.host = port, host
        self.pids_seen, self.max_children, self.t0, self.running = {4242}, 0, None, False

    def prepare(self):
        self.workdir.mkdir(parents=True, exist_ok=True)

    def start(self):
        import time
        self.t0 = time.time() + SKEW
        self.running = True

    def start_sampler(self):
        pass

    def alive(self):
        return self.running

    def stop(self):
        self.running = False
        self.psutil_out.write_text("t,pid,rss,vms,uss,num_fds,num_threads,cpu_percent,children,db_bytes\n"
                                   "5.00,4242,100,200,90,10,4,1.0,0,4096\n")
        self.instr_out.write_text(json.dumps({"kind": "snapshot", "ts": 1, "metrics": {"x": 1}}) + "\n")

    def db_state(self, full=False):
        return {"tables": ["associations"], "counts": {"associations": 7}}


def serve(agent, handler=None):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler or S.make_handler(agent))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def setup(ports=(9101, 9102)):
    agent = S.Agent(TOKEN, ports, tempfile.mkdtemp(), instance_cls=FakeInstance)
    httpd = serve(agent)
    return agent, httpd, f"127.0.0.1:{httpd.server_address[1]}"


def skewed_handler(agent):
    """Same handler, but /time answers with the server clock (client clock + SKEW)."""
    import time
    h = S.make_handler(agent)
    route = h._route

    def _route(self, method):
        if self.path == "/time" and agent.authorized(self.headers.get("X-Token")):
            return self._send(200, {"t": time.time() + SKEW})
        return route(self, method)
    h._route = _route
    return h


def test_unauthorized_rejected():
    _, httpd, addr = setup()
    try:
        Client(addr, "wrong-token").call("GET", "/time")
        assert False
    except RuntimeError as e:
        assert "401" in str(e)
    httpd.shutdown()


def test_app_dir_must_be_inside_apps():
    _, httpd, addr = setup()
    try:
        Client(addr, TOKEN).call("POST", "/arms/a/start", {"app_dir": "../../etc", "sample_s": 5})
        assert False
    except RuntimeError as e:
        assert "apps/" in str(e)
    httpd.shutdown()


def test_full_arm_cycle_and_clock_shift():
    agent = S.Agent(TOKEN, (9101, 9102), tempfile.mkdtemp(), instance_cls=FakeInstance)
    httpd = serve(agent, skewed_handler(agent))
    addr = f"127.0.0.1:{httpd.server_address[1]}"
    wd = Path(tempfile.mkdtemp()) / "armA"
    inst = RemoteInstance(Client(addr, TOKEN), "127.0.0.1", ROOT / "apps" / "credit-card-fastapi" / "instrumented",
                          wd, ROOT, sample_s=5.0, slot=0)
    inst.prepare()
    port = inst.start()
    assert port == 9101 and inst.base_url() == "http://127.0.0.1:9101"
    assert abs(inst.clock["before"]["offset_s"] - SKEW) < 0.05
    assert inst.alive() and inst.pids_seen == {4242}
    inst.stop()
    rows = inst.probe_rows()
    # sample at server t=5 s after the server t0 -> about 5 s after the client t0
    assert len(rows) == 1 and abs(rows[0]["t"] - 5.0) < 0.1, rows
    assert inst.db_state()["counts"]["associations"] == 7
    assert inst.instrumentation()[0]["metrics"] == {"x": 1}
    assert (wd / "clock.json").exists() and agent.free_ports == [9102, 9101]
    httpd.shutdown()


def test_port_pool_exhausted():
    _, httpd, addr = setup(ports=(9201,))
    c = Client(addr, TOKEN)
    app = "apps/credit-card-fastapi/instrumented"
    c.call("POST", "/arms/a/start", {"app_dir": app})
    try:
        c.call("POST", "/arms/b/start", {"app_dir": app})
        assert False
    except RuntimeError as e:
        assert "no free application port" in str(e)
    httpd.shutdown()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
