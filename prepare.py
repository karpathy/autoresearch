"""Fixed judge of AutoResearch-Aging. THE AGENT MUST NOT EDIT THIS FILE.

Same role as prepare.py in karpathy/autoresearch: fixed measurement and evaluation. The agent only edits
workload.py. This version covers detection and causes only (see program.md); there is no repair.

Commands (run from the repository root):
  python prepare.py setup --src <llm-aging-detection-agents>          # human, once
  python prepare.py check-oracles                                       # human: César's functional test on the app
  python prepare.py experiment --tag <tag> [--smoke]                    # full experiment (orchestrator), no human
  -- pieces used by the orchestrator (also runnable by hand, for debugging):
  python prepare.py controls [--minutes 60] [--with-r1]                 # phase 0: R0 and R0b (+ R1)
  python prepare.py detect [--minutes 60] [--seed 101] [--background]   # search round (run by the agent)
  python prepare.py check-load --run <run_id> --arm <arm> --seed <s>    # verification round (R1, workload, R0, R0b)
  python prepare.py freeze <run_id> <arm>                               # freeze the workload of one arm
  python prepare.py causes [--minutes 60] [--seed 301] [--background]   # causes round (run by the agent)
  python prepare.py rehash                                              # human: rebuild the manifest after editing the judge
  python prepare.py wait <run_id>                                       # blocks up to 9 min
  python prepare.py verify                                              # check the judge hashes
Use --smoke (2 min) only to check the flow; smoke runs never count.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# After setup, always run inside the pinned venv, even if the agent calls "python prepare.py".
_VPY = ROOT / ".cache" / "venv" / "bin" / "python"
if _VPY.exists() and Path(sys.prefix).resolve() != (ROOT / ".cache" / "venv").resolve():
    os.execv(str(_VPY), [str(_VPY), *sys.argv])

from harness.instance import Instance  # noqa: E402
from harness.remote import Client, RemoteInstance, check_server_manifest, server_config  # noqa: E402
from harness.load import (BASELINE, R0, R0B, WorkloadError, one_factor_diff, read_client, run_script,  # noqa: E402
                          same_as_default, summarize_client, validate_spec)
from harness.evaluate import evaluate_arm  # noqa: E402
from harness.functional_check import functional_check  # noqa: E402
from harness.workload_check import load_workload  # noqa: E402

APP = "credit-card-fastapi"
APPS = ROOT / "apps" / APP
INSTRUMENTED = APPS / "instrumented"
VENV = ROOT / ".cache" / "venv"
VENV_PY = VENV / "bin" / "python"
CESAR_REPO_DEFAULT = ROOT.parent / "llm-aging-detection-agents"
# Files copied unchanged from César's repository (source -> destination).
CESAR_FILES = {
    "findings/bax-fastapi-credit-card-F1.yaml": "findings/bax-fastapi-credit-card-F1.yaml",
    "apps/credit-card-fastapi/campaign.yaml": "findings/campaign.yaml",
    "reference/tests/test_credit_card_service.py": "oracle/test_credit_card_service.py",
    "workloads/bax-fastapi-credit-card-F1_targeted_v1.py": "workloads/bax-fastapi-credit-card-F1_targeted_v1.py",
    "workloads/bax-fastapi-credit-card-F1_targeted_v1.md": "workloads/bax-fastapi-credit-card-F1_targeted_v1.md",
    "workloads/bax-fastapi-credit-card-F1_control.py": "workloads/bax-fastapi-credit-card-F1_control.py",
}
HARNESS_REQS = ["psutil", "httpx", "requests", "pymannkendall", "numpy", "pyyaml"]
FROZEN = ROOT / "frozen" / "workload_frozen.py"
RESULTS = ROOT / "results"
JUDGE_DIR = ROOT / ".judge"
MANIFEST = JUDGE_DIR / "manifest.json"
JUDGE_LOG = RESULTS / "judge_log.jsonl"
SLOTS = 4
DEFAULT_MINUTES = 60
FIXED_GLOBS = ["prepare.py", "harness/*.py", "oracle/*.py", "workloads/*.py", "findings/*.yaml"]
AGENT_TIMEOUT_S = 3 * 3600
CAUSES_REF = "frozen_reference"
CHECK_LOAD = "workload"


# ---------------------------------------------------------------- integrity
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixed_files():
    files = []
    for g in FIXED_GLOBS:
        files += sorted(ROOT.glob(g))
    if FROZEN.exists():
        files.append(FROZEN)
    files += sorted(p for p in INSTRUMENTED.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    and p.suffix != ".pyc" and not p.name.startswith("db.sqlite3"))
    if (APPS / "SOURCE.json").exists():
        files.append(APPS / "SOURCE.json")
    return files


def _key(p):
    return str(p.relative_to(ROOT))


def _path(k):
    return ROOT / k


def write_manifest():
    JUDGE_DIR.mkdir(exist_ok=True)
    m = {_key(p): _sha(p) for p in fixed_files()}
    MANIFEST.write_text(json.dumps(m, indent=1))
    return m


def verify(quiet=False):
    if not MANIFEST.exists():
        sys.exit("manifest missing: run 'python prepare.py setup' (human)")
    m = json.loads(MANIFEST.read_text())
    now = {_key(p) for p in fixed_files()}
    bad = [k for k, v in m.items() if not _path(k).exists() or _sha(_path(k)) != v]
    bad += sorted(now - set(m))
    if bad:
        log_judge({"event": "judge_modified", "files": bad, "flag": "H5_judge_or_oracle_modified"})
        sys.exit("ABORTED: fixed files changed -> " + ", ".join(bad))
    if not quiet:
        print("verify: ok", len(m), "fixed files intact")


def log_judge(rec):
    RESULTS.mkdir(exist_ok=True)
    rec = dict(rec, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(JUDGE_LOG, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


# ---------------------------------------------------------------- setup
def _git_out(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


def cmd_setup(a):
    if MANIFEST.exists() and not a.force:
        sys.exit("setup already done. Use --force only as a human, outside a campaign.")
    src = Path(a.src).expanduser().resolve()
    cc = src / "apps" / APP
    if not (cc / "instrumented" / "app.py").exists():
        sys.exit(f"{cc}/instrumented/app.py not found: pass --src <clone of chas42/llm-aging-detection-agents>")
    # César's repository is only read; everything the judge uses is copied here and hashed.
    if INSTRUMENTED.exists():
        shutil.rmtree(INSTRUMENTED)
    shutil.copytree(cc / "instrumented", INSTRUMENTED, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "db.sqlite3*"))
    for rel_src, rel_dst in CESAR_FILES.items():
        (ROOT / rel_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel_src, ROOT / rel_dst)
    try:
        commit, dirty = _git_out(src, "rev-parse", "HEAD"), bool(_git_out(src, "status", "--porcelain"))
    except Exception:
        commit, dirty = "unknown", None
    (APPS / "SOURCE.json").write_text(json.dumps({
        "repo": "https://github.com/chas42/llm-aging-detection-agents", "commit": commit, "dirty": dirty,
        "copied": sorted(CESAR_FILES) + [f"apps/{APP}/instrumented"]}, indent=1))
    print("pinned venv (app requirements.txt + judge dependencies)...")
    if VENV.exists():
        shutil.rmtree(VENV)
    VENV.parent.mkdir(exist_ok=True)
    # Without ensurepip the venv has no pip, so packages are installed from outside with `pip --python`.
    subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", str(VENV)])
    pip = [sys.executable, "-m", "pip", "--python", str(VENV_PY)]
    subprocess.check_call(pip + ["install", "-q", "-r", str(INSTRUMENTED / "requirements.txt")])
    subprocess.check_call(pip + ["install", "-q", *HARNESS_REQS])
    with open(VENV.parent / "venv-freeze.txt", "w") as f:
        subprocess.check_call(pip + ["freeze"], stdout=f)
    m = write_manifest()
    print(f"setup ok: {len(m)} fixed files in the manifest. "
          "Next: python prepare.py check-oracles (starts the app and runs César's functional test).")


def cmd_check_oracles(a):
    """César's functional test against the instrumented app: checks the copy works before measuring."""
    fails = functional_check(INSTRUMENTED, Path(tempfile.mkdtemp()))
    print(f"instrumented: functional={fails or 'ok'}")
    if fails:
        sys.exit("César's functional test fails against the copied app; check the setup before going on")
    print("check-oracles: ok")


# ---------------------------------------------------------------- utilities
def frozen_spec():
    if not FROZEN.exists():
        sys.exit("frozen/workload_frozen.py missing: the orchestrator freezes the confirmed workload")
    try:
        ns = load_workload(FROZEN)
    except WorkloadError as e:
        sys.exit(f"frozen workload rejected: {e}")
    name = ns["FROZEN_ARM"]
    if name == BASELINE["name"]:
        return BASELINE
    for c in ns["CANDIDATES"]:
        if c["name"] == name:
            validate_spec(c)
            return c
    sys.exit(f"frozen candidate {name} not found")


def pins(i):
    """CPU pinning per slot (app i -> CPU i, load generator -> CPU 4+i), only with >= 8 cores and taskset."""
    if (os.cpu_count() or 0) >= 8 and shutil.which("taskset"):
        return str(i), str(4 + i)
    return None, None


def run_arms(arms, minutes, run_dir, smoke):
    """arms: list of {label, app_dir, spec, seed}. Runs all of them in parallel, one slot per arm."""
    seconds = int(minutes * 60)
    sample_s = 2.0 if smoke else 5.0
    results = {}

    server = server_config()
    client = None
    if server:
        client = Client(server["addr"], server["token"])
        check_server_manifest(client, json.loads(MANIFEST.read_text()))

    def one(slot, arm):
        wd = run_dir / arm["label"]
        if client:
            load_cpu = str(slot) if (os.cpu_count() or 0) >= SLOTS and shutil.which("taskset") else None
            inst = RemoteInstance(client, server["host"], arm["app_dir"], wd, ROOT, sample_s=sample_s, slot=slot)
        else:
            app_cpu, load_cpu = pins(slot)
            inst = Instance(arm["app_dir"], wd, sample_s=sample_s, cpus=app_cpu)
        try:
            inst.prepare()
            port = inst.start()
            inst.start_sampler()
            t0 = time.time()
            base_url = inst.base_url() if client else f"http://127.0.0.1:{port}"
            ran = run_script(arm["spec"], base_url, seconds, arm["seed"], wd, cpus=load_cpu)
            time.sleep(sample_s * 1.5)
            alive, pids, children = inst.alive(), set(inst.pids_seen), inst.max_children
        except Exception as e:
            results[arm["label"]] = {"arm": arm["label"], "valid": False, "flags": [f"error: {e}"]}
            inst.stop()
            return
        inst.stop()
        rows = read_client(ran["client_csv"])
        load = summarize_client(rows, t0)
        load.update({"script_exit": ran["exit"], "loadgen_cpu_p95": ran["loadgen_cpu_p95"]})
        snaps = inst.instrumentation()
        ev = evaluate_arm(arm["label"], inst.probe_rows(), rows, t0, load, inst.db_state(), pids, alive, seconds,
                          children, instr_last=(snaps[-1]["metrics"] if snaps else None))
        ev.update({"spec_name": arm["spec"]["name"], "hypothesis": arm["spec"].get("hypothesis", ""),
                   "seed": arm["seed"], "app_dir": str(arm["app_dir"]), "workload_cmd": ran["cmd"], "topology": "client-server" if client else "local"})
        if client:
            ev["clock"] = inst.clock
        (wd / "evaluation.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
        results[arm["label"]] = ev

    ths = [threading.Thread(target=one, args=(k, a)) for k, a in enumerate(arms)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results


def new_run_dir(kind, run_id=None):
    run_id = run_id or f"{kind}-{time.strftime('%m%d-%H%M')}-{uuid.uuid4().hex[:4]}"
    d = RESULTS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d


def background(argv, run_id, run_dir):
    argv = [a for a in argv if a != "--background"] + ["--run-id", run_id]
    with open(run_dir / "judge.log", "ab") as log:
        subprocess.Popen([sys.executable, str(ROOT / "prepare.py")] + argv, cwd=ROOT,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (run_dir / "STATUS").write_text("running")
    print(f"run_id: {run_id}\nstatus: running (use: python prepare.py wait {run_id})")


def fmt(v, nd=3):
    return "NA" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def hot_slope(ev):
    return (ev.get("hot_latency_p50") or {}).get("sen_slope_per_h")


def _ratio(num, den):
    return (num / den) if (num is not None and den not in (None, 0)) else None


def _load_candidates(path, reserved):
    """Reads and validates CANDIDATES from a workload file; raises WorkloadError (D3) on any violation."""
    wl = load_workload(path)
    cands = wl.get("CANDIDATES")
    if not isinstance(cands, list) or not 1 <= len(cands) <= SLOTS - 1:
        raise WorkloadError(f"CANDIDATES must be a list of 1 to {SLOTS - 1} items (1 slot is the reference)")
    if not all(isinstance(c, dict) for c in cands):
        raise WorkloadError("each candidate must be a dict")
    names = [c.get("name") for c in cands]
    if len(set(names)) != len(names) or reserved in names:
        raise WorkloadError("candidate names must be unique and different from the reference")
    for c in cands:
        validate_spec(c)
    return cands


def _reject(event, run_id, run_dir, error):
    rec = {"event": event, "run_id": run_id, "commit": git_head(), "valid": False,
           "flag": "D3_out_of_contract", "error": str(error)}
    log_judge(rec)
    (run_dir / "STATUS").write_text("done")
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print("INVALID:", error)


# ---------------------------------------------------------------- detection
def cmd_detect(a):
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("detect" + ("-smoke" if a.smoke else ""), a.run_id)
    try:
        cands = _load_candidates(ROOT / "workload.py", BASELINE["name"])
    except Exception as e:
        return _reject("detect", run_id, run_dir, e)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    shutil.copy(ROOT / "workload.py", run_dir / "workload.py")
    arms = [{"label": BASELINE["name"], "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": a.seed}]
    arms += [{"label": c["name"], "app_dir": INSTRUMENTED, "spec": c, "seed": a.seed} for c in cands]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    base = res[BASELINE["name"]]
    base_slope = hot_slope(base) if base.get("score_eligible") else None
    rows = []
    copies = {c["name"] for c in cands if same_as_default(c)}
    for label, ev in res.items():
        if label in copies:
            ev.setdefault("flags", []).append("D4_copy_of_reference")
        slope = hot_slope(ev)
        eligible = ev.get("score_eligible") and not any(f.startswith(("D2", "D4")) for f in ev.get("flags", []))
        ev["score_latency_ms_per_h"] = slope if eligible and label != BASELINE["name"] else None
        ev["ratio_vs_reference"] = (slope / base_slope) if (eligible and base_slope and base_slope > 0) else None
        rows.append(ev)
    rec = {"event": "detect", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "seed": a.seed, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "valid": r.get("valid"), "flags": r.get("flags"),
         "score": r.get("score_latency_ms_per_h"), "p": (r.get("hot_latency_p50") or {}).get("mk_p_upward")}
        for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_detect(run_id, rows)


def print_detect(run_id, rows, header=True):
    if header:
        print(f"---\nrun_id: {run_id}")
    for r in rows:
        lat, byt, rss = r.get("hot_latency_p50") or {}, r.get("hot_resp_bytes") or {}, r.get("rss_mb") or {}
        refs = " ".join(f"{k.replace('retrieve_', '')}={fmt(((r.get('client') or {}).get(k) or {}).get('p50', {}).get('sen_change_pct'), 1)}%"
                        for k in ("retrieve_ref4", "retrieve_ref200"))
        print(f"arm={r['arm']} valid={r.get('valid')} score_hot_latency_ms_per_h={fmt(r.get('score_latency_ms_per_h'))} "
              f"hot_lat_slope={fmt(lat.get('sen_slope_per_h'))} hot_lat_change={fmt(lat.get('sen_change_pct'), 1)}% "
              f"hr_p={fmt(lat.get('mk_p_upward'), 5)} hot_bytes_slope={fmt(byt.get('sen_slope_per_h'), 0)} "
              f"rss_slope={fmt(rss.get('sen_slope_per_h'))} rss_mk_p={fmt(rss.get('mk_p_upward'), 5)} "
              f"ref_change[{refs}] ttd_min={fmt(r.get('time_to_detect_min'))} ratio={fmt(r.get('ratio_vs_reference'))} "
              f"req={r.get('requests')} err={fmt(r.get('error_rate'), 4)} db_rows={r.get('db_rows')} "
              f"{ref4_diag_str(r)} flags={','.join(r.get('flags', [])) or '-'}")


# ---------------------------------------------------------------- freeze
def freeze_spec(spec, source):
    """Freezes a candidate workload in frozen/workload_frozen.py and rebuilds the manifest."""
    clean = {k: v for k, v in spec.items() if not k.startswith("_")}
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text('"""Frozen workload."""\n\n'
                      f"CANDIDATES = [{clean!r}]\n\nFROZEN_ARM = {clean['name']!r}\n"
                      f"FROZEN_FROM = {json.dumps(source, ensure_ascii=False)!r}\n")
    write_manifest()


def round_spec(run_id, arm):
    """Spec of one arm of a past search round, read from the workload.py saved with that round."""
    wl = load_workload(RESULTS / run_id / "workload.py")
    for c in wl.get("CANDIDATES", []):
        if c.get("name") == arm:
            validate_spec(c)
            return c
    sys.exit(f"arm {arm} not found in {run_id}")


def cmd_freeze(a):
    freeze_spec(round_spec(a.run_id, a.arm), {"run_id": a.run_id, "arm": a.arm})
    print(f"frozen workload: {a.arm} from {a.run_id}; manifest updated")


# ---------------------------------------------------------------- causes
def _interf_level(ev, ep):
    return ((ev.get("interference") or {}).get(ep) or {}).get("p95_level_ms")


def compare_to_reference(ref, ev):
    """Effect of a one-factor variation against the frozen workload of the same round and seed (variant/ref)."""
    byt = lambda e: (e.get("hot_resp_bytes") or {}).get("sen_slope_per_h")  # noqa: E731
    return {"lat_ratio": _ratio(hot_slope(ev), hot_slope(ref)),
            "bytes_ratio": _ratio(byt(ev), byt(ref)),
            "cost_ratio": _ratio(ev.get("hot_cost_ms_per_mb"), ref.get("hot_cost_ms_per_mb")),
            "assoc_p95_ratio": _ratio(_interf_level(ev, "associate_card"), _interf_level(ref, "associate_card")),
            "ref4_p95_ratio": _ratio(_interf_level(ev, "retrieve_ref4"), _interf_level(ref, "retrieve_ref4"))}


def cmd_causes(a):
    """Causes round: slot 0 = frozen workload (reference); slots 1..3 = variations changing exactly one arg."""
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("causes" + ("-smoke" if a.smoke else ""), a.run_id)
    ref_spec = dict(frozen_spec(), name=CAUSES_REF)
    try:
        cands = _load_candidates(ROOT / "workload.py", CAUSES_REF)
        factors = {c["name"]: one_factor_diff(ref_spec, c) for c in cands}
    except Exception as e:
        return _reject("causes", run_id, run_dir, e)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    shutil.copy(ROOT / "workload.py", run_dir / "workload.py")
    arms = [{"label": CAUSES_REF, "app_dir": INSTRUMENTED, "spec": ref_spec, "seed": a.seed}]
    arms += [{"label": c["name"], "app_dir": INSTRUMENTED, "spec": c, "seed": a.seed} for c in cands]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    ref = res[CAUSES_REF]
    ref_ok = bool(ref.get("valid") and ref.get("score_eligible"))
    rows = []
    for label, ev in res.items():
        if label != CAUSES_REF:
            k, v_ref, v_cand = factors[label]
            ev["factor"] = {"arg": k, "reference": v_ref, "variant": v_cand}
            ev["effect"] = compare_to_reference(ref, ev) if (ref_ok and ev.get("valid")) else None
        rows.append(ev)
    rec = {"event": "causes", "run_id": run_id, "commit": git_head(), "smoke": a.smoke, "minutes": minutes,
           "seed": a.seed, "reference_ok": ref_ok, "frozen_args": ref_spec.get("args", {}), "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "valid": r.get("valid"), "flags": r.get("flags"), "factor": r.get("factor"),
         "effect": r.get("effect")} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_causes(run_id, rows, ref_ok)


def print_causes(run_id, rows, ref_ok):
    print(f"---\nrun_id: {run_id}\nreference_ok: {ref_ok}")
    for r in rows:
        lat, byt = r.get("hot_latency_p50") or {}, r.get("hot_resp_bytes") or {}
        f, e = r.get("factor") or {}, r.get("effect") or {}
        fac = f"{f['arg']}:{f['reference']}->{f['variant']}" if f else "-"
        print(f"arm={r['arm']} factor={fac} valid={r.get('valid')} eligible={r.get('score_eligible')} "
              f"hot_lat_slope={fmt(lat.get('sen_slope_per_h'))} hot_lat_change={fmt(lat.get('sen_change_pct'), 1)}% "
              f"hr_p={fmt(lat.get('mk_p_upward'), 5)} hot_bytes_slope={fmt(byt.get('sen_slope_per_h'), 0)} "
              f"cost_ms_per_mb={fmt(r.get('hot_cost_ms_per_mb'), 1)} "
              f"lat_ratio={fmt(e.get('lat_ratio'))} bytes_ratio={fmt(e.get('bytes_ratio'))} "
              f"cost_ratio={fmt(e.get('cost_ratio'))} assoc_p95_ratio={fmt(e.get('assoc_p95_ratio'))} "
              f"ref4_p95_ratio={fmt(e.get('ref4_p95_ratio'))} db_rows={r.get('db_rows')} "
              f"err={fmt(r.get('error_rate'), 4)} flags={','.join(r.get('flags', [])) or '-'}")


# ---------------------------------------------------------------- ref4 diagnostic, controls and verification
def ref4_vs_control(rows, control="R0b"):
    """Diagnostic only (no effect on scores): ref4 latency of each arm against R0b in the same round and seed.
    Global drift raises ref4 in R0b too; interference from the aging raises it only in the accumulating arm."""
    ctl = next((r for r in rows if r.get("arm") == control), None)
    if not ctl or not ctl.get("ref4_diag"):
        return
    c = ctl["ref4_diag"]
    for r in rows:
        d = r.get("ref4_diag")
        if not d:
            continue
        if c["up"]:
            interp = "likely_global_drift"
        elif d["up"]:
            interp = "interference_arm_only"
        else:
            interp = "flat"
        r["ref4_vs_R0b"] = {"arm_p50_change_pct": d["p50_change_pct"], "arm_p50_mk_p": d["p50_mk_p"],
                            "R0b_p50_change_pct": c["p50_change_pct"], "R0b_p50_mk_p": c["p50_mk_p"],
                            "interpretation": interp}


def ref4_diag_str(r):
    d, v = r.get("ref4_diag") or {}, r.get("ref4_vs_R0b")
    s = f"ref4_diag[p50={fmt(d.get('p50_change_pct'), 1)}% p={fmt(d.get('p50_mk_p'), 5)}"
    if v:
        s += f" R0b={fmt(v['R0b_p50_change_pct'], 1)}% -> {v['interpretation']}"
    return s + "]"


def flat(ev):
    """Flat control: valid, with no rising latency, response size or RSS (same thresholds as the score)."""
    return bool(ev.get("valid") and not ev.get("latency_up") and not ev.get("bytes_up")
                and not ev.get("rss_upward_significant"))


def controls_ok(rows):
    """Phase 0 passes if R0 and R0b stay flat and R1 is eligible."""
    by = {r.get("arm"): r for r in rows}
    return bool(by.get("R0", {}).get("flat") and by.get("R0b", {}).get("flat")
                and by.get("R1", {}).get("score_eligible"))


def check_decision(rows):
    """Automatic decision of a verification round (R1, workload, R0, R0b, same seed): invalid if any arm is
    invalid (infrastructure); positive if R0 and R0b stay flat and R1 and the workload are eligible; else negative."""
    by = {r.get("arm"): r for r in rows}
    if len(by) < 4 or not all(r.get("valid") for r in by.values()):
        return "invalid"
    if (flat(by.get("R0", {})) and flat(by.get("R0b", {})) and by.get("R1", {}).get("score_eligible")
            and by.get(CHECK_LOAD, {}).get("score_eligible")):
        return "positive"
    return "negative"


def run_controls(seed, minutes=DEFAULT_MINUTES, smoke=False, with_r1=True, run_id=None):
    """R0 (F1_control) and R0b (F1_targeted_v1 --key-space 4), with R1 alongside. There is no positive control:
    if a control "detects" aging, the judge thresholds are too loose and no detection is valid."""
    verify(quiet=True)
    minutes = 2 if smoke else minutes
    run_id, run_dir = new_run_dir("controls" + ("-smoke" if smoke else ""), run_id)
    arms = [{"label": "R0", "app_dir": INSTRUMENTED, "spec": R0, "seed": seed},
            {"label": "R0b", "app_dir": INSTRUMENTED, "spec": R0B, "seed": seed}]
    if with_r1:
        arms.append({"label": "R1", "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": seed})
    res = run_arms(arms, minutes, run_dir, smoke)
    rows = []
    for arm in arms:
        ev = res[arm["label"]]
        ev["expected_flat"] = arm["label"] != "R1"
        ev["flat"] = flat(ev)
        rows.append(ev)
    ref4_vs_control(rows)
    ok = controls_ok(rows) if with_r1 else None
    rec = {"event": "controls", "run_id": run_id, "commit": git_head(), "smoke": smoke,
           "minutes": minutes, "seed": seed, "ok": ok, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "flat": r["flat"], "valid": r.get("valid"), "flags": r.get("flags"),
         "hot_lat_slope": hot_slope(r)} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_controls(run_id, rows)
    return {"run_id": run_id, "ok": ok}


def cmd_controls(a):
    if a.background:
        run_id, run_dir = new_run_dir("controls" + ("-smoke" if a.smoke else ""), a.run_id)
        return background(sys.argv[1:], run_id, run_dir)
    run_controls(a.seed, a.minutes, a.smoke, a.with_r1, a.run_id)


def print_controls(run_id, rows):
    print(f"---\nrun_id: {run_id}")
    for r in rows:
        lat, byt, rss = r.get("hot_latency_p50") or {}, r.get("hot_resp_bytes") or {}, r.get("rss_mb") or {}
        ok = r["flat"] == r["expected_flat"]
        print(f"arm={r['arm']} valid={r.get('valid')} expected={'flat' if r['expected_flat'] else 'rising'} "
              f"flat={r['flat']} ok={ok} hot_lat_slope={fmt(lat.get('sen_slope_per_h'))} "
              f"hot_lat_change={fmt(lat.get('sen_change_pct'), 1)}% hr_p={fmt(lat.get('mk_p_upward'), 5)} "
              f"bytes_change={fmt(byt.get('sen_change_pct'), 1)}% rss_slope={fmt(rss.get('sen_slope_per_h'))} "
              f"rss_mk_p={fmt(rss.get('mk_p_upward'), 5)} req={r.get('requests')} db_rows={r.get('db_rows')} "
              f"{ref4_diag_str(r)} flags={','.join(r.get('flags', [])) or '-'}")


def run_check(spec, seed, minutes=DEFAULT_MINUTES, smoke=False, run_id=None, source=None):
    """Verification round: the candidate workload next to R1, R0 and R0b, all with the same seed and a clean DB."""
    verify(quiet=True)
    minutes = 2 if smoke else minutes
    run_id, run_dir = new_run_dir("check" + ("-smoke" if smoke else ""), run_id)
    load = {k: v for k, v in spec.items() if not k.startswith("_")}
    load["name"] = CHECK_LOAD
    arms = [{"label": "R1", "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": seed},
            {"label": CHECK_LOAD, "app_dir": INSTRUMENTED, "spec": load, "seed": seed},
            {"label": "R0", "app_dir": INSTRUMENTED, "spec": R0, "seed": seed},
            {"label": "R0b", "app_dir": INSTRUMENTED, "spec": R0B, "seed": seed}]
    res = run_arms(arms, minutes, run_dir, smoke)
    rows = [res[x["label"]] for x in arms]
    for r in rows:
        if r.get("arm") in ("R0", "R0b"):
            r["flat"] = flat(r)
    ref4_vs_control(rows)
    decision = check_decision(rows)
    rec = {"event": "check", "run_id": run_id, "commit": git_head(), "smoke": smoke, "minutes": minutes,
           "seed": seed, "load_args": spec.get("args", {}), "load_name": spec.get("name"), "source": source,
           "decision": decision, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "valid": r.get("valid"), "eligible": r.get("score_eligible"), "flat": r.get("flat"),
         "flags": r.get("flags")} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_check(run_id, rows, decision)
    return {"run_id": run_id, "decision": decision}


def print_check(run_id, rows, decision):
    print(f"---\nrun_id: {run_id}\ndecision: {decision}")
    print_detect(run_id, rows, header=False)


def cmd_check(a):
    spec = round_spec(a.run, a.arm)
    run_check(spec, a.seed, a.minutes, a.smoke, a.run_id, source={"run_id": a.run, "arm": a.arm})


def cmd_rehash(a):
    if not MANIFEST.exists():
        sys.exit("manifest missing: run 'python prepare.py setup' (human)")
    m = write_manifest()
    print(f"manifest rebuilt: {len(m)} fixed files")


# ---------------------------------------------------------------- experiment (orchestrator)
def _judge_records():
    if not JUDGE_LOG.exists():
        return []
    out = []
    for line in JUDGE_LOG.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def find_round(event, seed, after_ts):
    """The round (round.json + candidates) the agent ran with this seed after after_ts, or None."""
    for rec in reversed(_judge_records()):
        if rec.get("ts", "") < after_ts:
            break
        if rec.get("event") == event and rec.get("seed") == seed and not rec.get("smoke") and "summary" in rec:
            d = RESULTS / rec["run_id"]
            rnd = json.loads((d / "round.json").read_text())
            cands = {}
            if (d / "workload.py").exists():
                for c in load_workload(d / "workload.py").get("CANDIDATES", []):
                    cands[c.get("name")] = c
            return {"run_id": rec["run_id"], "arms": rnd.get("arms", []), "candidates": cands}
    return None


def call_agent(campaign, prompt, tag):
    """One headless Claude Code call with the permissions of the campaign."""
    shutil.copy(ROOT / ".claude" / f"settings.{campaign}.json", ROOT / ".claude" / "settings.json")
    (ROOT / "logs").mkdir(exist_ok=True)
    log = ROOT / "logs" / f"{tag}-{campaign}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    with open(log, "wb") as f:
        try:
            subprocess.run(["claude", "-p", prompt, "--output-format", "stream-json", "--verbose"], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT, timeout=AGENT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            f.write(b'\n{"orchestrator": "agent call timed out"}\n')


def cmd_experiment(a):
    """Full experiment, no human in the middle: phase 0 -> search -> verification -> causes -> resume -> report."""
    from harness.load import normalize_args
    from harness import orchestrator as O
    verify(quiet=True)
    exp_dir = RESULTS / f"experiment-{a.tag}"
    if exp_dir.exists():
        sys.exit(f"{exp_dir} already exists: use another tag")
    exp_dir.mkdir(parents=True)
    branch = f"autoresearch/{a.tag}"
    if subprocess.run(["git", "checkout", "-b", branch], cwd=ROOT).returncode != 0:
        sys.exit(f"could not create branch {branch}")
    minutes, smoke = (2 if a.smoke else a.minutes), a.smoke
    log_judge({"event": "experiment_start", "tag": a.tag, "commit": git_head(), "smoke": smoke})

    def agent_round(campaign, round, seed, campaign_n=None):
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
        if campaign == "detect":
            prompt = f"Read program.md and run ONE round of the detect campaign. Round {round}, seed {seed}."
            event = "detect"
        else:
            prompt = (f"Read program.md and run ONE round of the causes campaign. Campaign {campaign_n}, "
                      f"round {round} of {O.CAUSES_ROUNDS}, seed {seed}.")
            event = "causes"
        if smoke:
            prompt += " Use --smoke."
        call_agent("detect" if campaign == "detect" else "causes", prompt, a.tag)
        return find_round(event, seed, t0)

    def agent_summary(campaign, campaign_n=None):
        if campaign == "detect":
            call_agent("detect", "Read program.md and write the summary of the detect campaign.", a.tag)
        else:
            call_agent("causes", f"Read program.md and write the summary of the causes campaign {campaign_n}.", a.tag)

    def save(state):
        (exp_dir / "state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False))

    orch = O.Orchestrator(
        run_controls=lambda seed: run_controls(seed, minutes, smoke),
        agent_round=agent_round,
        run_check=lambda spec, seed: run_check(spec, seed, minutes, smoke),
        freeze=freeze_spec, agent_summary=agent_summary,
        normalize=normalize_args, save=save, reference_name=BASELINE["name"])
    state = orch.run()
    (exp_dir / "report.md").write_text(O.report_markdown(state, a.tag))
    log_judge({"event": "experiment_end", "tag": a.tag, "status": state["status"],
               "confirmed": [c["load"]["name"] for c in state["confirmed"]]})
    print(f"experiment {a.tag}: {state['status']} · report in {exp_dir / 'report.md'}")


# ---------------------------------------------------------------- wait
def cmd_wait(a):
    d = RESULTS / a.run_id
    deadline = time.time() + min(a.max_seconds, 540)
    while time.time() < deadline:
        st = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "unknown"
        if st == "done":
            rec = json.loads((d / "round.json").read_text())
            if rec.get("event") == "check" and "arms" in rec:
                print_check(a.run_id, rec["arms"], rec.get("decision"))
            elif rec.get("event") == "causes" and "arms" in rec:
                print_causes(a.run_id, rec["arms"], rec.get("reference_ok"))
            elif rec.get("event") == "controls":
                print_controls(a.run_id, rec["arms"])
            elif "arms" in rec:
                print_detect(a.run_id, rec["arms"])
            else:
                print(json.dumps(rec, ensure_ascii=False))
            return
        time.sleep(15)
    print(f"status: running ({a.run_id}); call wait again")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("setup"); s.add_argument("--src", default=str(CESAR_REPO_DEFAULT)); s.add_argument("--force", action="store_true")
    sp.add_parser("check-oracles")
    s = sp.add_parser("detect")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--background", action="store_true")
    s.add_argument("--run-id")
    s = sp.add_parser("check-load", help="verification round for one arm of a search round")
    s.add_argument("--run", required=True); s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True); s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    s.add_argument("--smoke", action="store_true"); s.add_argument("--run-id")
    s = sp.add_parser("experiment", help="full experiment, no human in the middle")
    s.add_argument("--tag", required=True); s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    s.add_argument("--smoke", action="store_true")
    s = sp.add_parser("controls")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES); s.add_argument("--seed", type=int, default=100)
    s.add_argument("--with-r1", action="store_true", help="run R1 alongside (contrast)")
    s.add_argument("--smoke", action="store_true"); s.add_argument("--background", action="store_true")
    s.add_argument("--run-id")
    sp.add_parser("rehash")
    s = sp.add_parser("freeze"); s.add_argument("run_id"); s.add_argument("arm")
    s = sp.add_parser("causes")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--background", action="store_true")
    s.add_argument("--run-id")
    s = sp.add_parser("wait"); s.add_argument("run_id"); s.add_argument("--max-seconds", type=int, default=540)
    sp.add_parser("verify")
    a = ap.parse_args()
    {"setup": cmd_setup, "check-oracles": cmd_check_oracles, "detect": cmd_detect, "check-load": cmd_check,
     "experiment": cmd_experiment, "controls": cmd_controls, "rehash": cmd_rehash, "freeze": cmd_freeze,
     "causes": cmd_causes, "wait": cmd_wait, "verify": lambda _: verify()}[a.cmd](a)


if __name__ == "__main__":
    main()
