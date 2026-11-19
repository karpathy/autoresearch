"""Juiz fixo do AutoResearch-Aging. O AGENTE NÃO PODE EDITAR ESTE ARQUIVO.

Papel equivalente ao prepare.py do karpathy/autoresearch: medição e avaliação
fixas. O agente edita workload.py (detecção) ou candidates/vN/ (reparo).

Comandos (todos a partir da raiz do repositório):
  python prepare.py setup --src <llm-aging-detection-agents>     # humano, uma vez
  python prepare.py detect [--minutes 60] [--seed 1] [--background]
  python prepare.py freeze <run_id> <arm>                          # humano, fim da detecção
  python prepare.py repair-init                                    # cria candidates/v1..v3
  python prepare.py repair [--minutes 60] [--background]
  python prepare.py promote <vN>                                   # adota um candidato como base
  python prepare.py confirm --phase detect|repair [--minutes 60]   # 3 sementes + referência
  python prepare.py control [--rates 0,2.5,5,10] [--minutes 60]   # humano: vazamento sintético (controle positivo)
  python prepare.py rehash                                         # humano: refaz o manifesto após mudar o juiz
  python prepare.py wait <run_id>                                  # bloqueia até 9 min
  python prepare.py verify                                         # confere hashes do juiz
Use --smoke (2 min) só para checar o fluxo; smoke nunca entra no ranking.
"""

import argparse
import difflib
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
# depois do setup, tudo roda no venv travado (mesmas versões do Cesar), mesmo que o agente chame "python prepare.py"
_VPY = ROOT / ".cache" / "venv" / "bin" / "python"
if _VPY.exists() and Path(sys.prefix).resolve() != (ROOT / ".cache" / "venv").resolve():
    os.execv(str(_VPY), [str(_VPY), *sys.argv])

from harness.instance import Instance, NODE_MODULES, copy_app, copy_writable  # noqa: E402
from harness.load import BASELINE, LoadRun, validate_spec, WorkloadError  # noqa: E402
from harness.evaluate import evaluate_arm, ALPHA, MIN_EFFECT_MB_PER_H  # noqa: E402
from harness import gates  # noqa: E402
from harness.workload_check import load_workload, check_determinism  # noqa: E402

APP = "credit-card-fastapi"
APPS = ROOT / "apps" / APP
INSTRUMENTED = APPS / "instrumented"   # alvo da detecção e base do reparo (código do Cesar + sonda)
REPAIRED = APPS / "repaired"           # reparo de referência do Cesar: só para comparar, nunca entra num braço do agente
ORIGINAL = INSTRUMENTED                # nome histórico: "o código que o reparo tenta melhorar"
VENV = ROOT / ".cache" / "venv"
VENV_PY = VENV / "bin" / "python"
CESAR_REPO_DEFAULT = ROOT.parent / "llm-aging-detection-agents"
# arquivos copiados do repo do Cesar, exatamente como ele os usou (origem -> destino)
CESAR_FILES = {
    "findings/bax-fastapi-credit-card-F1.yaml": "findings/bax-fastapi-credit-card-F1.yaml",
    "apps/credit-card-fastapi/campaign.yaml": "findings/campaign.yaml",
    "reference/tests/test_credit_card_service.py": "oracle/test_credit_card_service.py",
    "workloads/bax-fastapi-credit-card-F1_targeted_v1.py": "workloads/bax-fastapi-credit-card-F1_targeted_v1.py",
    "workloads/bax-fastapi-credit-card-F1_targeted_v1.md": "workloads/bax-fastapi-credit-card-F1_targeted_v1.md",
    "workloads/bax-fastapi-credit-card-F1_control.py": "workloads/bax-fastapi-credit-card-F1_control.py",
}
HARNESS_REQS = ["psutil", "httpx", "requests", "pymannkendall", "numpy", "pyyaml"]
CANDS = ROOT / "candidates"
FROZEN = ROOT / "frozen" / "workload_frozen.py"
RESULTS = ROOT / "results"
JUDGE_DIR = ROOT / ".judge"
MANIFEST = JUDGE_DIR / "manifest.json"
JUDGE_LOG = RESULTS / "judge_log.jsonl"
SLOTS = 4                       # 4 instâncias em paralelo (decisão de 20/10)
DEFAULT_MINUTES = 60            # janela da reunião: sinal em ~1 h, não 48 h
CONFIRM_SEEDS = (11, 12, 13)
CONTROL_RATES = (0.0, 2.5, 5.0, 10.0)  # MiB/h injetados por harness/leak_inject.js
FIXED_GLOBS = ["prepare.py", "harness/*.py", "oracle/*.py", "workloads/*.py", "findings/*.yaml"]
HIDDEN_KEY = "@hidden/hidden_tests.py"  # fora do repo: harness/gates.py:HIDDEN


# ---------------------------------------------------------------- integridade
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixed_files():
    files = []
    for g in FIXED_GLOBS:
        files += sorted(ROOT.glob(g))
    if FROZEN.exists():
        files.append(FROZEN)
    for app in (INSTRUMENTED, REPAIRED):
        files += sorted(p for p in app.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                        and p.suffix != ".pyc" and not p.name.startswith("db.sqlite3"))
    if (APPS / "SOURCE.json").exists():
        files.append(APPS / "SOURCE.json")
    if gates.HIDDEN.exists():
        files.append(gates.HIDDEN)
    return files


def _key(p):
    return HIDDEN_KEY if p == gates.HIDDEN else str(p.relative_to(ROOT))


def _path(k):
    return gates.HIDDEN if k == HIDDEN_KEY else ROOT / k


def write_manifest():
    JUDGE_DIR.mkdir(exist_ok=True)
    if not gates.HIDDEN.exists():
        sys.exit(f"testes escondidos ausentes em {gates.HIDDEN}: defina AGING_HIDDEN_DIR ou mova o arquivo")
    m = {_key(p): _sha(p) for p in fixed_files()}
    MANIFEST.write_text(json.dumps(m, indent=1))
    return m


def verify(quiet=False):
    if not MANIFEST.exists():
        sys.exit("manifesto ausente: rode 'python prepare.py setup' (humano)")
    m = json.loads(MANIFEST.read_text())
    now = {_key(p) for p in fixed_files()}
    bad = [k for k, v in m.items() if not _path(k).exists() or _sha(_path(k)) != v]
    bad += sorted(now - set(m))
    if bad:
        log_judge({"event": "juiz_alterado", "files": bad, "flag": "H5_juiz_ou_oraculo_alterado"})
        sys.exit("ABORTADO: arquivos fixos alterados -> " + ", ".join(bad))
    if not quiet:
        print("verify: ok", len(m), "arquivos fixos íntegros")


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
        sys.exit("setup já feito. Use --force apenas como humano, fora da campanha.")
    src = Path(a.src).expanduser().resolve()
    cc = src / "apps" / APP
    if not (cc / "instrumented" / "app.py").exists():
        sys.exit(f"{cc}/instrumented/app.py não encontrado: passe --src <clone de chas42/llm-aging-detection-agents>")
    # o repo do Cesar é só LIDO; tudo que o juiz usa é copiado para cá e entra no manifesto
    for variant in ("instrumented", "repaired"):
        dst = APPS / variant
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(cc / variant, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "db.sqlite3*"))
    for rel_src, rel_dst in CESAR_FILES.items():
        (ROOT / rel_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel_src, ROOT / rel_dst)
    try:
        commit, dirty = _git_out(src, "rev-parse", "HEAD"), bool(_git_out(src, "status", "--porcelain"))
    except Exception:
        commit, dirty = "desconhecido", None
    (APPS / "SOURCE.json").write_text(json.dumps({
        "repo": "https://github.com/chas42/llm-aging-detection-agents", "commit": commit, "dirty": dirty,
        "copied": sorted(CESAR_FILES) + [f"apps/{APP}/instrumented", f"apps/{APP}/repaired"]}, indent=1))
    print("venv travado (requirements.txt da app + dependências do juiz)...")
    if VENV.exists():
        shutil.rmtree(VENV)
    VENV.parent.mkdir(exist_ok=True)
    # sem ensurepip (python3-venv ausente) o venv nasce sem pip: instalamos de fora com `pip --python`
    subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", str(VENV)])
    pip = [sys.executable, "-m", "pip", "--python", str(VENV_PY)]
    subprocess.check_call(pip + ["install", "-q", "-r", str(INSTRUMENTED / "requirements.txt")])
    subprocess.check_call(pip + ["install", "-q", *HARNESS_REQS])
    with open(VENV.parent / "venv-freeze.txt", "w") as f:
        subprocess.check_call(pip + ["freeze"], stdout=f)
    m = write_manifest()
    print(f"setup (cópia + venv) ok: {len(m)} arquivos fixos no manifesto. "
          "Próximo: python prepare.py check-oracles (sobe a app e roda os oráculos nas duas variantes).")


def cmd_check_oracles(a):
    """Oráculo funcional + testes escondidos contra instrumented e repaired do Cesar (ambos devem passar)."""
    bad = {}
    for name, app in (("instrumented", INSTRUMENTED), ("repaired", REPAIRED)):
        rep = gates.functional_and_hidden(app, Path(tempfile.mkdtemp()))
        print(f"{name}: funcional={rep[0] or 'ok'} escondidos={rep[1] or 'ok'} persistência={rep[2] or 'ok'}")
        if any(rep):
            bad[name] = rep
    if bad:
        sys.exit("oráculos falham contra o código do Cesar; revise os testes escondidos antes de seguir")
    print("check-oracles: ok")


# ---------------------------------------------------------------- utilidades
def frozen_spec():
    if not FROZEN.exists():
        sys.exit("frozen/workload_frozen.py ausente: o humano roda 'prepare.py freeze' ao fim da detecção")
    try:
        ns = load_workload(FROZEN)
    except WorkloadError as e:
        sys.exit(f"carga congelada rejeitada: {e}")
    name = ns["FROZEN_ARM"]
    if name == BASELINE["name"]:
        return BASELINE
    for c in ns["CANDIDATES"]:
        if c["name"] == name:
            return c
    sys.exit(f"candidato congelado {name} não encontrado")


def same_as_baseline(spec):
    """Carga equivalente à baseline (mesmos parâmetros e mesma sequência) não é hipótese nova."""
    import random as _r
    if spec["threads"] != BASELINE["threads"] or abs(spec["think_time_s"] - BASELINE["think_time_s"]) > 1e-9:
        return False
    for seed in (1, 2):
        ra, rb = _r.Random(seed), _r.Random(seed)
        for i in range(400):
            if spec["request"](ra, i) != BASELINE["request"](rb, i):
                return False
    return True


def run_arms(arms, minutes, run_dir, smoke):
    """arms: lista de dicts {label, app_dir, spec, seed}. Roda todos em paralelo."""
    seconds = int(minutes * 60)
    probe_ms = 2000 if smoke else 5000
    results = {}

    def one(arm):
        wd = run_dir / arm["label"]
        inst = Instance(arm["app_dir"], wd, probe_interval_ms=probe_ms, sample_s=probe_ms / 1000,
                        leak_mb_per_h=arm.get("leak_mb_per_h", 0.0))
        try:
            inst.prepare()
            port = inst.start()
            inst.start_sampler()
            load = LoadRun(arm["spec"], port, seconds, arm["seed"]).run()
            time.sleep(probe_ms / 1000 * 1.5)  # última amostra pós-GC
            alive = inst.alive()
            pids = set(inst.pids_seen)
            children = inst.max_children
        except Exception as e:
            results[arm["label"]] = {"arm": arm["label"], "valid": False, "flags": [f"erro: {e}"]}
            inst.stop()
            return
        inst.stop()
        ev = evaluate_arm(arm["label"], inst.probe_rows(), load, inst.db_state(), pids, alive, seconds,
                          children)
        ev.update({"spec_name": arm["spec"]["name"], "hypothesis": arm["spec"].get("hypothesis", ""),
                   "seed": arm["seed"], "app_dir": str(arm["app_dir"])})
        (wd / "buckets.json").write_text(json.dumps(load["buckets"]))
        (wd / "evaluation.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
        results[arm["label"]] = ev

    ths = [threading.Thread(target=one, args=(a,)) for a in arms]
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


# ---------------------------------------------------------------- detecção
def cmd_detect(a):
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("detect" + ("-smoke" if a.smoke else ""), a.run_id)
    try:
        wl = load_workload(ROOT / "workload.py")
        if not isinstance(wl.get("CANDIDATES"), list):
            raise WorkloadError("workload.py deve definir CANDIDATES como lista")
        cands = list(wl["CANDIDATES"])
        if not 1 <= len(cands) <= SLOTS - 1:
            raise WorkloadError(f"CANDIDATES deve ter de 1 a {SLOTS - 1} itens (1 slot é da baseline)")
        names = [c["name"] for c in cands]
        if len(set(names)) != len(names) or BASELINE["name"] in names:
            raise WorkloadError("nomes de candidatos devem ser únicos e diferentes da baseline")
        for c in cands:
            validate_spec(c)
            check_determinism(c, wl)
    except Exception as e:
        rec = {"event": "detect", "run_id": run_id, "commit": git_head(), "valid": False,
               "flag": "D3_fora_do_contrato", "error": str(e)}
        log_judge(rec)
        (run_dir / "STATUS").write_text("done")
        (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print("INVÁLIDO:", e)
        return
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    shutil.copy(ROOT / "workload.py", run_dir / "workload.py")
    arms = [{"label": BASELINE["name"], "app_dir": ORIGINAL, "spec": BASELINE, "seed": a.seed}]
    arms += [{"label": c["name"], "app_dir": ORIGINAL, "spec": c, "seed": a.seed} for c in cands]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    base = res[BASELINE["name"]]
    base_slope = base.get("rss_mb", {}).get("sen_slope_per_h") if base.get("valid") else None
    rows = []
    copies = {c["name"] for c in cands if same_as_baseline(c)}
    for label, ev in res.items():
        if label in copies:
            ev.setdefault("flags", []).append("D4_copia_da_baseline")
        slope = ev.get("rss_mb", {}).get("sen_slope_per_h")
        p = ev.get("rss_mb", {}).get("mk_p_upward")
        eligible = (ev.get("valid") and ev.get("rss_upward_significant")
                    and not any(f.startswith(("D2", "D4", "E2")) for f in ev.get("flags", [])))
        ev["score_rss_mb_per_h"] = slope if eligible and label != BASELINE["name"] else None
        ev["ratio_vs_baseline"] = (slope / base_slope) if (eligible and base_slope and base_slope > 0) else None
        rows.append(ev)
    rec = {"event": "detect", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "seed": a.seed, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "valid": r.get("valid"), "flags": r.get("flags"),
         "score": r.get("score_rss_mb_per_h"), "p": r.get("rss_mb", {}).get("mk_p_upward")} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_detect(run_id, rows)


def print_detect(run_id, rows):
    print(f"---\nrun_id: {run_id}")
    for r in rows:
        rss = r.get("rss_mb", {})
        print(f"arm={r['arm']} valid={r.get('valid')} score_rss_mb_per_h={fmt(r.get('score_rss_mb_per_h'))} "
              f"rss_slope={fmt(rss.get('sen_slope_per_h'))} mk_p={fmt(rss.get('mk_p_upward'), 5)} "
              f"heap_slope={fmt(r.get('heap_mb', {}).get('sen_slope_per_h'))} "
              f"handles_slope={fmt(r.get('handles', {}).get('sen_slope_per_h'))} "
              f"p95_slope={fmt(r.get('p95_ms', {}).get('sen_slope_per_h'))} "
              f"ttd_min={fmt(r.get('time_to_detect_min'))} ratio={fmt(r.get('ratio_vs_baseline'))} "
              f"req={r.get('requests')} err={fmt(r.get('error_rate'), 4)} db_rows={r.get('db_rows')} "
              f"flags={','.join(r.get('flags', [])) or '-'}")


# ---------------------------------------------------------------- congelar (humano)
def cmd_freeze(a):
    src = RESULTS / a.run_id / "workload.py"
    if not src.exists():
        sys.exit("run sem workload.py salvo")
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(src.read_text() + f"\n\nFROZEN_ARM = {a.arm!r}\nFROZEN_FROM = {a.run_id!r}\n")
    write_manifest()
    print(f"carga congelada: {a.arm} de {a.run_id}; manifesto atualizado")


# ---------------------------------------------------------------- reparo
def cmd_repair_init(a):
    base = CANDS / "base"
    if not base.exists():
        copy_writable(ORIGINAL, base)
    for v in ("v1", "v2", "v3"):
        d = CANDS / v
        if d.exists():
            shutil.rmtree(d)
        copy_writable(base, d)
    print("candidates/v1..v3 criados a partir de candidates/base")


def cmd_promote(a):
    src = CANDS / a.version
    base = CANDS / "base"
    shutil.rmtree(base)
    copy_writable(src, base)
    cmd_repair_init(a)
    print(f"{a.version} promovido a base")


def diff_lines(orig, cand):
    n = 0
    files = {p.relative_to(orig) for p in Path(orig).rglob("*") if p.is_file()}
    files |= {p.relative_to(cand) for p in Path(cand).rglob("*") if p.is_file() and "node_modules" not in p.parts}
    for f in sorted(files):
        a_ = (Path(orig) / f).read_text(errors="ignore").splitlines() if (Path(orig) / f).exists() else []
        b_ = (Path(cand) / f).read_text(errors="ignore").splitlines() if (Path(cand) / f).exists() else []
        n += sum(1 for l in difflib.unified_diff(a_, b_, lineterm="", n=0)
                 if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return n


def reference_replay(spec, trace):
    key = hashlib.sha256((json.dumps(trace) + _sha(ORIGINAL / "package.json")).encode()).hexdigest()[:16]
    cache = JUDGE_DIR / f"replay_ref_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ref = gates.replay(ORIGINAL, trace, Path(tempfile.mkdtemp()))
    cache.write_text(json.dumps(ref))
    return ref


def cmd_repair(a):
    verify(quiet=True)
    spec = frozen_spec()
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("repair" + ("-smoke" if a.smoke else ""), a.run_id)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    versions = [v for v in ("v1", "v2", "v3") if (CANDS / v).exists()]
    trace = gates.build_trace(spec)
    ref = reference_replay(spec, trace)
    gate_reports = {}
    for v in versions:
        gate_reports[v] = gates.run_gates(ORIGINAL, CANDS / v, ref, trace, run_dir)
        gate_reports[v]["diff_lines"] = diff_lines(ORIGINAL, CANDS / v)
        if gate_reports[v]["diff_lines"] == 0:
            gate_reports[v]["flags"].append("sem_alteracao")
            gate_reports[v]["passed"] = False
        print(f"gates {v}: passed={gate_reports[v]['passed']} flags={gate_reports[v]['flags']} "
              f"diff_lines={gate_reports[v]['diff_lines']}", flush=True)
    for v in versions:  # snapshot do que foi avaliado
        shutil.copytree(CANDS / v, run_dir / "snapshot" / v)
    arms = [{"label": "original", "app_dir": ORIGINAL, "spec": spec, "seed": a.seed}]
    arms += [{"label": v, "app_dir": CANDS / v, "spec": spec, "seed": a.seed}
             for v in versions if gate_reports[v]["passed"]]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    rows = [score_repair(res, v, gate_reports.get(v)) for v in ["original"] + versions]
    rec = {"event": "repair", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "frozen_arm": spec["name"], "rows": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "rows"} | {"summary": [
        {"arm": r["arm"], "score": r.get("score_reduction"), "flags": r.get("flags")} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_repair(run_id, rows)


def score_repair(res, v, gate):
    o = res.get("original", {})
    if v == "original":
        return {"arm": "original", "valid": o.get("valid"), "rss": o.get("rss_mb"),
                "throughput_mean_rps": o.get("throughput_mean_rps"), "p95_ms_median": o.get("p95_ms_median"),
                "flags": o.get("flags", [])}
    row = {"arm": v, "gates_passed": gate["passed"], "flags": list(gate["flags"]),
           "diff_lines": gate["diff_lines"], "gate_details": gate["details"], "score_reduction": None}
    ev = res.get(v)
    if not gate["passed"] or not ev:
        return row
    row["flags"] += ev.get("flags", [])
    s0 = o.get("rss_mb", {}).get("sen_slope_per_h")
    s1 = ev.get("rss_mb", {}).get("sen_slope_per_h")
    p1 = ev.get("rss_mb", {}).get("mk_p_upward")
    row.update({"rss": ev.get("rss_mb"), "heap": ev.get("heap_mb"), "handles": ev.get("handles"),
                "throughput_mean_rps": ev.get("throughput_mean_rps"), "p95_ms_median": ev.get("p95_ms_median")})
    if not ev.get("valid") or not o.get("valid"):
        return row
    thr_ok = ev["throughput_mean_rps"] >= 0.9 * o["throughput_mean_rps"]
    lat_ok = ev["p95_ms_median"] <= 1.1 * o["p95_ms_median"] + 1.0
    if not (thr_ok and lat_ok):
        row["flags"].append("H6_desempenho_degradado")
        return row
    # 5xx por SQLITE_BUSY variam entre execuções: tolerância relativa + absoluta
    if ev.get("server_error_rate", 0) > 1.5 * o.get("server_error_rate", 0) + 0.005:
        row["flags"].append("H7_mais_erros_sob_carga")
        return row
    if s0 is None or s0 < MIN_EFFECT_MB_PER_H or o.get("rss_mb", {}).get("mk_p_upward", 1) >= ALPHA:
        row["flags"].append("original_sem_tendencia_nesta_rodada")
        return row
    reduction = 1 - (s1 / s0)
    row["score_reduction"] = reduction
    row["fixed"] = (p1 >= 0.05) or (s1 < MIN_EFFECT_MB_PER_H) or reduction >= 0.8
    return row


def print_repair(run_id, rows):
    print(f"---\nrun_id: {run_id}")
    for r in rows:
        rss = r.get("rss") or {}
        print(f"arm={r['arm']} gates={r.get('gates_passed', '-')} score_reduction={fmt(r.get('score_reduction'))} "
              f"fixed={r.get('fixed', '-')} rss_slope={fmt(rss.get('sen_slope_per_h'))} "
              f"mk_p={fmt(rss.get('mk_p_upward'), 5)} thr={fmt(r.get('throughput_mean_rps'), 1)} "
              f"p95={fmt(r.get('p95_ms_median'), 2)} diff_lines={r.get('diff_lines', '-')} "
              f"flags={','.join(r.get('flags', [])) or '-'}")


# ---------------------------------------------------------------- confirmação
def cmd_confirm(a):
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir(f"confirm-{a.phase}" + ("-smoke" if a.smoke else ""), a.run_id)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    spec = frozen_spec()
    if a.phase == "detect":
        arms = [{"label": f"{spec['name']}_s{s}", "app_dir": ORIGINAL, "spec": spec, "seed": s} for s in CONFIRM_SEEDS]
        arms.append({"label": "baseline_s11", "app_dir": ORIGINAL, "spec": BASELINE, "seed": 11})
    else:
        arms = [{"label": f"base_s{s}", "app_dir": CANDS / "base", "spec": spec, "seed": s} for s in CONFIRM_SEEDS]
        arms.append({"label": "original_s11", "app_dir": ORIGINAL, "spec": spec, "seed": 11})
    res = run_arms(arms, minutes, run_dir, a.smoke)
    rec = {"event": f"confirm-{a.phase}", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "arms": list(res.values())}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"})
    (run_dir / "STATUS").write_text("done")
    print_detect(run_id, list(res.values()))


# ---------------------------------------------------------------- controle positivo (humano)
def cmd_control(a):
    """Baseline do artigo com vazamento sintético de taxa conhecida, um por slot.
    Responde: a partir de qual taxa o juiz detecta em `minutes`, e se a nota medida se aproxima da injetada."""
    verify(quiet=True)
    rates = [float(r) for r in a.rates.split(",")] if a.rates else list(CONTROL_RATES)
    if not 1 <= len(rates) <= SLOTS:
        sys.exit(f"--rates deve ter de 1 a {SLOTS} taxas")
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("control" + ("-smoke" if a.smoke else ""), a.run_id)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    arms = [{"label": f"leak_{r:g}", "app_dir": ORIGINAL, "spec": BASELINE, "seed": a.seed, "leak_mb_per_h": r}
            for r in rates]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    rows = []
    for arm in arms:
        ev = res[arm["label"]]
        ev["leak_injected_mb_per_h"] = arm["leak_mb_per_h"]
        ev["detected"] = bool(ev.get("valid") and ev.get("rss_upward_significant"))
        rows.append(ev)
    rec = {"event": "control", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "seed": a.seed, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "injected": r["leak_injected_mb_per_h"], "detected": r["detected"],
         "slope": r.get("rss_mb", {}).get("sen_slope_per_h"),
         "min_halves": r.get("rss_mb", {}).get("sen_min_halves_per_h")} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_control(run_id, rows)


def print_control(run_id, rows):
    print(f"---\nrun_id: {run_id}")
    for r in rows:
        rss = r.get("rss_mb", {})
        print(f"arm={r['arm']} injected={fmt(r.get('leak_injected_mb_per_h'), 1)} valid={r.get('valid')} "
              f"detected={r.get('detected')} rss_slope={fmt(rss.get('sen_slope_per_h'))} "
              f"min_halves={fmt(rss.get('sen_min_halves_per_h'))} mk_p={fmt(rss.get('mk_p_upward'), 5)} "
              f"external_slope={fmt(r.get('external_mb', {}).get('sen_slope_per_h'))} "
              f"heap_slope={fmt(r.get('heap_mb', {}).get('sen_slope_per_h'))} "
              f"flags={','.join(r.get('flags', [])) or '-'}")


def cmd_rehash(a):
    if not MANIFEST.exists():
        sys.exit("manifesto ausente: rode 'python prepare.py setup' (humano)")
    m = write_manifest()
    print(f"manifesto refeito: {len(m)} arquivos fixos")


# ---------------------------------------------------------------- espera
def cmd_wait(a):
    d = RESULTS / a.run_id
    deadline = time.time() + min(a.max_seconds, 540)
    while time.time() < deadline:
        st = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else "unknown"
        if st == "done":
            rec = json.loads((d / "round.json").read_text())
            if rec.get("event") == "repair":
                print_repair(a.run_id, rec["rows"])
            elif rec.get("event") == "control":
                print_control(a.run_id, rec["arms"])
            elif "arms" in rec:
                print_detect(a.run_id, rec["arms"])
            else:
                print(json.dumps(rec, ensure_ascii=False))
            return
        time.sleep(15)
    print(f"status: running ({a.run_id}); chame wait novamente")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("setup"); s.add_argument("--src", default=str(CESAR_REPO_DEFAULT)); s.add_argument("--force", action="store_true")
    sp.add_parser("check-oracles")
    for name in ("detect", "repair"):
        s = sp.add_parser(name)
        s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--background", action="store_true")
        s.add_argument("--run-id")
    s = sp.add_parser("confirm"); s.add_argument("--phase", choices=["detect", "repair"], required=True)
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES); s.add_argument("--smoke", action="store_true")
    s.add_argument("--background", action="store_true"); s.add_argument("--run-id")
    s = sp.add_parser("control"); s.add_argument("--rates", help="taxas em MiB/h separadas por vírgula")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--smoke", action="store_true"); s.add_argument("--background", action="store_true")
    s.add_argument("--run-id")
    sp.add_parser("rehash")
    s = sp.add_parser("freeze"); s.add_argument("run_id"); s.add_argument("arm")
    sp.add_parser("repair-init")
    s = sp.add_parser("promote"); s.add_argument("version", choices=["v1", "v2", "v3"])
    s = sp.add_parser("wait"); s.add_argument("run_id"); s.add_argument("--max-seconds", type=int, default=540)
    sp.add_parser("verify")
    a = ap.parse_args()
    {"setup": cmd_setup, "check-oracles": cmd_check_oracles, "detect": cmd_detect, "repair": cmd_repair, "confirm": cmd_confirm,
     "control": cmd_control, "rehash": cmd_rehash, "freeze": cmd_freeze, "repair-init": cmd_repair_init, "promote": cmd_promote,
     "wait": cmd_wait, "verify": lambda _: verify()}[a.cmd](a)


if __name__ == "__main__":
    main()
