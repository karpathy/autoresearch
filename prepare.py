"""Juiz fixo do AutoResearch-Aging. O AGENTE NÃO PODE EDITAR ESTE ARQUIVO.

Papel equivalente ao prepare.py do karpathy/autoresearch: medição e avaliação
fixas. O agente edita só workload.py. Esta versão é só de detecção (ver program.md): não há reparo.

Comandos (todos a partir da raiz do repositório):
  python prepare.py setup --src <llm-aging-detection-agents>     # humano, uma vez
  python prepare.py check-oracles                                  # humano: teste funcional do Cesar na app
  python prepare.py controls [--minutes 60] [--with-r1]            # controles negativos R0 e R0b do Cesar
  python prepare.py detect [--minutes 60] [--seed 1] [--background]
  python prepare.py freeze <run_id> <arm>                          # congela a carga do melhor braço
  python prepare.py confirm [--minutes 480] [--seed 201]           # 8 h: R0, R0b, R1 e o candidato congelado
  python prepare.py causes [--minutes 60] [--seed 301] [--background] # variações de 1 fator vs a carga congelada
  python prepare.py rehash                                         # humano: refaz o manifesto após mudar o juiz
  python prepare.py wait <run_id>                                  # bloqueia até 9 min
  python prepare.py verify                                         # confere hashes do juiz
Use --smoke (2 min) só para checar o fluxo; smoke nunca entra no ranking.
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
# depois do setup, tudo roda no venv travado (mesmas versões do Cesar), mesmo que o agente chame "python prepare.py"
_VPY = ROOT / ".cache" / "venv" / "bin" / "python"
if _VPY.exists() and Path(sys.prefix).resolve() != (ROOT / ".cache" / "venv").resolve():
    os.execv(str(_VPY), [str(_VPY), *sys.argv])

from harness.instance import Instance  # noqa: E402
from harness.load import (BASELINE, R0, R0B, WorkloadError, one_factor_diff, read_client, run_script,  # noqa: E402
                          same_as_default, summarize_client, validate_spec)
from harness.evaluate import evaluate_arm  # noqa: E402
from harness.functional_check import functional_check  # noqa: E402
from harness.workload_check import load_workload  # noqa: E402

APP = "credit-card-fastapi"
APPS = ROOT / "apps" / APP
INSTRUMENTED = APPS / "instrumented"   # alvo do experimento (código do Cesar + sonda aging_probe.py)
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
FROZEN = ROOT / "frozen" / "workload_frozen.py"
RESULTS = ROOT / "results"
JUDGE_DIR = ROOT / ".judge"
MANIFEST = JUDGE_DIR / "manifest.json"
JUDGE_LOG = RESULTS / "judge_log.jsonl"
SLOTS = 4                       # 4 instâncias em paralelo (decisão de 20/10)
DEFAULT_MINUTES = 60            # janela da reunião: sinal em ~1 h, não 48 h
CONFIRM_MINUTES = 480           # confirm: 8 h, como as rodadas do campaign.yaml do Cesar
FIXED_GLOBS = ["prepare.py", "harness/*.py", "oracle/*.py", "workloads/*.py", "findings/*.yaml"]


# ---------------------------------------------------------------- integridade
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
    if INSTRUMENTED.exists():
        shutil.rmtree(INSTRUMENTED)
    shutil.copytree(cc / "instrumented", INSTRUMENTED, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "db.sqlite3*"))
    for rel_src, rel_dst in CESAR_FILES.items():
        (ROOT / rel_dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel_src, ROOT / rel_dst)
    try:
        commit, dirty = _git_out(src, "rev-parse", "HEAD"), bool(_git_out(src, "status", "--porcelain"))
    except Exception:
        commit, dirty = "desconhecido", None
    (APPS / "SOURCE.json").write_text(json.dumps({
        "repo": "https://github.com/chas42/llm-aging-detection-agents", "commit": commit, "dirty": dirty,
        "copied": sorted(CESAR_FILES) + [f"apps/{APP}/instrumented"]}, indent=1))
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
          "Próximo: python prepare.py check-oracles (sobe a app e roda o teste funcional do Cesar).")


def cmd_check_oracles(a):
    """Teste funcional do Cesar contra a app instrumented: confirma que a cópia funciona antes de medir."""
    fails = functional_check(INSTRUMENTED, Path(tempfile.mkdtemp()))
    print(f"instrumented: funcional={fails or 'ok'}")
    if fails:
        sys.exit("o teste funcional do Cesar falha contra a app copiada; revise o setup antes de seguir")
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
            validate_spec(c)
            return c
    sys.exit(f"candidato congelado {name} não encontrado")


def pins(i):
    """Pinagem como a do Cesar (app num núcleo, gerador em outros), por slot: app i -> CPU i, carga -> CPU 4+i.
    Só com >= 8 núcleos e taskset; senão sem pinagem (a saturação do gerador é medida de qualquer forma)."""
    if (os.cpu_count() or 0) >= 8 and shutil.which("taskset"):
        return str(i), str(4 + i)
    return None, None


def run_arms(arms, minutes, run_dir, smoke):
    """arms: lista de dicts {label, app_dir, spec, seed}. Roda todos em paralelo, um slot por braço."""
    seconds = int(minutes * 60)
    sample_s = 2.0 if smoke else 5.0
    results = {}

    def one(slot, arm):
        wd = run_dir / arm["label"]
        app_cpu, load_cpu = pins(slot)
        inst = Instance(arm["app_dir"], wd, sample_s=sample_s, cpus=app_cpu)
        try:
            inst.prepare()
            port = inst.start()
            inst.start_sampler()
            t0 = time.time()
            ran = run_script(arm["spec"], f"http://127.0.0.1:{port}", seconds, arm["seed"], wd, cpus=load_cpu)
            time.sleep(sample_s * 1.5)  # última amostra
            alive, pids, children = inst.alive(), set(inst.pids_seen), inst.max_children
        except Exception as e:
            results[arm["label"]] = {"arm": arm["label"], "valid": False, "flags": [f"erro: {e}"]}
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
                   "seed": arm["seed"], "app_dir": str(arm["app_dir"]), "workload_cmd": ran["cmd"]})
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
            raise WorkloadError(f"CANDIDATES deve ter de 1 a {SLOTS - 1} itens (1 slot é da referência)")
        if not all(isinstance(c, dict) for c in cands):
            raise WorkloadError("cada candidato deve ser um dict")
        names = [c.get("name") for c in cands]
        if len(set(names)) != len(names) or BASELINE["name"] in names:
            raise WorkloadError("nomes de candidatos devem ser únicos e diferentes da referência")
        for c in cands:
            validate_spec(c)
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
    arms = [{"label": BASELINE["name"], "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": a.seed}]
    arms += [{"label": c["name"], "app_dir": INSTRUMENTED, "spec": c, "seed": a.seed} for c in cands]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    base = res[BASELINE["name"]]
    base_slope = hot_slope(base) if base.get("score_eligible") else None
    rows = []
    copies = {c["name"] for c in cands if same_as_default(c)}
    for label, ev in res.items():
        if label in copies:
            ev.setdefault("flags", []).append("D4_copia_da_referencia")
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


def print_detect(run_id, rows):
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


# ---------------------------------------------------------------- congelar (humano)
def cmd_freeze(a):
    src = RESULTS / a.run_id / "workload.py"
    if not src.exists():
        sys.exit("run sem workload.py salvo")
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(src.read_text() + f"\n\nFROZEN_ARM = {a.arm!r}\nFROZEN_FROM = {a.run_id!r}\n")
    write_manifest()
    print(f"carga congelada: {a.arm} de {a.run_id}; manifesto atualizado")


# ---------------------------------------------------------------- causas (fase 4 do program.md)
CAUSES_REF = "referencia_congelada"


def _ratio(num, den):
    return (num / den) if (num is not None and den not in (None, 0)) else None


def _interf_level(ev, ep):
    return ((ev.get("interference") or {}).get(ep) or {}).get("p95_level_ms")


def compare_to_reference(ref, ev):
    """Efeito de uma variação de um fator contra a carga congelada da mesma rodada e semente (razões cand/ref)."""
    byt = lambda e: (e.get("hot_resp_bytes") or {}).get("sen_slope_per_h")  # noqa: E731
    return {"lat_ratio": _ratio(hot_slope(ev), hot_slope(ref)),
            "bytes_ratio": _ratio(byt(ev), byt(ref)),
            "cost_ratio": _ratio(ev.get("hot_cost_ms_per_mb"), ref.get("hot_cost_ms_per_mb")),
            "assoc_p95_ratio": _ratio(_interf_level(ev, "associate_card"), _interf_level(ref, "associate_card")),
            "ref4_p95_ratio": _ratio(_interf_level(ev, "retrieve_ref4"), _interf_level(ref, "retrieve_ref4"))}


def cmd_causes(a):
    """Rodada da fase de causas: slot 0 = carga congelada (referência); slots 1..3 = variações que mudam
    exatamente um arg. Mede o efeito de cada fator contra a referência da mesma rodada e semente."""
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("causes" + ("-smoke" if a.smoke else ""), a.run_id)
    ref_spec = dict(frozen_spec(), name=CAUSES_REF)
    try:
        wl = load_workload(ROOT / "workload.py")
        cands = wl.get("CANDIDATES")
        if not isinstance(cands, list) or not 1 <= len(cands) <= SLOTS - 1:
            raise WorkloadError(f"CANDIDATES deve ser uma lista de 1 a {SLOTS - 1} itens (1 slot é da referência)")
        if not all(isinstance(c, dict) for c in cands):
            raise WorkloadError("cada candidato deve ser um dict")
        names = [c.get("name") for c in cands]
        if len(set(names)) != len(names) or CAUSES_REF in names:
            raise WorkloadError("nomes de candidatos devem ser únicos e diferentes da referência")
        factors = {}
        for c in cands:
            validate_spec(c)
            factors[c["name"]] = one_factor_diff(ref_spec, c)
    except Exception as e:
        rec = {"event": "causes", "run_id": run_id, "commit": git_head(), "valid": False,
               "flag": "D3_fora_do_contrato", "error": str(e)}
        log_judge(rec)
        (run_dir / "STATUS").write_text("done")
        (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print("INVÁLIDO:", e)
        return
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


# ---------------------------------------------------------------- confirmação (8 h, as 4 rodadas do campaign.yaml)
def ref4_vs_control(rows, control="R0b"):
    """Diagnóstico, sem efeito na nota: a latência do ref4 do braço comparada com a do controle (R0b) da mesma
    rodada e semente. Deriva global (disco, page cache, máquina) faz o ref4 subir também no R0b; a interferência
    do aging (travamentos das leituras quentes) só no braço que acumula cartões. Só existe onde há R0b na rodada
    (controls, confirm); nas rodadas de detect não há R0b e o campo fica ausente."""
    ctl = next((r for r in rows if r.get("arm") == control), None)
    if not ctl or not ctl.get("ref4_diag"):
        return
    c = ctl["ref4_diag"]
    for r in rows:
        d = r.get("ref4_diag")
        if not d:
            continue
        if c["up"]:
            interp = "deriva_global_provavel"
        elif d["up"]:
            interp = "so_no_braco_interferencia"
        else:
            interp = "plano"
        r["ref4_vs_R0b"] = {"arm_p50_change_pct": d["p50_change_pct"], "arm_p50_mk_p": d["p50_mk_p"],
                            "R0b_p50_change_pct": c["p50_change_pct"], "R0b_p50_mk_p": c["p50_mk_p"],
                            "interpretacao": interp}


def ref4_diag_str(r):
    d, v = r.get("ref4_diag") or {}, r.get("ref4_vs_R0b")
    s = f"ref4_diag[p50={fmt(d.get('p50_change_pct'), 1)}% p={fmt(d.get('p50_mk_p'), 5)}"
    if v:
        s += f" R0b={fmt(v['R0b_p50_change_pct'], 1)}% -> {v['interpretacao']}"
    return s + "]"


def cmd_confirm(a):
    """Confirmação longa da detecção: R0, R0b, R1 (F1_targeted_v1 padrão) e o candidato congelado, em paralelo,
    com a mesma semente nos quatro. O protocolo (program.md) roda duas execuções, sementes 201 e 202."""
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("confirm-detect" + ("-smoke" if a.smoke else ""), a.run_id)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    spec = frozen_spec()
    arms = [{"label": "R0", "app_dir": INSTRUMENTED, "spec": R0, "seed": a.seed},
            {"label": "R0b", "app_dir": INSTRUMENTED, "spec": R0B, "seed": a.seed},
            {"label": "R1", "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": a.seed},
            {"label": f"congelado_{spec['name']}", "app_dir": INSTRUMENTED, "spec": spec, "seed": a.seed}]
    res = run_arms(arms, minutes, run_dir, a.smoke)
    rows = list(res.values())
    ref4_vs_control(rows)
    rec = {"event": "confirm-detect", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "seed": a.seed, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"})
    (run_dir / "STATUS").write_text("done")
    print_detect(run_id, rows)


# ---------------------------------------------------------------- controles negativos (humano)
def cmd_controls(a):
    """R0 (F1_control) e R0b (F1_targeted_v1 --key-space 4), como no campaign.yaml do Cesar.
    NÃO há controle positivo. Responde: com a mesma métrica e os mesmos limiares, os controles ficam planos?
    Se um controle "detectar" aging, os limiares do juiz estão frouxos e nenhuma detecção vale."""
    verify(quiet=True)
    minutes = 2 if a.smoke else a.minutes
    run_id, run_dir = new_run_dir("controls" + ("-smoke" if a.smoke else ""), a.run_id)
    if a.background:
        return background(sys.argv[1:], run_id, run_dir)
    arms = [{"label": "R0", "app_dir": INSTRUMENTED, "spec": R0, "seed": a.seed},
            {"label": "R0b", "app_dir": INSTRUMENTED, "spec": R0B, "seed": a.seed}]
    if a.with_r1:  # R1 ao lado, para ver o contraste na mesma máquina
        arms.append({"label": "R1", "app_dir": INSTRUMENTED, "spec": BASELINE, "seed": a.seed})
    res = run_arms(arms, minutes, run_dir, a.smoke)
    rows = []
    for arm in arms:
        ev = res[arm["label"]]
        ev["expected_flat"] = arm["label"] != "R1"
        ev["flat"] = bool(ev.get("valid") and not ev.get("latency_up") and not ev.get("bytes_up")
                          and not ev.get("rss_upward_significant"))
        rows.append(ev)
    ref4_vs_control(rows)
    rec = {"event": "controls", "run_id": run_id, "commit": git_head(), "smoke": a.smoke,
           "minutes": minutes, "seed": a.seed, "arms": rows}
    (run_dir / "round.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    log_judge({k: v for k, v in rec.items() if k != "arms"} | {"summary": [
        {"arm": r["arm"], "flat": r["flat"], "valid": r.get("valid"), "flags": r.get("flags"),
         "hot_lat_slope": hot_slope(r)} for r in rows]})
    (run_dir / "STATUS").write_text("done")
    print_controls(run_id, rows)


def print_controls(run_id, rows):
    print(f"---\nrun_id: {run_id}")
    for r in rows:
        lat, byt, rss = r.get("hot_latency_p50") or {}, r.get("hot_resp_bytes") or {}, r.get("rss_mb") or {}
        ok = r["flat"] == r["expected_flat"]
        print(f"arm={r['arm']} valid={r.get('valid')} esperado={'plano' if r['expected_flat'] else 'sobe'} "
              f"flat={r['flat']} ok={ok} hot_lat_slope={fmt(lat.get('sen_slope_per_h'))} "
              f"hot_lat_change={fmt(lat.get('sen_change_pct'), 1)}% hr_p={fmt(lat.get('mk_p_upward'), 5)} "
              f"bytes_change={fmt(byt.get('sen_change_pct'), 1)}% rss_slope={fmt(rss.get('sen_slope_per_h'))} "
              f"rss_mk_p={fmt(rss.get('mk_p_upward'), 5)} req={r.get('requests')} db_rows={r.get('db_rows')} "
              f"{ref4_diag_str(r)} flags={','.join(r.get('flags', [])) or '-'}")


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
            if rec.get("event") == "causes" and "arms" in rec:
                print_causes(a.run_id, rec["arms"], rec.get("reference_ok"))
            elif rec.get("event") == "controls":
                print_controls(a.run_id, rec["arms"])
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
    s = sp.add_parser("detect")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--background", action="store_true")
    s.add_argument("--run-id")
    # --phase fica só por compatibilidade com os comandos já documentados (confirm --phase detect)
    s = sp.add_parser("confirm"); s.add_argument("--phase", choices=["detect"], default="detect")
    s.add_argument("--minutes", type=float, default=CONFIRM_MINUTES); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--smoke", action="store_true"); s.add_argument("--background", action="store_true"); s.add_argument("--run-id")
    s = sp.add_parser("controls")
    s.add_argument("--minutes", type=float, default=DEFAULT_MINUTES); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--with-r1", action="store_true", help="roda R1 ao lado (contraste)")
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
    {"setup": cmd_setup, "check-oracles": cmd_check_oracles, "detect": cmd_detect, "confirm": cmd_confirm,
     "controls": cmd_controls, "rehash": cmd_rehash, "freeze": cmd_freeze, "causes": cmd_causes, "wait": cmd_wait,
     "verify": lambda _: verify()}[a.cmd](a)


if __name__ == "__main__":
    main()
