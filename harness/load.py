"""Carga fixa: scripts do Cesar rodados como subprocesso, com o contrato dele, mais as travas do juiz.

Contrato do Cesar (harness/run_diagnostic.py):
    script --base-url URL --duration S --seed N --out client.csv [flags]
    client.csv: ts,endpoint,status,latency_ms,resp_bytes[,phase]

Quem escreve a carga:
- Referência (4º slot, R1) e controles (R0, R0b): scripts do Cesar, intactos, em workloads/.
- O agente (detecção) NÃO escreve código: workload.py declara CANDIDATES, cada um com `args`,
  flags do F1_targeted_v1. O juiz valida os valores aqui e roda o script fixo com a semente dele.
  Assim a carga do agente é uma variante do F1 que obedece ao mesmo contrato, sem código novo.

Limites contra reward hacking na detecção (o agente não pode "achar aging" só saturando a máquina):
  taxas não acima de 2x os defaults do Cesar; 5xx/exceções acima de 1% invalidam o braço.
"""

import csv
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import psutil

from .stats import percentile

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".cache" / "venv" / "bin" / "python"
TARGETED = "workloads/bax-fastapi-credit-card-F1_targeted_v1.py"
CONTROL = "workloads/bax-fastapi-credit-card-F1_control.py"
BUCKET_S = 60.0         # janela de agregação do cliente (o Cesar usa 60 s em analyze_run.py)
RATE_FACTOR = 2.0       # taxas até 2x os defaults do F1_targeted_v1


class WorkloadError(ValueError):
    pass


# Defaults do F1_targeted_v1 (argparse do script). O script em si é fixo e hasheado no manifesto.
DEFAULTS = {
    "hot_phones": 1, "new_pair_rate": 10.0, "grow_workers": 2, "key_space": 0,
    "refresh_rate": 2.0, "ref_sizes": (4, 200), "ref_rate": 2.0, "probe_interval": 2.0,
}
# flags que só o juiz define (sem isso o agente trocaria semente, destino ou duração)
JUDGE_ONLY = {"base_url", "duration", "seed", "out", "timeout", "phone_prefix", "progress_every"}
MAX_REF_SIZES = 3
MAX_TOTAL_RATE = RATE_FACTOR * (10.0 + 2.0 + 2.0 + (1 + 2) / 2.0)  # 31 req/s: todos os fluxos juntos


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise WorkloadError(f"args.{name} must be a number")
    return v


def normalize_args(args):
    """Valida `args` de um candidato e devolve o dict completo (defaults + valores)."""
    if not isinstance(args, dict):
        raise WorkloadError("args must be a dict")
    bad = sorted(set(args) - set(DEFAULTS))
    if bad:
        hint = " (set by the judge only)" if set(bad) <= JUDGE_ONLY else f" (allowed: {sorted(DEFAULTS)})"
        raise WorkloadError(f"args outside the contract: {bad}{hint}")
    a = dict(DEFAULTS)
    a.update(args)
    d = DEFAULTS
    if not (isinstance(a["hot_phones"], int) and 1 <= a["hot_phones"] <= RATE_FACTOR * d["hot_phones"]):
        raise WorkloadError(f"hot_phones must be an int in [1, {int(RATE_FACTOR * d['hot_phones'])}]")
    if not (isinstance(a["grow_workers"], int) and 1 <= a["grow_workers"] <= RATE_FACTOR * d["grow_workers"]):
        raise WorkloadError(f"grow_workers must be an int in [1, {int(RATE_FACTOR * d['grow_workers'])}]")
    if not (isinstance(a["key_space"], int) and 0 <= a["key_space"] <= 100000):
        raise WorkloadError("key_space must be an int in [0, 100000] (0 = always new cards)")
    for k in ("new_pair_rate", "refresh_rate", "ref_rate"):
        if not 0 <= _num(a[k], k) <= RATE_FACTOR * d[k]:
            raise WorkloadError(f"{k} must be in [0, {RATE_FACTOR * d[k]:g}] (2x the default)")
    if not _num(a["new_pair_rate"], "new_pair_rate") > 0:
        raise WorkloadError("new_pair_rate must be > 0")
    if not _num(a["probe_interval"], "probe_interval") >= d["probe_interval"] / RATE_FACTOR:
        raise WorkloadError(f"probe_interval must be >= {d['probe_interval'] / RATE_FACTOR:g} s (2x the default rate)")
    rs = a["ref_sizes"]
    if isinstance(rs, list):
        rs = tuple(rs)
    if not (isinstance(rs, tuple) and 1 <= len(rs) <= MAX_REF_SIZES
            and all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 2 * max(d["ref_sizes"]) for x in rs)):
        raise WorkloadError(f"ref_sizes must be a tuple of 1 to {MAX_REF_SIZES} ints in [1, {2 * max(d['ref_sizes'])}]")
    if tuple(sorted(rs)) != tuple(sorted(d["ref_sizes"])):
        # os telefones de referência são o controle interno (C1 e interferência leem retrieve_ref4/ref200):
        # mudá-los cegaria o controle, então não são um fator da carga
        raise WorkloadError(f"ref_sizes is fixed at {d['ref_sizes']} (internal control)")
    a["ref_sizes"] = rs
    total = (a["new_pair_rate"] + a["refresh_rate"] + a["ref_rate"]
             + (a["hot_phones"] + len(rs)) / a["probe_interval"])
    if total > MAX_TOTAL_RATE + 1e-9:
        raise WorkloadError(f"total nominal rate {total:.1f} req/s above {MAX_TOTAL_RATE:g} (2x the default)")
    return a


def validate_spec(spec):
    for key in ("name", "hypothesis", "args"):
        if key not in spec:
            raise WorkloadError(f"candidate missing required field '{key}'")
    if not isinstance(spec["name"], str) or not spec["name"].replace("_", "").replace("-", "").isalnum():
        raise WorkloadError("name must be alphanumeric (with _ or -)")
    spec["_args"] = normalize_args(spec["args"])


def args_to_argv(args):
    """args normalizados -> flags do script do Cesar (só o que difere do default, para o log ficar legível)."""
    argv = []
    for k, v in args.items():
        if v == DEFAULTS[k]:
            continue
        flag = "--" + k.replace("_", "-")
        argv += [flag, ",".join(map(str, v)) if isinstance(v, tuple) else str(v)]
    return argv


def same_as_default(spec):
    """Candidato com os mesmos parâmetros do F1_targeted_v1 padrão não é hipótese nova (D4)."""
    return normalize_args(spec.get("args", {})) == DEFAULTS


def one_factor_diff(ref_spec, cand_spec):
    """Fase de causas: o candidato precisa diferir da carga de referência (a congelada) em exatamente um arg.
    Devolve (fator, valor_na_referencia, valor_no_candidato) ou levanta WorkloadError (D3)."""
    ref = normalize_args(ref_spec.get("args", {}))
    cand = normalize_args(cand_spec.get("args", {}))
    diff = sorted(k for k in DEFAULTS if ref[k] != cand[k])
    if len(diff) != 1:
        raise WorkloadError(f"{cand_spec.get('name')}: must change exactly 1 arg relative to the frozen workload "
                            f"(changed {len(diff)}: {diff})")
    k = diff[0]
    return k, ref[k], cand[k]


# ---- braços fixos: a referência e os controles do Cesar (campaign.yaml) ----
# As hipóteses ficam neutras de propósito: aparecem nos resultados que o agente lê.
BASELINE = {"name": "F1_targeted_v1_reference", "script": TARGETED, "args": {},
            "hypothesis": "Reference: workload with the default parameters."}
R0 = {"name": "R0", "script": CONTROL, "args": {}, "argv": [],
      "hypothesis": "Negative control R0."}
R0B = {"name": "R0b", "script": TARGETED, "args": {"key_space": 4}, "argv": ["--key-space", "4"],
       "hypothesis": "Negative control R0b."}


def spec_argv(spec):
    return spec["argv"] if "argv" in spec else args_to_argv(normalize_args(spec.get("args", {})))


def spec_script(spec):
    return spec.get("script", TARGETED)


def run_script(spec, base_url, seconds, seed, run_dir, cpus=None):
    """Roda o script do Cesar como subprocesso e devolve {"exit", "client_csv", "loadgen_cpu_p95", ...}.
    Ambiente mínimo: o script não vê variáveis do agente nem do juiz."""
    run_dir = Path(run_dir)
    out = run_dir / "client.csv"
    cmd = [str(VENV_PY), str(ROOT / spec_script(spec)), "--base-url", base_url, "--duration", str(seconds),
           "--seed", str(seed), "--out", str(out), *spec_argv(spec)]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = {"PATH": "/usr/bin:/bin", "PYTHONUNBUFFERED": "1", "HOME": str(run_dir)}
    cpu = []
    with open(run_dir / "workload.log", "w") as wl:
        proc = subprocess.Popen(cmd, stdout=wl, stderr=subprocess.STDOUT, env=env, cwd=run_dir)
        try:
            p = psutil.Process(proc.pid)
            p.cpu_percent(None)
        except psutil.Error:
            p = None
        deadline = time.time() + seconds + 120
        while proc.poll() is None and time.time() < deadline:
            time.sleep(5)
            try:
                if p:
                    cpu.append(p.cpu_percent(None))
            except psutil.Error:
                pass
        if proc.poll() is None:
            proc.kill()
            exit_code = "timeout"
        else:
            exit_code = proc.returncode
    return {"exit": exit_code, "client_csv": str(out), "cmd": cmd,
            "loadgen_cpu_p95": percentile(cpu, 95) if cpu else None}


def read_client(path):
    """client.csv -> lista de (ts, endpoint, status, latency_ms, resp_bytes, phase)."""
    rows = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((float(r["ts"]), r["endpoint"], int(r["status"]), float(r["latency_ms"]),
                                 int(r["resp_bytes"]), r.get("phase", "")))
                except (KeyError, ValueError):
                    continue  # linha truncada no fim (o script faz flush a cada 1 s)
    except OSError:
        pass
    return rows


def summarize_client(rows, t0):
    """Totais e taxas de erro do cliente (a série por endpoint é calculada em evaluate.py)."""
    n = len(rows)
    exc = sum(1 for r in rows if r[2] == 0)
    e5 = sum(1 for r in rows if r[2] >= 500)
    c4 = sum(1 for r in rows if 400 <= r[2] < 500)
    ops = defaultdict(int)
    for r in rows:
        ops[r[1]] += 1
    return {"requests": n, "ops": dict(ops), "exception_rate": exc / max(1, n), "server_error_rate": e5 / max(1, n),
            "client_error_rate": c4 / max(1, n), "error_rate": (exc + e5) / max(1, n),
            "achieved_rps": n / max(1.0, (rows[-1][0] - t0) if rows else 1.0)}
