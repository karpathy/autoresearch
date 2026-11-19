"""Avaliação fixa de um braço (uma instância sob uma carga).

Duas famílias de série, como no analyze_run.py do Cesar:
- servidor (psutil, de fora da app): RSS, com MK + Sen (e as metades, robustas a degraus);
- cliente (client.csv do script): latência (p50/p95) e tamanho médio da resposta, por janela de 60 s,
  para retrieve_hot (o sintoma do F1) e retrieve_ref4/retrieve_ref200 (controle interno: mesma
  rota, tamanho constante; devem ficar planos). Aqui MK é o de Hamed-Rao + Sen.
"""

from collections import defaultdict

from .stats import trend, trend_hr, percentile, sen_slope

MB = 1024 * 1024
FINDING_ID = "bax-fastapi-credit-card-F1"
ALPHA = 0.01                 # significância exigida para "tendência de alta"
MAX_EXCEPTION_RATE = 0.01    # timeouts/conexões recusadas: saturação, não aging (invalida o braço)
MAX_SERVER_ERROR_RATE = 0.01 # 5xx da app acima disso: idem (decisão: "5xx/erros acima de 1% invalidam")
MAX_LOADGEN_CPU = 90.0       # % de um núcleo: gerador saturado mede o cliente, não a app
MIN_EFFECT_MB_PER_H = 5.0    # RSS: tamanho de efeito mínimo (A/A de 20/10 na versão Node; recalibrar com R0b)
MIN_LATENCY_CHANGE_PCT = 50.0   # latência de retrieve_hot: mudança implícita do Sen ao longo da série
MAX_REF_CHANGE_PCT = 25.0       # controle interno: acima disso, drift global e não crescimento por linha
CHECKPOINT_MIN = 5           # para calcular o tempo até a detecção
HOT = "retrieve_hot"
REFS = ("retrieve_ref4", "retrieve_ref200")

# D2 ("estado legítimo cresceu: mais de N linhas no banco, o braço não pontua") é configurável por finding.
# No F1 o crescimento de linhas É o finding (INSERT OR IGNORE sem retenção): D2 desligada. Os números de
# linhas continuam registrados (db_rows) como evidência de ativação.
D2_BY_FINDING = {
    FINDING_ID: {"enabled": False, "max_rows": None,
                 "why": "o crescimento de linhas é o próprio finding (associations nunca é podada)"},
}
D2_DEFAULT = {"enabled": True, "max_rows": 1000, "why": "dados novos podem inflar a memória por estado legítimo"}


def d2_policy(finding=FINDING_ID):
    return D2_BY_FINDING.get(finding, D2_DEFAULT)


def warmup_seconds(total_s):
    return min(300.0, 0.1 * total_s)


def window_seconds(total_s):
    return 60.0 if total_s >= 900 else 5.0    # smoke (2 min) usa janelas curtas


def _series(probe, key, warm):
    pts = [(r["t"], r[key]) for r in probe if r.get(key) is not None and r["t"] >= warm]
    return [t / 3600.0 for t, _ in pts], [v for _, v in pts]


def half_slopes(x, y):
    """Sen's slope de cada metade da série. Um degrau único infla a inclinação total,
    mas não as duas metades: a menor delas é robusta a degraus."""
    h = len(y) // 2
    if h < 10:
        return None
    return [sen_slope(x[:h], y[:h]), sen_slope(x[h:], y[h:])]


def client_windows(rows, t0, total_s, endpoint, warm_s=60.0):
    """Séries por janela de um endpoint: {'p50': (x, y), 'p95': ..., 'bytes': ...}, x em horas.
    Só respostas 2xx; descarta a janela parcial do início e a do fim, como o Cesar."""
    win = window_seconds(total_s)
    by = defaultdict(list)
    for ts, ep, status, lat, size, _phase in rows:
        if ep == endpoint and 200 <= status < 300:
            by[int((ts - t0) // win)].append((lat, size))
    keys = sorted(by)
    if len(keys) > 3:
        keys = keys[1:-1]
    keys = [k for k in keys if k * win >= min(warm_s, 0.1 * total_s)]
    xs = [k * win / 3600.0 for k in keys]
    return {"p50": (xs, [percentile([v[0] for v in by[k]], 50) for k in keys]),
            "p95": (xs, [percentile([v[0] for v in by[k]], 95) for k in keys]),
            "bytes": (xs, [sum(v[1] for v in by[k]) / len(by[k]) for k in keys])}


def significant_up(tr, min_change_pct):
    return bool(tr and tr.get("mk_p_upward", 1.0) < ALPHA and (tr.get("sen_change_pct") or 0) >= min_change_pct)


def time_to_detect(x, y):
    """Primeiro checkpoint (min) a partir do qual MK de alta é significativo e assim permanece."""
    if not x:
        return None
    first = None
    t = CHECKPOINT_MIN / 60.0
    end = x[-1]
    while t <= end + 1e-9:
        sub = [(a, b) for a, b in zip(x, y) if a <= t]
        sig = len(sub) >= 10 and trend([a for a, _ in sub], [b for _, b in sub])["mk_p_upward"] < ALPHA
        if sig and first is None:
            first = t * 60.0
        elif not sig:
            first = None
        t += CHECKPOINT_MIN / 60.0
    return first


def evaluate_arm(name, probe, client_rows, t0, load, db_state, pids_seen, alive_at_end, total_s,
                 max_children=0, instr_last=None, finding=FINDING_ID):
    """probe: amostras do psutil; client_rows/t0: client.csv lido; load: summarize_client + saída do script."""
    warm = warmup_seconds(total_s)
    flags = []
    out = {"arm": name, "warmup_s": warm, "requests": load["requests"], "max_children": max_children,
           "error_rate": load["error_rate"], "server_error_rate": load["server_error_rate"],
           "exception_rate": load["exception_rate"], "client_error_rate": load["client_error_rate"],
           "ops": load["ops"], "loadgen_cpu_p95": load.get("loadgen_cpu_p95"), "instr_last": instr_last}

    probe_pids = {r.get("pid") for r in probe}
    if load.get("violation") or load.get("script_exit") not in (0, None):
        flags.append("D3_fora_do_contrato")
    if (load["exception_rate"] > MAX_EXCEPTION_RATE or load["server_error_rate"] > MAX_SERVER_ERROR_RATE
            or (load.get("loadgen_cpu_p95") or 0) > MAX_LOADGEN_CPU):
        flags.append("D1_saturacao_ou_erros")
    if not alive_at_end or len(probe_pids | set(pids_seen)) > 1:
        flags.append("H4_processo_reiniciado_ou_morto")
    if max_children > 0:
        # Só o PID principal é medido: memória em filhos escapa do RSS e um filho
        # reciclado é rejuvenescimento, não correção.
        flags.append("H4_processo_filho")
    if len(probe) < 10:
        flags.append("sem_amostras_suficientes")

    # --- servidor: RSS (MK + Sen) ---
    x, y = _series(probe, "rss", warm)
    out["rss_mb"] = tr = trend(x, [v / MB for v in y])
    if y:
        tr["first"], tr["last"] = y[0] / MB, y[-1] / MB
    halves = half_slopes(x, [v / MB for v in y])
    tr["sen_halves_per_h"] = halves
    tr["sen_min_halves_per_h"] = min(halves) if halves else None
    x, y = _series(probe, "handles", warm)
    out["handles"] = trend(x, y)
    x, y = _series(probe, "db_bytes", warm)
    out["db_mb"] = trend(x, [v / MB for v in y])

    # --- cliente: retrieve_hot (sintoma) e retrieve_ref* (controle interno), MK Hamed-Rao + Sen ---
    ep_tr = {}
    for ep in (HOT,) + REFS:
        w = client_windows(client_rows, t0, total_s, ep)
        ep_tr[ep] = {m: trend_hr(*w[m]) for m in ("p50", "p95", "bytes")}
        ep_tr[ep]["_p50_series"] = w["p50"]
    out["client"] = {ep: {m: v for m, v in d.items() if m != "_p50_series"} for ep, d in ep_tr.items()}
    hot = ep_tr[HOT]
    out["hot_latency_p50"], out["hot_latency_p95"], out["hot_resp_bytes"] = hot["p50"], hot["p95"], hot["bytes"]
    out["time_to_detect_min"] = time_to_detect(*hot["_p50_series"])

    out["latency_up"] = significant_up(hot["p50"], MIN_LATENCY_CHANGE_PCT)
    out["bytes_up"] = significant_up(hot["bytes"], MIN_LATENCY_CHANGE_PCT)    # ativação: o telefone quente cresce
    out["rss_upward_significant"] = (out["rss_mb"]["mk_p_upward"] < ALPHA
                                     and out["rss_mb"]["sen_slope_per_h"] >= MIN_EFFECT_MB_PER_H)
    refs_up = [ep for ep in REFS if significant_up(ep_tr[ep]["p50"], MAX_REF_CHANGE_PCT)
               or significant_up(ep_tr[ep]["bytes"], MAX_REF_CHANGE_PCT)]
    out["internal_control_flat"] = not refs_up
    if refs_up:
        flags.append("C1_referencia_interna_subindo:" + ",".join(refs_up))
    if len(ep_tr[HOT]["_p50_series"][0]) < 5:
        flags.append("sem_janelas_do_cliente")

    # --- banco: linhas (D2 só se o finding a mantiver) ---
    rows = sum((db_state or {}).get("counts", {}).values()) if db_state else 0
    out["db_rows"] = rows
    out["db_state_counts"] = (db_state or {}).get("counts")
    pol = d2_policy(finding)
    out["d2_enabled"] = pol["enabled"]
    if pol["enabled"] and rows > pol["max_rows"]:
        flags.append("D2_crescimento_de_estado_legitimo")

    out["flags"] = flags
    out["valid"] = not any(f.startswith(("D1", "D3", "H4", "sem_")) for f in flags)
    # elegível a nota: sintoma subindo, mecanismo ativado, controle interno plano
    out["score_eligible"] = bool(out["valid"] and out["latency_up"] and out["bytes_up"] and out["internal_control_flat"])
    return out
