"""Teste da regra C1 (deriva global) com séries sintéticas de 8 h, no formato do client.csv do Cesar.

Números do Cesar (8 h, MK Hamed-Rao + Sen x duração / mediana, como harness/evaluate.py):
  R1 original: ref4 p50 +14,8% (p = 0,0002); ref200 e tamanho de resposta dos refs planos.
Regra nova: a latência do ref4 NÃO entra no C1 (sobe por interferência do próprio aging).

Casos:
  1. ref4 +14,8%, ref200 e tamanhos planos           -> C1 NÃO dispara
  2. ref4 +30% (passaria do limiar antigo de 25%)    -> C1 NÃO dispara
  3. ref200 (latência) sobe                          -> C1 dispara
  4. tamanho de resposta dos refs sobe               -> C1 dispara

Rodar:  .cache/venv/bin/python -m pytest -q tests/test_c1.py   (ou: .cache/venv/bin/python tests/test_c1.py)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.evaluate import evaluate_arm  # noqa: E402

TOTAL_S = 8 * 3600
WIN = 60.0
PER_WIN = 30  # leituras por janela e por endpoint (sonda a cada 2 s)


def growth_for_change(change):
    """Crescimento linear g (fim/início − 1) cuja mudança implícita Sen×duração/mediana é `change`.
    Série linear de a até a(1+g): mediana ≈ a(1+g/2), então change = g/(1+g/2)  =>  g = change/(1−change/2)."""
    return change / (1 - change / 2)


def make_rows(ref4_change=0.148, ref200_change=0.0, ref_bytes_change=0.0, seed=7):
    rng = random.Random(seed)
    g4, g200, gb = (growth_for_change(c) for c in (ref4_change, ref200_change, ref_bytes_change))
    rows = []
    nwin = int(TOTAL_S // WIN)
    for k in range(nwin):
        f = k / (nwin - 1)
        for i in range(PER_WIN):
            ts = k * WIN + i * (WIN / PER_WIN)
            noise = lambda: rng.lognormvariate(0, 0.15)  # noqa: E731
            # sintoma: latência e tamanho do telefone quente crescem muito (5 -> 50 ms, 15 kB -> 1,5 MB)
            rows.append((ts, "retrieve_hot", 200, 5.0 * (1 + 9 * f) * noise(), int(15000 * (1 + 99 * f)), "probe"))
            rows.append((ts + 0.66, "retrieve_ref4", 200, 4.7 * (1 + g4 * f) * noise(),
                         int(376 * (1 + gb * f)), "refread"))
            rows.append((ts + 1.33, "retrieve_ref200", 200, 6.0 * (1 + g200 * f) * noise(),
                         int(18000 * (1 + gb * f)), "refread"))
            rows.append((ts + 0.2, "associate_card", 201, 3.0 * noise(), 60, "grow"))
    rows.sort()
    return rows


def run(**kw):
    rows = make_rows(**kw)
    probe = [{"t": t, "pid": 1, "rss": int((60 + 0.001 * t) * 1024 * 1024), "handles": 10, "db_bytes": 1000}
             for t in range(0, TOTAL_S, 5)]
    load = {"requests": len(rows), "error_rate": 0.0, "server_error_rate": 0.0, "exception_rate": 0.0,
            "client_error_rate": 0.0, "ops": {}, "achieved_rps": 2.0, "loadgen_cpu_p95": 10.0,
            "violation": None, "script_exit": 0}
    return evaluate_arm("sintetico", probe, rows, 0.0, load, {"counts": {"associations": 100000}}, {1}, True, TOTAL_S)


def c1(ev):
    return [f for f in ev["flags"] if f.startswith("C1")]


def test_ref4_148_nao_dispara():
    ev = run(ref4_change=0.148)
    d = ev["ref4_diag"]
    assert d["p50_mk_p"] < 0.01 and 12 <= d["p50_change_pct"] <= 18, d   # o cenário do Cesar foi reproduzido
    assert not c1(ev), ev["flags"]
    assert ev["internal_control_flat"] and ev["score_eligible"]


def test_ref4_30_nao_dispara():
    ev = run(ref4_change=0.30)
    assert ev["ref4_diag"]["p50_change_pct"] > 25, ev["ref4_diag"]       # passaria do limiar antigo
    assert not c1(ev), ev["flags"]
    assert ev["score_eligible"]


def test_ref200_sobe_dispara():
    ev = run(ref4_change=0.148, ref200_change=0.40)
    assert c1(ev) and "retrieve_ref200.p50" in c1(ev)[0], ev["flags"]
    assert not ev["score_eligible"]


def test_ref_size_rising_triggers():
    ev = run(ref4_change=0.0, ref_bytes_change=0.40)
    flag = c1(ev)
    assert flag and "retrieve_ref4.bytes" in flag[0] and "retrieve_ref200.bytes" in flag[0], ev["flags"]
    assert not ev["score_eligible"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
