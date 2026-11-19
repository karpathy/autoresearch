"""Teste do critério de "fixed" do reparo (prepare.py:score_repair), com parâmetros provisórios
FIXED_MIN_REDUCTION = 0,30 e FIXED_P95_TOL = 0,10 (harness/evaluate.py).

Casos (o original sempre com sintoma; o candidato passou nos portões):
  1. redução de 43% (como o repaired/ do Cesar), interferência igual        -> fixed
  2. redução de 43%, p95 de associate_card +5% (dentro da tolerância)       -> fixed
  3. redução de 43%, p95 de associate_card +20%                              -> NÃO fixed
  4. redução de 20%, interferência menor                                      -> NÃO fixed
  5. o tamanho de resposta de retrieve_hot não entra: igual ao original, 43% -> fixed
E um teste de que evaluate_arm calcula a interferência a partir do client.csv.

Rodar:  .cache/venv/bin/python tests/test_repair_fixed.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prepare  # noqa: E402
from harness import evaluate as E  # noqa: E402

GATE = {"passed": True, "flags": [], "diff_lines": 12, "details": {}}
S0 = 40.0  # ms/h, inclinação do original


def arm(slope, assoc_p95, ref4_p95, latency_up=True, eligible=True):
    return {"valid": True, "achieved_rps": 30.0, "server_error_rate": 0.0, "flags": [], "score_eligible": eligible,
            "latency_up": latency_up, "hot_latency_p50": {"sen_slope_per_h": slope},
            "hot_resp_bytes": {"sen_slope_per_h": 180000.0},
            "rss_mb": {"sen_slope_per_h": 1.0, "mk_p_upward": 0.5},
            "client": {ep: {"p50": {"median": 3.0}} for ep in ("associate_card", "retrieve_ref4", "retrieve_ref200")},
            "interference": {"associate_card": {"p95_level_ms": assoc_p95}, "retrieve_ref4": {"p95_level_ms": ref4_p95}}}


def score(cand):
    res = {"original": arm(S0, 50.0, 43.0), "v1": cand}
    return prepare.score_repair(res, "v1", GATE)


def test_parametros_provisorios():
    assert E.FIXED_MIN_REDUCTION == 0.30 and E.FIXED_P95_TOL == 0.10


def test_reparo_do_cesar_43_passa():
    r = score(arm(S0 * 0.57, 50.0, 43.0))
    assert abs(r["score_reduction"] - 0.43) < 1e-9 and r["fixed"], r


def test_tolerancia_5pct_passa():
    r = score(arm(S0 * 0.57, 52.5, 43.0))
    assert r["fixed"] and r["interference_ok"], r


def test_interferencia_20pct_pior_reprova():
    r = score(arm(S0 * 0.57, 60.0, 43.0))
    assert not r["fixed"] and r["slope_ok"] and r["interference_ok"] is False, r


def test_reducao_20pct_reprova():
    r = score(arm(S0 * 0.80, 20.0, 10.0))
    assert not r["fixed"] and not r["slope_ok"], r


def test_tamanho_hot_nao_entra():
    cand = arm(S0 * 0.57, 50.0, 43.0)
    cand["hot_resp_bytes"] = {"sen_slope_per_h": 180000.0}  # igual ao original: a spec não deixa mudar
    assert score(cand)["fixed"]


def test_evaluate_calcula_interferencia():
    from test_c1 import run
    ev = run(ref4_change=0.148)
    for ep in E.INTERFERENCE_EPS:
        d = ev["interference"][ep]
        assert d["p95_level_ms"] and d["p95_level_ms"] > 0 and d["tail_windows"] >= 50, d  # 8 h: última hora


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
