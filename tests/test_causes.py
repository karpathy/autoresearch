"""Testes da fase de causas: regra de um fator, custo por MB e razões contra a referência.

Rodar:  .cache/venv/bin/python tests/test_causes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.load import BASELINE, WorkloadError, one_factor_diff  # noqa: E402
from test_c1 import run  # noqa: E402

import prepare  # noqa: E402


def cand(name, args):
    return {"name": name, "hypothesis": "teste", "args": args}


def test_one_factor_vs_default_reference():
    assert one_factor_diff(BASELINE, cand("a", {"new_pair_rate": 20.0})) == ("new_pair_rate", 10.0, 20.0)


def test_one_factor_vs_modified_frozen_workload():
    frozen = cand("frozen", {"new_pair_rate": 20.0, "hot_phones": 2})
    # volta só o hot_phones ao padrão: um fator
    assert one_factor_diff(frozen, cand("b", {"new_pair_rate": 20.0})) == ("hot_phones", 2, 1)


def test_zero_or_two_factors_rejected():
    for args in ({}, {"new_pair_rate": 20.0, "probe_interval": 1.0}):
        try:
            one_factor_diff(BASELINE, cand("c", args))
        except WorkloadError:
            continue
        raise AssertionError(f"deveria recusar {args}")


def test_cost_per_mb():
    # série sintética do test_c1: latência 5 -> 50 ms enquanto a resposta vai de 15 kB a 1,5 MB (~30 ms/MB)
    ev = run()
    c = ev["hot_cost_ms_per_mb"]
    assert c is not None and 25 <= c <= 36, c


def test_ratios_vs_reference():
    ref = {"hot_latency_p50": {"sen_slope_per_h": 10.0}, "hot_resp_bytes": {"sen_slope_per_h": 100.0},
           "hot_cost_ms_per_mb": 30.0,
           "interference": {"associate_card": {"p95_level_ms": 20.0}, "retrieve_ref4": {"p95_level_ms": 10.0}}}
    ev = {"hot_latency_p50": {"sen_slope_per_h": 5.0}, "hot_resp_bytes": {"sen_slope_per_h": 100.0},
          "hot_cost_ms_per_mb": 15.0,
          "interference": {"associate_card": {"p95_level_ms": 20.0}, "retrieve_ref4": {"p95_level_ms": 5.0}}}
    e = prepare.compare_to_reference(ref, ev)
    assert e == {"lat_ratio": 0.5, "bytes_ratio": 1.0, "cost_ratio": 0.5,
                 "assoc_p95_ratio": 1.0, "ref4_p95_ratio": 0.5}, e


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
