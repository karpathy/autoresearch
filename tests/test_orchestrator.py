"""Testes do orquestrador (harness/orchestrator.py) e da decisão automática da verificação (prepare.check_decision),
com juiz e agente simulados.

Rodar:  .cache/venv/bin/python tests/test_orchestrator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import orchestrator as O  # noqa: E402
from harness.load import normalize_args  # noqa: E402

REF = "F1_targeted_v1_referencia"


class Sim:
    """Agente e juiz simulados. eligible_at: {rodada: nome do candidato elegível};
    checks: lista de decisões das rodadas de verificação, consumidas em ordem."""

    def __init__(self, controls_ok=True, eligible_at=None, checks=None, args=None):
        self.controls_ok = controls_ok
        self.eligible_at = eligible_at or {}
        self.checks = list(checks or [])
        self.args = args or {}
        self.calls = []

    def run_controls(self, seed):
        self.calls.append(("controls", seed))
        return {"run_id": "ctl", "ok": self.controls_ok}

    def agent_round(self, kind, round, seed, campaign_n=None):
        self.calls.append((kind, round, seed, campaign_n))
        if kind == "causas":
            return {"run_id": f"causes-{campaign_n}-{round}", "arms": [], "candidates": {}}
        name = self.eligible_at.get(round)
        arms = [{"arm": REF, "score_eligible": True}]
        cands = {}
        if name:
            arms.append({"arm": name, "score_eligible": True, "flags": [], "ratio_vs_reference": 1.5,
                         "hot_latency_p50": {"sen_slope_per_h": 100.0}})
            cands[name] = {"name": name, "hypothesis": "h", "args": self.args.get(name, {"new_pair_rate": 15.0})}
        return {"run_id": f"detect-{round}", "arms": arms, "candidates": cands}

    def run_check(self, spec, seed):
        self.calls.append(("check", spec["name"], seed))
        return {"run_id": f"check-{seed}", "decision": self.checks.pop(0) if self.checks else "negativa"}

    def freeze(self, spec, source):
        self.calls.append(("freeze", spec["name"]))

    def agent_summary(self, kind, campaign_n=None):
        self.calls.append(("summary", kind, campaign_n))

    def orch(self):
        return O.Orchestrator(self.run_controls, self.agent_round, self.run_check, self.freeze, self.agent_summary,
                              normalize_args, lambda st: None, REF)


def kinds(sim, kind):
    return [c for c in sim.calls if c[0] == kind]


def test_fase0_falha_para_antes_da_busca():
    sim = Sim(controls_ok=False)
    st = sim.orch().run()
    assert st["status"] == "sem_calibracao" and not kinds(sim, "detect"), sim.calls


def test_sem_aging_roda_8_buscas():
    sim = Sim()
    st = sim.orch().run()
    det = kinds(sim, "detect")
    assert st["status"] == "aging_nao_confirmado"
    assert [c[1] for c in det] == list(range(1, 9)) and [c[2] for c in det] == list(range(101, 109))
    assert not kinds(sim, "check") and ("summary", "detect", None) in sim.calls


def test_confirma_na_rodada_2_roda_causas_e_retoma():
    sim = Sim(eligible_at={2: "c2"}, checks=["positiva", "positiva"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmado"
    assert [c[2] for c in kinds(sim, "check")] == [103, 104]           # verificação nas rodadas 3 e 4
    assert ("freeze", "c2") in sim.calls
    causas = kinds(sim, "causas")
    assert len(causas) == 6 and [c[2] for c in causas] == list(range(301, 307))
    assert [c[1] for c in kinds(sim, "detect")] == [1, 2, 5, 6, 7, 8]  # retoma nas rodadas que restam
    order = [c[0] for c in sim.calls]
    assert order.index("causas") < max(i for i, k in enumerate(order) if k == "detect")  # causas antes da retomada
    assert ("summary", "causas", 1) in sim.calls


def test_verificacao_negativa_volta_a_busca():
    sim = Sim(eligible_at={3: "c3"}, checks=["positiva", "negativa"])
    st = sim.orch().run()
    assert st["status"] == "aging_nao_confirmado" and not kinds(sim, "causas")
    assert [c[1] for c in kinds(sim, "detect")] == [1, 2, 3, 6, 7, 8]


def test_aging_tarde_estende_ate_10():
    sim = Sim(eligible_at={8: "c8"}, checks=["positiva", "positiva"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmado"
    assert [c[2] for c in kinds(sim, "check")] == [109, 110]


def test_sem_orcamento_para_verificar():
    # elegível na 8, verificação 9 inválida, repetição 10 positiva, mas falta a segunda positiva (rodada 11)
    sim = Sim(eligible_at={8: "c8"}, checks=["invalida", "positiva"])
    st = sim.orch().run()
    assert st["status"] == "aging_nao_confirmado"
    assert [c[2] for c in kinds(sim, "check")] == [109, 110]
    assert st["rounds"][-1]["decision"] == "sem_orcamento"


def test_invalida_repete_uma_vez():
    sim = Sim(eligible_at={1: "c1"}, checks=["invalida", "positiva", "positiva"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmado" and [c[2] for c in kinds(sim, "check")] == [102, 103, 104]


def test_duas_invalidas_nao_confirma():
    sim = Sim(eligible_at={1: "c1"}, checks=["invalida", "invalida"])
    st = sim.orch().run()
    assert st["status"] == "aging_nao_confirmado" and len(kinds(sim, "check")) == 2


def test_segunda_carga_confirmada_ganha_campanha_propria():
    sim = Sim(eligible_at={1: "a", 4: "b"}, checks=["positiva"] * 4,
              args={"a": {"new_pair_rate": 15.0}, "b": {"new_pair_rate": 18.0}})
    st = sim.orch().run()
    assert [c["load"]["name"] for c in st["confirmed"]] == ["a", "b"]
    seeds = [c[2] for c in kinds(sim, "causas")]
    assert seeds == list(range(301, 307)) + list(range(311, 317))


def test_mesma_carga_ja_confirmada_nao_reverifica():
    sim = Sim(eligible_at={1: "a", 4: "a2"}, checks=["positiva"] * 4,
              args={"a": {"new_pair_rate": 15.0}, "a2": {"new_pair_rate": 15.0}})
    st = sim.orch().run()
    assert len(st["confirmed"]) == 1 and len(kinds(sim, "check")) == 2


def test_rodada_do_agente_que_falha_consome_a_semente():
    class S(Sim):
        def agent_round(self, kind, round, seed, campaign_n=None):
            if kind == "detect" and round == 2:
                self.calls.append((kind, round, seed, campaign_n))
                return None
            return super().agent_round(kind, round, seed, campaign_n)
    sim = S()
    st = sim.orch().run()
    assert [r["status"] for r in st["rounds"]][1] == "falhou" and len(kinds(sim, "detect")) == 8


def test_decisao_da_verificacao():
    import prepare

    def arm(name, valid=True, eligible=False, lat=False, byt=False, rss=False):
        return {"arm": name, "valid": valid, "score_eligible": eligible, "latency_up": lat, "bytes_up": byt,
                "rss_upward_significant": rss}
    ok = [arm("R1", eligible=True), arm("carga", eligible=True), arm("R0"), arm("R0b")]
    assert prepare.check_decision(ok) == "positiva"
    assert prepare.check_decision([arm("R1", eligible=True), arm("carga"), arm("R0"), arm("R0b")]) == "negativa"
    assert prepare.check_decision([arm("R1", eligible=True), arm("carga", eligible=True), arm("R0", lat=True),
                                   arm("R0b")]) == "negativa"
    assert prepare.check_decision([arm("R1", eligible=True), arm("carga", eligible=True), arm("R0"),
                                   arm("R0b", valid=False)]) == "invalida"
    assert prepare.controls_ok([dict(arm("R0"), flat=True), dict(arm("R0b"), flat=True),
                                arm("R1", eligible=True)])
    assert not prepare.controls_ok([dict(arm("R0"), flat=False), dict(arm("R0b"), flat=True),
                                    arm("R1", eligible=True)])


def test_relatorio():
    sim = Sim(eligible_at={2: "c2"}, checks=["positiva", "positiva"])
    st = sim.orch().run()
    md = O.report_markdown(st, "teste")
    assert "aging_confirmado" in md and "c2" in md and "summary_causas_1.md" in md


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
