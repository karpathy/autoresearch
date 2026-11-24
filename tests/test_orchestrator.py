"""Testes do orquestrador (harness/orchestrator.py) e da decisão automática da verificação (prepare.check_decision),
com juiz e agente simulados.

Rodar:  .cache/venv/bin/python tests/test_orchestrator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import orchestrator as O  # noqa: E402
from harness.load import normalize_args  # noqa: E402

REF = "F1_targeted_v1_reference"


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
        if kind == "causes":
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
        return {"run_id": f"check-{seed}", "decision": self.checks.pop(0) if self.checks else "negative"}

    def freeze(self, spec, source):
        self.calls.append(("freeze", spec["name"]))

    def agent_summary(self, kind, campaign_n=None):
        self.calls.append(("summary", kind, campaign_n))

    def orch(self):
        return O.Orchestrator(self.run_controls, self.agent_round, self.run_check, self.freeze, self.agent_summary,
                              normalize_args, lambda st: None, REF)


def kinds(sim, kind):
    return [c for c in sim.calls if c[0] == kind]


def test_phase0_failure_stops_before_search():
    sim = Sim(controls_ok=False)
    st = sim.orch().run()
    assert st["status"] == "no_calibration" and not kinds(sim, "detect"), sim.calls


def test_no_aging_runs_8_searches():
    sim = Sim()
    st = sim.orch().run()
    det = kinds(sim, "detect")
    assert st["status"] == "aging_not_confirmed"
    assert [c[1] for c in det] == list(range(1, 9)) and [c[2] for c in det] == list(range(101, 109))
    assert not kinds(sim, "check") and ("summary", "detect", None) in sim.calls


def test_confirms_in_round_2_runs_causes_and_resumes():
    sim = Sim(eligible_at={2: "c2"}, checks=["positive", "positive"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmed"
    assert [c[2] for c in kinds(sim, "check")] == [103, 104]           # verificação nas rodadas 3 e 4
    assert ("freeze", "c2") in sim.calls
    causes_calls = kinds(sim, "causes")
    assert len(causes_calls) == 6 and [c[2] for c in causes_calls] == list(range(301, 307))
    assert [c[1] for c in kinds(sim, "detect")] == [1, 2, 5, 6, 7, 8]  # retoma nas rodadas que restam
    order = [c[0] for c in sim.calls]
    assert order.index("causes") < max(i for i, k in enumerate(order) if k == "detect")  # causes antes da retomada
    assert ("summary", "causes", 1) in sim.calls


def test_negative_verification_returns_to_search():
    sim = Sim(eligible_at={3: "c3"}, checks=["positive", "negative"])
    st = sim.orch().run()
    assert st["status"] == "aging_not_confirmed" and not kinds(sim, "causes")
    assert [c[1] for c in kinds(sim, "detect")] == [1, 2, 3, 6, 7, 8]


def test_aging_tarde_estende_ate_10():
    sim = Sim(eligible_at={8: "c8"}, checks=["positive", "positive"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmed"
    assert [c[2] for c in kinds(sim, "check")] == [109, 110]


def test_no_budget_to_verify():
    # elegível na 8, verificação 9 inválida, repetição 10 positiva, mas falta a segunda positiva (rodada 11)
    sim = Sim(eligible_at={8: "c8"}, checks=["invalid", "positive"])
    st = sim.orch().run()
    assert st["status"] == "aging_not_confirmed"
    assert [c[2] for c in kinds(sim, "check")] == [109, 110]
    assert st["rounds"][-1]["decision"] == "no_budget"


def test_invalida_repete_uma_vez():
    sim = Sim(eligible_at={1: "c1"}, checks=["invalid", "positive", "positive"])
    st = sim.orch().run()
    assert st["status"] == "aging_confirmed" and [c[2] for c in kinds(sim, "check")] == [102, 103, 104]


def test_two_invalid_do_not_confirm():
    sim = Sim(eligible_at={1: "c1"}, checks=["invalid", "invalid"])
    st = sim.orch().run()
    assert st["status"] == "aging_not_confirmed" and len(kinds(sim, "check")) == 2


def test_second_confirmed_workload_gets_own_campaign():
    sim = Sim(eligible_at={1: "a", 4: "b"}, checks=["positive"] * 4,
              args={"a": {"new_pair_rate": 15.0}, "b": {"new_pair_rate": 18.0}})
    st = sim.orch().run()
    assert [c["load"]["name"] for c in st["confirmed"]] == ["a", "b"]
    seeds = [c[2] for c in kinds(sim, "causes")]
    assert seeds == list(range(301, 307)) + list(range(311, 317))


def test_already_confirmed_workload_not_reverified():
    sim = Sim(eligible_at={1: "a", 4: "a2"}, checks=["positive"] * 4,
              args={"a": {"new_pair_rate": 15.0}, "a2": {"new_pair_rate": 15.0}})
    st = sim.orch().run()
    assert len(st["confirmed"]) == 1 and len(kinds(sim, "check")) == 2


def test_failed_agent_round_consumes_seed():
    class S(Sim):
        def agent_round(self, kind, round, seed, campaign_n=None):
            if kind == "detect" and round == 2:
                self.calls.append((kind, round, seed, campaign_n))
                return None
            return super().agent_round(kind, round, seed, campaign_n)
    sim = S()
    st = sim.orch().run()
    assert [r["status"] for r in st["rounds"]][1] == "failed" and len(kinds(sim, "detect")) == 8


def test_verification_decision():
    import prepare

    def arm(name, valid=True, eligible=False, lat=False, byt=False, rss=False):
        return {"arm": name, "valid": valid, "score_eligible": eligible, "latency_up": lat, "bytes_up": byt,
                "rss_upward_significant": rss}
    ok = [arm("R1", eligible=True), arm("workload", eligible=True), arm("R0"), arm("R0b")]
    assert prepare.check_decision(ok) == "positive"
    assert prepare.check_decision([arm("R1", eligible=True), arm("workload"), arm("R0"), arm("R0b")]) == "negative"
    assert prepare.check_decision([arm("R1", eligible=True), arm("workload", eligible=True), arm("R0", lat=True),
                                   arm("R0b")]) == "negative"
    assert prepare.check_decision([arm("R1", eligible=True), arm("workload", eligible=True), arm("R0"),
                                   arm("R0b", valid=False)]) == "invalid"
    assert prepare.controls_ok([dict(arm("R0"), flat=True), dict(arm("R0b"), flat=True),
                                arm("R1", eligible=True)])
    assert not prepare.controls_ok([dict(arm("R0"), flat=False), dict(arm("R0b"), flat=True),
                                    arm("R1", eligible=True)])


def test_report():
    sim = Sim(eligible_at={2: "c2"}, checks=["positive", "positive"])
    st = sim.orch().run()
    md = O.report_markdown(st, "teste")
    assert "aging_confirmed" in md and "c2" in md and "summary_causes_1.md" in md


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
