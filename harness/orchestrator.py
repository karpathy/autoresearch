"""Orquestrador do experimento (fixo; o agente não edita nem lê).

Encadeia, sem intervenção humana:
  Fase 0: controles de 60 min (R0, R0b, R1)  -> falha: fim "no_calibration"
  Busca: rodadas do agente (campanha detect), uma por chamada, até MAX_SEARCH
  Verificação: quando um candidato fica elegível, as rodadas seguintes repetem a carga dele com R1, R0 e R0b
               (mesma semente nos 4 braços); confirmado quando VERIFY_ROUNDS rodadas válidas dão positivo
  Causas: com o aging confirmado, a busca pausa, a carga é congelada e o agente roda a campanha de causas
  Retomada: a busca continua nas rodadas que restam; cada nova carga confirmada ganha a sua campanha de causas
  Relatório: ao fim do orçamento de busca

Regras de orçamento (docs/DESENHO-EXPERIMENTAL.md): as rodadas são numeradas em sequência e a rodada n usa a
semente 100 + n. Rodadas de busca só até MAX_SEARCH; rodadas de verificação podem ir até MAX_TOTAL. Uma rodada
de verificação inválida (infraestrutura) é repetida no máximo MAX_INVALID_RETRIES vezes por verificação.

A lógica fica em `Orchestrator.run()`, com as ações injetadas (juiz, agente, congelamento), para poder ser
testada sem subir a aplicação nem chamar o agente (tests/test_orchestrator.py).
"""

import json
import time

MAX_SEARCH = 8          # rodadas de busca
MAX_TOTAL = 10          # teto de rodadas, contando as de verificação
VERIFY_ROUNDS = 2       # rodadas de verificação positivas necessárias (com a rodada que achou: 3 de 3)
MAX_INVALID_RETRIES = 1 # repetições de rodada de verificação inválida, por verificação
CAUSES_ROUNDS = 6       # rodadas da campanha de causas
SEED_BASE = 100         # rodada n -> semente 100 + n
CONTROLS_SEED = 100     # Fase 0
CAUSES_SEED_BASE = 300  # campanha j, rodada r -> 300 + 10*(j-1) + r


def causes_seed(campaign, rnd):
    return CAUSES_SEED_BASE + 10 * (campaign - 1) + rnd


def pick_candidate(round_rec, reference_name, confirmed_args, normalize):
    """Melhor candidato elegível da rodada de busca (maior ratio; sem ratio, maior inclinação), ignorando a
    referência, braços com D4 e cargas com args idênticos aos de uma carga já confirmada.
    round_rec: {"arms": [...evaluation...], "candidates": {name: spec}}. Devolve o spec ou None."""
    best, best_key = None, None
    for ev in round_rec.get("arms", []):
        name = ev.get("arm")
        if name == reference_name or not ev.get("score_eligible"):
            continue
        if any(str(f).startswith("D4") for f in ev.get("flags", [])):
            continue
        spec = round_rec.get("candidates", {}).get(name)
        if spec is None:
            continue
        if normalize(spec.get("args", {})) in confirmed_args:
            continue
        slope = (ev.get("hot_latency_p50") or {}).get("sen_slope_per_h") or 0.0
        ratio = ev.get("ratio_vs_reference")
        key = (ratio is not None, ratio if ratio is not None else 0.0, slope)
        if best_key is None or key > best_key:
            best, best_key = spec, key
    return best


class Orchestrator:
    """Ações injetadas:
      run_controls(seed) -> {"run_id", "ok": bool}
      agent_round(kind, round, seed, campaign_n=None) -> dict da rodada ({"run_id", "arms", "candidates"}) ou None se falhou
      run_check(spec, seed) -> {"run_id", "decision": "positive"|"negative"|"invalid"}
      freeze(spec, source) -> None
      agent_summary(kind, campaign_n=None) -> None
      normalize(args) -> args normalizados (comparar cargas)
      save(state) -> None (grava o estado a cada passo)
    """

    def __init__(self, run_controls, agent_round, run_check, freeze, agent_summary, normalize, save,
                 reference_name, max_search=MAX_SEARCH, max_total=MAX_TOTAL, verify_rounds=VERIFY_ROUNDS,
                 max_invalid_retries=MAX_INVALID_RETRIES, causes_rounds=CAUSES_ROUNDS):
        self.run_controls, self.agent_round, self.run_check = run_controls, agent_round, run_check
        self.freeze, self.agent_summary, self.normalize, self.save = freeze, agent_summary, normalize, save
        self.reference_name = reference_name
        self.max_search, self.max_total = max_search, max_total
        self.verify_rounds, self.max_invalid_retries = verify_rounds, max_invalid_retries
        self.causes_rounds = causes_rounds
        self.state = {"status": "running", "phase0": None, "rounds": [], "confirmed": [], "started": _now()}

    def _log_round(self, **kw):
        self.state["rounds"].append(kw)
        self.save(self.state)

    def run(self):
        st = self.state
        ctl = self.run_controls(CONTROLS_SEED)
        st["phase0"] = ctl
        self.save(st)
        if not ctl.get("ok"):
            return self._finish("no_calibration")

        confirmed_args = []
        n = 1
        while n <= self.max_search:
            seed = SEED_BASE + n
            rec = self.agent_round("detect", round=n, seed=seed)
            self._log_round(n=n, seed=seed, kind="search", run_id=(rec or {}).get("run_id"),
                            status="ok" if rec else "failed")
            n += 1
            if not rec:
                continue
            cand = pick_candidate(rec, self.reference_name, confirmed_args, self.normalize)
            if cand is None:
                continue
            source = {"run_id": rec.get("run_id"), "arm": cand["name"]}
            n, ok = self._verify(cand, source, n)
            if not ok:
                continue
            j = len(st["confirmed"]) + 1
            entry = {"campaign": j, "load": cand, "source": source, "causes_rounds": []}
            st["confirmed"].append(entry)
            confirmed_args.append(self.normalize(cand.get("args", {})))
            self.freeze(cand, source)
            self.save(st)
            for r in range(1, self.causes_rounds + 1):
                cs = causes_seed(j, r)
                crec = self.agent_round("causes", round=r, seed=cs, campaign_n=j)
                entry["causes_rounds"].append({"round": r, "seed": cs, "run_id": (crec or {}).get("run_id"),
                                               "status": "ok" if crec else "failed"})
                self.save(st)
            self.agent_summary("causes", campaign_n=j)
        self.agent_summary("detect")
        return self._finish("aging_confirmed" if st["confirmed"] else "aging_not_confirmed")

    def _verify(self, cand, source, n):
        """Roda as rodadas de verificação a partir da rodada n. Devolve (próxima rodada, confirmado?)."""
        positives, retries = 0, 0
        while positives < self.verify_rounds:
            if n > self.max_total:
                self._log_round(n=None, seed=None, kind="verification", load=cand["name"], source=source,
                                decision="no_budget")
                return n, False
            seed = SEED_BASE + n
            res = self.run_check(cand, seed)
            dec = res.get("decision")
            self._log_round(n=n, seed=seed, kind="verification", load=cand["name"], source=source,
                            run_id=res.get("run_id"), decision=dec)
            n += 1
            if dec == "positive":
                positives += 1
            elif dec == "invalid" and retries < self.max_invalid_retries:
                retries += 1
            else:
                return n, False
        return n, True

    def _finish(self, status):
        self.state["status"] = status
        self.state["finished"] = _now()
        self.save(self.state)
        return self.state


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def report_markdown(state, tag):
    """Final human-readable report (the human reads this and the agent summaries)."""
    out = [f"# Experiment report `{tag}`", "",
           f"- Status: **{state['status']}**",
           f"- Started: {state.get('started')} · finished: {state.get('finished')}", ""]
    p0 = state.get("phase0") or {}
    out += ["## Phase 0: 60 min controls", "",
            f"- run_id: `{p0.get('run_id')}` · result: {'ok' if p0.get('ok') else 'failed'}", ""]
    out += ["## Rounds", "", "| Round | Seed | Kind | run_id | Verified workload | Result |",
            "|---|---|---|---|---|---|"]
    for r in state.get("rounds", []):
        out.append(f"| {r.get('n')} | {r.get('seed')} | {r.get('kind')} | `{r.get('run_id')}` | "
                   f"{r.get('load', '-')} | {r.get('decision', r.get('status', '-'))} |")
    out += ["", "## Confirmed aging", ""]
    if not state.get("confirmed"):
        out.append("None.")
    for c in state.get("confirmed", []):
        out += [f"### Workload {c['campaign']}: `{c['load']['name']}`", "",
                f"- args: `{json.dumps(c['load'].get('args', {}), ensure_ascii=False)}`",
                f"- found in: `{c['source']['run_id']}`",
                f"- causes campaign: {len(c['causes_rounds'])} rounds "
                f"(agent summary in `summary_causes_{c['campaign']}.md`)", ""]
    out += ["## Agent summaries", "", "- `summary_detect.md`"]
    out += [f"- `summary_causes_{c['campaign']}.md`" for c in state.get("confirmed", [])]
    return "\n".join(out) + "\n"
