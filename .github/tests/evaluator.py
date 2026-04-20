"""Pareto ratchet evaluator -- tiered keep/discard logic with ratcheting floors.

T1 (gate): task_success_rate, quality_gate_pass_rate
T2 (constraint): rework_rate | T3 (optimize): avg_token_consumption, time_per_turn
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RATCHET_PATH = Path(__file__).parent / "results" / "ratchet_state.json"
TIERS: dict[int, list[str]] = {
    1: ["task_success_rate", "quality_gate_pass_rate"],
    2: ["rework_rate"],
    3: ["avg_token_consumption", "time_per_turn"],
}
DIR: dict[str, str] = {
    "task_success_rate": "higher", "quality_gate_pass_rate": "higher",
    "rework_rate": "lower", "avg_token_consumption": "lower", "time_per_turn": "lower",
}
TIER_W = {1: 0.5, 2: 0.3, 3: 0.2}


def _better(val: float, ref: float, d: str) -> bool:
    return (d == "higher" and val > ref) or (d == "lower" and val < ref)


class RatchetState:
    """Loads and saves ratcheted metric floors with evaluation history."""
    def __init__(self, path: Path = RATCHET_PATH) -> None:
        self.path, self.floors, self.history = path, {}, []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.floors, self.history = data.get("floors", {}), data.get("history", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"floors": self.floors, "history": self.history}, indent=2) + "\n",
            encoding="utf-8")

    def record(self, metrics: dict[str, float], decision: str) -> None:
        self.history.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                             "metrics": metrics, "decision": decision})

    def update_floors(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            d = DIR.get(name)
            if d and (name not in self.floors or _better(value, self.floors[name], d)):
                self.floors[name] = value


def composite_score(metrics: dict[str, float]) -> float:
    """Weighted harmonic mean across tiers. T1=0.5, T2=0.3, T3=0.2."""
    w_sum, w_total = 0.0, 0.0
    for tier, names in TIERS.items():
        vals = [m if DIR[n] == "higher" else 1.0 / (1.0 + m)
                for n in names if (m := metrics.get(n, 0.0)) is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        if avg <= 0:
            return 0.0
        w = TIER_W[tier]
        w_sum += w / avg
        w_total += w
    return w_total / w_sum if w_sum > 0 else 0.0


def _check_tier(metrics: dict[str, float], state: RatchetState, tier: int) -> list[str]:
    return [f"{n}: {metrics.get(n, 0.0):.3f} vs floor {f:.3f}"
            for n in TIERS.get(tier, [])
            if (f := state.floors.get(n)) is not None
            and _better(f, metrics.get(n, 0.0), DIR[n])]

def _find_improvements(metrics: dict[str, float], state: RatchetState) -> list[str]:
    out: list[str] = []
    for tier in sorted(TIERS):
        for name in TIERS[tier]:
            val = metrics.get(name)
            if val is None:
                continue
            floor = state.floors.get(name)
            if floor is None:
                out.append(f"T{tier} {name}: {val:.3f} (new)")
            elif _better(val, floor, DIR[name]):
                out.append(f"T{tier} {name}: {val:.3f} (was {floor:.3f})")
    return out

def evaluate(metrics: dict[str, float]) -> dict:
    """Core decision function with tiered regression checks."""
    state = RatchetState()
    score = composite_score(metrics)
    result: dict = {"harness_score": round(score, 3)}
    for tier in (1, 2):
        viol = _check_tier(metrics, state, tier)
        if viol:
            result.update(decision="discard",
                          reason=f"T{tier} regression: {'; '.join(viol)}")
            state.record(metrics, "discard"); state.save()
            return result
    improvements = _find_improvements(metrics, state)
    if improvements:
        result.update(decision="keep", reason="; ".join(improvements))
    else:
        result.update(decision="discard", reason="No improvement in any metric")
    state.record(metrics, result["decision"]); state.save()
    return result

def _parse_metrics(text: str) -> dict[str, float]:
    return {n: float(m.group(1)) for n in DIR
            if (m := re.search(rf"^{re.escape(n)}:\s*([\d.]+)", text, re.MULTILINE))}

def _read_metrics() -> dict[str, float]:
    log = Path.cwd() / "run.log"
    if log.exists():
        return _parse_metrics(log.read_text(encoding="utf-8"))
    return _parse_metrics(sys.stdin.read()) if not sys.stdin.isatty() else {}

def reflect() -> None:
    """Analyze results.tsv discard patterns and output markdown summary."""
    path = Path.cwd() / "results" / "results.tsv"
    if not path.exists():
        print("No results/results.tsv found."); return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        print("results.tsv has no data rows."); return
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]
    dec_i = next((i for i, h in enumerate(header) if h.lower() == "decision"), None)
    reason_i = next((i for i, h in enumerate(header) if h.lower() == "reason"), None)
    if dec_i is None:
        print("No 'decision' column in results.tsv."); return
    total = len(rows)
    discards = [r for r in rows if len(r) > dec_i and r[dec_i] == "discard"]
    keeps = total - len(discards)
    reasons: Counter[str] = Counter()
    if reason_i is not None:
        for r in discards:
            if len(r) > reason_i:
                tag = re.match(r"(T\d [^\s:]+)", r[reason_i])
                reasons[tag.group(1) if tag else r[reason_i][:60]] += 1
    print("## Reflection: Discard Pattern Analysis\n")
    print(f"- Total evaluations: {total}")
    print(f"- Kept: {keeps}, Discarded: {len(discards)}")
    if total:
        print(f"- Keep rate: {keeps / total:.1%}\n")
    if not reasons:
        return
    print("### Top discard reasons\n")
    print("| Reason | Count | Share |\n|--------|-------|-------|")
    for reason, cnt in reasons.most_common(10):
        print(f"| {reason} | {cnt} | {cnt / len(discards):.0%} |")
    top = reasons.most_common(1)[0][0]
    hyp = {
        "T1": f"Primary blocker is T1 gate regression ({top})."
              " Stabilize core success metrics before optimizing T2/T3.",
        "T2": f"Primary blocker is T2 constraint ({top})."
              " Rework rate is the bottleneck -- focus on first-pass quality.",
    }
    msg = hyp.get(top[:2], f"Most frequent discard: {top}."
                            " Review recent change patterns for root cause.")
    print(f"\n### Hypotheses\n\n- {msg}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto ratchet evaluator")
    g = ap.add_mutually_exclusive_group(required=True)
    for flag, hlp in [("--extract-composite", "Print composite score from run.log or stdin"),
                      ("--check-ratchet", "Check metrics against saved floors"),
                      ("--update-floors", "Save current metrics as new floors"),
                      ("--reflect", "Analyze discard patterns from results.tsv")]:
        g.add_argument(flag, action="store_true", help=hlp)
    args = ap.parse_args()
    if args.reflect:
        reflect(); return
    metrics = _read_metrics()
    if args.extract_composite:
        print(f"harness_score: {composite_score(metrics):.3f}")
    elif args.check_ratchet:
        state = RatchetState()
        viols = [v for t in sorted(TIERS) for v in _check_tier(metrics, state, t)]
        if viols:
            for v in viols:
                print(f"VIOLATION: {v}")
            sys.exit(1)
        print("All floors maintained")
    elif args.update_floors:
        state = RatchetState()
        state.update_floors(metrics); state.save()
        print(f"Floors updated: {state.floors}")

if __name__== "__main__":
    main()
