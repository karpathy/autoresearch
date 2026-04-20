"""Pareto ratchet evaluator for harness metrics.

Implements tiered keep/discard logic:
  Tier 1 (gates):      task_success_rate, quality_gate_pass_rate
  Tier 2 (constraint): rework_rate
  Tier 3 (optimize):   avg_token_consumption, time_per_turn

Improvements to lower-tier metrics are accepted only if higher-tier
floors are maintained. Floors ratchet upward on improvement.

Usage:
    python tests/evaluator.py --extract-composite    # Print composite score
    python tests/evaluator.py --check-ratchet        # Check against ratcheted floors
    python tests/evaluator.py --update-floors        # Update floors after a keep
"""

import argparse
import json
import re
import sys
from pathlib import Path

RATCHET_PATH = Path(__file__).parent / "results" / "ratchet_state.json"

METRIC_TIERS: dict[int, list[str]] = {
    1: ["task_success_rate", "quality_gate_pass_rate"],
    2: ["rework_rate"],
    3: ["avg_token_consumption", "time_per_turn"],
}

METRIC_DIRECTION: dict[str, str] = {
    "task_success_rate": "higher",
    "quality_gate_pass_rate": "higher",
    "rework_rate": "lower",
    "avg_token_consumption": "lower",
    "time_per_turn": "lower",
}

TIER_WEIGHTS = {1: 0.5, 2: 0.3, 3: 0.2}


class RatchetState:
    """Loads and saves ratcheted metric floors."""

    def __init__(self, path: Path = RATCHET_PATH) -> None:
        self.path = path
        self.floors: dict[str, float] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.floors = data.get("floors", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"floors": self.floors}
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def update_floors(self, metrics: dict[str, float]) -> None:
        """Ratchet floors upward (or downward for lower-is-better metrics)."""
        for name, value in metrics.items():
            direction = METRIC_DIRECTION.get(name)
            if direction is None:
                continue
            current = self.floors.get(name)
            if current is None:
                self.floors[name] = value
            elif direction == "higher" and value > current:
                self.floors[name] = value
            elif direction == "lower" and value < current:
                self.floors[name] = value


def composite_score(metrics: dict[str, float]) -> float:
    """Weighted harmonic mean across tiers. T1=0.5, T2=0.3, T3=0.2."""
    weighted_sum = 0.0
    weight_total = 0.0

    for tier, names in METRIC_TIERS.items():
        values = []
        for name in names:
            raw = metrics.get(name, 0.0)
            direction = METRIC_DIRECTION[name]
            if direction == "higher":
                values.append(raw)
            else:
                values.append(1.0 / (1.0 + raw))

        avg = sum(values) / len(values) if values else 0.0
        if avg <= 0:
            return 0.0
        w = TIER_WEIGHTS[tier]
        weighted_sum += w / avg
        weight_total += w

    if weighted_sum <= 0:
        return 0.0
    return weight_total / weighted_sum


def check_floors(metrics: dict[str, float], state: RatchetState) -> list[str]:
    """Check metrics against ratcheted floors. Returns list of violations."""
    violations: list[str] = []
    for tier in sorted(METRIC_TIERS.keys()):
        for name in METRIC_TIERS[tier]:
            floor = state.floors.get(name)
            if floor is None:
                continue
            value = metrics.get(name, 0.0)
            direction = METRIC_DIRECTION[name]
            if direction == "higher" and value < floor:
                violations.append(
                    f"T{tier} {name}: {value:.3f} < floor {floor:.3f}"
                )
            elif direction == "lower" and value > floor:
                violations.append(
                    f"T{tier} {name}: {value:.3f} > floor {floor:.3f}"
                )
    return violations


def evaluate(metrics: dict[str, float]) -> dict:
    """Evaluate metrics against ratchet. Returns decision and reasoning."""
    state = RatchetState()
    violations = check_floors(metrics, state)
    score = composite_score(metrics)

    if violations:
        return {
            "decision": "discard",
            "harness_score": round(score, 3),
            "violations": violations,
            "reasoning": "Higher-tier floor(s) regressed",
        }
    return {
        "decision": "keep",
        "harness_score": round(score, 3),
        "violations": [],
        "reasoning": "All floors maintained",
    }


def extract_metrics_from_log(log_path: Path) -> dict[str, float]:
    """Extract metric values from a run.log file (line format: 'key: value')."""
    metrics: dict[str, float] = {}
    if not log_path.exists():
        return metrics
    text = log_path.read_text(encoding="utf-8")
    for name in METRIC_DIRECTION:
        pattern = rf"^{re.escape(name)}:\s*([\d.]+)"
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto ratchet evaluator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--extract-composite", action="store_true",
        help="Read run.log, extract metrics, print composite score",
    )
    group.add_argument(
        "--check-ratchet", action="store_true",
        help="Check current metrics against saved floors",
    )
    group.add_argument(
        "--update-floors", action="store_true",
        help="Save current metrics as new floors (after a keep)",
    )
    args = parser.parse_args()

    log_path = Path.cwd() / "run.log"
    metrics = extract_metrics_from_log(log_path)

    if args.extract_composite:
        score = composite_score(metrics)
        print(f"harness_score: {score:.3f}")

    elif args.check_ratchet:
        state = RatchetState()
        violations = check_floors(metrics, state)
        if violations:
            for v in violations:
                print(f"VIOLATION: {v}")
            sys.exit(1)
        else:
            print("All floors maintained")

    elif args.update_floors:
        state = RatchetState()
        state.update_floors(metrics)
        state.save()
        print(f"Floors updated: {state.floors}")


if __name__ == "__main__":
    main()
