"""
Proactive experiment suggestion engine for autoresearch.

Inspired by mdemg's Jiminy inner voice system and RSIC (Recursive
Self-Improvement Cycle), this module provides:

- Experiment suggestions based on Hebbian memory associations
- Plateau detection and radical change recommendations
- Frontier detection (unexplored hyperparameter regions)
- Contradiction surfacing (conflicting results that need investigation)
- Meta-cognitive assessment of experiment strategy effectiveness

Architecture from mdemg's Jiminy:
  Jiminy fires on every prompt with 4 parallel knowledge sources
  (consulting suggestions, correction vectors, contradiction edges,
  frontier detection) merged with a 6-second timeout. Results are
  deduplicated, confidence-filtered, and injected into the agent's
  context.

Here we adapt that pattern for experiment guidance: before each
experiment, the advisor assembles context from memory, resilience
status, and experiment history to suggest the most promising next
experiment.

From mdemg's RSIC:
  The Recursive Self-Improvement Cycle runs: assess -> reflect ->
  plan -> speculate -> execute. We simplify this to:
  assess (how are we doing?) -> suggest (what to try next?) ->
  with optional radical mode when plateaued.

Usage:
    from memory import ExperimentMemory
    from monitor import ExperimentTracker
    from resilience import ExperimentGuard

    advisor = ExperimentAdvisor(memory, tracker, guard)
    guidance = advisor.get_guidance()
    print(guidance["formatted"])
"""

import time
from dataclasses import dataclass, field
from typing import Optional

# Import types for type hints (these modules are in the same package)
# At runtime, actual instances are passed to the constructor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from memory import ExperimentMemory
    from monitor import ExperimentTracker
    from resilience import ExperimentGuard


# ---------------------------------------------------------------------------
# Suggestion types
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSuggestion:
    """A suggested next experiment with rationale."""
    category: str           # change category tag
    description: str        # human-readable suggestion
    rationale: str          # why this is suggested
    confidence: float       # 0-1, how confident we are this will help
    priority: int           # 1=highest priority
    source: str             # where this suggestion came from


@dataclass
class Contradiction:
    """A pair of conflicting experiment results."""
    description: str
    experiment_a: str       # description of first experiment
    experiment_b: str       # description of second experiment
    bpb_a: float
    bpb_b: float
    explanation: str        # why these conflict


@dataclass
class StrategyAssessment:
    """Meta-cognitive assessment of the experiment strategy.

    Inspired by mdemg's RSIC assess phase which evaluates the
    effectiveness of the current approach and recommends adjustments.
    """
    phase: str              # exploring | exploiting | plateaued | recovering
    effectiveness: float    # 0-1, how well the current strategy is working
    total_experiments: int
    improvements_found: int
    improvement_rate: float
    velocity_trend: str     # accelerating | stable | decelerating | stalled
    recommendation: str     # high-level strategy recommendation


# ---------------------------------------------------------------------------
# ExperimentAdvisor
# ---------------------------------------------------------------------------

class ExperimentAdvisor:
    """Proactive guidance engine for autonomous experiment research.

    Assembles context from memory (Hebbian associations, experiment
    history), monitoring (metrics, alerts), and resilience (circuit
    breaker, anomalies) to suggest the most promising next experiment.

    Inspired by mdemg's Jiminy inner voice which:
    1. Fans out to 4 knowledge sources in parallel
    2. Merges results with deduplication
    3. Filters by confidence threshold
    4. Formats for injection into agent context

    Here we adapt that to experiment guidance:
    1. Memory associations -> promising categories
    2. Experiment history -> what's been tried recently
    3. Anomaly status -> safety constraints
    4. Frontier analysis -> unexplored regions
    5. Contradiction detection -> conflicting results
    """

    def __init__(self, memory: Optional["ExperimentMemory"] = None,
                 tracker: Optional["ExperimentTracker"] = None,
                 guard: Optional["ExperimentGuard"] = None):
        self.memory = memory
        self.tracker = tracker
        self.guard = guard

    def get_guidance(self) -> dict:
        """Generate comprehensive guidance for the next experiment.

        This is the main entry point. Returns a dict with:
        - suggestions: ranked list of experiment suggestions
        - contradictions: conflicting results to investigate
        - assessment: meta-cognitive strategy assessment
        - warnings: safety warnings from resilience systems
        - formatted: human-readable formatted guidance string

        Inspired by mdemg's Jiminy which assembles guidance from
        multiple sources and formats it for injection into prompts.
        """
        suggestions = self._generate_suggestions()
        contradictions = self._detect_contradictions()
        assessment = self._assess_strategy()
        warnings = self._collect_warnings()

        guidance = {
            "suggestions": [
                {
                    "category": s.category,
                    "description": s.description,
                    "rationale": s.rationale,
                    "confidence": s.confidence,
                    "priority": s.priority,
                }
                for s in suggestions
            ],
            "contradictions": [
                {
                    "description": c.description,
                    "experiment_a": c.experiment_a,
                    "experiment_b": c.experiment_b,
                    "explanation": c.explanation,
                }
                for c in contradictions
            ],
            "assessment": {
                "phase": assessment.phase,
                "effectiveness": assessment.effectiveness,
                "velocity_trend": assessment.velocity_trend,
                "recommendation": assessment.recommendation,
            },
            "warnings": warnings,
            "formatted": self._format_guidance(
                suggestions, contradictions, assessment, warnings
            ),
            "generated_at": time.time(),
        }
        return guidance

    # -- Suggestion generation (Jiminy source 1: consulting) ----------------

    def _generate_suggestions(self) -> list[ExperimentSuggestion]:
        """Generate ranked experiment suggestions.

        Combines multiple signal sources:
        1. Hebbian associations from memory (what has worked)
        2. Unexplored frontiers (what hasn't been tried)
        3. Dead end avoidance (what to skip)
        4. Plateau breakers (radical changes when stuck)
        """
        suggestions = []
        priority = 1

        # Source 1: Promising directions from Hebbian memory
        if self.memory:
            promising = self.memory.get_promising_directions(top_k=3)
            for p in promising:
                cat = p["category"]
                score = p["score"]
                reason = p["reason"]

                if p["experiments"] == 0:
                    desc = f"Try a {cat} change — this category is unexplored"
                    rationale = "Frontier: no experiments in this category yet"
                else:
                    desc = f"Try a {cat} change — historically effective"
                    rationale = f"Hebbian association: {reason}"

                suggestions.append(ExperimentSuggestion(
                    category=cat, description=desc,
                    rationale=rationale, confidence=min(score, 1.0),
                    priority=priority, source="hebbian_memory",
                ))
                priority += 1

        # Source 2: Plateau breakers
        if self.memory:
            plateau = self.memory.get_plateaus()
            if plateau.get("plateau_detected"):
                suggestions.append(ExperimentSuggestion(
                    category="radical",
                    description="Try a radical architectural change to break plateau",
                    rationale=plateau.get("reason", "No improvements recently"),
                    confidence=0.3,
                    priority=priority,
                    source="plateau_detection",
                ))
                priority += 1

                # Also suggest combining prior improvements
                suggestions.append(ExperimentSuggestion(
                    category="combination",
                    description="Combine the top 2-3 near-miss experiments",
                    rationale="Near-misses may compound when combined",
                    confidence=0.4,
                    priority=priority,
                    source="plateau_detection",
                ))
                priority += 1

        # Source 3: Surprise-driven exploration
        if self.memory:
            surprises = self.memory.get_surprise_highlights(top_k=2)
            for s in surprises:
                if s["status"] == "discard" and s["surprise_score"] > 1.5:
                    # Surprisingly bad result — the inverse might work
                    suggestions.append(ExperimentSuggestion(
                        category=s["change_tags"][0] if s["change_tags"] else "misc",
                        description=f"Revisit '{s['description']}' — "
                                    f"surprisingly bad result, try the opposite",
                        rationale=f"Surprise score {s['surprise_score']:.2f} — "
                                  f"high surprise suggests unexplored dynamics",
                        confidence=0.35,
                        priority=priority,
                        source="surprise_analysis",
                    ))
                    priority += 1

        # Source 4: Dead end warnings (negative suggestions)
        if self.memory:
            dead_ends = self.memory.get_dead_ends()
            for de in dead_ends[:2]:
                suggestions.append(ExperimentSuggestion(
                    category=de["category"],
                    description=f"AVOID {de['category']} changes — "
                                f"consistently unpromising",
                    rationale=f"weight={de['weight']:.3f}, "
                              f"success={de['success_rate']:.0%} over "
                              f"{de['experiments']} experiments",
                    confidence=min(de["experiments"] / 10, 1.0),
                    priority=99,  # Low priority (avoidance, not action)
                    source="dead_end_detection",
                ))

        # Sort by priority
        suggestions.sort(key=lambda s: s.priority)
        return suggestions

    # -- Contradiction detection (Jiminy source 3) --------------------------

    def _detect_contradictions(self) -> list[Contradiction]:
        """Detect contradicting experiment results.

        Finds pairs of experiments where similar changes produced
        opposite outcomes. These contradictions suggest that the
        outcome depends on context (other hyperparameters, model
        state, etc.) and warrant investigation.

        Inspired by mdemg's CONTRADICTS edges which link nodes
        with conflicting information.
        """
        contradictions = []
        if not self.memory:
            return contradictions

        history = self.memory.get_experiment_history(last_n=50)
        if len(history) < 4:
            return contradictions

        # Group experiments by their primary change tag
        by_tag: dict[str, list[dict]] = {}
        for exp in history:
            if exp.get("status") == "crash":
                continue
            for tag in exp.get("change_tags", []):
                by_tag.setdefault(tag, []).append(exp)

        # Find contradictions: same category, opposite outcomes
        for tag, exps in by_tag.items():
            kept = [e for e in exps if e.get("status") == "keep"]
            discarded = [e for e in exps if e.get("status") == "discard"]

            if kept and discarded:
                # Find the best kept and worst discarded
                best_kept = min(kept, key=lambda e: e.get("val_bpb", float("inf")))
                worst_disc = max(discarded, key=lambda e: e.get("val_bpb", 0))

                if (worst_disc.get("val_bpb", 0) >
                        best_kept.get("val_bpb", float("inf")) + 0.001):
                    contradictions.append(Contradiction(
                        description=f"Contradicting results for '{tag}' changes",
                        experiment_a=best_kept.get("description", "?"),
                        experiment_b=worst_disc.get("description", "?"),
                        bpb_a=best_kept.get("val_bpb", 0),
                        bpb_b=worst_disc.get("val_bpb", 0),
                        explanation=(
                            f"Same category '{tag}' produced both improvement "
                            f"({best_kept.get('val_bpb', 0):.6f}) and regression "
                            f"({worst_disc.get('val_bpb', 0):.6f}). "
                            f"The outcome likely depends on context — investigate "
                            f"what differs between these experiments."
                        ),
                    ))

        return contradictions[:5]  # Limit to top 5 contradictions

    # -- Strategy assessment (RSIC assess phase) ----------------------------

    def _assess_strategy(self) -> StrategyAssessment:
        """Meta-cognitive assessment of experiment strategy.

        Evaluates the overall effectiveness of the current research
        approach and recommends strategic adjustments.

        Inspired by mdemg's RSIC which cycles through:
        assess -> reflect -> plan -> speculate -> execute

        Here we implement the assess phase, providing the agent with
        a high-level understanding of where it stands.
        """
        total = 0
        improvements = 0
        recent_improvements = 0
        velocity_trend = "stable"

        if self.tracker:
            summary = self.tracker.get_summary()
            total = summary.get("total_experiments", 0)
            improvements = summary.get("kept", 0)

        # Determine phase
        if total < 5:
            phase = "exploring"
            recommendation = ("Early exploration phase — try diverse changes "
                              "across different categories to map the landscape")
        elif self.memory:
            plateau = self.memory.get_plateaus()
            if plateau.get("plateau_detected"):
                phase = "plateaued"
                recommendation = ("Plateau detected — shift to radical changes: "
                                  "different architectures, unusual hyperparameter "
                                  "ranges, or combine multiple near-miss improvements")
            elif improvements / max(total, 1) > 0.25:
                phase = "exploiting"
                recommendation = ("High hit rate — continue exploiting promising "
                                  "directions, try finer-grained variations of "
                                  "what's working")
            else:
                phase = "exploring"
                recommendation = ("Low hit rate — broaden search to new categories, "
                                  "try changes you haven't attempted yet")
        else:
            phase = "exploring"
            recommendation = "Continue exploring — no memory system active"

        # Improvement rate
        improvement_rate = improvements / max(total, 1)

        # Velocity trend from tracker
        if self.tracker and len(self.tracker.experiments) >= 10:
            recent = self.tracker.experiments[-10:]
            earlier = self.tracker.experiments[-20:-10] if len(
                self.tracker.experiments) >= 20 else []

            recent_keeps = sum(1 for e in recent if e.status == "keep")
            if earlier:
                earlier_keeps = sum(1 for e in earlier if e.status == "keep")
                if recent_keeps > earlier_keeps:
                    velocity_trend = "accelerating"
                elif recent_keeps < earlier_keeps:
                    velocity_trend = "decelerating"
                if recent_keeps == 0 and earlier_keeps == 0:
                    velocity_trend = "stalled"
            recent_improvements = recent_keeps

        # Effectiveness score (0-1)
        effectiveness = min(improvement_rate * 2, 1.0)  # 50% keep rate = 1.0
        if velocity_trend == "stalled":
            effectiveness *= 0.5
        elif velocity_trend == "decelerating":
            effectiveness *= 0.75

        return StrategyAssessment(
            phase=phase,
            effectiveness=round(effectiveness, 3),
            total_experiments=total,
            improvements_found=improvements,
            improvement_rate=round(improvement_rate, 3),
            velocity_trend=velocity_trend,
            recommendation=recommendation,
        )

    # -- Warning collection -------------------------------------------------

    def _collect_warnings(self) -> list[str]:
        """Collect safety warnings from all resilience systems."""
        warnings = []

        if self.guard:
            status = self.guard.get_status()
            cb = status.get("circuit_breaker", {})
            if cb.get("state") != "closed":
                warnings.append(
                    f"Circuit breaker {cb.get('state', '?')} — "
                    f"{cb.get('consecutive_failures', 0)} consecutive failures")

            bp = status.get("backpressure", {})
            if bp.get("level") in ("warning", "critical"):
                warnings.append(
                    f"VRAM pressure {bp.get('level', '?')} — "
                    f"{bp.get('utilization', 0):.0%} utilized, "
                    f"trend: {bp.get('trend', '?')}")

            for anomaly in status.get("recent_anomalies", []):
                if anomaly.get("severity") in ("warning", "critical"):
                    warnings.append(anomaly.get("message", "Unknown anomaly"))

        if self.tracker:
            alerts = self.tracker.get_recent_alerts(5)
            for alert in alerts:
                if alert.get("severity") in ("warning", "critical"):
                    warnings.append(alert.get("message", "Unknown alert"))

        # Deduplicate
        seen = set()
        unique = []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique

    # -- Formatting ---------------------------------------------------------

    def _format_guidance(self, suggestions: list[ExperimentSuggestion],
                         contradictions: list[Contradiction],
                         assessment: StrategyAssessment,
                         warnings: list[str]) -> str:
        """Format guidance as a human-readable string.

        This formatted output is designed to be injected into the agent's
        context before each experiment decision, similar to how mdemg's
        Jiminy injects proactive guidance into every Claude prompt.
        """
        lines = [
            "=" * 60,
            "  EXPERIMENT GUIDANCE (auto-generated)",
            "=" * 60,
            "",
            f"  Phase: {assessment.phase.upper()}",
            f"  Effectiveness: {assessment.effectiveness:.0%}  "
            f"({assessment.improvements_found}/{assessment.total_experiments} kept)",
            f"  Velocity: {assessment.velocity_trend}",
            f"  Strategy: {assessment.recommendation}",
        ]

        if warnings:
            lines += ["", "  WARNINGS:"]
            for w in warnings:
                lines.append(f"    [!] {w}")

        if suggestions:
            # Separate actionable suggestions from avoidance
            actionable = [s for s in suggestions if s.priority < 99]
            avoidances = [s for s in suggestions if s.priority >= 99]

            if actionable:
                lines += ["", "  SUGGESTED NEXT EXPERIMENTS:"]
                for s in actionable[:5]:
                    lines.append(
                        f"    {s.priority}. [{s.category}] {s.description}")
                    lines.append(
                        f"       Rationale: {s.rationale}")
                    lines.append(
                        f"       Confidence: {s.confidence:.0%}")

            if avoidances:
                lines += ["", "  AVOID:"]
                for s in avoidances:
                    lines.append(f"    [-] {s.description}")
                    lines.append(f"        {s.rationale}")

        if contradictions:
            lines += ["", "  CONTRADICTIONS TO INVESTIGATE:"]
            for c in contradictions[:3]:
                lines.append(f"    [?] {c.description}")
                lines.append(f"        {c.explanation}")

        lines += ["", "=" * 60]
        return "\n".join(lines)
