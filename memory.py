"""
Cross-session experiment memory with Hebbian learning for autoresearch.

Inspired by mdemg's Conversation Memory System (CMS) and Hebbian learning
engine, this module provides:

- Persistent experiment knowledge base across research sessions
- Hebbian association tracking: strengthens connections between change
  categories and positive/negative outcomes
- Temporal decay: older experiments contribute less to association weights,
  keeping the system responsive to recent findings
- Surprise-weighted storage: unexpected results (large improvements or
  regressions from seemingly minor changes) are weighted more heavily
- Pattern extraction: identifies which categories of changes tend to
  improve BPB and which are dead ends

Architecture note from mdemg:
  mdemg uses a multi-layer graph (L0->Ln) with CO_ACTIVATED_WITH edges
  that strengthen via Hebbian learning (tanh soft-capping, multi-rate eta).
  We adapt this for experiment associations: when a change category leads
  to improvement, the association weight is strengthened. When it leads to
  regression, the weight is weakened. Temporal decay ensures the system
  doesn't get stuck on stale patterns.

Usage:
    memory = ExperimentMemory()
    memory.store_experiment(
        commit="a1b2c3d",
        description="increase matrix LR to 0.06",
        val_bpb=0.993,
        delta_bpb=-0.004,
        status="keep",
        change_tags=["learning_rate", "optimizer"],
    )
    directions = memory.get_promising_directions()
    memory.save()
"""

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Change category taxonomy
# ---------------------------------------------------------------------------

# Standardized tags for categorizing experiment changes. The agent (or a
# classifier in guidance.py) assigns one or more of these to each experiment.
CHANGE_CATEGORIES = [
    "architecture",      # model depth, width, layer structure
    "attention",         # attention mechanism changes (heads, windows, etc.)
    "activation",        # activation function changes (ReLU, GeLU, etc.)
    "optimizer",         # optimizer algorithm changes
    "learning_rate",     # learning rate tuning
    "schedule",          # warmup/cooldown/decay schedule changes
    "batch_size",        # batch size or gradient accumulation changes
    "initialization",    # weight initialization changes
    "regularization",    # weight decay, dropout, etc.
    "normalization",     # layer norm, RMS norm, etc.
    "embedding",         # embedding changes (value embeds, positional, etc.)
    "numerical",         # precision, softcap, numerical stability
    "simplification",    # code removal or simplification
    "combination",       # combining multiple prior improvements
    "radical",           # fundamental architectural departures
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentEntry:
    """A single experiment stored in memory."""
    commit: str
    description: str
    val_bpb: float
    delta_bpb: float  # negative = improvement
    status: str  # keep | discard | crash
    change_tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    surprise_score: float = 0.0  # how unexpected the result was
    peak_vram_mb: float = 0.0
    session_id: str = ""


@dataclass
class HebbianAssociation:
    """Tracks the association between a change category and outcomes.

    Inspired by mdemg's CO_ACTIVATED_WITH edges which strengthen when
    nodes are co-activated and decay over time. Here, "co-activation"
    means a change category was present in an experiment that improved BPB.

    Uses tanh soft-capping (from mdemg) instead of hard clamping to allow
    continuous learning without a hard saturation wall.
    """
    category: str
    # Positive outcomes (improvements)
    positive_count: int = 0
    positive_total_delta: float = 0.0  # sum of BPB improvements (positive values)
    # Negative outcomes (regressions)
    negative_count: int = 0
    negative_total_delta: float = 0.0  # sum of BPB regressions (positive values)
    # Current association weight (positive = promising, negative = unpromising)
    weight: float = 0.0
    # Last update timestamp (for temporal decay)
    last_updated: float = field(default_factory=time.time)
    # Total experiments involving this category
    total_experiments: int = 0


# ---------------------------------------------------------------------------
# Hebbian learning parameters (inspired by mdemg's learning engine)
# ---------------------------------------------------------------------------

@dataclass
class HebbianConfig:
    """Configuration for Hebbian association learning.

    Mirrors mdemg's learning rate configuration with:
    - eta: base learning rate for weight updates
    - wmax: soft saturation limit (used with tanh capping)
    - decay_rate: temporal decay applied to weights over time
    - surprise_multiplier: extra weight for surprising results
    """
    eta: float = 0.1            # base learning rate
    wmax: float = 1.0           # soft saturation limit for tanh capping
    decay_rate: float = 0.02    # per-hour decay rate
    decay_floor: float = 0.01   # minimum weight magnitude before zeroing
    surprise_multiplier: float = 2.0  # bonus for surprising results
    cautious_window_hours: float = 1.0  # recently-reinforced edges skip decay


# ---------------------------------------------------------------------------
# ExperimentMemory
# ---------------------------------------------------------------------------

class ExperimentMemory:
    """Persistent cross-session experiment knowledge base.

    Stores experiment results, tracks Hebbian associations between change
    categories and outcomes, and provides pattern extraction for guiding
    future experiments.

    Inspired by mdemg's CMS which provides:
    - Surprise-weighted learning (novel info retained longer)
    - Observation types (decision, correction, learning, etc.)
    - Resume/observe/correct/recall/consolidate APIs
    - Cross-session persistence

    Here we adapt those patterns for experiment research:
    - Surprise = unexpected BPB delta given the change category's history
    - Observation types = keep/discard/crash
    - Resume = load from disk at start of new session
    - Recall = query associations for promising directions
    - Consolidate = extract meta-patterns across experiments
    """

    def __init__(self, memory_dir: str = ".autoresearch/memory",
                 config: Optional[HebbianConfig] = None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or HebbianConfig()

        self.experiments: list[ExperimentEntry] = []
        self.associations: dict[str, HebbianAssociation] = {}
        self.session_id = f"session_{int(time.time())}"

        # Initialize associations for all known categories
        for cat in CHANGE_CATEGORIES:
            self.associations[cat] = HebbianAssociation(category=cat)

        # Load prior state
        self.load()

    # -- Core operations ----------------------------------------------------

    def store_experiment(self, commit: str, description: str,
                         val_bpb: float, delta_bpb: float, status: str,
                         change_tags: Optional[list[str]] = None,
                         peak_vram_mb: float = 0.0):
        """Store an experiment result and update Hebbian associations.

        Args:
            commit: Git commit hash (short)
            description: What the experiment tried
            val_bpb: Validation BPB achieved
            delta_bpb: Change from previous best (negative = improvement)
            status: keep | discard | crash
            change_tags: Categories of changes made
            peak_vram_mb: Peak VRAM usage
        """
        tags = change_tags or self._auto_tag(description)

        # Compute surprise score
        surprise = self._compute_surprise(tags, delta_bpb, status)

        entry = ExperimentEntry(
            commit=commit, description=description,
            val_bpb=val_bpb, delta_bpb=delta_bpb, status=status,
            change_tags=tags, surprise_score=surprise,
            peak_vram_mb=peak_vram_mb, session_id=self.session_id,
        )
        self.experiments.append(entry)

        # Update Hebbian associations (skip crashes — no signal)
        if status != "crash":
            self._hebbian_update(tags, delta_bpb, surprise)

        self.save()

    def get_associations(self) -> list[dict]:
        """Return all Hebbian associations sorted by weight (most promising first).

        Returns a list of dicts with category, weight, positive/negative counts,
        average improvement, and a confidence score based on sample size.
        """
        results = []
        for cat, assoc in sorted(self.associations.items(),
                                  key=lambda x: x[1].weight, reverse=True):
            if assoc.total_experiments == 0:
                continue
            avg_improvement = (assoc.positive_total_delta /
                               max(assoc.positive_count, 1))
            results.append({
                "category": cat,
                "weight": round(assoc.weight, 4),
                "positive_count": assoc.positive_count,
                "negative_count": assoc.negative_count,
                "total_experiments": assoc.total_experiments,
                "avg_improvement": round(avg_improvement, 6),
                "success_rate": round(assoc.positive_count /
                                      max(assoc.total_experiments, 1), 3),
                "confidence": min(assoc.total_experiments / 10, 1.0),
            })
        return results

    def get_promising_directions(self, top_k: int = 5) -> list[dict]:
        """Return the top-K most promising change categories.

        Combines Hebbian weight with confidence (sample size) to rank
        categories. Categories with high weight but low confidence are
        ranked lower (they need more experiments to be trusted).

        Inspired by mdemg's spreading activation which computes transient
        scores per query rather than relying solely on stored weights.
        """
        scored = []
        for cat, assoc in self.associations.items():
            if assoc.total_experiments == 0:
                # Unexplored categories get a small exploration bonus
                scored.append({
                    "category": cat,
                    "score": 0.1,  # exploration bonus
                    "reason": "unexplored — worth trying",
                    "experiments": 0,
                })
                continue

            confidence = min(assoc.total_experiments / 10, 1.0)
            # Blend weight with exploration bonus for low-sample categories
            exploration_bonus = 0.05 * (1 - confidence)
            score = assoc.weight * confidence + exploration_bonus

            if score > 0:
                reason = (f"weight={assoc.weight:.3f}, "
                          f"success_rate={assoc.positive_count}/{assoc.total_experiments}, "
                          f"confidence={confidence:.1f}")
                scored.append({
                    "category": cat,
                    "score": round(score, 4),
                    "reason": reason,
                    "experiments": assoc.total_experiments,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_dead_ends(self, min_experiments: int = 3) -> list[dict]:
        """Return categories that consistently fail to improve BPB.

        These are categories with negative Hebbian weight and enough
        experiments to be statistically meaningful. The agent should
        avoid these unless combining with other promising changes.
        """
        dead_ends = []
        for cat, assoc in self.associations.items():
            if (assoc.total_experiments >= min_experiments and
                    assoc.weight < -0.1):
                dead_ends.append({
                    "category": cat,
                    "weight": round(assoc.weight, 4),
                    "experiments": assoc.total_experiments,
                    "success_rate": round(assoc.positive_count /
                                          max(assoc.total_experiments, 1), 3),
                })
        dead_ends.sort(key=lambda x: x["weight"])
        return dead_ends

    def get_plateaus(self, window: int = 10) -> dict:
        """Detect improvement plateaus by analyzing recent experiment history.

        Returns analysis of whether improvements are slowing down,
        inspired by mdemg's anomaly detection for stale nodes.
        """
        if len(self.experiments) < window:
            return {"plateau_detected": False, "reason": "insufficient data"}

        recent = self.experiments[-window:]
        kept = [e for e in recent if e.status == "keep"]
        avg_delta = 0.0
        if kept:
            avg_delta = sum(e.delta_bpb for e in kept) / len(kept)

        # Compare to earlier improvement rate
        if len(self.experiments) > 2 * window:
            earlier = self.experiments[-2 * window:-window]
            earlier_kept = [e for e in earlier if e.status == "keep"]
            earlier_avg = 0.0
            if earlier_kept:
                earlier_avg = sum(e.delta_bpb for e in earlier_kept) / len(earlier_kept)

            velocity_ratio = abs(avg_delta) / max(abs(earlier_avg), 1e-8)
            if velocity_ratio < 0.3:
                return {
                    "plateau_detected": True,
                    "reason": f"improvement velocity dropped to {velocity_ratio:.0%} of earlier rate",
                    "recent_avg_delta": round(avg_delta, 6),
                    "earlier_avg_delta": round(earlier_avg, 6),
                    "kept_in_window": len(kept),
                }

        return {
            "plateau_detected": len(kept) == 0,
            "reason": "no improvements in window" if len(kept) == 0
                      else f"{len(kept)} improvements in last {window} experiments",
            "recent_avg_delta": round(avg_delta, 6),
            "kept_in_window": len(kept),
        }

    def decay(self):
        """Apply temporal decay to all Hebbian association weights.

        Inspired by mdemg's temporal decay system which uses exponential
        weight decay with cautious skipping of recently-reinforced edges.

        Called periodically (e.g., once per hour or after N experiments)
        to keep the system responsive to recent findings rather than
        being dominated by old results.
        """
        now = time.time()
        cfg = self.config
        for assoc in self.associations.values():
            hours_since_update = (now - assoc.last_updated) / 3600
            # Skip recently reinforced associations (cautious decay)
            if hours_since_update < cfg.cautious_window_hours:
                continue
            # Exponential decay
            decay_factor = math.exp(-cfg.decay_rate * hours_since_update)
            assoc.weight *= decay_factor
            # Floor: zero out negligible weights
            if abs(assoc.weight) < cfg.decay_floor:
                assoc.weight = 0.0
        self.save()

    def get_experiment_history(self, last_n: int = 20) -> list[dict]:
        """Return recent experiment history for context."""
        return [asdict(e) for e in self.experiments[-last_n:]]

    def get_surprise_highlights(self, top_k: int = 5) -> list[dict]:
        """Return the most surprising experiments (highest surprise scores).

        Surprise-weighted storage is inspired by mdemg's CMS which
        retains novel observations longer than routine ones. Surprising
        experiments often reveal important dynamics in the search space.
        """
        sorted_exps = sorted(self.experiments,
                             key=lambda e: e.surprise_score, reverse=True)
        return [
            {
                "commit": e.commit,
                "description": e.description,
                "val_bpb": e.val_bpb,
                "delta_bpb": round(e.delta_bpb, 6),
                "surprise_score": round(e.surprise_score, 4),
                "status": e.status,
                "change_tags": e.change_tags,
            }
            for e in sorted_exps[:top_k]
        ]

    # -- Persistence --------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """Persist memory state to disk."""
        out = Path(path) if path else self.memory_dir / "memory.json"
        data = {
            "experiments": [asdict(e) for e in self.experiments],
            "associations": {k: asdict(v) for k, v in self.associations.items()},
            "session_id": self.session_id,
            "saved_at": time.time(),
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: Optional[str] = None):
        """Load memory state from disk."""
        src = Path(path) if path else self.memory_dir / "memory.json"
        if not src.exists():
            return
        try:
            with open(src) as f:
                data = json.load(f)

            self.experiments = [
                ExperimentEntry(**e) for e in data.get("experiments", [])
            ]
            for cat, assoc_data in data.get("associations", {}).items():
                self.associations[cat] = HebbianAssociation(**assoc_data)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Corrupted memory — start fresh

    # -- Internal: Hebbian learning -----------------------------------------

    def _hebbian_update(self, tags: list[str], delta_bpb: float,
                        surprise: float):
        """Update Hebbian association weights based on experiment outcome.

        Uses tanh soft-capping from mdemg's learning engine:
            w_new = wmax * tanh((w_old + eta * signal) / wmax)

        This provides smooth saturation instead of hard clamping,
        allowing continuous learning even near the weight limits.

        The learning signal is:
        - Positive (strengthening) when BPB improved (delta_bpb < 0)
        - Negative (weakening) when BPB regressed (delta_bpb > 0)
        - Scaled by surprise for unexpected results
        """
        cfg = self.config
        now = time.time()

        for tag in tags:
            if tag not in self.associations:
                self.associations[tag] = HebbianAssociation(category=tag)

            assoc = self.associations[tag]
            assoc.total_experiments += 1
            assoc.last_updated = now

            if delta_bpb < 0:
                # Improvement: strengthen association
                assoc.positive_count += 1
                assoc.positive_total_delta += abs(delta_bpb)
                signal = abs(delta_bpb) * 100  # Scale up small BPB deltas
            else:
                # Regression: weaken association
                assoc.negative_count += 1
                assoc.negative_total_delta += abs(delta_bpb)
                signal = -abs(delta_bpb) * 100

            # Apply surprise multiplier
            if surprise > 1.0:
                signal *= cfg.surprise_multiplier

            # Tanh soft-capped update (mdemg's learning rule)
            raw = assoc.weight + cfg.eta * signal
            assoc.weight = cfg.wmax * math.tanh(raw / cfg.wmax)

    def _compute_surprise(self, tags: list[str], delta_bpb: float,
                          status: str) -> float:
        """Compute how surprising this experiment result is.

        Surprise is high when:
        1. The outcome contradicts the Hebbian association (e.g., a
           "dead end" category produces a big improvement)
        2. The BPB delta is unusually large
        3. A crash occurs in a previously stable category

        Inspired by mdemg's CMS surprise-weighted learning where novel
        observations are retained longer than routine ones.
        """
        if status == "crash":
            return 1.5  # crashes are moderately surprising

        if not tags:
            return 1.0  # no prior to compare against

        # Average expected direction from associations
        avg_weight = 0.0
        count = 0
        for tag in tags:
            if tag in self.associations and self.associations[tag].total_experiments > 0:
                avg_weight += self.associations[tag].weight
                count += 1

        if count == 0:
            return 1.0  # unknown categories are moderately surprising

        avg_weight /= count

        # Surprise = contradiction between expected and actual
        if avg_weight > 0 and delta_bpb > 0:
            # Expected improvement, got regression
            return 1.5 + abs(delta_bpb) * 100
        elif avg_weight < 0 and delta_bpb < 0:
            # Expected regression, got improvement
            return 2.0 + abs(delta_bpb) * 100
        else:
            # Outcome matched expectation — low surprise
            return 0.5 + abs(delta_bpb) * 50

    def _auto_tag(self, description: str) -> list[str]:
        """Automatically tag an experiment based on its description.

        Simple keyword matching to assign change categories. The agent
        can also provide explicit tags for more accuracy.
        """
        desc_lower = description.lower()
        tags = []

        keyword_map = {
            "architecture": ["layer", "depth", "width", "block", "head",
                             "dim", "model size", "n_layer", "n_embd"],
            "attention": ["attention", "window", "flash", "causal", "kv",
                          "gqa", "sliding", "sssl", "rope", "rotary"],
            "activation": ["relu", "gelu", "swish", "silu", "activation",
                           "squared", "tanh"],
            "optimizer": ["optimizer", "muon", "adamw", "adam", "sgd",
                          "momentum", "nesterov"],
            "learning_rate": ["lr", "learning rate", "learning_rate",
                              "matrix_lr", "embedding_lr", "scalar_lr"],
            "schedule": ["warmup", "cooldown", "warmdown", "schedule",
                         "cosine", "linear decay", "final_lr"],
            "batch_size": ["batch", "grad_accum", "accumulation",
                           "total_batch"],
            "initialization": ["init", "initialization", "weight init",
                               "xavier", "kaiming"],
            "regularization": ["decay", "weight_decay", "dropout",
                               "regulariz"],
            "normalization": ["norm", "layernorm", "rmsnorm", "prenorm",
                              "postnorm"],
            "embedding": ["embedding", "embed", "value embed", "wte",
                          "positional"],
            "numerical": ["precision", "softcap", "bfloat", "float16",
                          "numerical", "nan", "clamp"],
            "simplification": ["remov", "delet", "simplif", "clean",
                               "strip", "drop"],
            "combination": ["combin", "merge", "together", "plus",
                            "along with"],
            "radical": ["radical", "fundamental", "rewrite", "replace",
                        "entirely", "from scratch"],
        }

        for category, keywords in keyword_map.items():
            if any(kw in desc_lower for kw in keywords):
                tags.append(category)

        return tags if tags else ["misc"]
