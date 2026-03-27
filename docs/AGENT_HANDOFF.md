# Agent Handoff: autoresearch Observability & Intelligence Enhancement

**Date:** 2026-03-18
**Author:** reh3376
**Branch:** `feat/mdemg-observability-and-memory`
**PR:** https://github.com/karpathy/autoresearch/pull/329
**Target:** `karpathy/autoresearch:master`

---

## Project Context

**autoresearch** is Andrej Karpathy's framework for autonomous AI-driven LLM hyperparameter research. An AI agent modifies `train.py`, runs 5-minute GPU experiments, and keeps improvements — compounding gains overnight with zero human intervention.

This work adds production-grade observability, learning, and resilience infrastructure inspired by **mdemg** — a persistent memory system for AI coding agents built on Neo4j, Hebbian learning, and Prometheus metrics.

---

## Work Completed

### Research & Analysis Phase

1. **Deep dive into mdemg codebase** (~165K LOC, Go)
   - Analyzed 38 internal packages across monitoring, learning, resilience, and autonomy
   - Mapped 4 git submodules (homebrew, windows, menubar, autoresearch)
   - Identified key patterns: Hebbian learning with tanh soft-capping, circuit breakers with half-open recovery, Jiminy proactive guidance with 4-source parallel fan-out, CMS surprise-weighted storage, RSIC self-improvement cycle

2. **Deep dive into autoresearch codebase**
   - Full analysis of `prepare.py` (read-only data/eval), `train.py` (agent-editable model/optimizer), `program.md` (agent instructions)
   - Documented GPT architecture (RoPE, Flash Attention 3, MuonAdamW dual optimizer, value embeddings, sliding window attention)
   - Mapped the complete experiment lifecycle and agent autonomy constraints

3. **Gap analysis: mdemg patterns applicable to autoresearch**
   - Identified that autoresearch had no monitoring, no cross-session memory, no anomaly detection, no learning from experiment patterns, and only basic crash handling (NaN/loss > 100)

### Implementation Phase

Four new Python modules created (2,193 LOC total, zero new dependencies):

| Module | Lines | mdemg Inspiration | Key Classes |
|--------|-------|-------------------|-------------|
| `monitor.py` | 535 | `internal/metrics/` | `ExperimentTracker`, `AlertThresholds`, `format_dashboard()` |
| `memory.py` | 566 | `internal/conversation/`, `internal/learning/` | `ExperimentMemory`, `HebbianAssociation`, `HebbianConfig` |
| `resilience.py` | 571 | `internal/circuitbreaker/`, `internal/anomaly/`, `internal/backpressure/` | `CircuitBreaker`, `AnomalyDetector`, `BackpressureMonitor`, `ExperimentGuard` |
| `guidance.py` | 521 | `internal/jiminy/`, `internal/ape/` | `ExperimentAdvisor`, `StrategyAssessment`, `ExperimentSuggestion` |

### Documentation & Integration Phase

- **program.md**: +205 lines with full module documentation, usage examples, env var reference, and recommended integration pattern
- **analysis.ipynb**: +6 cells for Hebbian category effectiveness charts, monitoring dashboard with VRAM trends, and guidance report generation
- **.gitignore**: added `.autoresearch/` state directory exclusion

### Key Technical Decisions Made

1. **Zero new dependencies** — all modules use Python stdlib only (json, time, math, dataclasses)
2. **Non-intrusive design** — modules never modify train.py or prepare.py; they observe and advise
3. **Opt-in architecture** — each module is independently usable
4. **Tanh soft-capping** for Hebbian weights (from mdemg) instead of hard clamping — continuous learning without saturation walls
5. **Circuit breaker with exponential backoff** — CLOSED → OPEN → HALF_OPEN state machine with configurable cooldown multiplier

---

## Suggested Future Work

### High Priority

#### 1. Train.py Integration Hook
The modules are implemented but not yet wired into `train.py`. A lightweight integration wrapper that the agent can optionally call from the training loop would close the gap:

```python
# Suggested: add to train.py training loop
from monitor import ExperimentTracker
tracker = ExperimentTracker()
# ... call tracker.record_step() each step
```

This should remain optional — the agent decides whether to use it based on program.md guidance.

#### 2. Automated Test Suite
The modules have no unit tests yet. Recommended coverage:

- `test_monitor.py`: ExperimentTracker lifecycle, alert threshold triggering, Prometheus text format validation, JSON export round-trip
- `test_memory.py`: Hebbian weight updates (positive/negative signals), tanh soft-capping bounds, temporal decay, auto-tagging accuracy, persistence round-trip
- `test_resilience.py`: Circuit breaker state transitions (CLOSED→OPEN→HALF_OPEN→CLOSED), anomaly detection for each pattern (plateau, VRAM creep, regression, crash cluster), backpressure levels
- `test_guidance.py`: Suggestion ranking, contradiction detection, strategy phase classification

#### 3. Agent Instruction Enhancement for Module Usage
Update `program.md` to make the agent more opinionated about *when* to consult guidance vs. when to just run experiments. Current docs explain *how* but not *when* — e.g., "consult guidance every 5th experiment" or "check guidance after 3 consecutive discards."

### Medium Priority

#### 4. Loss Curve Comparison Visualization
The monitor captures per-step loss curves but analysis.ipynb doesn't yet overlay them for comparison. A cell that overlays the last N experiments' loss curves (color-coded by keep/discard) would help identify training dynamic patterns.

#### 5. Experiment Embedding Space
Rather than keyword auto-tagging, compute semantic embeddings of experiment descriptions (using the existing tokenizer or a lightweight model) and cluster experiments in embedding space. This would:
- Improve category assignment accuracy
- Reveal unexpected category structure
- Enable similarity-based "try something like experiment X" suggestions

#### 6. Multi-GPU / Multi-Agent Coordination
autoresearch currently runs on a single GPU. The monitoring and memory modules could be extended with:
- Shared memory store (SQLite or shared JSON) for multiple agents exploring in parallel
- Distributed circuit breaker (aggregate crash rates across agents)
- Cross-agent contradiction detection (agent A found X helps, agent B found X hurts)

#### 7. Prometheus + Grafana Dashboard Template
The `get_prometheus_text()` method is implemented but there's no accompanying Grafana dashboard JSON. A pre-built dashboard template with panels for:
- Experiment throughput (experiments/hour)
- BPB frontier progression
- VRAM pressure gauge
- Alert timeline
- Category effectiveness heatmap

Would make the monitoring immediately actionable for teams running autoresearch at scale.

### Lower Priority / Exploratory

#### 8. RSIC Full Implementation
The current guidance module implements only the *assess* phase of mdemg's RSIC (Recursive Self-Improvement Cycle). The full cycle is:

- **Assess** (implemented): evaluate strategy effectiveness
- **Reflect** (not implemented): identify why certain experiments succeeded/failed
- **Plan** (not implemented): generate a multi-experiment research plan
- **Speculate** (not implemented): predict outcomes of proposed experiments
- **Execute** (not implemented): autonomous plan execution with rollback

Full RSIC would make the agent significantly more strategic, moving from reactive (try something → evaluate) to proactive (plan a research trajectory → execute → adjust).

#### 9. Hebbian Network Visualization
The memory module stores pairwise associations between change categories but doesn't visualize the network. A graph visualization (networkx or d3.js) showing:
- Nodes = change categories, sized by experiment count
- Edges = co-occurrence in experiments, colored by joint success rate
- Node color = Hebbian weight (green = promising, red = dead end)

Would provide intuitive insight into the experiment search landscape.

#### 10. Historical Experiment Replay
Import historical `results.tsv` files into the memory system to bootstrap Hebbian associations for new sessions. This would let the agent start with learned priors rather than exploring from scratch — analogous to mdemg's space transfer feature.

#### 11. Adaptive Alert Thresholds
Current alert thresholds are static (configurable via env vars but fixed during a session). Implementing adaptive thresholds that learn from the experiment distribution — e.g., loss spike threshold based on rolling loss variance — would reduce false positives in early (volatile) vs. late (stable) experiment phases.

---

## Architecture Reference

```
autoresearch/
├── train.py            # Agent-modifiable (unchanged by this work)
├── prepare.py          # Read-only data/eval (unchanged)
├── program.md          # Agent instructions (updated with module docs)
├── analysis.ipynb      # Post-hoc analysis (enhanced with new cells)
├── monitor.py          # NEW: metrics, alerting, Prometheus export
├── memory.py           # NEW: Hebbian learning, cross-session memory
├── resilience.py       # NEW: circuit breakers, anomaly detection
├── guidance.py         # NEW: proactive suggestions, strategy assessment
├── .gitignore          # Updated: excludes .autoresearch/
├── .autoresearch/      # Runtime state (gitignored)
│   ├── metrics/        #   monitor session data + JSON exports
│   ├── memory/         #   experiment memory + Hebbian associations
│   └── resilience/     #   circuit breaker + anomaly state
└── docs/
    └── AGENT_HANDOFF.md  # This file
```

## Module Dependency Graph

```
guidance.py ──depends on──> memory.py     (Hebbian associations)
            ──depends on──> monitor.py    (session stats, alerts)
            ──depends on──> resilience.py (safety status)

resilience.py  (standalone — no module dependencies)
monitor.py     (standalone — no module dependencies)
memory.py      (standalone — no module dependencies)
```

All four modules are independently usable. `guidance.py` optionally integrates with the other three but gracefully handles `None` for any missing module.

---

## Key Files for Onboarding

If picking up this work, read in this order:

1. `program.md` — the "Observability & Intelligence Modules" section at the bottom
2. `monitor.py` — simplest module, good entry point for understanding the pattern
3. `memory.py` — core Hebbian learning logic, most novel contribution
4. `resilience.py` — circuit breaker state machine, anomaly detection patterns
5. `guidance.py` — synthesis layer, depends on understanding the other three
