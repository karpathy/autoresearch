# MnemeBrain Controlled Experiment Framework (v3)

## Context

Scientific comparison: does structured belief memory improve autonomous ML experimentation? The user runs `uv run train.py` manually, edits hyperparams between runs, and wants to prove that MnemeBrain helps make better experiment decisions.

**Key design principle:** All conditions run identical training code. MnemeBrain only influences what happens BETWEEN runs — never during training. This isolates the memory effect from optimizer effects.

## Research Docs Review

| Doc | Status | Key Finding |
|-----|--------|-------------|
| research_v1 | Solid | Correctly identifies forgetting problem. Gap analysis accurate. |
| research_v2 | Solid | Three core novelties (differentiable Belnap, AGM as loss, neural TMS) genuine. |
| research_v3 | Solid | Critical: LTN single-scalar, Evidential DL can't represent BOTH, Gumbel-Softmax incompatible with bilattice. |
| architecture_v4.1 | Good, gaps | Needs: training-loop telemetry spec, belief schema for experiment observations, prediction capability spec. |

**Doc improvements needed:**
- v4.1: Add "Experiment Telemetry" section — what signals to capture post-run
- v4.1: Add "Belief Schema" section — 3-level schema (observation, tendency, contextual)
- v4.1: Add "Prediction" section — sandbox-based outcome prediction before runs
- Sidecar weight formula (`0.5 + delta*10`) too narrow at ±0.05 BPB — use log-scale or adaptive capping

## Conditions (Revised — 4 Conditions)

### Condition A — No Memory, No Guidance
- **What:** Plain train.py. No sidecar. No history. No suggestions.
- **Changes to train.py:** None
- **During execution:** No memory, no guidance — stdout behavior unchanged
- **Post-run artifacting:** `NullHooks` writes a run result JSON to `results/` for analysis only (invisible to the operator during experiment sessions)
- **Purpose:** Raw baseline — how well does a human do with no memory?

### Condition B — Structured Logging (Control)
- **What:** Same training, but post-run telemetry is stored as structured JSON. Before next run, recent summaries are printed. No belief graph, no contradiction detection, no ranking.
- **Changes to train.py:** Post-run: write structured JSON summary to `experiment_log.jsonl`
- **Pre-run:** Print last N summaries from JSONL
- **What B captures:**
  ```json
  {
    "run_id": 12,
    "config": {"lr": 0.06, "wd": 0.1, "depth": 8},
    "outcome": {"val_bpb": 1.92, "diverged": true},
    "signals": {"grad_norm_max": 9.1, "loss_trend": "diverging", "final_step": 342},
    "summary": "LR 0.06 diverged around step 200"
  }
  ```
- **Purpose:** Controls for "more observability." If C beats B, the gain comes from belief structure, not just having notes.

### Condition C — Passive MnemeBrain
- **What:** MnemeBrain stores/retrieves beliefs across runs. Surfaces contradictions, confidence, similar prior runs. Does NOT suggest parameter changes.
- **Changes to train.py:** Post-run: record result as belief via sidecar. Pre-run: print belief context.
- **What C adds over B:**
  - Belnap four-valued logic: contradictions are first-class (BOTH state)
  - Evidence-weighted confidence ranking
  - Semantic similarity search ("find runs similar to this config")
  - Structured contradiction surfacing ("LR=0.03 worked in run 5 but failed in run 8 — context differed by warmup schedule")
  - Belief decay (stale evidence loses weight)
- **Purpose:** Isolates memory structure value

### Condition D — Active MnemeBrain
- **What:** Everything in C, plus pre-run recommendation and sandbox prediction
- **Changes to train.py:** Pre-run: query sidecar for prediction + recommendation
- **What D adds over C:**
  - **Prediction:** Before each run, fork a BeliefSandbox and predict outcome:
    ```
    prediction:
      expected_outcome: "improve modestly"
      confidence: 0.74
    ```
  - **Recommendation:** Suggest next experiment based on accumulated beliefs:
    ```
    recommendation:
      change: "reduce lr from 0.03 to 0.02"
      rationale:
        - 4 similar runs with lower grad instability
        - 2 contradictory runs explained by missing warmup
      risk: "low"
    ```
  - Log whether prediction/recommendation were correct
- **Purpose:** Isolates reasoning-on-top-of-memory value

**Critical rule: NO mid-training intervention in any condition.** Once `uv run train.py` starts, it runs identically regardless of condition. All condition logic happens before or after the run.

## Condition Order Control

The human operator learns across conditions — running A first, then B, then C, then D creates a learning confound that can fake part of the effect.

**Counterbalancing strategy:** Latin-square rotation across seed groups:
- Seed group 1: A → B → C → D
- Seed group 2: B → C → D → A
- Seed group 3: C → D → A → B
- Seed group 4: D → A → B → C

**Isolation rules:**
- Each condition gets its own **decision sheet** (a separate document/file)
- Before each run, write rationale in that condition's sheet only
- No access to prior-condition sheets during an active session
- Decision sheets are the auditable record — reviewers can verify no cross-condition leakage
- All decision rationale must be written down before the run starts (forces explicit reasoning, prevents implicit memory leakage)

## Config Search Space

All hyperparameter experiments operate within these bounds. Defined upfront so that "similar config," "L2 tolerance," and "wasted run" are well-specified.

| Parameter | Range | Scale | Normalization |
|-----------|-------|-------|---------------|
| lr | 0.001–0.1 | log | normalize in log10 space: `(log10(lr) - log10(0.001)) / (log10(0.1) - log10(0.001))` |
| wd | 0.0–0.2 | linear | min-max: `wd / 0.2` |
| depth | 4–16 | integer | min-max: `(depth - 4) / 12` |
| batch_size | 16–64 | discrete | min-max: `(bs - 16) / 48` |
| warmup | 0.0–0.2 | linear | min-max: `warmup / 0.2` |

**Config distance:** L2 over the normalized parameter vector (with LR in log space).

**"Meaningful step" thresholds** (for determining whether a near-bad-config run is a deliberate probe):
- LR: changed by ≥2× (one octave in log space)
- depth: changed by ≥2 layers
- warmup: changed by ≥0.05 absolute
- batch_size: changed by ≥1 discrete step (e.g., 16→32)
- wd: changed by ≥0.05 absolute

## Decomposition (Why This Design Is Strong)

| Comparison | What It Isolates |
|------------|-----------------|
| A vs B | Does telemetry/observability alone help? |
| B vs C | Does structured belief memory beat plain logs? |
| C vs D | Does active reasoning beat passive memory? |
| A vs D | Total effect of the full system |

If B ≈ A → telemetry alone doesn't explain gains.
If C ≈ B → memory structure isn't doing enough.
If D ≈ C → active reasoning is weak.

## Belief Schema (3 Levels)

### Level 1 — Run-Observation Beliefs
Direct observations from a single run:
```
claim: "Run 12 showed gradient norm spike > 8 at step 180"
evidence: { run_id: 12, step: 180, grad_norm: 8.3 }
polarity: SUPPORTS
type: FACT
```

### Level 2 — Causal/Tendency Beliefs
Patterns across multiple runs:
```
claim: "LR 0.06 tends to destabilize training"
evidence: [run 12 (diverged), run 15 (diverged), run 8 (marginal)]
polarity: SUPPORTS
confidence: 0.82
type: INFERENCE
```

### Level 3 — Contextual Beliefs
Context-dependent patterns that resolve contradictions through contextual revision:

**Before contextual analysis** — the broad claim is contradictory:
```
claim: "LR 0.03 improves training"
truth_state: BOTH
evidence:
  - { polarity: SUPPORTS, source: run 5, warmup: 200, val_bpb: 1.85 }
  - { polarity: ATTACKS, source: run 8, warmup: 0, val_bpb: 2.31 }
type: INFERENCE
```

**After contextual revision** — the contradiction resolves into two contextualized claims:
```
claim: "LR 0.03 improves training when warmup > 100 steps"
truth_state: TRUE
evidence:
  - { polarity: SUPPORTS, source: run 5, warmup: 200, val_bpb: 1.85 }
type: INFERENCE
context: { warmup: ">100 steps" }
```
```
claim: "LR 0.03 harms training when warmup = 0"
truth_state: TRUE
evidence:
  - { polarity: SUPPORTS, source: run 8, warmup: 0, val_bpb: 2.31 }
type: INFERENCE
context: { warmup: "0" }
```

Note: `truth_state` is a Belnap four-valued state (TRUE, FALSE, BOTH, NEITHER). `polarity` is per-evidence (SUPPORTS or ATTACKS). BOTH emerges from conflicting evidence on broad claims — contextual revision is the mechanism that resolves BOTH into conditional TRUE/FALSE claims with narrower scope.

This third level is crucial — it prevents overgeneralization and demonstrates belief model advantage. The key insight: MnemeBrain doesn't just store contradictions, it *resolves* them by discovering the contextual variable that explains the divergence.

## Outcome Metrics

### Primary
1. **Runs-to-threshold** — How many experiments to reach val_bpb <= X? (strongest metric)
2. **Best achieved val_bpb** — Classic end result after fixed N runs

### Secondary
3. **Improvement rate** — Average val_bpb delta per experiment
4. **Bad-run rate** — Fraction of runs that diverge or clearly regress
5. **Wasted-run rate** — A run is **wasted** if it satisfies any of:
   - Diverged (loss → NaN or exceeded 10× initial loss)
   - val_bpb regressed by more than ε=0.02 from the prior run's val_bpb
   - Config is within L2 tolerance (τ_waste, TBD from pilot) of a previously observed bad config in normalized param space
   - Exact duplicate of a previously tested config with no new rationale

   **Exception:** A run near a known-bad config is **not wasted** if:
   - `rationale_tag` is non-empty in `RunConfig` (hypothesis written before the run), AND
   - At least one focal parameter changed beyond a "meaningful step" threshold (see Config Search Space section)

   This exception protects deliberate ablation probes near failure boundaries. The `rationale_tag` field makes this auditable.

### Condition D Only
6. **Recommendation hit rate** — How often did the suggested change improve results?
7. **Prediction calibration** — Brier score on outcome predictions

## Results Directory

All experiment results are written to `results/` at the project root (`/Users/in615bac/Documents/MnemeBrain/mnemebrain-labs/autoresearch/results/`). This provides a single location for both human and AI review.

### Directory Structure

```
results/
├── runs/                          # Individual run results
│   ├── condition_A/
│   │   ├── seed_01/
│   │   │   ├── run_001.json       # Full run result (config + results + telemetry)
│   │   │   ├── run_002.json
│   │   │   └── ...
│   │   ├── seed_02/
│   │   └── ...
│   ├── condition_B/
│   ├── condition_C/
│   └── condition_D/
├── logs/                          # Raw training stdout/stderr per run
│   └── condition_B/seed_01/run_001.log
├── experiment_log.jsonl           # Condition B append-only structured log
├── summaries/                     # Per-condition aggregate summaries (auto-generated)
│   ├── condition_A_summary.json   # Best val_bpb, run count, wasted runs, improvement curve
│   ├── condition_B_summary.json
│   ├── condition_C_summary.json
│   └── condition_D_summary.json
├── comparisons/                   # Cross-condition analysis (auto-generated after each condition completes)
│   ├── pilot_comparison.json      # Pilot N=5 results for promotion decision
│   └── full_comparison.json       # Final N=20 results
└── plots/                         # Optional visualization outputs
    ├── val_bpb_by_condition.png
    ├── runs_to_threshold.png
    └── wasted_run_rate.png
```

### Run Result Schema (`runs/condition_X/seed_Y/run_NNN.json`)

```json
{
  "run_id": 1,
  "condition": "B",
  "seed": 42,
  "timestamp": "2026-03-16T14:32:00Z",
  "config": {
    "lr": 0.03,
    "wd": 0.1,
    "depth": 8,
    "batch_size": 32,
    "warmup": 0.1
  },
  "results": {
    "val_bpb": 1.872,
    "steps": 1450,
    "peak_vram_mb": 4200,
    "final_loss": 2.34,
    "mfu": 0.38,
    "diverged": false,
    "loss_trend": "improving",
    "grad_norm_max": 3.2
  },
  "rationale_tag": "ablate_lr_near_failure",
  "rationale": "Testing lower LR after run 0 showed instability at 0.06",
  "wasted": false,
  "delta_from_best": -0.048,
  "pre_run_context": {
    "condition": "B",
    "summaries": ["Run 0: LR=0.06, diverged at step 200"]
  }
}
```

### Condition Summary Schema (`summaries/condition_X_summary.json`)

```json
{
  "condition": "B",
  "total_runs": 12,
  "seeds_completed": [1, 2, 3, 4, 5],
  "best_val_bpb": 1.824,
  "mean_val_bpb": 1.891,
  "std_val_bpb": 0.034,
  "wasted_runs": 3,
  "wasted_run_rate": 0.25,
  "diverged_runs": 2,
  "improvement_rate": -0.0056,
  "runs_to_threshold": 8,
  "val_bpb_trajectory": [2.10, 1.95, 1.92, 1.88, 1.87, 1.85, 1.84, 1.83, 1.83, 1.82, 1.83, 1.82]
}
```

### Comparison Schema (`comparisons/pilot_comparison.json`)

```json
{
  "protocol": "protocol_1",
  "phase": "pilot",
  "n_seeds": 5,
  "timestamp": "2026-03-17T10:00:00Z",
  "conditions": {
    "A": { "mean_best_val_bpb": 1.91, "std": 0.04, "mean_wasted_rate": 0.35 },
    "B": { "mean_best_val_bpb": 1.87, "std": 0.03, "mean_wasted_rate": 0.28 },
    "C": { "mean_best_val_bpb": 1.83, "std": 0.02, "mean_wasted_rate": 0.15 },
    "D": { "mean_best_val_bpb": 1.80, "std": 0.02, "mean_wasted_rate": 0.10 }
  },
  "pairwise_tests": {
    "A_vs_B": { "cohens_d": 0.42, "directional": true },
    "B_vs_C": { "cohens_d": 0.55, "directional": true },
    "C_vs_D": { "cohens_d": 0.38, "directional": true }
  },
  "promotion_decision": "promote",
  "promotion_rationale": "All 4 promotion criteria met"
}
```

### Design Principles

1. **One run = one file.** No appending to shared files during training — avoids corruption from crashes. `experiment_log.jsonl` (Condition B) is the sole exception, written post-run.
2. **Summaries are derived, not primary.** Summary and comparison files are regenerated from individual run files. Deleting them loses nothing.
3. **Human-scannable.** `cat results/summaries/condition_B_summary.json | python -m json.tool` gives immediate overview. `ls results/runs/condition_A/seed_01/` shows run count at a glance.
4. **AI-parseable.** Consistent JSON schemas mean an AI agent can `glob results/runs/**/*.json`, parse, and compare across conditions without custom code.
5. **Git-ignored.** Add `results/` to `.gitignore` — results are experiment artifacts, not source code.

### Hook Integration

Each hook implementation writes to the results directory:

| Hook | Writes to |
|------|-----------|
| `NullHooks` | `results/runs/condition_A/seed_Y/run_NNN.json` (config + results only) |
| `LoggingHooks` | Same + `results/experiment_log.jsonl` (append) |
| `PassiveHooks` | Same as NullHooks (beliefs stored in sidecar, not results dir) |
| `ActiveHooks` | Same + `pre_run_context` includes prediction/recommendation fields |

Summary and comparison files are generated by a separate `results_analyzer.py` utility, not by the hooks — keeps hook code minimal.

## Execution Protocols

### Protocol 1 — Fixed Search Budget
Give each condition 12 runs. Measure: best final val_bpb, regressions, wasted runs.

### Protocol 2 — Threshold Race
Stop when condition hits target val_bpb. Measure: runs needed, wall-clock, failure count.

**Threshold determination (locked after Step 1b):**
- Run 3-5 baseline Condition A runs
- Compute mean and std of best val_bpb across those runs
- Set threshold = mean_best - 1σ (i.e., a target that baseline reaches ~16% of the time)
- Alternative: use the 25th percentile of baseline best-of-12 as threshold
- Do NOT precommit to a threshold number before seeing baseline data

### Protocol 3 — Contradictory Regime
Deliberately create cases where the same hyperparameter works in one context and fails in another:
- LR=0.03 good WITH warmup → LR=0.03 bad WITHOUT warmup
- batch_size=32 good at depth=8 → batch_size=32 bad at depth=4

This is where MnemeBrain's BOTH state and contextual beliefs should shine.

### Pilot Design and Promotion Criteria

The N=5 pilot is **not** a confirmatory significance test. It estimates:
- **Variance:** within-condition val_bpb variance across seeds
- **Effect size:** directional separation between conditions on primary metrics
- **Feasibility:** runtime, crash rate, telemetry completeness
- **Hook stability:** all 4 hook implementations run without errors

**Promotion to full run (N=20) requires ALL of:**
1. Hooks are stable — zero crashes across pilot runs
2. Telemetry is complete — all fields populated in experiment_log.jsonl
3. Directional separation appears — at least **two** of the following show consistent ordering across conditions:
   - runs-to-threshold
   - best achieved val_bpb
   - wasted-run rate
4. Estimated effect size justifies compute — Cohen's d > 0.3 estimated from pilot variance, suggesting the full run has reasonable power

If the pilot shows no directional signal at all, revisit the experimental design before scaling up.

### Repetitions
Run each protocol with N=20 seeds (after pilot promotion). Report mean and variance.

## Implementation

### New: `mnemebrain_hooks.py` — Between-run integration

#### Typed Data Contracts

```python
from dataclasses import dataclass, field

@dataclass
class RunConfig:
    lr: float
    wd: float
    depth: int
    batch_size: int
    warmup: float
    seed: int
    rationale_tag: str = ""  # Short structured label for audit/exception logic (e.g., "ablate_lr_near_failure")
    rationale: str = ""      # Optional human-readable explanation stored in run artifacts

@dataclass
class RunResults:
    val_bpb: float
    steps: int
    peak_vram_mb: float
    final_loss: float
    mfu: float
    diverged: bool
    loss_trend: str          # "improving" | "flat" | "diverging"
    grad_norm_max: float

@dataclass
class PredictionResult:
    # expected_outcome quantitative definitions (ε = 0.02 val_bpb):
    #   "improve"  — val_bpb beats current best by more than ε
    #   "marginal" — within ±ε of current best
    #   "regress"  — worse than current best by more than ε
    #   "diverge"  — NaN / unstable / catastrophic loss growth (>10× initial)
    expected_outcome: str    # "improve" | "regress" | "diverge" | "marginal"
    confidence: float
    similar_runs: list[int]
    risks: list[str]
    source_run_ids: list[int] = field(default_factory=list)  # Which prior runs informed this prediction

@dataclass
class RecommendationResult:
    suggested_change: str
    rationale: list[str]
    risk_level: str          # "low" | "medium" | "high"
    source_run_ids: list[int] = field(default_factory=list)  # Which prior runs informed this recommendation

@dataclass
class PreRunContext:
    condition: str           # "A" | "B" | "C" | "D"
    summaries: list[str] = field(default_factory=list)
    similar_runs: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    prediction: PredictionResult | None = None
    recommendation: RecommendationResult | None = None
```

#### Hook Protocol and Implementations

```python
class ExperimentHooks(Protocol):
    def pre_run(self, config: RunConfig) -> PreRunContext:
        """Called BEFORE training. Returns context/suggestions."""
    def post_run(self, config: RunConfig, results: RunResults) -> None:
        """Called AFTER training. Records results."""

class NullHooks:       # Condition A — returns empty PreRunContext, no-op post_run
class LoggingHooks:    # Condition B — JSONL read/write, summaries in PreRunContext
class PassiveHooks:    # Condition C — sidecar believe/search/context, contradictions populated
class ActiveHooks:     # Condition D — sidecar + sandbox + predict + recommend (all fields populated)
```

### Changes to train.py (minimal, ~8 lines)

```python
# Near top:
from mnemebrain_hooks import create_hooks, RunConfig, RunResults
hooks = create_hooks()  # reads AUTORESEARCH_CONDITION env var (A/B/C/D)

# Before training loop:
run_config = RunConfig(
    lr=MATRIX_LR, wd=WEIGHT_DECAY, depth=DEPTH,
    batch_size=TOTAL_BATCH_SIZE, warmup=WARMUP_RATIO, seed=SEED,
)
run_context = hooks.pre_run(run_config)

# After training + eval (end of file):
hooks.post_run(
    config=run_config,
    results=RunResults(
        val_bpb=val_bpb, steps=step, peak_vram_mb=peak_vram_mb,
        final_loss=debiased_smooth_loss, mfu=steady_state_mfu,
        diverged=diverged, loss_trend=loss_trend, grad_norm_max=grad_norm_max,
    ),
)
```

**No mid-training hooks.** Training dynamics (loss trend, grad norms) are computed post-run from the training log or final state only.

### Post-run telemetry capture

Instead of mid-training HTTP calls, capture telemetry at the end:
- Final smoothed loss
- Total steps completed
- Whether loss was trending down/flat/up in final 20%
- Peak gradient norm (requires adding a single line to accumulate max grad norm)
- MFU percentage

This is computed from values already available in train.py at end of training.

## Files to Create

| File | Location | Purpose |
|------|----------|---------|
| `mnemebrain_hooks.py` | autoresearch/ | ExperimentHooks protocol + 4 implementations |
| `results_analyzer.py` | autoresearch/ | Generates summaries and comparisons from individual run JSONs |
| `results/` | autoresearch/ | Results directory (git-ignored, created by hooks on first run) |

## Files to Modify

| File | Change |
|------|--------|
| `train.py` | Add ~8 lines: import hooks, pre_run before loop, post_run after eval |
| `autoresearch-mnemebrain/server.py` | Add `POST /predict` and `GET /recommend` endpoints (Condition D) |
| `models/__init__.py` | No change needed for core experiment |

## Files NOT Modified
- `prepare.py` (frozen)
- `models/nanochat.py`, `models/base.py`
- `platform_config.py`

## Sidecar Additions Needed

Existing endpoints sufficient for Conditions B and C. Condition D needs:

```
POST /predict
  Body: { config: {lr, wd, depth, batch_size, warmup} }
  → Forks BeliefSandbox, checks config against known beliefs
  → Returns: { expected_outcome, confidence, similar_runs, risks }

GET /recommend
  → Returns: { suggested_change, rationale, risk_level }
  → Based on: goals, untested regions, contradiction resolution
```

## Execution Sequence

See **Revised Execution Sequence** in Review Notes below — the original steps have been superseded.

## Verification

```bash
# Condition A (baseline, must match current exactly):
AUTORESEARCH_CONDITION=A uv run train.py

# Condition B (structured logging):
AUTORESEARCH_CONDITION=B uv run train.py
cat experiment_log.jsonl  # verify structured output

# Condition C (passive MnemeBrain — start sidecar first):
cd ../autoresearch-mnemebrain && uv run python -m server &
AUTORESEARCH_CONDITION=C uv run train.py
curl http://localhost:7432/context  # verify beliefs stored

# Condition D (active MnemeBrain — start full stack):
docker-compose -f ../autoresearch-mnemebrain/docker-compose.yml up -d
AUTORESEARCH_CONDITION=D uv run train.py
# Verify: prediction + recommendation printed before training starts

```

## Existing Code to Reuse

- `autoresearch-mnemebrain/server.py` — 6 existing endpoints (needs 2 new for D)
- `mnemebrain-python` SDK — MnemeBrainClient, sandbox, revision, attacks, goals
- `models/__init__.py` — Model registry + create_model() factory
- `models/base.py` — TrainableModel protocol
- `platform_config.py` — Hardware detection singleton

## Paper Claim

Not: "MnemeBrain improves model training."

Better: **"Structured belief memory reduces wasted experiments and reaches good configurations faster than no memory, plain logs, or passive memory alone."**

## Review Notes (2026-03-16, v2→v3)

### Resolved Issues

1. **Statistical Power** — N=20 × 12 runs × 4 conditions = 80 hours compute. Start with N=5 pilot (20 hours) to validate signal before full run.

2. **Human-in-the-loop Confound** — Conditions C/D measure human+MnemeBrain, not MnemeBrain alone.
   - **Main experiment:** Human-in-the-loop for all conditions (ecological validity — this is how the tool would actually be used)
   - **Supplemental experiment:** Condition D* where MnemeBrain recommendations are followed autonomously with no human override (isolates system judgment from human judgment)
   - D* is clearly optional but provides a clean answer when reviewers ask "how much is the human vs the system?"

3. **Missing Baseline Anchor** — Run unmodified `train.py` 3-5 times before designing Protocol 2 threshold. Record mean and variance of val_bpb.

4. **Sidecar Weight Formula** — Replace `0.5 + delta*10` with log-scale: `weight = 0.5 + 0.5 * tanh(log10(abs(delta) + 1e-4) + 2)`. Delta is relative to **current best val_bpb** (not previous run — avoids noisy oscillation). Sign matters: negative delta (improvement) → weight > 0.5, positive delta (regression) → weight < 0.5. Finalize formula before implementing Condition C.

   **Calibration table** (delta = val_bpb - best_val_bpb):
   | delta | weight | interpretation |
   |-------|--------|----------------|
   | -0.05 | ~0.88 | strong improvement → high-confidence belief |
   | -0.01 | ~0.76 | modest improvement |
   | -0.001 | ~0.60 | marginal improvement |
   | +0.001 | ~0.40 | marginal regression |
   | +0.01 | ~0.24 | modest regression |
   | +0.05 | ~0.12 | strong regression → low-confidence belief |

5. **Condition B Schema Gap** — Add `timestamp` (ISO 8601) and `seed` fields to `experiment_log.jsonl` schema:
   ```json
   {
     "run_id": 12,
     "timestamp": "2026-03-16T14:32:00Z",
     "seed": 42,
     "config": {"lr": 0.06, "wd": 0.1, "depth": 8},
     "outcome": {"val_bpb": 1.92, "diverged": true},
     "signals": {"grad_norm_max": 9.1, "loss_trend": "diverging", "final_step": 342},
     "summary": "LR 0.06 diverged around step 200"
   }
   ```

6. **Condition A Limitation** — Human memory is a confound. Acknowledge in paper: "Condition A measures human-with-implicit-memory, not zero-memory. A true no-memory baseline would require randomized configs or amnesic participants."

7. **BSM Transformer Deferred** — Step 6 (BSM-augmented model) is a separate research question. Defer entirely until Protocol 1 results are analyzed. Remove from critical path.

### Revised Execution Sequence

1. **Step 1:** Create `mnemebrain_hooks.py` with NullHooks + LoggingHooks (Conditions A & B)
2. **Step 1b:** Run 3-5 Condition A runs → establish baseline val_bpb mean/variance
3. **Step 2:** Implement PassiveHooks (Condition C) + finalize sidecar weight formula
4. **Step 2b:** Run 3-5 Condition B then C runs → sanity check pipeline
5. **Step 3:** Implement ActiveHooks (Condition D) + predict/recommend endpoints
6. **Step 4:** Full Protocol 1 (N=5 pilot first, then N=20 if signal found)
7. **Step 5:** Protocol 3 (contradictory regime) — only if Protocol 1 shows signal
8. **Step 6:** BSM Transformer — only after Protocol 1 analysis complete
9. **Step 7:** Full analysis + statistical significance tests

## Expected Result Pattern

| Condition | Expected |
|-----------|----------|
| A | Slow, noisy, many repeated mistakes |
| B | Slightly better due to observability |
| C | Meaningfully better due to contradiction-aware memory |
| D | Best, due to better next-step choice + prediction |

## Future Work / Separate Branch (Not on Critical Path)

### BSM Transformer (`experiment/bsm-transformer`)
A separate research question: can belief states be embedded directly into the transformer architecture?

- New model `models/bsm_gpt.py` implementing TrainableModel protocol
- Registered as `"bsm"` in model registry
- Run under Condition D infrastructure: `AUTORESEARCH_MODEL=bsm AUTORESEARCH_CONDITION=D uv run train.py`
- **Only pursue after Protocol 1 analysis is complete and the hooks experiment has clear results**

This is a different paper claim ("belief-augmented architectures") than the core claim ("belief memory improves experiment selection"). Mixing them weakens both.

## v3 Patch Log (2026-03-16)

Applied 5 design patches based on two rounds of review:

| Patch | Section | What Changed |
|-------|---------|-------------|
| 1. Counterbalancing | New: "Condition Order Control" | Latin-square rotation, isolation rules, mandatory written rationale |
| 2. Config space | New: "Config Search Space" | Bounds, scales, normalization, distance metric, meaningful-step thresholds |
| 3. Pilot criteria | New: "Pilot Design and Promotion Criteria" | Explicit goals (variance, effect size, feasibility, stability), 4-point promotion gate |
| 4. Typed hooks API | Updated: "mnemebrain_hooks.py" | Raw dicts → `RunConfig`, `RunResults`, `PreRunContext`, `PredictionResult`, `RecommendationResult` dataclasses |
| 5. Level 3 beliefs | Updated: "Belief Schema Level 3" | Contextualized claims now resolve BOTH → conditional TRUE/FALSE, demonstrating revision mechanism |
| 6. Results directory | New: "Results Directory" | Structured `results/` dir with per-run JSONs, summaries, comparisons, and `results_analyzer.py` |
| 7. NEITHER terminology | Updated: all Belnap references | `NONE` → `NEITHER` (standard Belnap term) |
| 8. Sheet isolation | Updated: "Condition Order Control" | Cooldown rule → auditable decision-sheet isolation per condition |
| 9. Pilot gate tightened | Updated: "Pilot Promotion Criteria" | 1 metric → 2-of-3 metrics required for promotion |
| 10. Condition A clarity | Updated: "Condition A" | Disambiguated: stdout-only during session, JSON written post-run for analysis only |
| 11. `rationale_tag` | Updated: `RunConfig` | First-class field for wasted-run exception audit |
| 12. `source_run_ids` | Updated: `PredictionResult`, `RecommendationResult` | Traceability for Condition D outputs |

Additional refinements:
- Wasted-run exception for deliberate ablation probes (documented hypothesis + meaningful parameter change)
- D* positioned as pre-registered supplemental condition, not part of core A/B/C/D ladder

**Status: Design frozen. Ready for Step 1 implementation.**
