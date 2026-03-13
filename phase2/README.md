# Phase 2: Agent with Optimization Tools

**Date**: March 11, 2026
**Objective**: Test whether an AI agent can strategically use Bayesian and Genetic optimization tools to outperform pure optimization methods or pure agent experimentation.

---

## Setup

### Fair Baseline

The current `train.py` uses **Karpathy's original hyperparameters** from the H100 config, with only memory-critical adjustments for RTX 2060 compatibility:

**Original hyperparameters (unchanged)**:
- `EMBEDDING_LR = 0.6`
- `UNEMBEDDING_LR = 0.004`
- `MATRIX_LR = 0.04`
- `SCALAR_LR = 0.5`
- `WEIGHT_DECAY = 0.2`
- `WARMDOWN_RATIO = 0.5`

**Memory adjustments only** (required for 6GB VRAM):
- `DEPTH = 4` (vs 8 original)
- `DEVICE_BATCH_SIZE = 4` (vs 128 original)
- `TOTAL_BATCH_SIZE = 2^16` (vs 2^19 original)
- `WINDOW_PATTERN = "L"` (vs "SSSL" original)

This ensures the agent does NOT benefit from Phase 1 discoveries (best config: val_bpb=1.371).

### Phase 1 Results (Hidden)

All Phase 1 results are archived in `.phase1_archive/` (hidden from agent):
- Agent method: 15 trials, best 1.421
- Bayesian (TPE): 10 trials, best 1.371 ⭐
- Genetic (CMA-ES): 10 trials, best 1.426

### New Tools Available

Two optimization skills are available to the agent:

#### `/bayesian-optimize`
Run Bayesian (TPE) optimization with Optuna
- Sample-efficient probabilistic search
- Good for limited trial budgets
- High exploration, finds novel configurations

#### `/genetic-optimize`
Run Genetic (CMA-ES) optimization with Optuna
- Evolutionary population-based search
- Adapts search distribution over time
- Good for smooth objective landscapes

---

## Research Questions

1. **Strategic Tool Use**: Will the agent learn to use optimization tools strategically vs. manual experimentation?

2. **Hybrid Performance**: Can agent + tools beat:
   - Pure agent (best: 1.421)
   - Pure Bayesian (best: 1.371)
   - Pure Genetic (best: 1.426)

3. **Meta-Learning**: Will the agent develop insights about:
   - When to use Bayesian vs Genetic vs manual trials?
   - How many optimization trials to run?
   - How to interpret and build on optimization results?

4. **Efficiency**: Can the agent reach Phase 1's best performance (1.371) faster by using tools?

---

## Expected Agent Behavior

### Possible Strategies

**Strategy 1: Tool-First**
- Run Bayesian optimization early to map promising regions
- Manually refine best configs found
- Use Genetic for final polishing

**Strategy 2: Manual-First**
- Start with manual experiments to build intuition
- Use tools when stuck or to validate hypotheses
- Iterate between manual and automated search

**Strategy 3: Hybrid Parallel**
- Run optimization in background while manually experimenting
- Combine insights from both approaches
- Test hybrid configurations

**Strategy 4: Tool-Only**
- Delegate all exploration to optimization tools
- Agent acts as orchestrator/interpreter
- Focus on analysis rather than manual trials

### Success Metrics

- **Peak performance**: Best val_bpb achieved
- **Efficiency**: Trials needed to reach Phase 1 best (1.371)
- **Reliability**: Success rate and consistency
- **Insight generation**: Quality of conclusions drawn

---

## Execution Plan

1. **Launch agent** with `program.md` instructions
2. **Agent discovers** optimization skills are available
3. **Agent decides** when/how to use tools vs manual experimentation
4. **Monitor** agent's strategy and decision-making
5. **Compare** Phase 2 results vs Phase 1 methods

---

## Files

- `program.md` - Agent instructions (updated with tool info)
- `.claude/skills/bayesian-optimize.md` - Bayesian optimization skill
- `.claude/skills/genetic-optimize.md` - Genetic optimization skill
- `run_optuna.py` - Optuna runner (unchanged from Phase 1)
- `train_wrapper.py` - Training wrapper (unchanged from Phase 1)
- `results.tsv` - Agent's experiment log (will be created)

---

## Important Notes

- Agent has NO access to `.phase1_archive/` (hidden directory)
- Current `train.py` baseline: expected val_bpb ~1.45 (unoptimized)
- Phase 1 best (1.371) is the target to beat
- Each trial takes ~5-10 minutes
- Agent can run indefinitely (as per `program.md`)

---

## Hypothesis

**Null**: Agent + tools performs same as pure methods (no synergy)
**Alternative**: Agent + tools outperforms pure methods by combining:
- Agent's creative intuition and pattern recognition
- Bayesian's efficient exploration of hyperparameter space
- Genetic's adaptive refinement of promising regions

**Prediction**: Agent will likely:
1. Run Bayesian early to identify promising regions (10-20 trials)
2. Manually test edge cases and validate findings
3. Use Genetic for fine-tuning best configurations
4. Achieve better performance than Phase 1 within 30-40 total trials
