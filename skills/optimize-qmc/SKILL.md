---
description: Quasi-Monte Carlo exploration using low-discrepancy sequences for systematic sampling and warm-starting adaptive methods
---

When the user needs initial exploration, design of experiments, or warm-starting for other algorithms:

## What This Skill Does

Runs Quasi-Monte Carlo (QMC) sampling using Sobol or Halton low-discrepancy sequences for systematic exploration. Provides O(1/N) convergence vs O(1/√N) for random sampling.

## When to Use

Use this skill when the user:
- Wants initial exploration (first 50-100 trials before adaptive methods)
- Needs Design of Experiments (DoE) for sensitivity analysis
- Wants to warm-start Bayesian or CMA-ES optimization
- Has massively parallel workloads (perfect parallelization)
- Needs uniform coverage of search space

**Best Practice:** Recommend using QMC followed by adaptive optimization:
- QMC (50 trials) for exploration
- Then Bayesian/CMA-ES (150 trials) for exploitation

Don't use as only optimizer - always combine with adaptive method.

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Optional, for plots
```

## How to Run

### QMC exploration only:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-qmc.py --n_trials 50
```

### QMC → Bayesian hybrid (recommended):
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-qmc.py \
  --n_trials 50 \
  --then bayesian \
  --n_adaptive 150
```

### QMC → CMA-ES (for continuous problems):
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-qmc.py \
  --n_trials 100 \
  --then cmaes \
  --n_adaptive 400
```

### Use Halton for low-dimensional:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-qmc.py --qmc_type halton --n_trials 50
```

## Key Parameters

- `--n_trials`: Number of QMC trials (50-200 typical)
- `--qmc_type`: 'sobol' (d>6, default) or 'halton' (d≤6)
- `--then`: Follow with 'bayesian' or 'cmaes' (recommended)
- `--n_adaptive`: Trials for adaptive phase (if --then specified)

## QMC Type Selection

Auto-select based on dimensionality:
- **Sobol**: Best for d > 6 dimensions (most cases)
- **Halton**: Best for d ≤ 6 dimensions (degrades in high-dim)

The script auto-suggests if user chooses wrong type.

## Recommended Strategies

### Strategy 1: QMC → Bayesian (default)
```bash
--n_trials 50 --then bayesian --n_adaptive 150
```
Best for: General optimization with mixed parameters

### Strategy 2: QMC → CMA-ES
```bash
--n_trials 100 --then cmaes --n_adaptive 400
```
Best for: Pure continuous optimization

### Strategy 3: QMC only (rare)
```bash
--n_trials 100
```
Best for: DoE, sensitivity analysis, initial exploration only

## Why QMC Works Well

Explain to user:
- **Systematic coverage**: Unlike random, QMC uniformly covers space
- **O(1/N) convergence**: Faster than random's O(1/√N)
- **Perfect parallelization**: All trials independent, no coordination
- **Deterministic**: Reproducible results

Visual analogy:
```
Random: ●   ●  ●     ●●  ●   ● (clusters and gaps)
QMC:    ● ● ● ● ● ● ● ● ● ● (uniform grid)
```

## When to Suggest Hybrid Approach

If user says:
- "I want to try optimization" → Suggest QMC→Bayesian
- "Complex/multimodal landscape" → Suggest QMC for exploration first
- "Not sure about parameters" → QMC shows sensitivity
- "Large parallel cluster" → QMC is embarrassingly parallel

## Output

Results saved to `qmc_output/` (or `qmc_adaptive_output/` if hybrid):
- QMC exploration results
- If hybrid: best result from both phases
- Uniform sampling of search space

## After QMC

If running QMC only:
1. Show exploration results
2. Suggest following with adaptive method
3. Recommend `/autoresearch-optimization:optimize-bayesian` or CMA-ES

If running hybrid:
1. Show which phase found best result
2. Note the benefit of exploration + exploitation
3. Run `/autoresearch-optimization:analyze-optimization`
