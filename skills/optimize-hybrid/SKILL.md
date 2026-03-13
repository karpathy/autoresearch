---
description: Hybrid optimization with auto-strategy selection for mixed variables, neural architecture search, and adaptive algorithm selection
---

When the user has mixed parameter types, does neural architecture search, or is unsure which algorithm to use:

## What This Skill Does

Runs hybrid optimization that automatically selects the best algorithm based on search space analysis. Supports mixed continuous/integer/categorical parameters and provides adaptive strategies.

## When to Use

Use this skill when the user:
- Has mixed continuous + integer + categorical parameters
- Does neural architecture search (NAS) or AutoML
- Has complex heterogeneous search spaces
- **Is unsure which algorithm to use** (strategy='auto' analyzes and picks)
- Needs best-in-class mixed-variable optimization (CatCMA)

Recommend this as the "smart default" when user is uncertain.

Don't use when:
- Pure continuous only → Use CMA-ES (faster)
- Pure discrete only → Use Bayesian
- Single parameter type → Use specialized algorithm

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Optional, for plots
pip install optunahub   # Optional, enables CatCMA (best for mixed)
```

## How to Run

### Auto-select strategy (recommended):
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-hybrid/optimize-hybrid.py --strategy auto --n_trials 200
```

### Sequential multi-phase:
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-hybrid/optimize-hybrid.py --strategy sequential --n_trials 500
```

### Force CatCMA (best for mixed variables):
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-hybrid/optimize-hybrid.py --strategy catcma --n_trials 300
```

## Key Parameters

- `--strategy`: 'auto' (recommended), 'catcma', 'sequential', 'tpe_only', 'cmaes_only'
- `--n_trials`: Number of trials (200-500 typical)
- `--output_dir`: Where to save results

## Strategy Explanation

### 'auto' (Recommended)
Analyzes search space and picks best algorithm:
```
All continuous → CMA-ES
Mixed + optunahub installed → CatCMA
Otherwise → TPE (Bayesian)
```

### 'catcma'
Best-in-class for mixed variables (requires optunahub):
- 1.4x faster than TPE on mixed-variable benchmarks
- Handles continuous/integer/categorical jointly
- Research-backed (GECCO 2025)

### 'sequential'
Multi-phase strategy:
- Phase 1 (20%): QMC exploration
- Phase 2 (50%): TPE exploitation
- Phase 3 (30%): CMA-ES refinement (if continuous)

### 'tpe_only' / 'cmaes_only'
Force specific algorithm (use when you know what you want).

## When to Recommend This Skill

If user describes:
- "Mixed parameter types" → Use this with strategy='auto'
- "Neural architecture search" → Use this with strategy='catcma'
- "Not sure which optimizer" → Use this with strategy='auto'
- "Large trial budget (500+)" → Suggest strategy='sequential'

## Search Space Analysis

The script analyzes the user's search space:
```
Analyzing search space...
  Continuous: 2, Integer: 3, Categorical: 2
  Selected strategy: catcma
```

Explain to user what was detected and why that strategy was chosen.

## Example Use Case: Neural Architecture Search

When user does NAS, guide them:
```python
search_space = {
    'n_layers': (2, 10),              # integer - architecture
    'layer_type': ['conv', 'dense'],  # categorical - architecture
    'hidden_size': (64, 512),         # integer - architecture
    'dropout': (0.0, 0.5),            # continuous - regularization
    'learning_rate': (1e-4, 1e-2),    # continuous - training
    'optimizer': ['adam', 'sgd'],     # categorical - training
}
```

This is perfect for hybrid with strategy='auto' or 'catcma'.

## Output

Results saved to `hybrid_output/`:
- Which strategy was selected and why
- Optimization results
- If sequential: results from each phase

## Performance Tips

- Install optunahub for CatCMA (best mixed-variable performance)
- Use strategy='auto' when uncertain (good default)
- Use strategy='sequential' for budgets >500 trials
- Monitor which strategy was selected and verify it makes sense

## After Optimization

1. Show which strategy was used
2. Explain why it was selected (based on search space)
3. Show results
4. If sequential, note which phase found best result
5. Run `/autoresearch-optimization:analyze-optimization`

## Fallback Behavior

If CatCMA requested but optunahub not installed:
```
⚠️ CatCMA requires optunahub. Falling back to TPE.
Install with: pip install optunahub
```

Guide user to install if they want CatCMA benefits.
