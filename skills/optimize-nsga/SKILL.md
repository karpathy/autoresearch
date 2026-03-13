---
description: Multi-objective optimization with NSGA-II/III for finding Pareto frontiers and trade-off analysis
---

When the user has multiple competing objectives or needs trade-off analysis:

## What This Skill Does

Runs NSGA-II or NSGA-III multi-objective optimization to find Pareto-optimal frontiers. Returns a set of solutions representing optimal trade-offs between competing objectives.

## When to Use

Use this skill when the user:
- Has multiple competing objectives (accuracy vs latency, cost vs performance)
- Needs entire Pareto frontier (not single "best" solution)
- Wants trade-off analysis
- Has 2-10 objectives to optimize simultaneously

Don't use when:
- Single objective → Use Bayesian or CMA-ES
- Very small budget (<100 trials)

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Optional, for plots
```

## How to Run

### Bi-objective optimization:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-nsga.py --n_objectives 2 --n_trials 200
```

### Many-objective (4+):
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-nsga.py --n_objectives 5 --variant nsga3 --n_trials 500
```

### Custom directions (e.g., maximize accuracy, minimize latency):
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-nsga.py \
  --n_objectives 2 \
  --directions maximize minimize \
  --n_trials 200
```

## Key Parameters

- `--n_objectives`: Number of objectives (2-10)
- `--directions`: Space-separated list of 'minimize' or 'maximize' for each objective
- `--variant`: 'nsga2' (2-3 obj), 'nsga3' (4+ obj), or 'auto' (recommended)
- `--n_trials`: Number of trials (recommend 200+ for good Pareto coverage)

## Understanding Multi-Objective Results

**Critical:** Explain to the user that multi-objective optimization returns a **Pareto frontier**, not a single "best":

- Each solution on the Pareto front is equally "optimal"
- Improving one objective means worsening another
- User must choose based on their preferences/constraints

Example interpretation:
```
Pareto frontier: 15 solutions
Solution 1: accuracy=0.95, latency=150ms
Solution 2: accuracy=0.92, latency=80ms
Solution 3: accuracy=0.88, latency=50ms

All are optimal - user picks based on latency constraint.
```

## How to Help User Choose from Pareto Front

Guide the user to filter or select:

1. **By constraint**: "Which solutions have latency < 100ms?"
2. **By compromise**: "Find the balanced middle solution"
3. **By preference**: "I prioritize accuracy over speed"
4. **Visualize**: Show trade-off plot if possible

The script outputs the Pareto front in `metadata['pareto_front']`.

## Variant Selection

- **NSGA-II**: Use for 2-3 objectives (crowding distance for diversity)
- **NSGA-III**: Use for 4+ objectives (reference points for diversity)
- **Auto** (default): Automatically selects based on n_objectives

## Output

Results saved to `nsga_output/`:
- `optimization_result.json` includes full Pareto frontier
- `metadata['pareto_front']` contains all Pareto-optimal solutions
- `metadata['pareto_front_size']` shows number of solutions

## Performance Tips

- Population size auto-computed as ~50×(n_objectives-1)
- Need ≥20×population_size trials for good convergence
- Use NSGA-III for 4+ objectives (better diversity)
- Normalize objectives to similar scales if possible

## After Optimization

1. Show user the Pareto frontier size and solutions
2. Help filter by constraints if needed
3. Visualize trade-offs if requested
4. Run `/autoresearch-optimization:analyze-optimization` for detailed analysis
