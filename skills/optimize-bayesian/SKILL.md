---
description: Bayesian optimization (TPE/GP) for sample-efficient hyperparameter tuning with mixed parameter types
---

When the user requests Bayesian optimization, hyperparameter tuning, or mentions TPE/GP optimization:

## What This Skill Does

This skill runs Bayesian optimization using Tree-structured Parzen Estimator (TPE) or Gaussian Process (GP) for sample-efficient hyperparameter search. Best for general ML tuning with mixed parameter types and 100-1000 trial budgets.

## When to Use

Use this skill when the user:
- Needs general hyperparameter optimization
- Has mixed continuous/discrete/categorical parameters
- Has limited trial budget (100-1000 trials)
- Wants sample-efficient optimization (2.5x faster than random search)
- Is unsure which optimization algorithm to use (good default)

Don't use when:
- All parameters are continuous with known correlations → Suggest CMA-ES instead
- Multiple competing objectives → Suggest NSGA instead
- Very small budget (<20 trials) → Suggest QMC exploration first

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Optional, for plots
```

## How to Run

### Demo optimization:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-bayesian.py --n_trials 50
```

### With ML training integration:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-bayesian.py --ml --n_trials 20 --multivariate
```

### Custom parameters:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/optimize-bayesian.py \
  --n_trials 100 \
  --variant tpe \
  --multivariate \
  --output_dir my_optimization
```

## Key Parameters

- `--n_trials`: Number of optimization trials (default: 50)
- `--variant`: 'tpe' (fast, default) or 'gp' (better for <500 trials)
- `--multivariate`: Enable parameter interaction modeling
- `--ml`: Use ML training wrapper integration
- `--output_dir`: Where to save results

## Integration with User Code

If the user has a custom objective function, guide them to use the Python API:

```python
from optuna_algorithms import optimize_bayesian

def objective(trial, params):
    # User's evaluation code
    result = evaluate(params)
    return result['loss']

result = optimize_bayesian(
    objective,
    search_space={'param1': (0, 1), 'param2': (10, 100)},
    n_trials=50,
)
```

## Output

Results are saved to the output directory:
- `optimization_result.json` - Complete results
- `all_trials.csv` - Trial data
- `best_config.json` - Best parameters
- `plots/` - Visualization plots

After running, suggest analyzing with `/autoresearch-optimization:analyze-optimization`

## Performance Tips

- Enable `--multivariate` if parameters interact
- Use `--variant gp` for <500 trials with complex landscapes
- Set n_trials to at least 50 for meaningful results
- Follow with analysis to understand results
