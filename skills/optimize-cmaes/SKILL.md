---
description: CMA-ES evolution strategy for continuous optimization with parameter correlations and noisy objectives
---

When the user requests CMA-ES optimization, evolution strategies, or has purely continuous parameters with correlations:

## What This Skill Does

Runs Covariance Matrix Adaptation Evolution Strategy (CMA-ES) - the gold standard for continuous optimization. Learns parameter correlations and is robust to noisy objectives.

## When to Use

Use this skill when the user:
- Has **ALL** parameters continuous (float only, no integers/categories)
- Suspects parameter correlations or interactions
- Has noisy objective functions
- Has large trial budget (500-10000 trials)
- Works on robotics, control, or continuous policy search

**Critical:** CMA-ES ONLY works with continuous parameters. If any parameter is integer or categorical, recommend Bayesian or Hybrid instead.

Don't use when:
- Any discrete/categorical parameters → Use Bayesian
- Small budget (<200 trials) → Use Bayesian
- Multi-objective → Use NSGA

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Optional, for plots
```

## How to Run

### Standard CMA-ES:
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-cmaes/optimize-cmaes.py --n_trials 500
```

### With restart strategy for multimodal problems:
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-cmaes/optimize-cmaes.py --restart ipop --n_trials 1000
```

### BIPOP variant:
```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-cmaes/optimize-cmaes.py --restart bipop --n_trials 2000
```

## Key Parameters

- `--n_trials`: Number of trials (recommend 500+)
- `--restart`: None (default), 'ipop', or 'bipop'
  - 'ipop': Increases population on restart (multimodal problems)
  - 'bipop': Alternates population sizes (complex landscapes)
- `--output_dir`: Where to save results

## Validation

**Important:** Before running, verify ALL parameters are continuous:

```python
# ❌ This will FAIL - has integer
search_space = {
    'x': (0.0, 1.0),      # OK
    'batch_size': (16, 256)  # ❌ FAIL - integer!
}

# ✅ This works - all continuous
search_space = {
    'x': (0.0, 1.0),
    'y': (-5.0, 5.0),
}
```

If user has mixed types, recommend `/autoresearch-optimization:optimize-hybrid` instead.

## When to Suggest Restart Strategies

- User mentions "stuck in local minimum" → Suggest `--restart ipop`
- Complex/multimodal landscape → Suggest `--restart bipop`
- Unknown landscape → Start without restart, add if needed

## Output

Results saved to `cmaes_output/`:
- Complete optimization results
- Best parameters
- Convergence plots

Follow with `/autoresearch-optimization:analyze-optimization` for detailed analysis.

## Performance Tips

- Need at least 50×sqrt(n_params) trials for good results
- Use restart strategies if optimization plateaus
- CMA-ES excels at learning parameter correlations
- Works best with trial budgets of 500-10000
