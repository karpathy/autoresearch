---
description: Analyze and visualize optimization results with convergence plots, parameter importance, and comprehensive reports
---

When the user completes an optimization run and wants to understand the results, or wants to compare multiple optimization runs:

## What This Skill Does

Analyzes optimization results from any algorithm and generates:
- Summary statistics (best value, parameters, trial counts)
- Convergence plots (best value over trials)
- Parameter importance (which parameters matter most)
- Parameter distribution plots
- Best config export (JSON)
- Markdown reports

Also supports comparing multiple optimization runs side-by-side.

## When to Use

Use this skill when the user:
- Completed an optimization and wants analysis
- Asks "what were the results?"
- Wants to visualize convergence
- Needs to understand which parameters matter
- Wants to compare different algorithms/runs
- Needs to export best configuration

**Always suggest this after any optimization completes.**

## Installation Requirements

```bash
pip install optuna pandas numpy
pip install matplotlib  # Highly recommended for plots
```

## How to Run

### Analyze single run:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/analyze-optimization.py bayesian_output
```

### Compare multiple runs:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/analyze-optimization.py \
  --compare \
  bayesian_output \
  cmaes_output \
  hybrid_output
```

### Custom plot directory:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/analyze-optimization.py \
  bayesian_output \
  --plot_dir my_analysis_plots
```

## What Gets Analyzed

### 1. Summary Statistics
- Total trials, completed, failed, success rate
- Best value and parameters
- Time statistics (total, per trial)
- Value distribution (min, max, mean, median, std)

### 2. Convergence Plot
Shows best value found over trials:
- Identifies when optimization converged
- Shows if more trials would help
- Visualizes exploration vs exploitation

### 3. Parameter Importance
Correlation between parameters and objective:
- Which parameters have strongest effect
- Helps focus future search
- Identifies irrelevant parameters

### 4. Parameter Distributions
Shows explored ranges:
- How parameters were sampled
- Best parameter values (marked)
- Coverage of search space

## Output Files

Analysis creates:
```
{output_dir}/
├── plots/
│   ├── convergence.png
│   ├── parameter_importance.png
│   └── parameter_distributions.png
├── best_config.json
└── optimization_report.md
```

## Interpreting Results for User

### Convergence Analysis
```
If plot plateaus early:
→ "Optimization converged quickly. Parameters found are robust."

If still improving at end:
→ "Still improving. Consider running more trials."

If erratic/noisy:
→ "Noisy objective. Consider increasing trials or using CMA-ES."
```

### Parameter Importance
```
High correlation (>0.7):
→ "This parameter strongly affects performance. Focus tuning here."

Low correlation (<0.3):
→ "This parameter has minimal effect. Could simplify search space."
```

### Distribution Analysis
```
Narrow distribution around best:
→ "Optimizer focused on promising region."

Uniform distribution:
→ "Parameter had similar performance across range."

Best value at boundary:
→ "Consider expanding search space in this direction."
```

## Comparison Mode

When comparing multiple runs, help user understand:
1. Which algorithm converged fastest
2. Which found best final value
3. Which was most sample-efficient
4. Trade-offs (speed vs quality)

The comparison generates:
- Side-by-side convergence plot
- Summary table of all runs
- Saved to comparison output directory

## Common User Questions

### "Did it work?"
Check summary statistics and best value.

### "Which algorithm was better?"
Use comparison mode to compare side-by-side.

### "Which parameters matter?"
Show parameter importance plot.

### "Should I run more trials?"
Check convergence plot - if still improving, yes.

### "Can I use these parameters?"
Export best_config.json for easy reuse.

## Integration with Other Skills

After analysis, you can:
1. Suggest re-running with refined search space
2. Recommend different algorithm based on results
3. Suggest more trials if not converged
4. Export config for production use

## Error Handling

If no matplotlib:
```
⚠️ matplotlib not installed, skipping plots
Install with: pip install matplotlib
```
Still generates JSON exports and reports.

If optimization_result.json missing:
```
❌ No optimization results found in {directory}
Make sure optimization completed successfully.
```

## Output Explanation

Always explain the key findings to the user:
- "Converged after 30 trials with best value of 0.123"
- "Parameter 'learning_rate' has highest importance (correlation: 0.85)"
- "All trials completed successfully (100% success rate)"
- "Optimization took 5.2 minutes (3.1 seconds per trial)"

Then suggest next steps based on results.
