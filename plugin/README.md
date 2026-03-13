# Autoresearch Optimization Plugin

Complete Claude Code plugin for hyperparameter optimization with 6 algorithms covering all optimization scenarios.

## Skills

- **optimize-bayesian** - TPE/GP Bayesian optimization for mixed parameters
- **optimize-cmaes** - CMA-ES evolution strategy for continuous optimization
- **optimize-nsga** - Multi-objective NSGA-II/III for Pareto frontiers
- **optimize-qmc** - QMC exploration with adaptive follow-up
- **optimize-hybrid** - Auto-strategy selection for uncertain cases
- **analyze-optimization** - Results analysis with plots and reports

## Installation

### From GitHub

```bash
# Add the marketplace
/plugin marketplace add lejinvarghese/autoresearch

# Install the plugin
/plugin install autoresearch-optimization@autoresearch-plugins
```

### Local Installation (for development)

```bash
# From the repository root
/plugin install ./plugin --scope local

# Or use --plugin-dir flag
claude --plugin-dir ./plugin
```

## Usage

After installation, skills are available as:

```bash
/autoresearch-optimization:optimize-bayesian
/autoresearch-optimization:optimize-cmaes
/autoresearch-optimization:optimize-nsga
/autoresearch-optimization:optimize-qmc
/autoresearch-optimization:optimize-hybrid
/autoresearch-optimization:analyze-optimization
```

## Requirements

```bash
pip install optuna>=3.0.0 pandas>=1.0.0 numpy>=1.19.0
pip install matplotlib  # Optional, for plots
pip install optunahub  # Optional, enables CatCMA for hybrid
```

## Example

```bash
# Run Bayesian optimization
/autoresearch-optimization:optimize-bayesian

# Or directly call the script
python ${CLAUDE_PLUGIN_ROOT}/skills/optimize-bayesian/optimize-bayesian.py --n_trials 50
```

## Framework

Built on a modular `optuna_algorithms/` framework with:
- `BaseOptimizer` - Unified interface for all algorithms
- Algorithm modules - Bayesian, CMA-ES, NSGA, QMC, Hybrid
- Analysis tools - Convergence plots, parameter importance, reports
- YAML configs - Detailed documentation for each algorithm

## License

MIT
