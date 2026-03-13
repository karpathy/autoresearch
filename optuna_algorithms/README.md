# Optuna Algorithms: Modular Optimization Framework

A comprehensive, modular optimization framework providing 5 core algorithms for maximum problem coverage.

## 🎯 Quick Start

```python
from optuna_algorithms import BayesianOptimizer

def objective(trial, params):
    x = params['x']
    y = params['y']
    return (x - 2)**2 + (y + 3)**2

optimizer = BayesianOptimizer(
    objective_func=objective,
    search_space={'x': (-10, 10), 'y': (-10, 10)},
    output_dir='my_optimization',
)

result = optimizer.optimize(n_trials=50)
print(f"Best: {result.best_params} = {result.best_value:.6f}")
```

## 📚 Table of Contents

- [Overview](#overview)
- [5 Core Algorithms](#5-core-algorithms)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Algorithm Selection Guide](#algorithm-selection-guide)
- [API Reference](#api-reference)
- [Analysis & Visualization](#analysis--visualization)
- [Configuration Files](#configuration-files)

## Overview

This framework provides **5 optimization algorithms** that collectively solve the most diverse set of optimization problems:

| Algorithm | Best For | Key Strength |
|-----------|----------|-------------|
| **Bayesian (TPE/GP)** | General ML tuning | Sample-efficient, handles all types |
| **CMA-ES** | Continuous optimization | Learns correlations, robust to noise |
| **NSGA-II/III** | Multi-objective | Produces Pareto frontier |
| **QMC** | Exploration/DoE | Perfect parallelization, systematic |
| **Hybrid (CatCMA)** | Mixed variables | Handles heterogeneous search spaces |

## 5 Core Algorithms

### 1. Bayesian Optimization (TPE/GP)

**When to use:**
- General hyperparameter tuning (default choice)
- Limited budget (100-1000 trials)
- Mixed continuous/discrete/categorical parameters

**Strengths:**
- 2.5x faster than random search
- Handles 100+ dimensions
- Excellent parallelization

```python
from optuna_algorithms import optimize_bayesian

result = optimize_bayesian(
    objective_func=my_objective,
    search_space={
        'learning_rate': (0.0001, 0.1),
        'batch_size': (16, 256),
        'optimizer': ['adam', 'sgd'],
    },
    n_trials=100,
    variant='tpe',
)
```

### 2. Evolution Strategies (CMA-ES)

**When to use:**
- Purely continuous optimization
- Parameter correlations/interactions
- Noisy objectives
- Large budgets (1000+ trials)

**Strengths:**
- Gold standard for continuous optimization
- Learns covariance matrix
- Massively parallel

```python
from optuna_algorithms import optimize_cmaes

result = optimize_cmaes(
    objective_func=my_objective,
    search_space={
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0),
        'z': (-5.0, 5.0),
    },
    n_trials=500,
    restart_strategy='ipop',
)
```

### 3. Multi-Objective (NSGA-II/III)

**When to use:**
- Multiple competing objectives
- Trade-off analysis
- Pareto frontier needed

**Strengths:**
- Produces complete Pareto front
- NSGA-III for 4+ objectives
- Population-based parallelism

```python
from optuna_algorithms import optimize_nsga

def multi_obj(trial, params):
    accuracy = train_model(params)
    latency = measure_latency(params)
    return accuracy, latency  # Both to minimize

result = optimize_nsga(
    objective_func=multi_obj,
    search_space={'lr': (0.001, 0.1), 'size': (64, 512)},
    n_objectives=2,
    directions=['maximize', 'minimize'],
    n_trials=200,
)

# Access Pareto frontier
pareto_front = result.metadata['pareto_front']
```

### 4. Quasi-Monte Carlo (QMC)

**When to use:**
- Initial exploration (first 50-100 trials)
- Design of Experiments
- Warm-start for adaptive methods

**Strengths:**
- O(1/N) convergence vs O(1/√N) random
- Perfect parallelization
- Zero computational overhead

```python
from optuna_algorithms import optimize_qmc, qmc_then_adaptive

# QMC exploration
result = optimize_qmc(
    objective_func=my_objective,
    search_space={'x': (-10, 10), 'y': (-10, 10)},
    n_trials=50,
    qmc_type='sobol',
)

# Or: QMC → Bayesian hybrid
result = qmc_then_adaptive(
    objective_func=my_objective,
    search_space={'x': (-10, 10), 'y': (-10, 10)},
    n_qmc_trials=50,
    n_adaptive_trials=150,
    adaptive_method='bayesian',
)
```

### 5. Hybrid Optimization

**When to use:**
- Mixed continuous/integer/categorical
- Neural architecture search
- AutoML pipelines
- Unsure which algorithm to use

**Strengths:**
- CatCMA for mixed variables
- Sequential strategies (QMC → TPE → CMA-ES)
- Auto-selection based on search space

```python
from optuna_algorithms import optimize_hybrid

result = optimize_hybrid(
    objective_func=my_objective,
    search_space={
        'learning_rate': (0.0001, 0.1),  # continuous
        'batch_size': (16, 256),          # integer
        'optimizer': ['adam', 'sgd'],     # categorical
    },
    n_trials=200,
    strategy='auto',  # Automatically selects best approach
)
```

## Installation

```bash
# Basic installation
pip install optuna pandas numpy

# Optional: For visualization
pip install matplotlib

# Optional: For GP sampler and CatCMA
pip install torch botorch optunahub

# Install as package
cd optuna_algorithms
pip install -e .
```

## Usage Examples

### Example 1: Simple Function Optimization

```python
from optuna_algorithms import BayesianOptimizer

def sphere(trial, params):
    return sum(v**2 for v in params.values())

optimizer = BayesianOptimizer(
    objective_func=sphere,
    search_space={
        'x': (-5, 5),
        'y': (-5, 5),
        'z': (-5, 5),
    },
)

result = optimizer.optimize(n_trials=50)
```

### Example 2: ML Training with Prune on Failure

```python
from optuna_algorithms import BayesianOptimizer
import optuna

def train_objective(trial, params):
    result = train_model(**params)

    if result['status'] != 'success':
        raise optuna.TrialPruned()

    return result['val_loss']

optimizer = BayesianOptimizer(
    objective_func=train_objective,
    search_space={
        'embedding_lr': (0.1, 1.0),
        'matrix_lr': (0.01, 0.1),
        'weight_decay': (0.0, 0.5),
    },
)

result = optimizer.optimize(n_trials=100)
```

### Example 3: Multi-Objective with Analysis

```python
from optuna_algorithms import optimize_nsga, OptimizationAnalyzer

def multi_obj(trial, params):
    accuracy = train_model(params)
    size = count_parameters(params)
    return accuracy, size  # maximize, minimize

result = optimize_nsga(
    objective_func=multi_obj,
    search_space={'depth': (2, 12), 'width': (64, 512)},
    n_objectives=2,
    directions=['maximize', 'minimize'],
    n_trials=100,
)

# Analyze
analyzer = OptimizationAnalyzer('nsga_output')
analyzer.print_summary()
analyzer.plot_all(output_dir='plots')
```

## Algorithm Selection Guide

### Decision Flowchart

```
Do you have multiple competing objectives?
├─ YES → NSGA-II/III (Multi-objective)
└─ NO ↓

Do you have fewer than 50 trials?
├─ YES → QMC (Exploration) → then switch to adaptive
└─ NO ↓

Are all parameters continuous (no integers/categoricals)?
├─ YES ↓
│   Do parameters likely correlate?
│   ├─ YES → CMA-ES (Evolution Strategies)
│   └─ NO → Bayesian (TPE)
└─ NO ↓
    Mixed continuous + discrete + categorical?
    ├─ YES → Hybrid (CatCMA or auto)
    └─ NO → Bayesian (TPE) [handles all types]
```

### Coverage Matrix

| Problem Type | Algorithms (ranked) |
|--------------|-------------------|
| **Continuous** | CMA-ES > Bayesian > Hybrid |
| **Discrete/Categorical** | Bayesian > Hybrid > QMC |
| **Mixed Variables** | Hybrid (CatCMA) > Bayesian |
| **Multi-objective** | NSGA-II/III |
| **High-dimensional** | CMA-ES > Bayesian > QMC |
| **Noisy objectives** | CMA-ES > Bayesian |
| **Small budget (<50)** | QMC > Bayesian |
| **Large budget (1000+)** | CMA-ES > Bayesian |

## API Reference

### Base Optimizer

All optimizers inherit from `BaseOptimizer`:

```python
optimizer = AlgorithmOptimizer(
    objective_func: Callable,      # (trial, params) -> float
    search_space: Dict,            # Parameter definitions
    direction: str = 'minimize',   # or 'maximize'
    output_dir: str = '...',       # Where to save results
    **kwargs                       # Algorithm-specific
)

result = optimizer.optimize(
    n_trials: int = 50,
    timeout: float = None,         # Time limit in seconds
    show_progress: bool = True,
    callbacks: List = None,
)
```

### OptimizationResult

```python
result.best_params      # Dict of best parameters
result.best_value       # Best objective value
result.n_trials         # Total trials run
result.n_complete       # Successfully completed
result.elapsed_time_s   # Total time in seconds
result.all_trials       # List of all trial data

result.save(output_dir) # Save to JSON
result.summary()        # Print summary
```

## Analysis & Visualization

### OptimizationAnalyzer

```python
from optuna_algorithms import OptimizationAnalyzer

analyzer = OptimizationAnalyzer('optimization_output')

# Print summary
analyzer.print_summary()

# Generate plots
analyzer.plot_convergence()
analyzer.plot_parameter_importances()
analyzer.plot_parameter_distributions()
analyzer.plot_all(output_dir='plots')

# Export
analyzer.export_best_config('best_config.json')
analyzer.to_markdown_report('report.md')

# Compare optimizations
from optuna_algorithms.analysis import compare_optimizations
compare_optimizations(
    ['bayesian_output', 'cmaes_output', 'nsga_output'],
    output_dir='comparison'
)
```

## Configuration Files

Each algorithm has a detailed YAML configuration template in `configs/`:

```
optuna_algorithms/configs/
├── bayesian_config.yaml
├── cmaes_config.yaml
├── nsga_config.yaml
├── qmc_config.yaml
└── hybrid_config.yaml
```

These files document:
- When to use the algorithm
- Strengths and limitations
- All configuration parameters
- Example search spaces
- Performance tips

## Project Structure

```
optuna_algorithms/
├── __init__.py
├── base.py                    # BaseOptimizer class
├── bayesian.py                # TPE/GP Bayesian optimization
├── evolution_strategies.py    # CMA-ES
├── multi_objective.py         # NSGA-II/III
├── qmc.py                     # Quasi-Monte Carlo
├── hybrid.py                  # CatCMA + adaptive strategies
├── analysis.py                # Result analysis & visualization
├── configs/                   # Configuration templates
│   ├── bayesian_config.yaml
│   ├── cmaes_config.yaml
│   ├── nsga_config.yaml
│   ├── qmc_config.yaml
│   └── hybrid_config.yaml
└── README.md                  # This file
```

## Claude Skills

Use Claude skills for quick access:

- `/optimize-bayesian` - Bayesian optimization
- `/optimize-cmaes` - CMA-ES optimization
- `/optimize-nsga` - Multi-objective optimization
- `/optimize-qmc` - QMC exploration
- `/optimize-hybrid` - Hybrid/adaptive optimization
- `/analyze-optimization` - Analyze results

## Performance Tips

1. **Start with Bayesian (TPE)** - Good default for most problems
2. **QMC warmstart** - Use QMC for first 20-30% of trials
3. **Multivariate TPE** - Enable if parameters interact
4. **CMA-ES for continuous** - Best for pure continuous optimization
5. **Parallel evaluation** - All algorithms support parallelization
6. **Prune early** - Raise `optuna.TrialPruned()` for failed trials
7. **Log scale** - Use for parameters spanning orders of magnitude
8. **Normalize objectives** - Scale objectives to similar ranges

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE: Bergstra et al. 2011](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [CMA-ES: Hansen & Ostermeier 2001](https://direct.mit.edu/evco/article-abstract/9/2/159/908/Completely-Derandomized-Self-Adaptation-in)
- [NSGA-II: Deb et al. 2002](https://ieeexplore.ieee.org/document/996017)
- [NSGA-III: Deb & Jain 2014](https://ieeexplore.ieee.org/document/6600851)
- [OptunaHub](https://hub.optuna.org/)

## License

MIT License - See parent project for details.

## Contributing

This framework is part of the autoresearch project. See main README for contribution guidelines.
