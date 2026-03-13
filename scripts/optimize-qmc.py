#!/usr/bin/env python3
"""
QMC (Quasi-Monte Carlo) Exploration Skill

Usage:
    python optimize-qmc.py --n_trials 50
    python optimize-qmc.py --n_trials 50 --then bayesian --n_adaptive 150

When to use:
    ✅ Initial exploration, DoE, warm-start adaptive methods, perfect parallelization
    ❌ As only optimizer (combine with adaptive), very large budgets
"""

import sys
import argparse

try:
    from optuna_algorithms import optimize_qmc, qmc_then_adaptive, OptimizationAnalyzer
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstall dependencies: pip install optuna pandas numpy")
    sys.exit(1)


def run_qmc_exploration(
    objective_func=None,
    search_space=None,
    n_trials=50,
    qmc_type='sobol',
    output_dir='qmc_output',
    analyze=True,
):
    """
    Run QMC exploration with low-discrepancy sequences.

    Args:
        objective_func: Function(trial, params) -> float
        search_space: Dict of parameter bounds
        n_trials: Number of trials (50-200 typical for exploration)
        qmc_type: 'sobol' (d>6) or 'halton' (d<=6)
        output_dir: Where to save results
        analyze: Generate analysis

    Returns:
        OptimizationResult
    """

    # Demo objective
    if objective_func is None:
        print("Using demo objective: Rastrigin function (highly multimodal)\n")
        def objective_func(trial, params):
            """Rastrigin: needs good exploration"""
            import math
            A = 10
            values = [v for v in params.values() if isinstance(v, (int, float))]
            n = len(values)
            return A * n + sum(v**2 - A * math.cos(2 * math.pi * v) for v in values)

    # Demo search space
    if search_space is None:
        search_space = {
            'x1': (-5.12, 5.12),
            'x2': (-5.12, 5.12),
            'x3': (-5.12, 5.12),
        }

    n_dims = len(search_space)
    if n_dims > 6 and qmc_type == 'halton':
        print(f"⚠️  Warning: Halton degrades for d>6. Using Sobol instead.\n")
        qmc_type = 'sobol'

    print(f"\n{'='*80}")
    print("QMC EXPLORATION")
    print(f"{'='*80}")
    print(f"Type: {qmc_type.upper()} sequence")
    print(f"Dimensions: {n_dims}")
    print(f"Trials: {n_trials}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run QMC
    result = optimize_qmc(
        objective_func=objective_func,
        search_space=search_space,
        n_trials=n_trials,
        qmc_type=qmc_type,
        output_dir=output_dir,
    )

    print(f"\n{'='*80}")
    print("✅ QMC EXPLORATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best value found: {result.best_value:.6f}")
    print(f"Best parameters: {result.best_params}")
    print(f"\nResults: {output_dir}/")
    print(f"\n💡 Tip: Follow with adaptive optimization for better results")
    print(f"{'='*80}\n")

    # Analysis
    if analyze:
        try:
            from pathlib import Path
            analyzer = OptimizationAnalyzer(output_dir)
            plot_dir = Path(output_dir) / 'plots'
            analyzer.plot_all(output_dir=str(plot_dir), show=False)
            print(f"📊 Plots: {plot_dir}/\n")
        except Exception as e:
            print(f"⚠️  Analysis skipped: {e}\n")

    return result


def run_hybrid_qmc_adaptive(
    objective_func=None,
    search_space=None,
    n_qmc_trials=50,
    n_adaptive_trials=150,
    adaptive_method='bayesian',
    output_dir='qmc_adaptive_output',
):
    """
    Hybrid: QMC exploration → Adaptive optimization.

    Args:
        objective_func: Objective function
        search_space: Parameter bounds
        n_qmc_trials: QMC exploration trials
        n_adaptive_trials: Adaptive optimization trials
        adaptive_method: 'bayesian' or 'cmaes'
        output_dir: Output directory

    Returns:
        Best result from both phases
    """

    # Demo objective if none provided
    if objective_func is None:
        def objective_func(trial, params):
            import math
            A = 10
            values = [v for v in params.values() if isinstance(v, (int, float))]
            n = len(values)
            return A * n + sum(v**2 - A * math.cos(2 * math.pi * v) for v in values)

    if search_space is None:
        search_space = {
            'x1': (-5.12, 5.12),
            'x2': (-5.12, 5.12),
            'x3': (-5.12, 5.12),
        }

    print(f"\n{'='*80}")
    print(f"HYBRID: QMC → {adaptive_method.upper()}")
    print(f"{'='*80}")
    print(f"Phase 1: QMC exploration ({n_qmc_trials} trials)")
    print(f"Phase 2: {adaptive_method.upper()} optimization ({n_adaptive_trials} trials)")
    print(f"Total trials: {n_qmc_trials + n_adaptive_trials}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    result = qmc_then_adaptive(
        objective_func=objective_func,
        search_space=search_space,
        n_qmc_trials=n_qmc_trials,
        n_adaptive_trials=n_adaptive_trials,
        adaptive_method=adaptive_method,
        output_dir=output_dir,
    )

    print(f"\n✅ Hybrid strategy complete!")
    print(f"   Best value: {result.best_value:.6f}")
    print(f"   Best params: {result.best_params}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="QMC Exploration Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QMC exploration only
  python optimize-qmc.py --n_trials 50

  # QMC → Bayesian hybrid (recommended)
  python optimize-qmc.py --n_trials 50 --then bayesian --n_adaptive 150

  # QMC → CMA-ES (for continuous problems)
  python optimize-qmc.py --n_trials 100 --then cmaes --n_adaptive 400

  # Use Halton for low-dimensional
  python optimize-qmc.py --qmc_type halton --n_trials 50

QMC Types:
  - sobol: Best for d > 6 dimensions (default)
  - halton: Best for d ≤ 6 dimensions
        """
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of QMC trials'
    )

    parser.add_argument(
        '--qmc_type',
        choices=['sobol', 'halton'],
        default='sobol',
        help='QMC sequence type'
    )

    parser.add_argument(
        '--then',
        choices=['bayesian', 'cmaes'],
        default=None,
        help='Follow with adaptive optimization'
    )

    parser.add_argument(
        '--n_adaptive',
        type=int,
        default=150,
        help='Trials for adaptive phase (if --then specified)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory'
    )

    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip analysis'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'qmc_adaptive_output' if args.then else 'qmc_output'

    if args.then:
        # Hybrid strategy
        result = run_hybrid_qmc_adaptive(
            n_qmc_trials=args.n_trials,
            n_adaptive_trials=args.n_adaptive,
            adaptive_method=args.then,
            output_dir=args.output_dir,
        )
    else:
        # QMC only
        result = run_qmc_exploration(
            n_trials=args.n_trials,
            qmc_type=args.qmc_type,
            output_dir=args.output_dir,
            analyze=not args.no_analyze,
        )

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
