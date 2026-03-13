#!/usr/bin/env python3
"""
CMA-ES Optimization Skill

Usage:
    python optimize-cmaes.py --n_trials 500
    python optimize-cmaes.py --restart ipop --n_trials 1000

When to use:
    ✅ ALL parameters continuous, correlations expected, noisy objectives
    ❌ Any discrete/categorical params (use Bayesian), multi-objective (use NSGA)
"""

import sys
import argparse
from pathlib import Path

try:
    from optuna_algorithms import optimize_cmaes, OptimizationAnalyzer
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstall dependencies: pip install optuna pandas numpy")
    sys.exit(1)


def run_cmaes_optimization(
    objective_func=None,
    search_space=None,
    n_trials=500,
    restart_strategy=None,
    output_dir='cmaes_output',
    analyze=True,
):
    """
    Run CMA-ES optimization for continuous parameters.

    Args:
        objective_func: Function(trial, params) -> float
        search_space: Dict of {param_name: (min_float, max_float)}
        n_trials: Number of trials (CMA-ES needs 500+ typically)
        restart_strategy: None, 'ipop', or 'bipop'
        output_dir: Where to save results
        analyze: Generate analysis plots

    Returns:
        OptimizationResult
    """

    # Demo objective
    if objective_func is None:
        print("Using demo objective: Rosenbrock function\n")
        def objective_func(trial, params):
            """Rosenbrock: has strong parameter correlation"""
            x = params.get('x', 0)
            y = params.get('y', 0)
            return (1 - x)**2 + 100*(y - x**2)**2

    # Demo search space
    if search_space is None:
        search_space = {
            'x': (-2.0, 2.0),
            'y': (-1.0, 3.0),
        }

    # Validate: CMA-ES requires ALL continuous parameters
    for param_name, bounds in search_space.items():
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            print(f"❌ Error: {param_name} must have (min, max) bounds")
            print("   CMA-ES only works with continuous parameters!")
            sys.exit(1)

    print(f"\n{'='*80}")
    print("CMA-ES OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Trials: {n_trials}")
    print(f"Restart strategy: {restart_strategy or 'None'}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run optimization
    result = optimize_cmaes(
        objective_func=objective_func,
        search_space=search_space,
        n_trials=n_trials,
        restart_strategy=restart_strategy,
        output_dir=output_dir,
    )

    print(f"\n{'='*80}")
    print("✅ CMA-ES COMPLETE")
    print(f"{'='*80}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nResults: {output_dir}/")
    print(f"{'='*80}\n")

    # Analysis
    if analyze:
        try:
            analyzer = OptimizationAnalyzer(output_dir)
            plot_dir = Path(output_dir) / 'plots'
            analyzer.plot_all(output_dir=str(plot_dir), show=False)
            print(f"📊 Plots: {plot_dir}/\n")
        except Exception as e:
            print(f"⚠️  Analysis skipped: {e}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CMA-ES Optimization Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard CMA-ES
  python optimize-cmaes.py --n_trials 500

  # With restart strategy for multimodal problems
  python optimize-cmaes.py --restart ipop --n_trials 1000

  # BIPOP restart (alternates population sizes)
  python optimize-cmaes.py --restart bipop --n_trials 2000

Restart strategies:
  - None: Standard CMA-ES
  - ipop: Increases population on restart (multimodal problems)
  - bipop: Alternates large/small populations (complex landscapes)
        """
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=500,
        help='Number of trials (CMA-ES needs 500+ for good results)'
    )

    parser.add_argument(
        '--restart',
        choices=['ipop', 'bipop'],
        default=None,
        help='Restart strategy for multimodal problems'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='cmaes_output',
        help='Output directory'
    )

    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip analysis'
    )

    args = parser.parse_args()

    result = run_cmaes_optimization(
        n_trials=args.n_trials,
        restart_strategy=args.restart,
        output_dir=args.output_dir,
        analyze=not args.no_analyze,
    )

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
