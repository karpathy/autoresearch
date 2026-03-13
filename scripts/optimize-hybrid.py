#!/usr/bin/env python3
"""
Hybrid Optimization Skill

Usage:
    python optimize-hybrid.py --strategy auto --n_trials 200
    python optimize-hybrid.py --strategy sequential --n_trials 500

When to use:
    ✅ Mixed continuous+integer+categorical, NAS, AutoML, unsure which algorithm
    ❌ Pure continuous (use CMA-ES), single parameter type (use specialized)
"""

import sys
import argparse
from pathlib import Path

try:
    from optuna_algorithms import optimize_hybrid, OptimizationAnalyzer
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstall dependencies: pip install optuna pandas numpy")
    sys.exit(1)


def run_hybrid_optimization(
    objective_func=None,
    search_space=None,
    n_trials=200,
    strategy='auto',
    output_dir='hybrid_output',
    analyze=True,
):
    """
    Run hybrid/adaptive optimization.

    Args:
        objective_func: Function(trial, params) -> float
        search_space: Dict with mixed parameter types
        n_trials: Number of trials
        strategy: 'auto', 'catcma', 'sequential', 'tpe_only'
        output_dir: Output directory
        analyze: Generate analysis

    Returns:
        OptimizationResult
    """

    # Demo objective
    if objective_func is None:
        print("Using demo objective with mixed parameter types\n")
        def objective_func(trial, params):
            """Mixed variables: continuous, integer, categorical"""
            x = params.get('x', 0)
            y = params.get('y', 0)
            n = params.get('n', 5)
            mode = params.get('mode', 'square')

            base = (x - 1)**2 + (y + 1)**2 + (n - 5)**2

            if mode == 'square':
                return base
            elif mode == 'sqrt':
                return base**0.5
            else:  # 'log'
                return (base + 1)**0.1

    # Demo search space with mixed types
    if search_space is None:
        search_space = {
            'x': (-5.0, 5.0),  # continuous
            'y': (-5.0, 5.0),  # continuous
            'n': (1, 10),      # integer
            'mode': ['square', 'sqrt', 'log'],  # categorical
        }

    print(f"\n{'='*80}")
    print(f"HYBRID OPTIMIZATION (strategy={strategy})")
    print(f"{'='*80}")
    print(f"Trials: {n_trials}")
    print(f"Strategy: {strategy}")
    print(f"Search space: {list(search_space.keys())}")

    # Analyze search space
    n_cont = sum(1 for b in search_space.values()
                 if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], float))
    n_int = sum(1 for b in search_space.values()
                if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], int))
    n_cat = sum(1 for b in search_space.values() if isinstance(b, list))

    print(f"  Continuous: {n_cont}, Integer: {n_int}, Categorical: {n_cat}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run optimization
    result = optimize_hybrid(
        objective_func=objective_func,
        search_space=search_space,
        n_trials=n_trials,
        strategy=strategy,
        output_dir=output_dir,
    )

    print(f"\n{'='*80}")
    print("✅ HYBRID OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
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
        description="Hybrid Optimization Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select strategy based on search space
  python optimize-hybrid.py --strategy auto --n_trials 200

  # Sequential multi-phase (QMC → TPE → CMA-ES)
  python optimize-hybrid.py --strategy sequential --n_trials 500

  # Use CatCMA for mixed variables (needs optunahub)
  python optimize-hybrid.py --strategy catcma --n_trials 300

Strategies:
  - auto: Automatically select based on parameter types (recommended)
  - catcma: Best for mixed variables (needs: pip install optunahub)
  - sequential: Multi-phase QMC → TPE → CMA-ES
  - tpe_only: Fallback to TPE (handles all types)

Auto-selection logic:
  - All continuous → CMA-ES
  - Mixed types + optunahub → CatCMA
  - Otherwise → TPE
        """
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=200,
        help='Number of trials'
    )

    parser.add_argument(
        '--strategy',
        choices=['auto', 'catcma', 'sequential', 'tpe_only', 'cmaes_only'],
        default='auto',
        help='Optimization strategy'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='hybrid_output',
        help='Output directory'
    )

    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip analysis'
    )

    args = parser.parse_args()

    result = run_hybrid_optimization(
        n_trials=args.n_trials,
        strategy=args.strategy,
        output_dir=args.output_dir,
        analyze=not args.no_analyze,
    )

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
