#!/usr/bin/env python3
"""
NSGA Multi-Objective Optimization Skill

Usage:
    python optimize-nsga.py --n_objectives 2 --n_trials 200
    python optimize-nsga.py --n_objectives 5 --variant nsga3 --n_trials 500

When to use:
    ✅ Multiple competing objectives, need Pareto frontier, trade-off analysis
    ❌ Single objective (use Bayesian/CMA-ES), very small budget
"""

import sys
import argparse
from pathlib import Path

try:
    from optuna_algorithms import optimize_nsga, OptimizationAnalyzer
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstall dependencies: pip install optuna pandas numpy")
    sys.exit(1)


def run_nsga_optimization(
    objective_func=None,
    search_space=None,
    n_objectives=2,
    directions=None,
    n_trials=200,
    variant='auto',
    output_dir='nsga_output',
):
    """
    Run NSGA-II/III multi-objective optimization.

    Args:
        objective_func: Function(trial, params) -> tuple of objective values
        search_space: Dict of parameter bounds
        n_objectives: Number of objectives to optimize
        directions: List of 'minimize' or 'maximize' for each objective
        n_trials: Number of trials
        variant: 'nsga2', 'nsga3', or 'auto'
        output_dir: Where to save results

    Returns:
        OptimizationResult with Pareto frontier in metadata
    """

    # Demo objective
    if objective_func is None:
        print("Using demo: minimize distance from (0,0) and (2,2)\n")
        def objective_func(trial, params):
            """Bi-objective: minimize distance from two points"""
            x = params.get('x', 0)
            y = params.get('y', 0)
            obj1 = x**2 + y**2  # Distance from origin
            obj2 = (x-2)**2 + (y-2)**2  # Distance from (2,2)
            return obj1, obj2

    # Demo search space
    if search_space is None:
        search_space = {
            'x': (-1.0, 3.0),
            'y': (-1.0, 3.0),
        }

    # Default directions
    if directions is None:
        directions = ['minimize'] * n_objectives

    print(f"\n{'='*80}")
    print(f"NSGA-{variant.upper()} MULTI-OBJECTIVE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Objectives: {n_objectives}")
    print(f"Directions: {directions}")
    print(f"Trials: {n_trials}")
    print(f"Variant: {variant}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run optimization
    result = optimize_nsga(
        objective_func=objective_func,
        search_space=search_space,
        n_objectives=n_objectives,
        directions=directions,
        n_trials=n_trials,
        variant=variant,
        output_dir=output_dir,
    )

    print(f"\n{'='*80}")
    print("✅ NSGA OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Pareto frontier: {result.metadata['pareto_front_size']} solutions")
    print(f"\nTop 5 Pareto solutions:")
    for i, point in enumerate(result.metadata['pareto_front'][:5], 1):
        print(f"\n{i}. Objectives: {point['values']}")
        print(f"   Parameters: {point['params']}")
    print(f"\nAll solutions saved to: {output_dir}/")
    print(f"{'='*80}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="NSGA Multi-Objective Optimization Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bi-objective (accuracy vs latency)
  python optimize-nsga.py --n_objectives 2 --n_trials 200

  # Many-objective (4+ objectives, use NSGA-III)
  python optimize-nsga.py --n_objectives 5 --variant nsga3 --n_trials 500

  # Custom directions (maximize first, minimize second)
  python optimize-nsga.py --n_objectives 2 --directions maximize minimize

Note:
  - NSGA-II: Best for 2-3 objectives
  - NSGA-III: Best for 4+ objectives
  - Auto: Automatically selects based on n_objectives
        """
    )

    parser.add_argument(
        '--n_objectives',
        type=int,
        default=2,
        help='Number of objectives to optimize'
    )

    parser.add_argument(
        '--directions',
        nargs='+',
        choices=['minimize', 'maximize'],
        default=None,
        help='Direction for each objective (default: all minimize)'
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=200,
        help='Number of trials'
    )

    parser.add_argument(
        '--variant',
        choices=['nsga2', 'nsga3', 'auto'],
        default='auto',
        help='NSGA variant (auto selects based on n_objectives)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='nsga_output',
        help='Output directory'
    )

    args = parser.parse_args()

    # Validate directions
    if args.directions and len(args.directions) != args.n_objectives:
        print(f"❌ Error: {args.n_objectives} objectives need {args.n_objectives} directions")
        sys.exit(1)

    result = run_nsga_optimization(
        n_objectives=args.n_objectives,
        directions=args.directions,
        n_trials=args.n_trials,
        variant=args.variant,
        output_dir=args.output_dir,
    )

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
