#!/usr/bin/env python3
"""
Bayesian Optimization Skill

Usage:
    python optimize-bayesian.py --n_trials 50
    python optimize-bayesian.py --ml --n_trials 20 --multivariate

When to use:
    ✅ General ML tuning, mixed parameter types, 100-1000 trial budget
    ❌ Pure continuous with correlations (use CMA-ES), multi-objective (use NSGA)
"""

import sys
import argparse
from pathlib import Path

# Import optimization framework
try:
    from optuna_algorithms import optimize_bayesian, OptimizationAnalyzer
    import optuna
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nMake sure you're in the autoresearch directory and optuna is installed:")
    print("  cd /path/to/autoresearch")
    print("  pip install optuna pandas numpy")
    sys.exit(1)


def run_bayesian_optimization(
    objective_func=None,
    search_space=None,
    n_trials=50,
    variant='tpe',
    multivariate=False,
    output_dir='bayesian_output',
    analyze=True,
):
    """
    Run Bayesian optimization with TPE or GP.

    Args:
        objective_func: Function(trial, params) -> float to minimize
        search_space: Dict of {param_name: (min, max)} or [choices]
        n_trials: Number of optimization trials
        variant: 'tpe' (fast, scalable) or 'gp' (better for <500 trials)
        multivariate: Enable parameter interaction modeling
        output_dir: Where to save results
        analyze: Generate plots and analysis

    Returns:
        OptimizationResult with best_params, best_value, all_trials, etc.
    """

    # Demo objective if none provided
    if objective_func is None:
        print("Using demo objective: minimize sum of squares\n")
        def objective_func(trial, params):
            return sum(v**2 for v in params.values() if isinstance(v, (int, float)))

    # Demo search space if none provided
    if search_space is None:
        search_space = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
        }

    print(f"\n{'='*80}")
    print("BAYESIAN OPTIMIZATION (TPE/GP)")
    print(f"{'='*80}")
    print(f"Variant: {variant.upper()}")
    print(f"Trials: {n_trials}")
    print(f"Multivariate: {multivariate}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run optimization
    result = optimize_bayesian(
        objective_func=objective_func,
        search_space=search_space,
        n_trials=n_trials,
        variant=variant,
        multivariate=multivariate,
        n_startup_trials=min(10, max(2, n_trials // 5)),
        output_dir=output_dir,
    )

    print(f"\n{'='*80}")
    print("✅ OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Best parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"{'='*80}\n")

    # Analysis
    if analyze:
        try:
            analyzer = OptimizationAnalyzer(output_dir)
            plot_dir = Path(output_dir) / 'plots'
            analyzer.plot_all(output_dir=str(plot_dir), show=False)
            analyzer.export_best_config(Path(output_dir) / 'best_config.json')
            print(f"📊 Analysis complete:")
            print(f"   - Plots: {plot_dir}/")
            print(f"   - Best config: {output_dir}/best_config.json\n")
        except ImportError:
            print("⚠️  matplotlib not installed, skipping plots")
            print("   Install with: pip install matplotlib\n")
        except Exception as e:
            print(f"⚠️  Analysis failed: {e}\n")

    return result


def run_ml_optimization(n_trials=20, multivariate=True, search_space=None):
    """
    Run Bayesian optimization for ML training.
    Integrates with train_wrapper.py if available.
    """
    try:
        from train_wrapper import run_training_with_params

        def objective(trial, params):
            """Objective function for ML training"""
            result = run_training_with_params(
                **params,
                output_dir=Path("trials") / f"trial_{trial.number}",
                trial_id=trial.number,
            )

            # Prune failed trials
            if result['status'] != 'success':
                print(f"Trial {trial.number} failed, pruning...")
                raise optuna.TrialPruned()

            return result['val_bpb']

        # Default ML search space
        if search_space is None:
            search_space = {
                'embedding_lr': (0.1, 1.0),
                'matrix_lr': (0.01, 0.15),
                'weight_decay': (0.0, 0.5),
                'warmdown_ratio': (0.0, 0.5),
                'depth': (2, 12),
            }

        print("🤖 ML Training Mode: Using train_wrapper integration\n")

        return run_bayesian_optimization(
            objective_func=objective,
            search_space=search_space,
            n_trials=n_trials,
            multivariate=multivariate,
            output_dir='ml_bayesian_optimization',
        )

    except ImportError:
        print("⚠️  train_wrapper.py not found. Running demo optimization instead.\n")
        return run_bayesian_optimization(n_trials=n_trials, multivariate=multivariate)


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple demo
  python optimize-bayesian.py --n_trials 50

  # ML training mode
  python optimize-bayesian.py --ml --n_trials 20 --multivariate

  # Use GP variant
  python optimize-bayesian.py --variant gp --n_trials 100

  # Custom output directory
  python optimize-bayesian.py --output_dir my_optimization --n_trials 30
        """
    )

    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )

    parser.add_argument(
        '--variant',
        choices=['tpe', 'gp'],
        default='tpe',
        help='Sampler: tpe (fast) or gp (better for <500 trials)'
    )

    parser.add_argument(
        '--multivariate',
        action='store_true',
        help='Enable parameter interaction modeling (slower but more accurate)'
    )

    parser.add_argument(
        '--ml',
        action='store_true',
        help='Use ML training integration with train_wrapper.py'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: bayesian_output or ml_bayesian_optimization)'
    )

    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip analysis and plotting'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = 'ml_bayesian_optimization' if args.ml else 'bayesian_output'

    # Run optimization
    if args.ml:
        result = run_ml_optimization(
            n_trials=args.n_trials,
            multivariate=args.multivariate,
        )
    else:
        result = run_bayesian_optimization(
            n_trials=args.n_trials,
            variant=args.variant,
            multivariate=args.multivariate,
            output_dir=args.output_dir,
            analyze=not args.no_analyze,
        )

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
