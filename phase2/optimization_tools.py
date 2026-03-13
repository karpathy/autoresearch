"""
Phase 2: Optimization tools for agent use
Provides Bayesian and Genetic optimizers as callable tools
"""

import json
import time
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler, CmaEsSampler

from train_wrapper import run_training_with_params


def bayesian_optimize(
    parameters_to_optimize,
    n_trials=5,
    output_dir="tool_runs/bayesian",
    **fixed_params
):
    """
    Bayesian optimization tool for agent use.

    Args:
        parameters_to_optimize: dict mapping param names to (min, max) ranges
            Example: {'embedding_lr': (0.1, 1.0), 'matrix_lr': (0.01, 0.1)}
        n_trials: number of optimization trials to run
        output_dir: where to save results
        **fixed_params: parameters to keep fixed (not optimize)

    Returns:
        dict: {
            'best_params': {...},
            'best_value': float,
            'all_results': [...],
            'tool_summary': {...}
        }
    """

    print(f"\n{'='*80}")
    print(f"🔧 BAYESIAN OPTIMIZATION TOOL")
    print(f"   Optimizing: {list(parameters_to_optimize.keys())}")
    print(f"   Trials: {n_trials}")
    print(f"   Fixed params: {list(fixed_params.keys())}")
    print(f"{'='*80}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        # Suggest parameters
        suggested = {}
        for param_name, (min_val, max_val) in parameters_to_optimize.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                suggested[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif param_name.endswith('_lr'):
                suggested[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                suggested[param_name] = trial.suggest_float(param_name, min_val, max_val)

        # Combine with fixed params
        all_params = {**fixed_params, **suggested}

        # Run training
        result = run_training_with_params(
            **all_params,
            output_dir=output_dir / f"trial_{trial.number}",
            trial_id=trial.number,
        )

        if result['status'] != 'success':
            raise optuna.TrialPruned()

        return result['val_bpb']

    # Run optimization
    sampler = TPESampler(seed=42, n_startup_trials=min(2, n_trials // 2))
    study = optuna.create_study(
        sampler=sampler,
        direction='minimize',
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start_time

    # Collect results
    all_results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_results.append({
                'trial': trial.number,
                'params': trial.params,
                'value': trial.value,
            })

    tool_result = {
        'best_params': study.best_params if study.best_trial else {},
        'best_value': study.best_value if study.best_trial else float('inf'),
        'all_results': all_results,
        'tool_summary': {
            'method': 'bayesian_tpe',
            'n_trials': n_trials,
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'elapsed_time_s': elapsed,
            'parameters_optimized': list(parameters_to_optimize.keys()),
            'fixed_parameters': list(fixed_params.keys()),
        }
    }

    # Save results
    with open(output_dir / "tool_results.json", 'w') as f:
        json.dump(tool_result, f, indent=2)

    print(f"\n✅ Bayesian optimization complete!")
    print(f"   Best val_bpb: {tool_result['best_value']:.6f}")
    print(f"   Best params: {tool_result['best_params']}")
    print(f"   Time: {elapsed:.1f}s\n")

    return tool_result


def genetic_optimize(
    parameters_to_optimize,
    n_trials=5,
    output_dir="tool_runs/genetic",
    **fixed_params
):
    """
    Genetic algorithm optimization tool for agent use.

    Args:
        parameters_to_optimize: dict mapping param names to (min, max) ranges
        n_trials: number of optimization trials (generations)
        output_dir: where to save results
        **fixed_params: parameters to keep fixed

    Returns:
        dict: {
            'best_params': {...},
            'best_value': float,
            'all_results': [...],
            'tool_summary': {...}
        }
    """

    print(f"\n{'='*80}")
    print(f"🧬 GENETIC OPTIMIZATION TOOL")
    print(f"   Optimizing: {list(parameters_to_optimize.keys())}")
    print(f"   Generations: {n_trials}")
    print(f"   Fixed params: {list(fixed_params.keys())}")
    print(f"{'='*80}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        # Suggest parameters (same as Bayesian)
        suggested = {}
        for param_name, (min_val, max_val) in parameters_to_optimize.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                suggested[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif param_name.endswith('_lr'):
                suggested[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                suggested[param_name] = trial.suggest_float(param_name, min_val, max_val)

        all_params = {**fixed_params, **suggested}

        result = run_training_with_params(
            **all_params,
            output_dir=output_dir / f"trial_{trial.number}",
            trial_id=trial.number,
        )

        if result['status'] != 'success':
            raise optuna.TrialPruned()

        return result['val_bpb']

    # Run optimization with CMA-ES
    sampler = CmaEsSampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        direction='minimize',
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - start_time

    # Collect results
    all_results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_results.append({
                'trial': trial.number,
                'params': trial.params,
                'value': trial.value,
            })

    tool_result = {
        'best_params': study.best_params if study.best_trial else {},
        'best_value': study.best_value if study.best_trial else float('inf'),
        'all_results': all_results,
        'tool_summary': {
            'method': 'genetic_cmaes',
            'n_trials': n_trials,
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'elapsed_time_s': elapsed,
            'parameters_optimized': list(parameters_to_optimize.keys()),
            'fixed_parameters': list(fixed_params.keys()),
        }
    }

    # Save results
    with open(output_dir / "tool_results.json", 'w') as f:
        json.dump(tool_result, f, indent=2)

    print(f"\n✅ Genetic optimization complete!")
    print(f"   Best val_bpb: {tool_result['best_value']:.6f}")
    print(f"   Best params: {tool_result['best_params']}")
    print(f"   Time: {elapsed:.1f}s\n")

    return tool_result


if __name__ == "__main__":
    # Example usage for testing
    print("Testing optimization tools...\n")

    # Test Bayesian
    result = bayesian_optimize(
        parameters_to_optimize={
            'embedding_lr': (0.3, 0.8),
            'matrix_lr': (0.02, 0.06),
        },
        n_trials=3,
        output_dir="test_bayesian",
        depth=4,
        device_batch_size=4,
        total_batch_size=2**16,
        weight_decay=0.2,
        warmdown_ratio=0.5,
        window_pattern='L',
    )

    print(f"\nBayesian test result: {result['best_value']:.6f}")
