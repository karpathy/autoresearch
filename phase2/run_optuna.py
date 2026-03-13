"""
Phase 1: Optuna hyperparameter optimization (Bayesian TPE vs Genetic CMA-ES)
Sequential execution for single-GPU systems.
"""

import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler, CmaEsSampler
import pandas as pd

from train_wrapper import run_training_with_params


def objective(trial, method_name, output_dir, trial_counter):
    """Optuna objective function."""

    # Define search space (narrowed based on manual experimentation findings)
    params = {
        'depth': 4,  # Fixed: clearly optimal from manual trials
        'device_batch_size': 4,  # Fixed: baseline value works well
        'total_batch_size': 2**16,  # Fixed: 65536 is optimal
        'embedding_lr': trial.suggest_float('embedding_lr', 0.55, 0.75),  # Narrowed around 0.65
        'matrix_lr': trial.suggest_float('matrix_lr', 0.065, 0.095),  # Narrowed around 0.07-0.08
        'weight_decay': trial.suggest_float('weight_decay', 0.1, 0.3),  # Narrowed around 0.2
        'warmdown_ratio': trial.suggest_float('warmdown_ratio', 0.2, 0.4),  # Narrowed around 0.3
        'window_pattern': 'L',  # Keep simple for RTX 2060
    }

    # Run training
    print(f"\n{'='*80}")
    print(f"Method: {method_name} | Trial {trial_counter['count']}/{trial_counter['total']}")
    print(f"Optuna Trial #{trial.number}")
    print(f"Params: {json.dumps({k: v for k, v in params.items() if k != 'window_pattern'}, indent=2)}")
    print(f"{'='*80}\n")

    start_time = time.time()
    results = run_training_with_params(
        **params,
        output_dir=output_dir / f"trial_{trial.number:03d}",
        trial_id=trial.number,
    )
    wall_time = time.time() - start_time

    # Log to JSON Lines
    log_entry = {
        'trial_number': trial_counter['count'],
        'optuna_trial': trial.number,
        'method': method_name,
        'timestamp': datetime.now().isoformat(),
        'params': params,
        'results': results,
        'wall_time_s': wall_time,
    }

    with open(output_dir / "results.jsonl", 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    trial_counter['count'] += 1

    # Set user attributes for later analysis
    trial.set_user_attr('peak_vram_mb', results.get('peak_vram_mb', 0))
    trial.set_user_attr('wall_time_s', wall_time)
    trial.set_user_attr('training_time_s', results.get('training_time_s', 0))
    trial.set_user_attr('status', results.get('status', 'unknown'))

    # Handle failures
    if results.get('status') != 'success':
        print(f"❌ Trial failed: {results.get('error', 'Unknown error')}")
        raise optuna.TrialPruned()

    val_bpb = results['val_bpb']
    print(f"\n✅ Trial complete: val_bpb = {val_bpb:.6f}\n")

    return val_bpb


def run_optimization(method, n_trials, output_dir):
    """Run optimization with specified method."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# Phase 1: {method.upper()} Optimization")
    print(f"# Trials: {n_trials}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*80}\n")

    # Create sampler
    if method == 'bayesian':
        sampler = TPESampler(seed=42, n_startup_trials=3)
        study_name = 'bayesian_tpe'
    elif method == 'genetic':
        sampler = CmaEsSampler(seed=42)
        study_name = 'genetic_cmaes'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create study
    storage_path = f"sqlite:///{output_dir}/study.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        sampler=sampler,
        direction='minimize',
        load_if_exists=True,
    )

    # Run optimization
    trial_counter = {'count': 1, 'total': n_trials}
    start_time = time.time()

    study.optimize(
        lambda trial: objective(trial, method, output_dir, trial_counter),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    elapsed_time = time.time() - start_time

    # Save summary
    summary = {
        'method': method,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'best_trial_number': study.best_trial.number if study.best_trial else None,
        'best_value': study.best_value if study.best_trial else float('inf'),
        'best_params': study.best_params if study.best_trial else {},
        'total_time_s': elapsed_time,
        'avg_time_per_trial_s': elapsed_time / len(study.trials) if study.trials else 0,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'#'*80}")
    print(f"# {method.upper()} Optimization Complete!")
    print(f"#")
    print(f"# Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print(f"# Completed trials: {summary['n_complete']}/{n_trials}")
    print(f"# Failed trials: {summary['n_failed']}")
    print(f"# Best val_bpb: {summary['best_value']:.6f}")
    print(f"# Best params:")
    for k, v in summary['best_params'].items():
        print(f"#   {k}: {v}")
    print(f"{'#'*80}\n")

    return study, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Optuna Optimization")
    parser.add_argument(
        'method',
        choices=['bayesian', 'genetic'],
        help='Optimization method: bayesian (TPE), genetic (CMA-ES)'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=10,
        help='Number of trials'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: experiments/phase1/{method})'
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"experiments/phase1/{args.method}"

    run_optimization(args.method, args.n_trials, args.output_dir)
