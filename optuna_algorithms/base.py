"""
Base optimizer class with common functionality for all algorithms
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Callable, Optional, Tuple
from datetime import datetime

import optuna
import pandas as pd


@dataclass
class OptimizationResult:
    """Standard result format for all optimizers"""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_complete: int
    n_failed: int
    elapsed_time_s: float
    algorithm_name: str
    algorithm_config: Dict[str, Any]
    study_name: str
    all_trials: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)

    def save(self, output_dir: Path):
        """Save results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "optimization_result.json", 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Also save as DataFrame for easy analysis
        if self.all_trials:
            df = pd.DataFrame(self.all_trials)
            df.to_csv(output_dir / "all_trials.csv", index=False)

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"{'='*80}",
            f"OPTIMIZATION SUMMARY: {self.algorithm_name}",
            f"{'='*80}",
            f"Study: {self.study_name}",
            f"Trials: {self.n_complete}/{self.n_trials} complete ({self.n_failed} failed)",
            f"Time: {self.elapsed_time_s:.1f}s ({self.elapsed_time_s/60:.1f} min)",
            f"Best value: {self.best_value:.6f}",
            f"Best parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"{'='*80}")
        return "\n".join(lines)


class BaseOptimizer:
    """
    Base class for all optimization algorithms.
    Provides common interface and utilities.
    """

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, Tuple],
        direction: str = 'minimize',
        output_dir: str = 'optimization_output',
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            objective_func: Function to optimize, takes trial and returns float
            search_space: Dict mapping parameter names to (min, max) tuples
            direction: 'minimize' or 'maximize'
            output_dir: Directory to save results
            study_name: Name for the Optuna study
            storage: Optuna storage URL (e.g., 'sqlite:///study.db')
            **kwargs: Algorithm-specific parameters
        """
        self.objective_func = objective_func
        self.search_space = search_space
        self.direction = direction
        self.output_dir = Path(output_dir)
        self.study_name = study_name or f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.kwargs = kwargs

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # To be set by subclasses
        self.sampler = None
        self.algorithm_name = "Base"

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create algorithm-specific sampler. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_sampler()")

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters from search space using Optuna's suggest methods.
        Automatically handles int, float, categorical, and log-scale parameters.
        """
        params = {}
        for param_name, bounds in self.search_space.items():
            if isinstance(bounds, (list, tuple)):
                if len(bounds) == 2:
                    min_val, max_val = bounds
                    # Determine type and suggest appropriately
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    elif param_name.endswith('_lr') or param_name.startswith('lr_'):
                        # Learning rates use log scale by default
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                else:
                    # Categorical
                    params[param_name] = trial.suggest_categorical(param_name, bounds)
            else:
                raise ValueError(f"Invalid bounds format for {param_name}: {bounds}")

        return params

    def wrapped_objective(self, trial: optuna.Trial) -> float:
        """
        Wrapper around user's objective function.
        Handles parameter suggestion and error logging.
        """
        params = self.suggest_params(trial)

        try:
            value = self.objective_func(trial, params)

            # Log trial info
            trial_info = {
                'trial_number': trial.number,
                'params': params,
                'value': value,
                'state': 'COMPLETE',
                'timestamp': datetime.now().isoformat(),
            }

            # Append to JSONL log
            with open(self.output_dir / "trials.jsonl", 'a') as f:
                f.write(json.dumps(trial_info) + '\n')

            return value

        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")

            # Log failure
            trial_info = {
                'trial_number': trial.number,
                'params': params,
                'state': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }

            with open(self.output_dir / "trials.jsonl", 'a') as f:
                f.write(json.dumps(trial_info) + '\n')

            raise optuna.TrialPruned()

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        show_progress: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Time budget in seconds (optional)
            show_progress: Show progress bar
            callbacks: List of callback functions

        Returns:
            OptimizationResult object with all trial data
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting {self.algorithm_name} Optimization")
        print(f"{'='*80}")
        print(f"Study: {self.study_name}")
        print(f"Direction: {self.direction}")
        print(f"Trials: {n_trials}")
        print(f"Search space: {list(self.search_space.keys())}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")

        # Create sampler
        self.sampler = self.create_sampler()

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            sampler=self.sampler,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=True,
        )

        # Run optimization
        start_time = time.time()

        study.optimize(
            self.wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            callbacks=callbacks,
        )

        elapsed_time = time.time() - start_time

        # Collect results
        all_trials = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value if trial.value is not None else float('inf'),
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration_s': trial.duration.total_seconds() if trial.duration else None,
            }
            all_trials.append(trial_data)

        result = OptimizationResult(
            best_params=study.best_params if study.best_trial else {},
            best_value=study.best_value if study.best_trial else float('inf'),
            n_trials=len(study.trials),
            n_complete=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            n_failed=len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            elapsed_time_s=elapsed_time,
            algorithm_name=self.algorithm_name,
            algorithm_config=self.get_algorithm_config(),
            study_name=self.study_name,
            all_trials=all_trials,
            metadata={
                'timeout': timeout,
                'n_trials_requested': n_trials,
                'search_space': self.search_space,
                'direction': self.direction,
            }
        )

        # Save results
        result.save(self.output_dir)

        # Print summary
        print(result.summary())

        return result

    def get_algorithm_config(self) -> Dict[str, Any]:
        """Return algorithm-specific configuration. Override in subclasses."""
        return {'sampler': self.sampler.__class__.__name__}
