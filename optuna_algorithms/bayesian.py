"""
Bayesian Optimization using TPE and GP samplers

**When to use:**
- General-purpose hyperparameter tuning (default choice)
- Limited trial budget (100-1000 trials)
- Expensive objective evaluations
- Mixed continuous/discrete/categorical parameters
- High-dimensional spaces (TPE handles 100+ dims well)
- When you don't know the problem structure

**Strengths:**
- 2.5x faster convergence than random search
- Sample-efficient (learns from all previous trials)
- Handles all parameter types
- Excellent parallelization support
- Low computational overhead

**Limitations:**
- Less effective than CMA-ES for purely continuous problems with correlations
- GP version doesn't scale well beyond ~500 trials (O(n³))
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler, GPSampler
from .base import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using Tree-structured Parzen Estimator (TPE)
    or Gaussian Process (GP) models.
    """

    def __init__(
        self,
        objective_func,
        search_space: Dict,
        direction: str = 'minimize',
        output_dir: str = 'bayesian_output',
        variant: str = 'tpe',  # 'tpe' or 'gp'
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        multivariate: bool = False,
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            variant: 'tpe' or 'gp'
                - tpe: Fast, scalable, default choice
                - gp: Better for <500 trials, supports constraints
            n_startup_trials: Random exploration trials before using model
            n_ei_candidates: Number of candidates for Expected Improvement (TPE only)
            multivariate: Use multivariate TPE to model parameter interactions (TPE only)
            seed: Random seed for reproducibility
        """
        super().__init__(objective_func, search_space, direction, output_dir, **kwargs)

        self.algorithm_name = f"Bayesian-{variant.upper()}"
        self.variant = variant
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.multivariate = multivariate
        self.seed = seed

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create TPE or GP sampler based on variant"""

        if self.variant == 'tpe':
            return TPESampler(
                n_startup_trials=self.n_startup_trials,
                n_ei_candidates=self.n_ei_candidates,
                multivariate=self.multivariate,
                seed=self.seed,
            )

        elif self.variant == 'gp':
            # GP sampler (BoTorch integration)
            # Note: Requires torch and botorch installed
            try:
                return GPSampler(
                    n_startup_trials=self.n_startup_trials,
                    seed=self.seed,
                )
            except ImportError:
                print("⚠️  GPSampler requires torch and botorch. Falling back to TPE.")
                return TPESampler(
                    n_startup_trials=self.n_startup_trials,
                    seed=self.seed,
                )

        else:
            raise ValueError(f"Unknown variant: {self.variant}. Use 'tpe' or 'gp'.")

    def get_algorithm_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'Bayesian',
            'variant': self.variant,
            'n_startup_trials': self.n_startup_trials,
            'n_ei_candidates': self.n_ei_candidates,
            'multivariate': self.multivariate,
            'seed': self.seed,
        }


# Convenience function for quick usage
def optimize_bayesian(
    objective_func,
    search_space: Dict,
    n_trials: int = 50,
    variant: str = 'tpe',
    output_dir: str = 'bayesian_output',
    **sampler_kwargs
):
    """
    Quick Bayesian optimization function.

    Example:
        def objective(trial, params):
            x = params['x']
            y = params['y']
            return (x - 2)**2 + (y + 3)**2

        result = optimize_bayesian(
            objective,
            search_space={'x': (-10, 10), 'y': (-10, 10)},
            n_trials=100,
            variant='tpe',
        )

        print(f"Best params: {result.best_params}")
        print(f"Best value: {result.best_value}")
    """
    optimizer = BayesianOptimizer(
        objective_func=objective_func,
        search_space=search_space,
        output_dir=output_dir,
        variant=variant,
        **sampler_kwargs
    )

    return optimizer.optimize(n_trials=n_trials)


# Decision helper
def should_use_bayesian(
    n_trials: int,
    param_types: str,  # 'mixed', 'continuous', 'discrete'
    has_correlations: bool = False,
    n_objectives: int = 1,
) -> bool:
    """
    Helper function to decide if Bayesian optimization is appropriate.

    Returns:
        True if Bayesian is recommended, False otherwise
    """
    # Multi-objective? Use NSGA instead
    if n_objectives > 1:
        return False

    # Very few trials? Use QMC for exploration
    if n_trials < 20:
        return False

    # Continuous with known correlations? CMA-ES might be better
    if param_types == 'continuous' and has_correlations and n_trials > 200:
        return False

    # Otherwise, Bayesian (TPE) is a great default choice
    return True


if __name__ == "__main__":
    # Example usage
    print("Example: Bayesian Optimization with TPE\n")

    def sphere_function(trial, params):
        """Simple test function: minimize sum of squares"""
        x = params['x']
        y = params['y']
        z = params['z']
        return x**2 + y**2 + z**2

    result = optimize_bayesian(
        objective_func=sphere_function,
        search_space={
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
            'z': (-5.0, 5.0),
        },
        n_trials=30,
        variant='tpe',
        n_startup_trials=5,
        output_dir='test_bayesian',
    )

    print(f"\n✅ Best value: {result.best_value:.6f}")
    print(f"Best params: {result.best_params}")
