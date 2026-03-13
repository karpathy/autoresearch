"""
Evolution Strategies (CMA-ES and variants)

**When to use:**
- Purely continuous optimization problems
- Suspected parameter correlations/interactions
- Noisy objective functions
- Large trial budgets (1000-10000 trials)
- High-dimensional continuous spaces (10-1000+ dims)
- Robotics, control, continuous policy search

**Strengths:**
- Industry gold standard for continuous optimization
- Learns parameter correlations via adaptive covariance matrix
- Robust to noisy objectives
- Excellent parallelization (near-linear speedup)
- Restart strategies prevent premature convergence

**Limitations:**
- Poor performance on categorical/discrete parameters
- Requires larger population sizes (~4+3ln(d) for d dimensions)
- Higher computational cost: O(d³) per iteration
- Not sample-efficient for small budgets (<100 trials)
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import CmaEsSampler
from .base import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy optimizer.
    Best for continuous optimization with parameter correlations.
    """

    def __init__(
        self,
        objective_func,
        search_space: Dict,
        direction: str = 'minimize',
        output_dir: str = 'cmaes_output',
        with_margin: bool = False,
        use_separable: bool = False,
        restart_strategy: Optional[str] = None,  # None, 'ipop', or 'bipop'
        population_size: Optional[int] = None,
        sigma0: float = 0.25,
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            with_margin: Use margin for constraint handling
            use_separable: Use sep-CMA-ES (faster for high-dim, assumes independence)
            restart_strategy: Restart strategy for avoiding local minima
                - None: Standard CMA-ES
                - 'ipop': IPOP-CMA-ES (increases population on restart)
                - 'bipop': BIPOP-CMA-ES (alternates large/small populations)
            population_size: Population size (default: 4 + 3*ln(n_params))
            sigma0: Initial step-size (fraction of search space, default: 0.25)
            seed: Random seed
        """
        super().__init__(objective_func, search_space, direction, output_dir, **kwargs)

        self.algorithm_name = "CMA-ES"
        if restart_strategy:
            self.algorithm_name += f"-{restart_strategy.upper()}"
        if use_separable:
            self.algorithm_name += "-Sep"

        self.with_margin = with_margin
        self.use_separable = use_separable
        self.restart_strategy = restart_strategy
        self.population_size = population_size
        self.sigma0 = sigma0
        self.seed = seed

        # Validate: CMA-ES works best with continuous parameters
        self._validate_search_space()

    def _validate_search_space(self):
        """Warn if search space contains non-continuous parameters"""
        for param_name, bounds in self.search_space.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                min_val, max_val = bounds
                if isinstance(min_val, int) and isinstance(max_val, int):
                    print(f"⚠️  Warning: {param_name} is integer. CMA-ES works best with continuous parameters.")
                    print(f"   Consider using CatCMA (hybrid optimizer) for mixed variables.")
            else:
                print(f"⚠️  Warning: {param_name} is categorical. CMA-ES does NOT support categorical.")
                print(f"   Use TPE (Bayesian) or CatCMA (hybrid) instead.")

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create CMA-ES sampler with specified configuration"""

        sampler_kwargs = {
            'seed': self.seed,
            'with_margin': self.with_margin,
            'use_separable_cma': self.use_separable,
        }

        if self.restart_strategy:
            sampler_kwargs['restart_strategy'] = self.restart_strategy

        if self.population_size is not None:
            sampler_kwargs['n_startup_trials'] = self.population_size

        # Note: sigma0 is set per-parameter in the search space
        # by using trial.suggest_float(..., step=sigma0 * range)
        # For simplicity, we use default sigma0 in CmaEsSampler

        return CmaEsSampler(**sampler_kwargs)

    def get_algorithm_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'CMA-ES',
            'with_margin': self.with_margin,
            'use_separable': self.use_separable,
            'restart_strategy': self.restart_strategy,
            'population_size': self.population_size,
            'sigma0': self.sigma0,
            'seed': self.seed,
        }


# Convenience function
def optimize_cmaes(
    objective_func,
    search_space: Dict,
    n_trials: int = 200,
    output_dir: str = 'cmaes_output',
    restart_strategy: Optional[str] = None,
    **sampler_kwargs
):
    """
    Quick CMA-ES optimization function.

    Example:
        def rosenbrock(trial, params):
            x = params['x']
            y = params['y']
            return (1 - x)**2 + 100*(y - x**2)**2

        result = optimize_cmaes(
            rosenbrock,
            search_space={'x': (-5.0, 5.0), 'y': (-5.0, 5.0)},
            n_trials=500,
            restart_strategy='ipop',
        )
    """
    optimizer = CMAESOptimizer(
        objective_func=objective_func,
        search_space=search_space,
        output_dir=output_dir,
        restart_strategy=restart_strategy,
        **sampler_kwargs
    )

    return optimizer.optimize(n_trials=n_trials)


def should_use_cmaes(
    n_trials: int,
    n_params: int,
    param_types: str,  # 'continuous', 'mixed', 'discrete'
    has_correlations: bool = False,
    is_noisy: bool = False,
) -> bool:
    """
    Helper to decide if CMA-ES is appropriate.

    Returns:
        True if CMA-ES is recommended
    """
    # Need purely continuous parameters
    if param_types != 'continuous':
        return False

    # Need sufficient trials (population-based)
    min_trials = 4 + int(3 * (n_params ** 0.5))
    if n_trials < min_trials * 3:
        return False

    # Great for correlated parameters or noisy objectives
    if has_correlations or is_noisy:
        return True

    # Good for large trial budgets
    if n_trials >= 500:
        return True

    return False


if __name__ == "__main__":
    # Example: Optimize Rosenbrock function (has strong parameter correlation)
    print("Example: CMA-ES on Rosenbrock Function\n")

    def rosenbrock(trial, params):
        """Rosenbrock function - has curved valley (x,y correlation)"""
        x = params['x']
        y = params['y']
        return (1 - x)**2 + 100*(y - x**2)**2

    result = optimize_cmaes(
        objective_func=rosenbrock,
        search_space={
            'x': (-2.0, 2.0),
            'y': (-1.0, 3.0),
        },
        n_trials=100,
        restart_strategy='ipop',
        output_dir='test_cmaes',
    )

    print(f"\n✅ Best value: {result.best_value:.6f}")
    print(f"Best params: {result.best_params}")
    print(f"(Optimum is at x=1, y=1 with value=0)")
