"""
Quasi-Monte Carlo Sampling (QMC) for Design of Experiments

**When to use:**
- Initial exploration phase (first 50-100 trials)
- Design of Experiments (DoE)
- Warm-starting other adaptive algorithms
- Sensitivity analysis
- Need uniform coverage of search space
- Embarrassingly parallel workloads

**Strengths:**
- O(1/N) convergence vs O(1/√N) for random sampling
- Perfect parallelization (no coordination needed)
- Deterministic and reproducible
- Zero computational overhead
- Systematic exploration of search space

**Limitations:**
- Non-adaptive (doesn't learn from trials)
- Not an optimizer, just generates well-distributed samples
- Worse sample efficiency than Bayesian/CMA-ES for targeted optimization
- Best as exploration phase, then switch to adaptive method
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import QMCSampler
from .base import BaseOptimizer


class QMCOptimizer(BaseOptimizer):
    """
    Quasi-Monte Carlo sampler for systematic exploration.
    Uses low-discrepancy sequences (Sobol or Halton).
    """

    def __init__(
        self,
        objective_func,
        search_space: Dict,
        direction: str = 'minimize',
        output_dir: str = 'qmc_output',
        qmc_type: str = 'sobol',  # 'sobol' or 'halton'
        scramble: bool = True,
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            qmc_type: Type of low-discrepancy sequence
                - 'sobol': Best for d > 6 dimensions (recommended)
                - 'halton': Best for d <= 6 dimensions
            scramble: Use scrambled sequence (improves uniformity)
            seed: Random seed for scrambling
        """
        super().__init__(objective_func, search_space, direction, output_dir, **kwargs)

        self.algorithm_name = f"QMC-{qmc_type.capitalize()}"
        self.qmc_type = qmc_type
        self.scramble = scramble
        self.seed = seed

        # Determine dimensionality
        self.n_dims = len(search_space)

        # Suggest optimal QMC type based on dimensionality
        if self.n_dims > 6 and qmc_type == 'halton':
            print(f"⚠️  Warning: Halton sequences degrade for d > 6.")
            print(f"   Consider using qmc_type='sobol' for {self.n_dims} dimensions.")

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create QMC sampler"""

        return QMCSampler(
            qmc_type=self.qmc_type,
            scramble=self.scramble,
            seed=self.seed,
        )

    def get_algorithm_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'QMC',
            'qmc_type': self.qmc_type,
            'scramble': self.scramble,
            'n_dimensions': self.n_dims,
            'seed': self.seed,
        }


def optimize_qmc(
    objective_func,
    search_space: Dict,
    n_trials: int = 100,
    qmc_type: str = 'sobol',
    output_dir: str = 'qmc_output',
    **sampler_kwargs
):
    """
    Quick QMC sampling function.

    Example:
        def objective(trial, params):
            return sum(v**2 for v in params.values())

        # Initial exploration with QMC
        qmc_result = optimize_qmc(
            objective,
            search_space={'x': (-5, 5), 'y': (-5, 5), 'z': (-5, 5)},
            n_trials=50,
            qmc_type='sobol',
        )

        # Then use best region for adaptive optimization
        best_x = qmc_result.best_params['x']
        # ... narrow search space around best_x ...
    """
    optimizer = QMCOptimizer(
        objective_func=objective_func,
        search_space=search_space,
        output_dir=output_dir,
        qmc_type=qmc_type,
        **sampler_kwargs
    )

    return optimizer.optimize(n_trials=n_trials)


def qmc_then_adaptive(
    objective_func,
    search_space: Dict,
    n_qmc_trials: int = 50,
    n_adaptive_trials: int = 150,
    adaptive_method: str = 'bayesian',
    output_dir: str = 'qmc_adaptive_output',
):
    """
    Hybrid strategy: QMC exploration → adaptive optimization.

    Example:
        result = qmc_then_adaptive(
            objective_func=my_objective,
            search_space={'x': (-10, 10), 'y': (-10, 10)},
            n_qmc_trials=50,
            n_adaptive_trials=150,
            adaptive_method='bayesian',
        )
    """
    from pathlib import Path
    import time

    print(f"\n{'='*80}")
    print(f"🎯 HYBRID STRATEGY: QMC → {adaptive_method.upper()}")
    print(f"{'='*80}")
    print(f"Phase 1: QMC exploration ({n_qmc_trials} trials)")
    print(f"Phase 2: {adaptive_method.upper()} optimization ({n_adaptive_trials} trials)")
    print(f"{'='*80}\n")

    output_path = Path(output_dir)

    # Phase 1: QMC exploration
    print("🔍 Phase 1: QMC Exploration\n")
    qmc_result = optimize_qmc(
        objective_func=objective_func,
        search_space=search_space,
        n_trials=n_qmc_trials,
        output_dir=str(output_path / "phase1_qmc"),
    )

    # Phase 2: Adaptive optimization
    print(f"\n🎯 Phase 2: {adaptive_method.upper()} Optimization\n")

    if adaptive_method == 'bayesian':
        from .bayesian import optimize_bayesian
        adaptive_result = optimize_bayesian(
            objective_func=objective_func,
            search_space=search_space,
            n_trials=n_adaptive_trials,
            output_dir=str(output_path / "phase2_bayesian"),
            n_startup_trials=0,  # Already explored with QMC
        )
    elif adaptive_method == 'cmaes':
        from .evolution_strategies import optimize_cmaes
        adaptive_result = optimize_cmaes(
            objective_func=objective_func,
            search_space=search_space,
            n_trials=n_adaptive_trials,
            output_dir=str(output_path / "phase2_cmaes"),
        )
    else:
        raise ValueError(f"Unknown adaptive method: {adaptive_method}")

    # Return best result
    if adaptive_result.best_value < qmc_result.best_value:
        print(f"\n✅ Best from {adaptive_method.upper()}: {adaptive_result.best_value:.6f}")
        return adaptive_result
    else:
        print(f"\n✅ Best from QMC: {qmc_result.best_value:.6f}")
        return qmc_result


if __name__ == "__main__":
    # Example: QMC sampling for uniform exploration
    print("Example: QMC Sampling with Sobol Sequence\n")

    def rastrigin(trial, params):
        """Rastrigin function - highly multimodal, needs good exploration"""
        import math
        A = 10
        n = len(params)
        return A * n + sum(v**2 - A * math.cos(2 * math.pi * v) for v in params.values())

    # QMC is great for initial exploration of complex landscapes
    result = optimize_qmc(
        objective_func=rastrigin,
        search_space={
            'x1': (-5.12, 5.12),
            'x2': (-5.12, 5.12),
            'x3': (-5.12, 5.12),
        },
        n_trials=50,
        qmc_type='sobol',
        output_dir='test_qmc',
    )

    print(f"\n✅ Best value found: {result.best_value:.6f}")
    print(f"Best params: {result.best_params}")
    print(f"(Global optimum is at origin with value=0)")

    # Demonstrate hybrid approach
    print("\n" + "="*80)
    print("Demonstrating QMC → Bayesian hybrid strategy\n")

    hybrid_result = qmc_then_adaptive(
        objective_func=rastrigin,
        search_space={
            'x1': (-5.12, 5.12),
            'x2': (-5.12, 5.12),
            'x3': (-5.12, 5.12),
        },
        n_qmc_trials=30,
        n_adaptive_trials=70,
        adaptive_method='bayesian',
        output_dir='test_qmc_hybrid',
    )

    print(f"\n✅ Hybrid best: {hybrid_result.best_value:.6f}")
