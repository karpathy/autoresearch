"""
Hybrid and Adaptive Optimization Strategies

**When to use:**
- Mixed continuous + discrete + categorical parameters
- Neural architecture search
- AutoML with diverse hyperparameter types
- Complex pipelines with heterogeneous search spaces
- When problem doesn't fit cleanly into continuous or discrete categories
- When you're unsure which algorithm to use

**Strengths:**
- Handles mixed-variable optimization
- Adaptive to problem structure
- Can combine multiple algorithms (ensemble, sequential)
- Flexibility for complex real-world problems

**Limitations:**
- More complex setup
- May require algorithm-specific tuning
- CatCMA requires installation from OptunaHub
"""

from typing import Dict, Any, Optional, List
import optuna
from .base import BaseOptimizer
from pathlib import Path


class HybridOptimizer(BaseOptimizer):
    """
    Hybrid optimizer supporting multiple strategies:
    1. CatCMA - Mixed continuous/discrete/categorical (best-in-class)
    2. Sequential strategies (QMC → TPE, TPE → CMA-ES)
    3. Ensemble approaches (parallel algorithms)
    """

    def __init__(
        self,
        objective_func,
        search_space: Dict,
        direction: str = 'minimize',
        output_dir: str = 'hybrid_output',
        strategy: str = 'auto',  # 'catcma', 'sequential', 'ensemble', 'auto'
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            strategy: Hybrid strategy to use
                - 'catcma': CatCMA-ES for mixed variables (requires OptunaHub)
                - 'sequential': Multi-phase optimization (QMC → TPE → CMA-ES)
                - 'ensemble': Run multiple algorithms in parallel
                - 'auto': Automatically choose based on search space
        """
        super().__init__(objective_func, search_space, direction, output_dir, **kwargs)

        self.strategy = strategy
        self.seed = seed

        # Analyze search space
        self.space_analysis = self._analyze_search_space()

        # Auto-select strategy
        if self.strategy == 'auto':
            self.strategy = self._auto_select_strategy()

        self.algorithm_name = f"Hybrid-{self.strategy.upper()}"

        print(f"\n📊 Search Space Analysis:")
        print(f"   Continuous: {self.space_analysis['n_continuous']}")
        print(f"   Integer: {self.space_analysis['n_integer']}")
        print(f"   Categorical: {self.space_analysis['n_categorical']}")
        print(f"   Selected strategy: {self.strategy}\n")

    def _analyze_search_space(self) -> Dict[str, Any]:
        """Analyze parameter types in search space"""
        n_continuous = 0
        n_integer = 0
        n_categorical = 0

        for param_name, bounds in self.search_space.items():
            if isinstance(bounds, (list, tuple)):
                if len(bounds) == 2:
                    min_val, max_val = bounds
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        n_integer += 1
                    else:
                        n_continuous += 1
                else:
                    n_categorical += 1

        param_types = []
        if n_continuous > 0:
            param_types.append('continuous')
        if n_integer > 0:
            param_types.append('integer')
        if n_categorical > 0:
            param_types.append('categorical')

        return {
            'n_continuous': n_continuous,
            'n_integer': n_integer,
            'n_categorical': n_categorical,
            'total': n_continuous + n_integer + n_categorical,
            'param_types': param_types,
            'is_mixed': len(param_types) > 1,
            'is_pure_continuous': len(param_types) == 1 and 'continuous' in param_types,
        }

    def _auto_select_strategy(self) -> str:
        """Automatically select best strategy based on search space"""

        if self.space_analysis['is_pure_continuous']:
            return 'cmaes_only'

        if self.space_analysis['is_mixed']:
            # Try CatCMA for mixed variables
            try:
                import optunahub
                return 'catcma'
            except ImportError:
                print("⚠️  CatCMA requires OptunaHub. Falling back to TPE.")
                return 'tpe_only'

        # Default: TPE (handles all types reasonably well)
        return 'tpe_only'

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create sampler based on strategy"""

        if self.strategy == 'catcma':
            return self._create_catcma_sampler()

        elif self.strategy in ['tpe_only', 'sequential']:
            from optuna.samplers import TPESampler
            return TPESampler(seed=self.seed, multivariate=True)

        elif self.strategy == 'cmaes_only':
            from optuna.samplers import CmaEsSampler
            return CmaEsSampler(seed=self.seed)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _create_catcma_sampler(self):
        """Create CatCMA sampler from OptunaHub"""
        try:
            import optunahub

            # Load CatCMA from OptunaHub
            module = optunahub.load_module("samplers/catcma")
            CatCmaSampler = module.CatCmaSampler

            return CatCmaSampler(seed=self.seed)

        except ImportError:
            print("❌ CatCMA requires optunahub package.")
            print("   Install with: pip install optunahub")
            print("   Falling back to TPE sampler.")

            from optuna.samplers import TPESampler
            return TPESampler(seed=self.seed)

        except Exception as e:
            print(f"❌ Failed to load CatCMA: {e}")
            print("   Falling back to TPE sampler.")

            from optuna.samplers import TPESampler
            return TPESampler(seed=self.seed)

    def optimize_sequential(
        self,
        n_trials: int = 200,
        phase_splits: List[float] = [0.2, 0.5, 0.3],  # QMC, TPE, CMA-ES
    ):
        """
        Sequential multi-phase optimization.

        Phase 1: QMC exploration
        Phase 2: TPE exploitation
        Phase 3: CMA-ES refinement (if continuous)
        """
        from .qmc import QMCOptimizer
        from .bayesian import BayesianOptimizer
        from .evolution_strategies import CMAESOptimizer

        n_qmc = int(n_trials * phase_splits[0])
        n_tpe = int(n_trials * phase_splits[1])
        n_cmaes = n_trials - n_qmc - n_tpe

        print(f"\n{'='*80}")
        print(f"🔄 SEQUENTIAL HYBRID OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Phase 1: QMC ({n_qmc} trials) - Exploration")
        print(f"Phase 2: TPE ({n_tpe} trials) - Exploitation")
        if self.space_analysis['is_pure_continuous']:
            print(f"Phase 3: CMA-ES ({n_cmaes} trials) - Refinement")
        print(f"{'='*80}\n")

        output_path = Path(self.output_dir)

        # Phase 1: QMC
        print("🔍 Phase 1: QMC Exploration\n")
        qmc = QMCOptimizer(self.objective_func, self.search_space,
                           direction=self.direction,
                           output_dir=str(output_path / "phase1_qmc"))
        qmc_result = qmc.optimize(n_trials=n_qmc, show_progress=True)

        # Phase 2: TPE
        print("\n🎯 Phase 2: TPE Exploitation\n")
        tpe = BayesianOptimizer(self.objective_func, self.search_space,
                                direction=self.direction,
                                output_dir=str(output_path / "phase2_tpe"),
                                n_startup_trials=0)  # Use QMC results
        tpe_result = tpe.optimize(n_trials=n_tpe, show_progress=True)

        # Phase 3: CMA-ES (if pure continuous)
        if self.space_analysis['is_pure_continuous'] and n_cmaes > 0:
            print("\n⚡ Phase 3: CMA-ES Refinement\n")
            cmaes = CMAESOptimizer(self.objective_func, self.search_space,
                                   direction=self.direction,
                                   output_dir=str(output_path / "phase3_cmaes"))
            cmaes_result = cmaes.optimize(n_trials=n_cmaes, show_progress=True)

            # Return best across all phases
            best_result = min([qmc_result, tpe_result, cmaes_result],
                            key=lambda r: r.best_value)
        else:
            best_result = min([qmc_result, tpe_result], key=lambda r: r.best_value)

        print(f"\n{'='*80}")
        print(f"✅ Sequential optimization complete!")
        print(f"   Best value: {best_result.best_value:.6f}")
        print(f"   From phase: {best_result.algorithm_name}")
        print(f"{'='*80}\n")

        return best_result

    def get_algorithm_config(self) -> Dict[str, Any]:
        return {
            'algorithm': 'Hybrid',
            'strategy': self.strategy,
            'space_analysis': self.space_analysis,
            'seed': self.seed,
        }


def optimize_hybrid(
    objective_func,
    search_space: Dict,
    n_trials: int = 200,
    strategy: str = 'auto',
    output_dir: str = 'hybrid_output',
    **kwargs
):
    """
    Quick hybrid optimization function.

    Example:
        # Mixed parameter types
        result = optimize_hybrid(
            objective_func=my_objective,
            search_space={
                'learning_rate': (1e-5, 1e-1),  # continuous
                'batch_size': (16, 256),  # integer
                'optimizer': ['adam', 'sgd', 'rmsprop'],  # categorical
            },
            n_trials=200,
            strategy='auto',  # Automatically selects best approach
        )
    """
    optimizer = HybridOptimizer(
        objective_func=objective_func,
        search_space=search_space,
        output_dir=output_dir,
        strategy=strategy,
        **kwargs
    )

    if strategy == 'sequential':
        return optimizer.optimize_sequential(n_trials=n_trials)
    else:
        return optimizer.optimize(n_trials=n_trials)


if __name__ == "__main__":
    # Example: Mixed variable optimization
    print("Example: Hybrid Optimization with Mixed Variables\n")

    def mixed_objective(trial, params):
        """Objective with continuous, integer, and categorical parameters"""
        x = params['x']  # continuous
        n = params['n']  # integer
        mode = params['mode']  # categorical

        base = x**2 + (n - 5)**2

        if mode == 'square':
            return base
        elif mode == 'sqrt':
            return base**0.5
        else:  # 'log'
            return (base + 1)**0.1

    result = optimize_hybrid(
        objective_func=mixed_objective,
        search_space={
            'x': (-5.0, 5.0),  # continuous
            'n': (1, 10),  # integer
            'mode': ['square', 'sqrt', 'log'],  # categorical
        },
        n_trials=50,
        strategy='auto',
        output_dir='test_hybrid',
    )

    print(f"\n✅ Best value: {result.best_value:.6f}")
    print(f"Best params: {result.best_params}")
