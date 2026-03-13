"""
Multi-Objective Optimization using NSGA-II and NSGA-III

**When to use:**
- Multiple competing objectives (accuracy vs latency, performance vs cost)
- Trade-off analysis (need entire Pareto frontier, not single solution)
- Many-objective problems (3-10 objectives, use NSGA-III)
- Decision-making with multiple criteria

**Strengths:**
- Produces complete Pareto-optimal frontier
- NSGA-III superior for many-objective (4+ objectives)
- Handles all parameter types
- Embarrassingly parallel (population-based)
- Well-established, proven algorithm

**Limitations:**
- Less sample-efficient than Bayesian for single-objective
- Requires larger populations for many objectives
- Returns set of solutions (not single best)
"""

from typing import Dict, Any, Optional, List, Callable
import optuna
from optuna.samplers import NSGAIISampler, NSGAIIISampler
from .base import BaseOptimizer, OptimizationResult


class NSGAOptimizer(BaseOptimizer):
    """
    Multi-objective optimizer using NSGA-II or NSGA-III.

    Note: For multi-objective, objective_func should return a tuple/list of values,
    one for each objective.
    """

    def __init__(
        self,
        objective_func,
        search_space: Dict,
        n_objectives: int = 2,
        directions: Optional[List[str]] = None,  # e.g., ['minimize', 'maximize']
        output_dir: str = 'nsga_output',
        variant: str = 'auto',  # 'nsga2', 'nsga3', or 'auto'
        population_size: Optional[int] = None,
        mutation_prob: Optional[float] = None,
        crossover_prob: float = 0.9,
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            n_objectives: Number of objectives to optimize
            directions: List of 'minimize' or 'maximize' for each objective
            variant: 'nsga2', 'nsga3', or 'auto'
                - nsga2: Best for 2-3 objectives
                - nsga3: Best for 4+ objectives (many-objective)
                - auto: Automatically choose based on n_objectives
            population_size: Size of population (default: 50 for NSGA-II, 100 for NSGA-III)
            mutation_prob: Probability of mutation per parameter
            crossover_prob: Probability of crossover
            seed: Random seed
        """
        # For multi-objective, we pass directions list instead of single direction
        super().__init__(
            objective_func, search_space,
            direction='minimize',  # Dummy, will be overridden
            output_dir=output_dir,
            **kwargs
        )

        self.n_objectives = n_objectives
        self.directions = directions or ['minimize'] * n_objectives

        if len(self.directions) != n_objectives:
            raise ValueError(f"directions must have {n_objectives} elements")

        # Auto-select variant
        if variant == 'auto':
            self.variant = 'nsga3' if n_objectives >= 4 else 'nsga2'
        else:
            self.variant = variant

        self.algorithm_name = f"{self.variant.upper()}"

        # Set default population sizes
        if population_size is None:
            self.population_size = 100 if self.variant == 'nsga3' else 50
        else:
            self.population_size = population_size

        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.seed = seed

    def create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create NSGA-II or NSGA-III sampler"""

        sampler_kwargs = {
            'population_size': self.population_size,
            'crossover_prob': self.crossover_prob,
            'seed': self.seed,
        }

        if self.mutation_prob is not None:
            sampler_kwargs['mutation_prob'] = self.mutation_prob

        if self.variant == 'nsga2':
            return NSGAIISampler(**sampler_kwargs)
        elif self.variant == 'nsga3':
            return NSGAIIISampler(**sampler_kwargs)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        show_progress: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        Note: For multi-objective, best_value and best_params represent
        one point on the Pareto frontier (not necessarily unique "best").
        Use all_trials to access the full Pareto frontier.
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting {self.algorithm_name} Multi-Objective Optimization")
        print(f"{'='*80}")
        print(f"Objectives: {self.n_objectives}")
        print(f"Directions: {self.directions}")
        print(f"Population: {self.population_size}")
        print(f"Trials: {n_trials}")
        print(f"{'='*80}\n")

        # Create sampler
        self.sampler = self.create_sampler()

        # Create multi-objective study
        study = optuna.create_study(
            study_name=self.study_name,
            sampler=self.sampler,
            directions=self.directions,  # Multi-objective!
            storage=self.storage,
            load_if_exists=True,
        )

        # Run optimization
        import time as time_module
        start_time = time_module.time()

        study.optimize(
            self.wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            callbacks=callbacks,
        )

        elapsed_time = time_module.time() - start_time

        # Collect results
        all_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    'number': trial.number,
                    'params': trial.params,
                    'values': trial.values,  # Multi-objective: list of values
                    'state': trial.state.name,
                }
                all_trials.append(trial_data)

        # Get Pareto front
        pareto_trials = [t for t in study.best_trials]

        print(f"\n📊 Pareto Front: {len(pareto_trials)} solutions")

        # For compatibility with single-objective interface, use first Pareto point
        if pareto_trials:
            best_trial = pareto_trials[0]
            best_params = best_trial.params
            best_value = best_trial.values[0]  # First objective
        else:
            best_params = {}
            best_value = float('inf')

        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=len(study.trials),
            n_complete=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            n_failed=len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            elapsed_time_s=elapsed_time,
            algorithm_name=self.algorithm_name,
            algorithm_config=self.get_algorithm_config(),
            study_name=self.study_name,
            all_trials=all_trials,
            metadata={
                'n_objectives': self.n_objectives,
                'directions': self.directions,
                'pareto_front_size': len(pareto_trials),
                'pareto_front': [{'params': t.params, 'values': t.values} for t in pareto_trials],
            }
        )

        result.save(self.output_dir)
        print(result.summary())

        return result

    def get_algorithm_config(self) -> Dict[str, Any]:
        return {
            'algorithm': self.variant.upper(),
            'n_objectives': self.n_objectives,
            'directions': self.directions,
            'population_size': self.population_size,
            'mutation_prob': self.mutation_prob,
            'crossover_prob': self.crossover_prob,
            'seed': self.seed,
        }


def optimize_nsga(
    objective_func,
    search_space: Dict,
    n_objectives: int = 2,
    directions: Optional[List[str]] = None,
    n_trials: int = 100,
    variant: str = 'auto',
    output_dir: str = 'nsga_output',
    **sampler_kwargs
):
    """
    Quick multi-objective optimization function.

    Example:
        def multi_objective(trial, params):
            x = params['x']
            y = params['y']
            # Minimize both objectives
            obj1 = x**2 + y**2  # Minimize distance from origin
            obj2 = (x-1)**2 + (y-1)**2  # Minimize distance from (1,1)
            return obj1, obj2

        result = optimize_nsga(
            multi_objective,
            search_space={'x': (-5, 5), 'y': (-5, 5)},
            n_objectives=2,
            directions=['minimize', 'minimize'],
            n_trials=200,
        )

        # Access Pareto frontier
        pareto_front = result.metadata['pareto_front']
        for point in pareto_front:
            print(f"Params: {point['params']}, Objectives: {point['values']}")
    """
    optimizer = NSGAOptimizer(
        objective_func=objective_func,
        search_space=search_space,
        n_objectives=n_objectives,
        directions=directions,
        output_dir=output_dir,
        variant=variant,
        **sampler_kwargs
    )

    return optimizer.optimize(n_trials=n_trials)


if __name__ == "__main__":
    # Example: Bi-objective optimization
    print("Example: NSGA-II for Bi-Objective Optimization\n")

    def bi_objective(trial, params):
        """
        Trade-off: minimize distance from origin vs distance from (2,2)
        Pareto front is the line segment connecting them.
        """
        x = params['x']
        y = params['y']

        obj1 = x**2 + y**2  # Distance from (0,0)
        obj2 = (x-2)**2 + (y-2)**2  # Distance from (2,2)

        return obj1, obj2

    result = optimize_nsga(
        objective_func=bi_objective,
        search_space={'x': (-1, 3), 'y': (-1, 3)},
        n_objectives=2,
        directions=['minimize', 'minimize'],
        n_trials=50,
        variant='nsga2',
        output_dir='test_nsga',
    )

    print(f"\n✅ Pareto front has {result.metadata['pareto_front_size']} solutions")
    print("\nFirst 5 Pareto solutions:")
    for point in result.metadata['pareto_front'][:5]:
        print(f"  {point}")
