"""
Optimization Analysis and Visualization Tools

Provides comprehensive analysis of optimization results:
- Convergence plots
- Parameter importance
- Parallel coordinate plots
- Pareto fronts (multi-objective)
- Statistical summaries
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np


class OptimizationAnalyzer:
    """
    Analyze and visualize optimization results.
    Works with results from any optimizer.
    """

    def __init__(self, result_dir: Union[str, Path]):
        """
        Args:
            result_dir: Directory containing optimization_result.json and trials.jsonl
        """
        self.result_dir = Path(result_dir)

        # Load results
        with open(self.result_dir / "optimization_result.json", 'r') as f:
            self.result = json.load(f)

        # Load trials DataFrame
        if (self.result_dir / "all_trials.csv").exists():
            self.df = pd.read_csv(self.result_dir / "all_trials.csv")
        else:
            self.df = pd.DataFrame(self.result['all_trials'])

        self.algorithm = self.result['algorithm_name']
        self.n_trials = self.result['n_trials']
        self.n_complete = self.result['n_complete']

    def print_summary(self):
        """Print comprehensive summary statistics"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION ANALYSIS: {self.algorithm}")
        print(f"{'='*80}\n")

        print(f"📊 Trial Statistics:")
        print(f"   Total trials: {self.n_trials}")
        print(f"   Completed: {self.n_complete}")
        print(f"   Failed: {self.result['n_failed']}")
        print(f"   Success rate: {100*self.n_complete/self.n_trials:.1f}%")
        print(f"   Time: {self.result['elapsed_time_s']:.1f}s ({self.result['elapsed_time_s']/60:.1f} min)")
        print(f"   Avg time/trial: {self.result['elapsed_time_s']/self.n_trials:.1f}s\n")

        print(f"🎯 Best Result:")
        print(f"   Best value: {self.result['best_value']:.6f}")
        print(f"   Best parameters:")
        for k, v in self.result['best_params'].items():
            print(f"      {k}: {v}")
        print()

        # Value statistics
        if len(self.df) > 0:
            values = self.df['value'].dropna()
            if len(values) > 0:
                print(f"📈 Value Distribution:")
                print(f"   Best (min): {values.min():.6f}")
                print(f"   Worst (max): {values.max():.6f}")
                print(f"   Mean: {values.mean():.6f}")
                print(f"   Median: {values.median():.6f}")
                print(f"   Std dev: {values.std():.6f}")
                print()

        # Parameter ranges explored
        print(f"🔍 Parameter Exploration:")
        params = self.result['best_params'].keys()
        for param in params:
            if param in self.df.columns:
                param_values = self.df[param].dropna()
                if len(param_values) > 0:
                    print(f"   {param}:")
                    print(f"      Range: [{param_values.min():.4f}, {param_values.max():.4f}]")
                    print(f"      Mean: {param_values.mean():.4f}")
        print()

        print(f"{'='*80}\n")

    def get_convergence_data(self) -> Dict[str, List]:
        """
        Get convergence data (best value over trials).

        Returns:
            Dict with 'trial_numbers' and 'best_values' lists
        """
        values = []
        trials = []

        best_so_far = float('inf')

        for _, row in self.df.iterrows():
            if pd.notna(row['value']):
                trial = row['number']
                value = row['value']

                if value < best_so_far:
                    best_so_far = value

                trials.append(trial)
                values.append(best_so_far)

        return {'trial_numbers': trials, 'best_values': values}

    def plot_convergence(self, output_path: Optional[str] = None, show: bool = True):
        """
        Plot optimization convergence.

        Args:
            output_path: Path to save plot (optional)
            show: Show plot interactively
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
            return

        conv_data = self.get_convergence_data()

        plt.figure(figsize=(10, 6))
        plt.plot(conv_data['trial_numbers'], conv_data['best_values'], 'b-', linewidth=2)
        plt.xlabel('Trial Number', fontsize=12)
        plt.ylabel('Best Value', fontsize=12)
        plt.title(f'Optimization Convergence: {self.algorithm}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Convergence plot saved to {output_path}")

        if show:
            plt.show()

        plt.close()

    def plot_parameter_importances(self, output_path: Optional[str] = None, show: bool = True):
        """
        Plot parameter importance (correlation with objective).

        Args:
            output_path: Path to save plot
            show: Show plot interactively
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
            return

        # Calculate correlations
        params = list(self.result['best_params'].keys())
        correlations = {}

        for param in params:
            if param in self.df.columns:
                param_values = self.df[param].dropna()
                value_aligned = self.df.loc[param_values.index, 'value'].dropna()

                if len(param_values) > 1 and len(value_aligned) > 1:
                    corr = abs(np.corrcoef(param_values, value_aligned)[0, 1])
                    correlations[param] = corr

        if not correlations:
            print("⚠️  No correlations could be calculated")
            return

        # Sort by importance
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        importances = [p[1] for p in sorted_params]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(param_names, importances, color='steelblue')
        plt.xlabel('Importance (|correlation|)', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.title(f'Parameter Importance: {self.algorithm}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Parameter importance plot saved to {output_path}")

        if show:
            plt.show()

        plt.close()

    def plot_parameter_distributions(self, output_path: Optional[str] = None, show: bool = True):
        """
        Plot distribution of parameter values explored.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  Matplotlib not available")
            return

        params = list(self.result['best_params'].keys())
        n_params = len(params)

        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
        if n_params == 1:
            axes = [axes]

        for i, param in enumerate(params):
            if param in self.df.columns:
                param_values = self.df[param].dropna()

                axes[i].hist(param_values, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
                axes[i].axvline(self.result['best_params'][param], color='red',
                              linestyle='--', linewidth=2, label='Best')
                axes[i].set_xlabel(param, fontsize=11)
                axes[i].set_ylabel('Frequency', fontsize=11)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.suptitle(f'Parameter Distributions: {self.algorithm}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Parameter distribution plot saved to {output_path}")

        if show:
            plt.show()

        plt.close()

    def plot_all(self, output_dir: Optional[str] = None, show: bool = False):
        """
        Generate all standard plots.

        Args:
            output_dir: Directory to save plots
            show: Show plots interactively
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.plot_convergence(str(output_path / "convergence.png"), show=show)
            self.plot_parameter_importances(str(output_path / "parameter_importance.png"), show=show)
            self.plot_parameter_distributions(str(output_path / "parameter_distributions.png"), show=show)

            print(f"\n✅ All plots saved to {output_dir}\n")
        else:
            self.plot_convergence(show=show)
            self.plot_parameter_importances(show=show)
            self.plot_parameter_distributions(show=show)

    def compare_trials(self, trial_numbers: List[int]) -> pd.DataFrame:
        """
        Compare specific trials side-by-side.

        Args:
            trial_numbers: List of trial numbers to compare

        Returns:
            DataFrame with trial comparisons
        """
        trials = self.df[self.df['number'].isin(trial_numbers)]
        return trials.set_index('number')

    def export_best_config(self, output_path: str):
        """
        Export best parameters as JSON config file.

        Args:
            output_path: Path to save config
        """
        config = {
            'best_parameters': self.result['best_params'],
            'best_value': self.result['best_value'],
            'algorithm': self.algorithm,
            'metadata': {
                'n_trials': self.n_trials,
                'success_rate': self.n_complete / self.n_trials,
                'elapsed_time_s': self.result['elapsed_time_s'],
            }
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Best config exported to {output_path}")

    def to_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report of optimization results.

        Returns:
            Markdown string (also saves to file if output_path provided)
        """
        lines = [
            f"# Optimization Report: {self.algorithm}",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Trials:** {self.n_trials}",
            f"- **Completed:** {self.n_complete} ({100*self.n_complete/self.n_trials:.1f}%)",
            f"- **Failed:** {self.result['n_failed']}",
            f"- **Total Time:** {self.result['elapsed_time_s']:.1f}s ({self.result['elapsed_time_s']/60:.1f} min)",
            f"- **Avg Time/Trial:** {self.result['elapsed_time_s']/self.n_trials:.2f}s",
            "",
            "## Best Result",
            "",
            f"**Best Value:** `{self.result['best_value']:.6f}`",
            "",
            "**Best Parameters:**",
            "",
        ]

        for k, v in self.result['best_params'].items():
            lines.append(f"- `{k}`: {v}")

        lines.append("")
        lines.append("## Algorithm Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.result['algorithm_config'], indent=2))
        lines.append("```")
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✅ Markdown report saved to {output_path}")

        return report


def compare_optimizations(result_dirs: List[Union[str, Path]], output_dir: Optional[str] = None):
    """
    Compare multiple optimization runs.

    Args:
        result_dirs: List of result directories to compare
        output_dir: Directory to save comparison plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Matplotlib required for comparison plots")
        return

    analyzers = [OptimizationAnalyzer(d) for d in result_dirs]

    # Convergence comparison
    plt.figure(figsize=(12, 6))

    for analyzer in analyzers:
        conv_data = analyzer.get_convergence_data()
        plt.plot(conv_data['trial_numbers'], conv_data['best_values'],
                label=analyzer.algorithm, linewidth=2)

    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Best Value', fontsize=12)
    plt.title('Optimization Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "convergence_comparison.png", dpi=150, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {output_dir}")

    plt.show()
    plt.close()

    # Summary table
    summary_data = []
    for analyzer in analyzers:
        summary_data.append({
            'Algorithm': analyzer.algorithm,
            'Best Value': analyzer.result['best_value'],
            'Trials': analyzer.n_trials,
            'Time (s)': analyzer.result['elapsed_time_s'],
            'Success Rate': f"{100*analyzer.n_complete/analyzer.n_trials:.1f}%",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

    return summary_df


if __name__ == "__main__":
    print("Optimization Analysis Tools\n")
    print("Example usage:")
    print("""
    from optuna_algorithms.analysis import OptimizationAnalyzer

    # Analyze single optimization
    analyzer = OptimizationAnalyzer('bayesian_output')
    analyzer.print_summary()
    analyzer.plot_all(output_dir='plots', show=True)

    # Generate report
    analyzer.to_markdown_report('report.md')
    analyzer.export_best_config('best_config.json')

    # Compare multiple runs
    from optuna_algorithms.analysis import compare_optimizations
    compare_optimizations(
        ['bayesian_output', 'cmaes_output', 'nsga_output'],
        output_dir='comparison_plots'
    )
    """)
