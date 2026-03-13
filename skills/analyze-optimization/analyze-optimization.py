#!/usr/bin/env python3
"""
Optimization Analysis Skill

Usage:
    python analyze-optimization.py bayesian_output
    python analyze-optimization.py --compare bayesian_output cmaes_output nsga_output

Analyzes optimization results with plots, statistics, and exports.
"""

import sys
import argparse
from pathlib import Path

try:
    from optuna_algorithms import OptimizationAnalyzer
    from optuna_algorithms.analysis import compare_optimizations
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nInstall dependencies: pip install optuna pandas numpy matplotlib")
    sys.exit(1)


def analyze_single_run(result_dir, plot_dir=None, export=True):
    """
    Analyze a single optimization run.

    Args:
        result_dir: Directory containing optimization results
        plot_dir: Where to save plots (default: result_dir/plots)
        export: Export best config and report

    Returns:
        OptimizationAnalyzer instance
    """
    result_dir = Path(result_dir)

    if not result_dir.exists():
        print(f"❌ Error: {result_dir} not found")
        sys.exit(1)

    if not (result_dir / 'optimization_result.json').exists():
        print(f"❌ Error: No optimization_result.json in {result_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"ANALYZING: {result_dir}")
    print(f"{'='*80}\n")

    # Load and analyze
    analyzer = OptimizationAnalyzer(str(result_dir))

    # Print summary
    analyzer.print_summary()

    # Generate plots
    if plot_dir is None:
        plot_dir = result_dir / 'plots'

    try:
        analyzer.plot_all(output_dir=str(plot_dir), show=False)
        print(f"\n📊 Plots saved to: {plot_dir}/")
        print("   - convergence.png")
        print("   - parameter_importance.png")
        print("   - parameter_distributions.png")
    except ImportError:
        print("\n⚠️  matplotlib not installed, skipping plots")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"\n⚠️  Plotting failed: {e}")

    # Export results
    if export:
        try:
            config_path = result_dir / 'best_config.json'
            report_path = result_dir / 'optimization_report.md'

            analyzer.export_best_config(str(config_path))
            analyzer.to_markdown_report(str(report_path))

            print(f"\n📄 Exports:")
            print(f"   - Best config: {config_path}")
            print(f"   - Report: {report_path}")
        except Exception as e:
            print(f"\n⚠️  Export failed: {e}")

    print(f"\n{'='*80}\n")

    return analyzer


def compare_multiple_runs(result_dirs, output_dir='comparison'):
    """
    Compare multiple optimization runs.

    Args:
        result_dirs: List of result directories to compare
        output_dir: Where to save comparison plots

    Returns:
        Comparison summary DataFrame
    """
    print(f"\n{'='*80}")
    print(f"COMPARING {len(result_dirs)} OPTIMIZATION RUNS")
    print(f"{'='*80}\n")

    for i, result_dir in enumerate(result_dirs, 1):
        print(f"{i}. {result_dir}")

    print(f"\nComparison output: {output_dir}/")
    print(f"{'='*80}\n")

    try:
        summary_df = compare_optimizations(
            result_dirs=result_dirs,
            output_dir=output_dir,
        )

        print(f"\n📊 Comparison plots saved to: {output_dir}/")
        print(f"\n{'='*80}\n")

        return summary_df

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Optimization Analysis Skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single run
  python analyze-optimization.py bayesian_output

  # Analyze with custom plot directory
  python analyze-optimization.py bayesian_output --plot_dir my_plots

  # Compare multiple runs
  python analyze-optimization.py --compare bayesian_output cmaes_output

  # Compare and save to custom directory
  python analyze-optimization.py --compare run1 run2 run3 --output comparison_results

Features:
  - Summary statistics (trials, best value, parameters)
  - Convergence plots (best value over trials)
  - Parameter importance (correlation with objective)
  - Parameter distributions (explored ranges)
  - Best config export (JSON)
  - Markdown report generation
  - Multi-run comparison
        """
    )

    parser.add_argument(
        'result_dirs',
        nargs='*',
        help='Result directories to analyze'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple runs (provide multiple directories)'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default=None,
        help='Custom plot directory for single run analysis'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='comparison',
        help='Output directory for comparison (with --compare)'
    )

    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip exporting best config and report'
    )

    args = parser.parse_args()

    if not args.result_dirs:
        parser.print_help()
        sys.exit(1)

    if args.compare:
        # Compare multiple runs
        if len(args.result_dirs) < 2:
            print("❌ Error: Need at least 2 directories to compare")
            sys.exit(1)

        compare_multiple_runs(args.result_dirs, args.output_dir)

    else:
        # Analyze single run
        if len(args.result_dirs) != 1:
            print("❌ Error: Provide exactly 1 directory (or use --compare for multiple)")
            sys.exit(1)

        analyze_single_run(
            args.result_dirs[0],
            plot_dir=args.plot_dir,
            export=not args.no_export,
        )


if __name__ == "__main__":
    main()
