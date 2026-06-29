#!/usr/bin/env python3
"""CLI to compare a baseline agent run against a ResearchFS agent run."""

import argparse
import sys

from evals.nanochat_bpb.metrics import (
    aggregate_seeded_runs,
    compare_run_to_benchmarks,
    compare_runs,
    detect_reward_hacks,
    format_benchmark_comparison,
    format_comparison,
    format_reward_hack_report,
    format_seeded_aggregate,
    list_recent_runs,
    load_run,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ResearchFS runs and optionally screen them against published NanoChat benchmarks."
    )
    parser.add_argument(
        "run_ids",
        type=int,
        nargs="*",
        help="agent_run_id values to compare",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent agent runs instead of comparing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent runs to list",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write the comparison report to this file",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Append benchmark comparison for both runs to the normal comparison",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Compare one run ID against published benchmarks",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate two or more run IDs as multi-seed evaluations of one recipe",
    )

    args = parser.parse_args()

    if args.list:
        runs = list_recent_runs(args.limit)
        print(f"{'ID':<6} {'Mode':<12} {'Model':<25} {'Started'}")
        print("-" * 80)
        for run in runs:
            print(
                f"{run['agent_run_id']:<6} "
                f"{run['mode']:<12} "
                f"{run['model'] or 'N/A':<25} "
                f"{run['started_at']}"
            )
        return

    exclusive_modes = [args.benchmarks, args.benchmark_only, args.aggregate_seeds]
    if sum(exclusive_modes) > 1:
        print(
            "Error: --benchmarks, --benchmark-only, and --aggregate-seeds are mutually exclusive.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    if args.aggregate_seeds:
        if len(args.run_ids) < 2:
            print(
                "Error: --aggregate-seeds requires at least two run IDs.",
                file=sys.stderr,
            )
            parser.print_help()
            sys.exit(1)
        aggregate = aggregate_seeded_runs(args.run_ids)
        report = format_seeded_aggregate(aggregate)
    elif args.benchmark_only:
        if len(args.run_ids) != 1:
            print(
                "Error: --benchmark-only requires exactly one run ID.",
                file=sys.stderr,
            )
            parser.print_help()
            sys.exit(1)
        run_id = args.run_ids[0]
        run = load_run(run_id)
        report = "\n\n".join(
            [
                format_benchmark_comparison(compare_run_to_benchmarks(run_id)),
                format_reward_hack_report(run, detect_reward_hacks(run)),
            ]
        )
    else:
        if len(args.run_ids) != 2:
            print(
                "Error: exactly two run IDs are required unless --list, --benchmark-only, or --aggregate-seeds is used.",
                file=sys.stderr,
            )
            parser.print_help()
            sys.exit(1)

        baseline_run_id, researchfs_run_id = args.run_ids
        baseline = load_run(baseline_run_id)
        researchfs = load_run(researchfs_run_id)
        comparison = compare_runs(baseline_run_id, researchfs_run_id)
        report_parts = [format_comparison(comparison)]
        if args.benchmarks:
            report_parts.extend(
                [
                    "## Benchmark comparison: Baseline",
                    format_benchmark_comparison(
                        compare_run_to_benchmarks(baseline_run_id)
                    ),
                    format_reward_hack_report(
                        baseline, detect_reward_hacks(baseline)
                    ),
                    "## Benchmark comparison: ResearchFS",
                    format_benchmark_comparison(
                        compare_run_to_benchmarks(researchfs_run_id)
                    ),
                    format_reward_hack_report(
                        researchfs, detect_reward_hacks(researchfs)
                    ),
                ]
            )
        else:
            report_parts.extend(
                [
                    format_reward_hack_report(baseline, detect_reward_hacks(baseline)),
                    format_reward_hack_report(
                        researchfs, detect_reward_hacks(researchfs)
                    ),
                ]
            )
        report = "\n\n".join(report_parts)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
