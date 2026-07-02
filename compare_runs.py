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


def build_comparison_report(
    baseline_run_id: int,
    researchfs_run_id: int,
    *,
    include_benchmarks: bool = False,
    conn=None,
) -> str:
    """Build a two-run comparison report without CLI side effects."""
    comparison = compare_runs(baseline_run_id, researchfs_run_id, conn=conn)
    baseline = comparison["baseline"]
    researchfs = comparison["researchfs"]

    report_parts = [format_comparison(comparison)]
    if include_benchmarks:
        report_parts.extend(
            [
                format_benchmark_comparison(
                    compare_run_to_benchmarks(baseline_run_id, conn=conn),
                    title="Benchmark comparison: Baseline",
                    heading_level=2,
                ),
                format_reward_hack_report(baseline, detect_reward_hacks(baseline)),
                format_benchmark_comparison(
                    compare_run_to_benchmarks(researchfs_run_id, conn=conn),
                    title="Benchmark comparison: ResearchFS",
                    heading_level=2,
                ),
                format_reward_hack_report(researchfs, detect_reward_hacks(researchfs)),
            ]
        )
    else:
        report_parts.extend(
            [
                format_reward_hack_report(baseline, detect_reward_hacks(baseline)),
                format_reward_hack_report(researchfs, detect_reward_hacks(researchfs)),
            ]
        )
    return "\n\n".join(report_parts)


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

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    if args.list:
        if (
            args.run_ids
            or args.output
            or args.benchmarks
            or args.benchmark_only
            or args.aggregate_seeds
        ):
            parser.error(
                "--list cannot be combined with run IDs, output, or comparison modes"
            )
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

    if args.benchmarks and (args.benchmark_only or args.aggregate_seeds):
        parser.error(
            "--benchmarks can only be used with the normal two-run comparison"
        )
    if args.benchmark_only and args.aggregate_seeds:
        parser.error("--benchmark-only and --aggregate-seeds are mutually exclusive")

    try:
        if args.aggregate_seeds:
            if len(args.run_ids) < 2:
                parser.error("--aggregate-seeds requires at least two run IDs")
            aggregate = aggregate_seeded_runs(args.run_ids)
            report = format_seeded_aggregate(aggregate)
        elif args.benchmark_only:
            if len(args.run_ids) != 1:
                parser.error("--benchmark-only requires exactly one run ID")
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
                parser.error(
                    "exactly two run IDs are required unless --list, "
                    "--benchmark-only, or --aggregate-seeds is used"
                )

            baseline_run_id, researchfs_run_id = args.run_ids
            report = build_comparison_report(
                baseline_run_id,
                researchfs_run_id,
                include_benchmarks=args.benchmarks,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(report)
        except OSError as exc:
            print(
                f"Error writing report to {args.output}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
