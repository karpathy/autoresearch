#!/usr/bin/env python3
"""CLI to compare a baseline agent run against a ResearchFS agent run."""

import argparse
import sys

from evals.nanochat_bpb.metrics import (
    compare_runs,
    format_comparison,
    list_recent_runs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and ResearchFS agent runs."
    )
    parser.add_argument(
        "baseline_run_id",
        type=int,
        nargs="?",
        help="agent_run_id of the baseline run",
    )
    parser.add_argument(
        "researchfs_run_id",
        type=int,
        nargs="?",
        help="agent_run_id of the ResearchFS (loopgraph) run",
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

    if args.baseline_run_id is None or args.researchfs_run_id is None:
        print(
            "Error: baseline_run_id and researchfs_run_id are required unless --list is used.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)

    comparison = compare_runs(args.baseline_run_id, args.researchfs_run_id)
    report = format_comparison(comparison)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
