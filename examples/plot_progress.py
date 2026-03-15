#!/usr/bin/env python3
"""
Generate a progress chart from an evaluation history database.

Thin wrapper around autoanything.plotting.generate_chart.

Usage:
    python examples/plot_progress.py evaluator/history.db
    python examples/plot_progress.py evaluator/history.db -o chart.png
    python examples/plot_progress.py path/to/history.db --title "My Run"
"""

import argparse
import os

from autoanything.plotting import generate_chart


def main():
    parser = argparse.ArgumentParser(
        description="Generate progress chart from evaluation history"
    )
    parser.add_argument("db_path", help="Path to history.db")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: <db_dir>/progress.png)")
    parser.add_argument("--title", default=None, help="Custom chart title")
    parser.add_argument("--direction", default="minimize",
                        choices=["minimize", "maximize"])
    parser.add_argument("--score-label", default="Score",
                        help="Y-axis label (default: Score)")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(args.db_path) or ".", "progress.png"
    )
    generate_chart(args.db_path, output, args.title, args.direction,
                   args.score_label)
    print(f"Chart saved to {output}")


if __name__ == "__main__":
    main()
