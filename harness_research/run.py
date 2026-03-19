"""Entry point for the harness research pipeline."""

import argparse
import logging
import sys

from .evaluator import evaluate_all
from .generator import generate_candidates
from .ranker import find_winner
from .reporter import write_report

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Harness research pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip git operations and PR creation",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="Number of candidate variations to generate (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    log.info("=== Harness Research Pipeline ===")
    log.info("Mode: %s", "dry-run" if args.dry_run else "live")

    # Step 1: Generate candidate harness configs
    log.info("Step 1: Generating %d candidate harness configs...", args.candidates)
    candidates = generate_candidates(n=args.candidates)
    log.info("Generated %d candidates", len(candidates))

    # Step 2: Evaluate each candidate against all scenarios
    log.info("Step 2: Evaluating candidates against environment scenarios...")
    results = evaluate_all(candidates)

    # Step 3: Rank and find winner
    log.info("Step 3: Ranking candidates...")
    ranked, winner, champion_score, best_score = find_winner(results)

    for i, (r, score) in enumerate(ranked, 1):
        log.info("  #%d %s: %.4f — %s", i, r.candidate_id, score, r.description)

    # Step 4: Report
    log.info("Step 4: Writing report...")
    had_winner = write_report(ranked, winner, champion_score, dry_run=args.dry_run)

    if had_winner:
        log.info("Winner found and reported: %s (%.4f > %.4f)", winner.candidate_id, best_score, champion_score)
    else:
        log.info("No improvement over champion (best=%.4f, champion=%.4f)", best_score, champion_score)

    log.info("=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
