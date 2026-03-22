"""Entry point for the harness research pipeline.

Supports three modes:
  1. Standard (no flags): generate + evaluate candidates using Gemini, rank, report
  2. --hardware-tier <tier>: same as (1) but seeded and prompted for a specific hardware tier
  3. --fleet: read fleet contribute_eval submissions from Firestore instead of generating candidates

Flags:
  --dry-run           Use synthetic data, skip git/Firestore writes
  --candidates N      Number of Gemini candidates to generate (default 5)
  --hardware-tier T   Target a specific hardware tier (e.g. pi5-hailo8l)
  --all-tiers         Loop over all known hardware tiers
  --fleet             Use fleet submission data from Firestore instead of generating candidates
  --fleet-lookback N  Days of fleet data to include (default 7)
  --promote           Also run the promoter after a winning config is found
  --push-to-queue     Push generated candidates to Firestore queue for fleet eval (skips local eval)
  --max-evaluations N Max evaluations per candidate when pushing to queue (default 5)
  --model-research    Expand each candidate with all RESEARCH_MODELS as a new dimension
  --model-id MODEL    Scope results to a specific model (e.g. gemini-2.5-flash)
"""

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def _push_candidates_to_queue_for_tier(args, hardware_tier: str | None) -> int:
    """Generate candidates and push to Firestore queue; return count pushed."""
    from .generator import generate_candidates
    from .queue_manager import get_queue_status, push_candidates_to_queue

    tier_label = hardware_tier or "generic"
    effective_tier = hardware_tier or "pi5-8gb"

    log.info("=== Push-to-queue for tier: %s ===", tier_label)
    log.info("Step 1: Generating %d candidates (tier=%s)...", args.candidates, tier_label)
    candidates = generate_candidates(
        n=args.candidates,
        dry_run=args.dry_run,
        hardware_tier=hardware_tier,
        model_research=args.model_research,
    )
    log.info("Generated %d candidates", len(candidates))

    pushed = push_candidates_to_queue(
        candidates,
        hardware_tier=effective_tier,
        max_evaluations=args.max_evaluations,
        model_id=args.model_id,
    )

    status = get_queue_status(hardware_tier=effective_tier)
    tier_status = status.get(effective_tier, {})
    log.info(
        "Pushed %d candidates to Firestore queue for tier %s. "
        "Queue: pending=%d assigned=%d completed=%d total=%d. "
        "Robots will evaluate via 'castor contribute'.",
        pushed,
        effective_tier,
        tier_status.get("pending", 0),
        tier_status.get("assigned", 0),
        tier_status.get("completed", 0),
        tier_status.get("total", 0),
    )
    print(
        f"Pushed {pushed} candidates to Firestore queue for tier {effective_tier}. "
        f"Robots will evaluate via 'castor contribute'."
    )
    return pushed


def _run_for_tier(args, hardware_tier: str | None) -> bool:
    """Run the full pipeline for one hardware tier (or generic). Returns True if winner found."""
    from .evaluator import evaluate_all
    from .generator import generate_candidates
    from .ranker import find_winner
    from .reporter import write_report

    tier_label = hardware_tier or "generic"
    log.info("=== Pipeline for tier: %s ===", tier_label)

    log.info("Step 1: Generating %d candidates (tier=%s)...", args.candidates, tier_label)
    candidates = generate_candidates(
        n=args.candidates,
        dry_run=args.dry_run,
        hardware_tier=hardware_tier,
        model_research=args.model_research,
    )
    log.info("Generated %d candidates", len(candidates))

    log.info("Step 2: Evaluating candidates...")
    results = evaluate_all(candidates)

    log.info("Step 3: Ranking candidates...")
    ranked, winner, champion_score, best_score = find_winner(
        results, hardware_tier=hardware_tier, model_id=args.model_id,
    )

    for i, (r, score) in enumerate(ranked, 1):
        log.info("  #%d %s: %.4f — %s", i, r.candidate_id, score, r.description)

    log.info("Step 4: Writing report...")
    had_winner = write_report(
        ranked, winner, champion_score,
        dry_run=args.dry_run,
        hardware_tier=hardware_tier,
        model_id=args.model_id,
    )

    if had_winner:
        log.info("Winner: %s (%.4f > %.4f)", winner.candidate_id, best_score, champion_score)

        # OHB-1 real eval: validate top candidates before promoting
        if getattr(args, "real_eval", False):
            log.info("Step 4b: OHB-1 real LLM eval on top-5 finalists...")
            from .benchmark import validate_finalists
            top_candidates = [{"id": r.candidate_id, "config": r.config,
                               "description": r.description} for r, _ in ranked[:5]]
            obh_results = validate_finalists(top_candidates, top_n=5)
            # Use OHB-1 winner if it beats simulation winner
            if obh_results and obh_results[0].candidate_id != winner.candidate_id:
                log.info(
                    "OHB-1 reranked: simulation winner=%s (%.4f) → "
                    "OHB-1 winner=%s (%.4f)",
                    winner.candidate_id, best_score,
                    obh_results[0].candidate_id, obh_results[0].composite_score,
                )
                winner_id = obh_results[0].candidate_id
                # Find the winner result object from ranked list
                for r, score in ranked:
                    if r.candidate_id == winner_id:
                        winner = r
                        break
            log.info(
                "OHB-1 validated champion: %s composite=%.4f success=%.2f safety=%.2f cost=$%.5f",
                obh_results[0].candidate_id,
                obh_results[0].composite_score,
                obh_results[0].success_rate,
                obh_results[0].safety_rate,
                obh_results[0].estimated_cost_usd,
            )

        if args.promote:
            from .promoter import promote
            log.info("Step 5: Promoting winner to OpenCastor...")
            promote(dry_run=args.dry_run, hardware_tier=hardware_tier, model_id=args.model_id)
    else:
        log.info("No improvement (best=%.4f, champion=%.4f)", best_score, champion_score)

    return had_winner


def _run_fleet_mode(args, hardware_tier: str | None) -> dict[str, bool]:
    """Run fleet-eval aggregation mode. Returns dict of tier → had_winner."""
    from .contribute_eval import run_fleet_research

    results = run_fleet_research(
        hardware_tier=hardware_tier,
        dry_run=args.dry_run,
        lookback_days=args.fleet_lookback,
    )

    if args.promote:
        from .promoter import promote, promote_all_profiles
        if hardware_tier:
            promote(dry_run=args.dry_run, hardware_tier=hardware_tier)
        else:
            promote_all_profiles(dry_run=args.dry_run)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harness research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data, skip git/Firestore writes")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Number of candidate variations to generate (default: 5)")
    parser.add_argument("--hardware-tier",
                        help="Target a specific hardware tier (e.g. pi5-hailo8l, pi4-8gb)")
    parser.add_argument("--all-tiers", action="store_true",
                        help="Loop over all known hardware tiers")
    parser.add_argument("--fleet", action="store_true",
                        help="Use fleet Firestore submissions instead of generating candidates")
    parser.add_argument("--fleet-lookback", type=int, default=7,
                        help="Days of fleet data to include (default: 7)")
    parser.add_argument("--promote", action="store_true",
                        help="Run promoter after a winner is found")
    parser.add_argument("--push-to-queue", action="store_true",
                        help="Push generated candidates to Firestore queue for fleet eval (skips local eval)")
    parser.add_argument("--max-evaluations", type=int, default=5,
                        help="Max evaluations per candidate when pushing to queue (default: 5)")
    parser.add_argument("--model-research", action="store_true",
                        help="Expand each candidate with all RESEARCH_MODELS as a new dimension")
    parser.add_argument("--model-id",
                        help="Scope results to a specific model (e.g. gemini-2.5-flash)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Show the harness research dashboard and exit")
    parser.add_argument("--search-space-status", action="store_true",
                        help="Print search space size, explored count/pct, and champion as JSON")
    parser.add_argument("--real-eval", action="store_true",
                        help="Run OHB-1 real LLM benchmark on top-5 finalists before promotion "
                             "(requires GEMINI_API_KEY; ~$0.01 per run)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run OHB-1 benchmark on a single candidate config and print results. "
                             "Pass --candidates 1 and a specific config via --hardware-tier.")
    args = parser.parse_args()

    if args.dashboard:
        from .dashboard import main as dashboard_main
        return dashboard_main()

    if args.search_space_status:
        import json as _json
        from .search_space import status_dict
        print(_json.dumps(status_dict(), indent=2))
        return 0

    if args.benchmark:
        import json as _json
        from .benchmark import evaluate_candidate_real
        from .generator import generate_candidates
        # Evaluate the current champion config as a baseline
        candidates = generate_candidates(n=1, dry_run=True,
                                         hardware_tier=args.hardware_tier)
        if not candidates:
            print("No candidates generated")
            return 1
        cand = candidates[0]
        print(f"Running OHB-1 benchmark on candidate: {cand['id']}")
        result = evaluate_candidate_real(cand)
        print(_json.dumps(result.summary(), indent=2))
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from .generator import HARDWARE_TIERS

    log.info("=== Harness Research Pipeline ===")
    log.info("Mode: %s | dry-run: %s", "fleet" if args.fleet else "generate", args.dry_run)

    if args.all_tiers:
        tiers = HARDWARE_TIERS
        log.info("Running for all %d hardware tiers: %s", len(tiers), tiers)
    elif args.hardware_tier:
        tiers = [args.hardware_tier]
        log.info("Running for hardware tier: %s", args.hardware_tier)
    else:
        tiers = [None]  # generic
        log.info("Running generic (no hardware tier)")

    any_winner = False
    for tier in tiers:
        if args.push_to_queue:
            _push_candidates_to_queue_for_tier(args, hardware_tier=tier)
            # skip local eval — robots evaluate via 'castor contribute'
        elif args.fleet:
            results = _run_fleet_mode(args, hardware_tier=tier)
            any_winner = any_winner or any(results.values())
        else:
            had = _run_for_tier(args, hardware_tier=tier)
            any_winner = any_winner or had

    if args.push_to_queue:
        log.info("=== Push-to-queue complete — robots will evaluate via 'castor contribute' ===")
        return 0

    log.info("=== Pipeline complete — winner found: %s ===", any_winner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
