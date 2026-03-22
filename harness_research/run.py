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
import os
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
    results = evaluate_all(candidates, real_eval=getattr(args, "real_eval", False))

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
                        help="Use OHB-1 real LLM evaluation via Ollama (gemma3:1b) instead of "
                             "simulation. Also re-ranks top-5 finalists before promotion.")
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
        from pathlib import Path as _Path
        import yaml as _yaml
        from .benchmark import run_benchmark
        from .evaluator import _load_scenarios

        # Load current champion config as the baseline candidate
        ops_dir = _Path(os.environ.get("OPENCASTOR_OPS_DIR",
                                        _Path.home() / "opencastor-ops"))
        champion_path = ops_dir / "harness-research" / "champion.yaml"
        if champion_path.exists():
            raw = _yaml.safe_load(champion_path.read_text()) or {}
            candidate = {
                "id": raw.get("candidate_id", raw.get("id", "champion")),
                "config": raw.get("config", {}),
                "description": raw.get("description", "Current champion"),
            }
        else:
            # Fall back to default harness
            from .generator import _load_seed
            candidate = {
                "id": "default_seed",
                "config": _load_seed(),
                "description": "Default harness seed",
            }

        scenarios = _load_scenarios()
        print(f"OHB-1 Benchmark — model={os.environ.get('OHB_MODEL','gemma3:1b')} "
              f"candidate={candidate['id']} tasks={len(scenarios)}")
        print()
        result = run_benchmark(candidate, scenarios, verbose=True)
        print()
        summary = result.to_dict()
        print(f"Composite score : {summary['composite_score']:.4f}")
        print(f"Tasks passed    : {summary['tasks_passed']}/{summary['tasks_total']}")
        print(f"Success rate    : {summary['success_rate']:.2%}")
        print(f"Safety rate     : {summary['safety_rate']:.2%}")
        print(f"Avg tokens      : {summary['avg_tokens']:.0f}")
        print(f"Avg latency     : {summary['avg_latency_ms']:.0f}ms")
        print()
        print("By environment:")
        for env, stats in summary["by_environment"].items():
            print(f"  {env:<12} {stats['passed']}/{stats['total']} tasks  "
                  f"composite={stats['composite']:.3f}")
        print()
        if args.dry_run:
            print("[dry-run] Full JSON result:")
            print(_json.dumps(summary, indent=2))
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
