#!/usr/bin/env python3
"""AutoResearch — Autonomous Bot Optimization Runner

Inspired by karpathy/autoresearch. Reads program.md for a bot,
proposes parameter changes, runs backtests, keeps improvements.

This script is meant to be called by the OpenClaw nightly cron
which spawns a sub-agent to run the research loop.

Usage:
    python3 researcher.py --bot kalshi_weather --experiments 5
    python3 researcher.py --bot arbiter --experiments 3
    python3 researcher.py --all --experiments 2  # 2 per bot, rotating
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROGRAMS_DIR = BASE_DIR / "programs"
RESULTS_DIR = BASE_DIR / "results"
BASELINES_DIR = BASE_DIR / "baselines"
LOGS_DIR = BASE_DIR / "logs"

# Ensure dirs exist
for d in [RESULTS_DIR, BASELINES_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

BOTS = [
    "arbiter",
    "kalshi_weather",
    "polymarket_weather",
    "gex_credit_spread",
    "weekly_condor",
    "king_node_pin",
    "earnings_iv_crush",
    "crypto_15min",
]


def get_program(bot_name: str) -> str:
    """Read the program.md for a bot."""
    path = PROGRAMS_DIR / f"{bot_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No program.md found for {bot_name} at {path}")
    return path.read_text()


def get_results(bot_name: str) -> str:
    """Read existing results TSV for a bot."""
    path = RESULTS_DIR / f"{bot_name}_results.tsv"
    if path.exists():
        return path.read_text()
    return ""


def get_baseline(bot_name: str) -> dict:
    """Read current baseline params for a bot."""
    path = BASELINES_DIR / f"{bot_name}_baseline.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_baseline(bot_name: str, baseline: dict) -> None:
    """Save updated baseline after a successful experiment."""
    path = BASELINES_DIR / f"{bot_name}_baseline.json"
    path.write_text(json.dumps(baseline, indent=2))


def append_result(bot_name: str, row: str) -> None:
    """Append a result row to the bot's TSV."""
    path = RESULTS_DIR / f"{bot_name}_results.tsv"
    if not path.exists():
        # Write header
        path.write_text("date\texperiment\tparam_changed\told_value\tnew_value\t"
                       "metric_before\tmetric_after\tstatus\tdescription\n")
    with open(path, "a") as f:
        f.write(row + "\n")


def log_session(bot_name: str, content: str) -> None:
    """Write nightly research session log."""
    today = datetime.date.today().isoformat()
    path = LOGS_DIR / f"{today}-{bot_name}.md"
    with open(path, "a") as f:
        f.write(content + "\n")


def build_research_prompt(bot_name: str, num_experiments: int) -> str:
    """Build the prompt for the research sub-agent."""
    program = get_program(bot_name)
    results = get_results(bot_name)
    baseline = get_baseline(bot_name)

    prompt = f"""# AutoResearch Session: {bot_name}
Date: {datetime.date.today().isoformat()}

## Your Program
{program}

## Past Experiment Results
{results if results else "No experiments yet — this is the first session. Start by establishing the baseline."}

## Current Baseline Parameters
{json.dumps(baseline, indent=2) if baseline else "No baseline yet — run the current config as baseline first."}

## Instructions

You are an autonomous trading researcher. Your loop:

1. If no baseline exists, run the current bot config and record it as baseline.
2. Review past experiments. Identify what worked and what didn't.
3. Propose ONE parameter change based on the experiment ideas in your program.
4. Run a backtest or evaluate recent paper performance.
5. Compare the metric to baseline.
6. If improved → KEEP (update baseline, log as "keep")
7. If worse or equal → DISCARD (revert, log as "discard")
8. Log the result in TSV format.
9. Repeat for {num_experiments} experiments total.

## Rules
- ONE change at a time. Never change multiple params simultaneously.
- Always measure before and after with the SAME evaluation method.
- If a backtest crashes, log as "crash" and move on. Don't get stuck.
- Prefer experiments that have clear hypotheses over random exploration.
- Read your past results — don't repeat failed experiments.
- Simpler is better. If removing a feature gets equal results, that's a win.
- NEVER STOP until you've completed {num_experiments} experiments.

## Output
For each experiment, output:
1. Hypothesis (what you're testing and why)
2. Change made (exact parameter, old → new)
3. Evaluation result (metric before → after)
4. Decision (keep/discard/crash)
5. TSV row for results file

At the end, output a summary of the session.
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="AutoResearch Bot Optimizer")
    parser.add_argument("--bot", type=str, help="Bot name to research")
    parser.add_argument("--all", action="store_true", help="Research all bots (rotating)")
    parser.add_argument("--experiments", type=int, default=3,
                       help="Number of experiments per bot (default: 3)")
    parser.add_argument("--list", action="store_true", help="List available bots")
    parser.add_argument("--status", action="store_true", help="Show experiment counts")

    args = parser.parse_args()

    if args.list:
        print("Available bots:")
        for b in BOTS:
            prog = PROGRAMS_DIR / f"{b}.md"
            results = RESULTS_DIR / f"{b}_results.tsv"
            n_experiments = 0
            if results.exists():
                n_experiments = len(results.read_text().strip().split("\n")) - 1
            status = "✅" if prog.exists() else "❌"
            print(f"  {status} {b} ({n_experiments} experiments)")
        return

    if args.status:
        print("AutoResearch Status:")
        for b in BOTS:
            results = RESULTS_DIR / f"{b}_results.tsv"
            baseline = BASELINES_DIR / f"{b}_baseline.json"
            n_exp = 0
            if results.exists():
                n_exp = len(results.read_text().strip().split("\n")) - 1
            has_baseline = "✅" if baseline.exists() else "❌"
            print(f"  {b}: {n_exp} experiments, baseline: {has_baseline}")
        return

    if args.all:
        # Rotate through bots — pick 2 per night
        today = datetime.date.today()
        day_index = today.toordinal() % len(BOTS)
        selected = [BOTS[day_index], BOTS[(day_index + 1) % len(BOTS)]]
        print(f"Tonight's research targets: {', '.join(selected)}")
        for bot in selected:
            prompt = build_research_prompt(bot, args.experiments)
            print(f"\n{'='*60}")
            print(f"RESEARCH PROMPT FOR: {bot}")
            print(f"{'='*60}")
            print(prompt)
    elif args.bot:
        if args.bot not in BOTS:
            print(f"Unknown bot: {args.bot}. Available: {', '.join(BOTS)}")
            sys.exit(1)
        prompt = build_research_prompt(args.bot, args.experiments)
        print(prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
