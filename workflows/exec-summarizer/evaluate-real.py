#!/usr/bin/env python3
"""
Evaluation wrapper for real-world articles.
Runs the same evaluation logic as evaluate.py but against articles-real.json.

Usage: python evaluate-real.py > run.log 2>&1
"""

import json
import sys
from pathlib import Path

# Patch the articles path before importing evaluate logic
script_dir = Path(__file__).parent
real_articles_path = script_dir / "articles-real.json"

if not real_articles_path.exists():
    print(f"Error: articles-real.json not found at {real_articles_path}", file=sys.stderr)
    sys.exit(1)

# Import and run evaluate with real articles
import evaluate

async def main():
    """Run evaluation against real articles."""
    prompt_path = script_dir / "prompt.txt"

    if not prompt_path.exists():
        print(f"Error: prompt.txt not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    articles = json.loads(real_articles_path.read_text(encoding="utf-8"))

    if not articles:
        print("Error: No articles found in articles-real.json", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(articles)} real-world articles...", file=sys.stderr)
    print(f"Prompt: {prompt_path}", file=sys.stderr)
    print(f"Articles: {real_articles_path}", file=sys.stderr)
    print(file=sys.stderr)

    # Generate summaries
    print("Generating summaries...", file=sys.stderr)
    summaries = []
    for article in articles:
        result = await evaluate.generate_summary(system_prompt, article)
        summaries.append(result)
        status = "OK" if not result["error"] else "FAILED"
        print(f"  {result['article_id']}: {status}", file=sys.stderr)

    # Score summaries
    print("Scoring summaries...", file=sys.stderr)
    scores = await evaluate.score_summaries(summaries)

    # Output results
    print("\n--- Article Results ---")
    import re
    total_score = 0.0
    dim_totals = {"conciseness": 0, "relevance": 0, "provenance": 0, "ecq": 0}
    for score in scores:
        dims = [score["conciseness"], score["relevance"], score["provenance"], score["ecq"]]
        nonzero = [d for d in dims if d > 0]
        if nonzero:
            avg = len(nonzero) / sum(1.0 / d for d in nonzero)
        else:
            avg = 0.0
        total_score += avg
        for k in dim_totals:
            dim_totals[k] += score[k]
        print(
            f"{score['article_id']}: summary_score={avg:.1f} "
            f"(conciseness={score['conciseness']} "
            f"relevance={score['relevance']} "
            f"provenance={score['provenance']} "
            f"ecq={score['ecq']})"
        )

    n = len(scores) if scores else 1
    quality_score = total_score / n

    print("\n--- Dimension Averages ---")
    for k in ["conciseness", "relevance", "provenance", "ecq"]:
        print(f"avg_{k}: {dim_totals[k] / n:.2f}")

    print("\n---")
    print(f"quality_score: {quality_score:.2f}")
    print(f"total_articles: {len(articles)}")

    # Print summaries for review
    print("\n--- Generated Summaries ---")
    for item in summaries:
        print(f"\n[{item['article_id']}]")
        print(item['summary'] if not item['error'] else f"ERROR: {item['error']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
