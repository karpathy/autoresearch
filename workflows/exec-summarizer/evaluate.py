#!/usr/bin/env python3
"""
Evaluation harness for exec-summarizer workflow.

Loads system prompt and test articles, generates summaries via Copilot SDK,
scores them with a rubric, and outputs quality metrics.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from copilot import CopilotClient

MODEL = "gpt-5.4-mini"

SCORING_RUBRIC = """You are evaluating executive summaries of articles.

Rate each summary on a 1-10 scale across 4 dimensions:
1. Conciseness (1-10): Is it 2-3 sentences? No filler, no unnecessary words.
2. Executive Relevance (1-10): Does it surface impact, implications, and "so what?" for decision-makers?
3. Source Provenance (1-10): Are data points, figures, and quotes attributed with URI links?
4. ECQ Alignment (1-10): Does the framing connect to Executive Core Qualifications (driving efficiency, achieving results, advancing merit, leading people, commitment to rule of law)?

For each summary, respond with ONLY a JSON array. Each element must have these keys:
"article_id", "conciseness", "relevance", "provenance", "ecq"

Example: [{"article_id": "article-1", "conciseness": 8, "relevance": 7, "provenance": 6, "ecq": 8}]

Respond with ONLY the JSON array. No other text.
"""


def approve_all(request, context):
    """Auto-approve all permission requests."""
    return {"kind": "approved", "rules": []}


async def call_llm(prompt: str) -> str:
    """Call LLM via Copilot SDK. Creates a fresh client+session per call."""
    client = CopilotClient()
    await client.start()
    try:
        session = await client.create_session({
            "model": MODEL,
            "on_permission_request": approve_all,
        })

        response_parts = []
        done = asyncio.Event()

        def on_event(event):
            if event.type.value == "assistant.message":
                if hasattr(event.data, 'content') and event.data.content:
                    response_parts.append(event.data.content)
            elif event.type.value == "session.idle":
                done.set()

        session.on(on_event)
        await session.send({"prompt": prompt})

        try:
            await asyncio.wait_for(done.wait(), timeout=120)
        except asyncio.TimeoutError:
            print("Warning: LLM call timed out after 120s", file=sys.stderr)

        return "".join(response_parts)
    finally:
        await client.stop()


async def generate_summary(system_prompt: str, article: dict) -> dict:
    """Generate a summary for one article."""
    article_id = article.get("id", "unknown")
    try:
        prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

---

Article Title: {article.get('title', 'Untitled')}
Source: {article.get('source', 'Unknown')}
URL: {article.get('url', 'No URL')}

{article.get('text', '')}

---

Provide an executive summary following the system instructions above."""

        summary = await call_llm(prompt)
        return {"article_id": article_id, "summary": summary.strip(), "error": None}
    except Exception as e:
        print(f"Error generating summary for {article_id}: {e}", file=sys.stderr)
        return {"article_id": article_id, "summary": "", "error": str(e)}


async def score_summaries(summaries: list) -> list:
    """Score all summaries in a single batch call."""
    batch_content = []
    for item in summaries:
        if item["error"]:
            batch_content.append(f"\n=== {item['article_id']} ===\n[GENERATION FAILED]")
        else:
            batch_content.append(f"\n=== {item['article_id']} ===\n{item['summary']}")

    scoring_prompt = SCORING_RUBRIC + "\n\nSummaries to evaluate:" + "".join(batch_content)

    try:
        response = await call_llm(scoring_prompt)

        json_text = response.strip()
        match = re.search(r'\[.*\]', json_text, re.DOTALL)
        if match:
            json_text = match.group(0)

        scores = json.loads(json_text)
        score_map = {s["article_id"]: s for s in scores}
        results = []

        for item in summaries:
            aid = item["article_id"]
            if aid in score_map and not item["error"]:
                results.append(score_map[aid])
            else:
                results.append({"article_id": aid, "conciseness": 0, "relevance": 0, "provenance": 0, "ecq": 0})

        return results

    except Exception as e:
        print(f"Error scoring summaries: {e}", file=sys.stderr)
        return [
            {"article_id": item["article_id"], "conciseness": 0, "relevance": 0, "provenance": 0, "ecq": 0}
            for item in summaries
        ]


async def main():
    """Main evaluation workflow."""
    script_dir = Path(__file__).parent
    prompt_path = script_dir / "prompt.txt"
    articles_path = script_dir / "articles.json"

    if not prompt_path.exists():
        print(f"Error: prompt.txt not found at {prompt_path}", file=sys.stderr)
        sys.exit(1)

    system_prompt = prompt_path.read_text(encoding="utf-8")

    if not articles_path.exists():
        print(f"Error: articles.json not found at {articles_path}", file=sys.stderr)
        sys.exit(1)

    articles = json.loads(articles_path.read_text(encoding="utf-8"))
    if not articles:
        print("Error: No articles found in articles.json", file=sys.stderr)
        sys.exit(1)

    # Generate summaries
    print("Generating summaries...", file=sys.stderr)
    summaries = []
    for article in articles:
        result = await generate_summary(system_prompt, article)
        summaries.append(result)
        status = "OK" if not result["error"] else "FAILED"
        print(f"  {result['article_id']}: {status}", file=sys.stderr)

    # Score summaries
    print("Scoring summaries...", file=sys.stderr)
    scores = await score_summaries(summaries)

    # Output results
    print("\n--- Article Results ---")
    total_score = 0.0
    for score in scores:
        avg = (score["conciseness"] + score["relevance"] + score["provenance"] + score["ecq"]) / 4.0
        total_score += avg
        print(
            f"{score['article_id']}: summary_score={avg:.1f} "
            f"(conciseness={score['conciseness']} "
            f"relevance={score['relevance']} "
            f"provenance={score['provenance']} "
            f"ecq={score['ecq']})"
        )

    quality_score = total_score / len(scores) if scores else 0.0
    print("\n---")
    print(f"quality_score: {quality_score:.2f}")
    print(f"total_articles: {len(articles)}")


if __name__ == "__main__":
    asyncio.run(main())
