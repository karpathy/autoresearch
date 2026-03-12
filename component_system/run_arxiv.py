#!/usr/bin/env python3
"""
arXiv search/fetch script using the arxiv Python library.
Supports CLI args for query, id_list, max_results, sort, output format, and PDF download.

In this project the arxiv dependency is provided by uv. Run with:
  uv run python component_system/run_arxiv.py --query "machine learning" --max-results 5
  uv run python component_system/run_arxiv.py --id 1605.08386v1 --output json
  uv run python component_system/run_arxiv.py --query "transformer" --download-dir ./papers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import arxiv
except ImportError:
    print("Install the arxiv package: pip install arxiv", file=sys.stderr)
    sys.exit(1)


def _sort_criterion(s: str) -> arxiv.SortCriterion:
    m = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitteddate": arxiv.SortCriterion.SubmittedDate,
        "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
    }
    key = s.strip().lower().replace(" ", "")
    if key not in m:
        raise ValueError(f"Invalid sort_by: {s}. Choose: relevance, submittedDate, lastUpdatedDate")
    return m[key]


def _sort_order(s: str) -> arxiv.SortOrder:
    m = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending,
    }
    key = s.strip().lower()
    if key not in m:
        raise ValueError(f"Invalid sort_order: {s}. Choose: ascending, descending")
    return m[key]


def _result_to_dict(r: arxiv.Result) -> dict:
    return {
        "entry_id": r.entry_id,
        "title": r.title,
        "summary": (r.summary or "").strip(),
        "authors": [a.name for a in r.authors],
        "published": r.published.isoformat() if r.published else None,
        "updated": r.updated.isoformat() if r.updated else None,
        "primary_category": getattr(r, "primary_category", None) or "",
        "categories": getattr(r, "categories", []) or [],
        "pdf_url": getattr(r, "pdf_url", None) or "",
        "links": [{"href": l.href, "title": getattr(l, "title", None)} for l in (r.links or [])],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search or fetch arXiv papers via the arxiv Python library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Search query (e.g. 'transformer' or 'au:smith AND ti:neural'). Ignored if --id is set.",
    )
    parser.add_argument(
        "--id",
        dest="id_list",
        type=str,
        nargs="+",
        default=None,
        metavar="ARXIV_ID",
        help="One or more arXiv IDs (e.g. 1605.08386v1). If set, --query is ignored.",
    )
    parser.add_argument(
        "--max-results",
        "-n",
        type=int,
        default=10,
        help="Maximum number of results to return.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="relevance",
        choices=["relevance", "submittedDate", "lastUpdatedDate"],
        help="Sort criterion for results.",
    )
    parser.add_argument(
        "--sort-order",
        type=str,
        default="descending",
        choices=["ascending", "descending"],
        help="Sort order.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output format: text (one line per paper) or json.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="If set, download PDF for each result into this directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress (e.g. download paths).",
    )
    args = parser.parse_args()

    if args.id_list:
        search = arxiv.Search(id_list=args.id_list, max_results=len(args.id_list) or None)
    else:
        if not args.query.strip():
            parser.error("Either --query or --id must be provided.")
        sort_by = _sort_criterion(args.sort_by)
        sort_order = _sort_order(args.sort_order)
        search = arxiv.Search(
            query=args.query,
            max_results=args.max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    client = arxiv.Client()
    results = list(client.results(search))

    if args.download_dir:
        d = Path(args.download_dir)
        d.mkdir(parents=True, exist_ok=True)
        for r in results:
            try:
                path = r.download_pdf(dirpath=str(d))
                if args.verbose and path:
                    print(f"Downloaded: {path}", file=sys.stderr)
            except Exception as e:
                print(f"Download failed for {r.entry_id}: {e}", file=sys.stderr)

    if args.output == "json":
        out = [_result_to_dict(r) for r in results]
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for r in results:
            print(r.title)
            print(f"  {r.entry_id}  {getattr(r, 'pdf_url', '') or ''}")
            if r.summary:
                summary = (r.summary or "").strip()
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                print(f"  {summary}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
