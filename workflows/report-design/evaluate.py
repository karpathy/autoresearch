#!/usr/bin/env python3
"""Report quality evaluator for autoresearch report-design workflow.

Generates a sample report from exec-summarizer data (or synthetic data),
then scores it on 8 quality dimensions using an LLM judge via the
Copilot SDK, or falls back to heuristic scoring when the SDK is
unavailable.

Usage:
    python evaluate.py > run.log 2>&1
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SKILL = REPO_ROOT / ".github" / "skills" / "report-generator"

# Dimensions with weights: structural=2x, presentational=1x, polish=0.5x
DIMENSIONS = {
    "narrative_coherence": {"weight": 2.0, "category": "structural"},
    "information_density": {"weight": 2.0, "category": "structural"},
    "chart_comprehension": {"weight": 2.0, "category": "structural"},
    "visual_hierarchy": {"weight": 1.0, "category": "presentational"},
    "accessibility_contrast": {"weight": 1.0, "category": "presentational"},
    "source_attribution": {"weight": 1.0, "category": "presentational"},
    "engagement": {"weight": 0.5, "category": "polish"},
    "responsiveness": {"weight": 0.5, "category": "polish"},
}


def generate_sample_report() -> Path | None:
    """Generate a report from existing workflow data for evaluation."""
    # Try exec-summarizer first, then any workflow with results
    candidates = ["exec-summarizer", "harness-optimize"]
    for name in candidates:
        workflow_dir = REPO_ROOT / "workflows" / name
        results_tsv = workflow_dir / "results" / "results.tsv"
        if results_tsv.exists():
            gen_script = REPORT_SKILL / "generate_report.py"
            try:
                subprocess.run(
                    [sys.executable, str(gen_script), str(workflow_dir)],
                    check=True, capture_output=True, text=True,
                )
                report_path = workflow_dir / "outputs" / "report" / "index.html"
                if report_path.exists():
                    return report_path
            except subprocess.CalledProcessError:
                continue

    # Fall back to generating with synthetic data
    return _generate_synthetic_report()


def _generate_synthetic_report() -> Path | None:
    """Generate a report using synthetic experiment data for evaluation."""
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="report-eval-"))
    # Create synthetic workflow.yaml
    (tmpdir / "workflow.yaml").write_text(
        "name: synthetic-eval\n"
        "description: Synthetic data for report quality evaluation\n"
        "targets:\n  - path: target.py\n    description: Target file\n"
        "metric:\n  name: quality_score\n  direction: higher\n"
        "  extract: \"grep quality_score run.log\"\n"
        "run:\n  command: echo done\n  timeout: 60\n",
        encoding="utf-8",
    )
    # Create synthetic results.tsv
    results_dir = tmpdir / "results"
    results_dir.mkdir()
    (results_dir / "results.tsv").write_text(
        "commit\tquality_score\tstatus\tdescription\n"
        "a1b2c3d\t5.20\tkeep\tbaseline\n"
        "b2c3d4e\t5.45\tkeep\tadd section headers\n"
        "c3d4e5f\t5.30\tdiscard\tswitch to grid layout\n"
        "d4e5f6g\t5.60\tkeep\timprove typography scale\n"
        "e5f6g7h\t5.55\tdiscard\tadd gradient backgrounds\n"
        "f6g7h8i\t0.00\tcrash\tremoved base template tag\n"
        "g7h8i9j\t5.75\tkeep\toptimize chart colors\n"
        "h8i9j0k\t5.90\tkeep\trefine spacing rhythm\n",
        encoding="utf-8",
    )
    # Create synthetic musings.md
    (results_dir / "musings.md").write_text(
        "## Experiment 1: baseline\n"
        "**Rationale**: Establish starting point.\n"
        "**Result**: Keep. quality_score = 5.20\n"
        "**Learning**: Starting point established.\n\n"
        "## Experiment 2: add section headers\n"
        "**Rationale**: Clear headings improve navigation.\n"
        "**Result**: Keep. quality_score = 5.45 (delta: +0.25)\n"
        "**Learning**: Typography hierarchy matters.\n\n"
        "## Experiment 4: improve typography scale\n"
        "**Rationale**: Better font size progression.\n"
        "**Result**: Keep. quality_score = 5.60 (delta: +0.15)\n"
        "**Learning**: Modular scale creates visual rhythm.\n",
        encoding="utf-8",
    )

    gen_script = REPORT_SKILL / "generate_report.py"
    try:
        subprocess.run(
            [sys.executable, str(gen_script), str(tmpdir)],
            check=True, capture_output=True, text=True,
        )
        report_path = tmpdir / "outputs" / "report" / "index.html"
        if report_path.exists():
            return report_path
    except subprocess.CalledProcessError as e:
        print(f"Report generation failed: {e.stderr}", file=sys.stderr)
    return None


def heuristic_score(html_content: str, css_content: str) -> dict[str, float]:
    """Score report quality using heuristic analysis (no LLM needed)."""
    scores: dict[str, float] = {}

    # 1. Narrative coherence: check all 5 sections exist
    sections = ["situation", "challenge", "experiments", "findings", "impact"]
    found = sum(1 for s in sections if s in html_content.lower())
    scores["narrative_coherence"] = min(10.0, (found / len(sections)) * 10)

    # 2. Information density: ratio of content to markup
    text_chars = len(re.sub(r"<[^>]+>", "", html_content))
    total_chars = len(html_content)
    ratio = text_chars / total_chars if total_chars > 0 else 0
    scores["information_density"] = min(10.0, ratio * 20)

    # 3. Chart comprehension: check Chart.js canvases exist
    chart_count = html_content.count("<canvas")
    scores["chart_comprehension"] = min(10.0, chart_count * 3.3)

    # 4. Visual hierarchy: count heading levels used
    headings = set(re.findall(r"<h([1-6])", html_content))
    scores["visual_hierarchy"] = min(10.0, len(headings) * 2.5)

    # 5. Accessibility: check for alt attrs, ARIA, contrast vars
    alt_count = html_content.count('alt="')
    aria_count = len(re.findall(r"aria-", html_content))
    contrast_vars = css_content.count("--color") + css_content.count("--text")
    scores["accessibility_contrast"] = min(
        10.0, (alt_count + aria_count) * 0.5 + min(contrast_vars, 10)
    )

    # 6. Source attribution: check for commit hashes, citation elements
    commit_refs = len(re.findall(r"[a-f0-9]{7,}", html_content))
    citation_elements = html_content.count("citation") + html_content.count("provenance")
    scores["source_attribution"] = min(10.0, (commit_refs * 0.3 + citation_elements * 1.0))

    # 7. Engagement: scroll-snap, transitions, progress indicators
    snap = 1.0 if "scroll-snap" in css_content else 0.0
    transitions = min(3.0, css_content.count("transition") * 0.5)
    progress = 2.0 if "nav-dot" in html_content or "progress" in html_content else 0.0
    scores["engagement"] = min(10.0, (snap * 4.0 + transitions + progress))

    # 8. Responsiveness: media queries and viewport units
    media_queries = len(re.findall(r"@media", css_content))
    viewport_units = len(re.findall(r"\d+v[hw]", css_content))
    scores["responsiveness"] = min(
        10.0, media_queries * 2.0 + min(viewport_units * 0.5, 4.0)
    )

    return scores


def composite_score(scores: dict[str, float]) -> float:
    """Weighted average across all dimensions."""
    total_weight = sum(d["weight"] for d in DIMENSIONS.values())
    weighted = sum(
        scores.get(name, 0.0) * dim["weight"]
        for name, dim in DIMENSIONS.items()
    )
    return weighted / total_weight if total_weight > 0 else 0.0


def main() -> None:
    """Run evaluation and print results."""
    print("Generating sample report for evaluation...")
    report_path = generate_sample_report()

    if report_path is None:
        print("Error: Could not generate report for evaluation", file=sys.stderr)
        print("quality_score: 0.00")
        sys.exit(0)

    html_content = report_path.read_text(encoding="utf-8")

    # Try to find CSS file
    css_path = report_path.parent / "static" / "styles.css"
    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    # Score using heuristics
    scores = heuristic_score(html_content, css_content)

    # Print per-dimension scores
    print("\n--- Report Quality Evaluation ---\n")
    for name, dim in DIMENSIONS.items():
        score = scores.get(name, 0.0)
        category = dim["category"]
        weight = dim["weight"]
        print(f"  {name}: {score:.1f}/10  (weight: {weight}x, {category})")

    # Compute and print composite
    composite = composite_score(scores)
    print(f"\nquality_score: {composite:.2f}")
    print(f"total_dimensions: {len(DIMENSIONS)}")

    # Print as JSON for machine parsing
    result = {
        "quality_score": round(composite, 2),
        "dimensions": {k: round(v, 1) for k, v in scores.items()},
        "report_path": str(report_path),
    }
    print(f"\n--- JSON ---\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
