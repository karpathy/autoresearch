"""
Autoresearch Report Generator — produces structured experiment reports.

Generates reports after each supervisor round or on demand, summarizing:
- What was tested and why (from director directives)
- Results (kept, discarded, crashed)
- Trend analysis (improving, plateauing, regressing)
- Recommendations for next round

Reports are written to autoresearch/reports/ as timestamped markdown files.

Usage:
    from report import ReportGenerator
    rg = ReportGenerator()
    rg.generate(results_tsv, directives, round_num)
"""

import os
from datetime import UTC, datetime
from pathlib import Path

REPORTS_DIR = Path(os.getenv("AUTORESEARCH_REPORTS_DIR", "reports"))


def _parse_results(results_tsv: str) -> list[dict]:
    """Parse results.tsv text into rows."""
    lines = results_tsv.strip().splitlines()
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        row = dict(zip(headers, parts, strict=False))
        for key in ("val_bpb", "memory_gb"):
            if key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError) as exc:
                    import logging

                    logging.debug("results_parse_float_failed", key=key, error=str(exc))
                    pass
        rows.append(row)
    return rows


def _trend_analysis(results: list[dict]) -> str:
    """Analyze val_bpb trend over kept experiments."""
    keeps = [r for r in results if r.get("status") == "keep"]
    if len(keeps) < 2:
        return "Insufficient data for trend analysis."

    bpbs = []
    for k in keeps:
        try:
            bpbs.append(float(k["val_bpb"]))
        except (ValueError, TypeError, KeyError) as exc:
            import logging

            logging.debug("trend_analysis_val_bpb_failed", error=str(exc))
            pass

    if len(bpbs) < 2:
        return "Insufficient numeric data for trend analysis."

    first_half = bpbs[: len(bpbs) // 2]
    second_half = bpbs[len(bpbs) // 2 :]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)

    total_improvement = bpbs[0] - bpbs[-1]
    recent_improvement = (first_half[-1] if first_half else bpbs[0]) - bpbs[-1]

    if avg_second < avg_first:
        trend = "IMPROVING"
    elif abs(avg_second - avg_first) < 0.0005:
        trend = "PLATEAUING"
    else:
        trend = "REGRESSION (recent experiments worse than earlier ones)"

    return (
        f"Trend: **{trend}**\n"
        f"- Total improvement: {total_improvement:+.6f} val_bpb\n"
        f"- Recent improvement (2nd half): {recent_improvement:+.6f} val_bpb\n"
        f"- Best: {min(bpbs):.6f} | Worst kept: {max(bpbs):.6f}"
    )


class ReportGenerator:
    """Generates markdown experiment reports."""

    def generate(
        self,
        results_tsv: str,
        directives: list[dict] | None = None,
        round_num: int = 0,
        tag: str = "",
    ) -> str:
        """Generate a report and write to disk.

        Args:
            results_tsv: Full contents of results.tsv
            directives: List of director directives issued this round
            round_num: Supervisor round number
            tag: Experiment tag

        Returns:
            Path to the written report file
        """
        results = _parse_results(results_tsv)
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        total = len(results)
        keeps = [r for r in results if r.get("status") == "keep"]
        discards = [r for r in results if r.get("status") == "discard"]
        crashes = [r for r in results if r.get("status") == "crash"]

        best_bpb = None
        for r in keeps:
            try:
                bpb = float(r["val_bpb"])
                if bpb > 0 and (best_bpb is None or bpb < best_bpb):
                    best_bpb = bpb
            except (ValueError, TypeError, KeyError) as exc:
                import logging

                logging.debug("report_gen_best_bpb_failed", error=str(exc))
                pass

        best_bpb_display = f"{best_bpb:.6f}" if best_bpb is not None else "N/A"

        # Build report
        lines = [
            f"# Autoresearch Report — Round {round_num}",
            "",
            f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Tag:** {tag or 'N/A'}",
            f"**Total experiments:** {total}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Kept | {len(keeps)} |",
            f"| Discarded | {len(discards)} |",
            f"| Crashed | {len(crashes)} |",
            f"| Best val_bpb | {best_bpb_display} |",
            f"| Keep rate | {len(keeps) / total * 100:.0f}% |" if total else "| Keep rate | N/A |",
            "",
            "## Trend Analysis",
            "",
            _trend_analysis(results),
            "",
        ]

        # Directives section
        if directives:
            lines.extend(
                [
                    "## Director Directives This Round",
                    "",
                ]
            )
            for i, d in enumerate(directives, 1):
                lines.extend(
                    [
                        f"### Directive {i}: {d.get('category', 'unknown')}",
                        f"- **Hypothesis:** {d.get('hypothesis', 'N/A')}",
                        f"- **Priority:** {d.get('priority', 'N/A')}",
                        f"- **Risk:** {d.get('risk', 'N/A')}",
                        f"- **Rationale:** {d.get('rationale', 'N/A')}",
                        "",
                    ]
                )

        # Full results table
        lines.extend(
            [
                "## Experiment Log",
                "",
                "| # | Commit | val_bpb | Memory GB | Status | Description |",
                "|---|---|---|---|---|---|",
            ]
        )
        for i, r in enumerate(results, 1):
            bpb = r.get("val_bpb", "N/A")
            if isinstance(bpb, float):
                bpb = f"{bpb:.6f}"
            mem = r.get("memory_gb", "N/A")
            if isinstance(mem, float):
                mem = f"{mem:.1f}"
            status = r.get("status", "?")
            status_icon = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(status, "?")
            lines.append(
                f"| {i} | `{r.get('commit', '?')}` | {bpb} | {mem} | "
                f"{status_icon} {status} | {r.get('description', '')} |"
            )

        lines.extend(["", "---", "*Report generated by autoresearch supervisor*", ""])

        report_text = "\n".join(lines)

        # Write to disk
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"report_r{round_num}_{ts}.md"
        report_path = REPORTS_DIR / filename
        report_path.write_text(report_text)

        return str(report_path)

    def generate_from_file(
        self,
        results_file: str = "results.tsv",
        round_num: int = 0,
        tag: str = "",
    ) -> str:
        """Convenience: read results.tsv from disk and generate report."""
        results_tsv = Path(results_file).read_text() if Path(results_file).exists() else ""
        return self.generate(results_tsv, round_num=round_num, tag=tag)
