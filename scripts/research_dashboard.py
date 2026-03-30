from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import webbrowser
from collections import Counter
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
DATASET_ROOT = RESULTS_ROOT / "datasets"
DEFAULT_DASHBOARD_PATH = RESULTS_ROOT / "dashboard" / "index.html"
MARKDOWN_SEPARATOR_RE = re.compile(r"^\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and serve autoresearch run artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Print a terminal summary of runs and datasets.")
    summary.add_argument("--limit-runs", type=int, default=10, help="Maximum recent runs to print.")
    summary.add_argument("--limit-datasets", type=int, default=5, help="Maximum recent datasets to print.")

    export_html = subparsers.add_parser("export-html", help="Render the dashboard HTML to disk.")
    export_html.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DASHBOARD_PATH,
        help="Output HTML file path.",
    )

    serve = subparsers.add_parser("serve", help="Render the dashboard and serve the repo over HTTP.")
    serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    serve.add_argument("--port", type=int, default=8765, help="HTTP port.")
    serve.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DASHBOARD_PATH,
        help="Output HTML file path before serving.",
    )
    serve.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_timestamp(value: str | None) -> str:
    dt = parse_iso8601(value)
    if dt is None:
        return "n/a"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def shorten_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def relative_link(base_file: Path, target: Path) -> str:
    return os.path.relpath(target, start=base_file.parent).replace("\\", "/")


def analyze_markdown_report(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"table_rows": 0, "not_confirmed_rows": 0, "category_headings": 0}

    table_rows = 0
    not_confirmed_rows = 0
    category_headings = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            category_headings += 1
        if not line.startswith("|"):
            continue
        if line.startswith("| # |") or MARKDOWN_SEPARATOR_RE.match(line):
            continue
        table_rows += 1
        if "Not confirmed" in line:
            not_confirmed_rows += 1
    return {
        "table_rows": table_rows,
        "not_confirmed_rows": not_confirmed_rows,
        "category_headings": category_headings,
    }


def count_csv_rows(csv_path: Path) -> tuple[int, int, Counter[str]]:
    row_count = 0
    unconfirmed = 0
    per_phase: Counter[str] = Counter()
    if not csv_path.exists():
        return row_count, unconfirmed, per_phase
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            phase_number = (row.get("phase_number") or "").strip()
            if phase_number:
                per_phase[phase_number] += 1
            if (row.get("image_url") or "").strip() == "Not confirmed":
                unconfirmed += 1
    return row_count, unconfirmed, per_phase


def normalize_status(metadata: dict[str, Any], report_path: Path) -> str:
    exit_code = metadata.get("exit_code")
    finished_at = metadata.get("finished_at")
    if exit_code == 0 and finished_at and report_path.exists():
        return "completed"
    if exit_code not in (None, 0):
        return "failed"
    if finished_at and not report_path.exists():
        return "missing-report"
    if report_path.exists():
        return "partial"
    return "incomplete"


def collect_runs(results_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not results_root.exists():
        return runs

    for run_json in sorted(results_root.glob("*/run.json")):
        metadata = load_json(run_json)
        output_dir = run_json.parent
        final_report = Path(metadata.get("final_report") or output_dir / "final_report.md")
        started_at = parse_iso8601(metadata.get("started_at"))
        finished_at = parse_iso8601(metadata.get("finished_at"))
        duration_seconds = None
        if started_at and finished_at:
            duration_seconds = (finished_at - started_at).total_seconds()

        report_stats = analyze_markdown_report(final_report)
        vars_payload = metadata.get("vars") or {}
        runs.append(
            {
                "folder_name": output_dir.name,
                "output_dir": str(output_dir),
                "output_dir_rel": shorten_path(output_dir),
                "name": metadata.get("name") or output_dir.name,
                "tag": metadata.get("tag") or "",
                "spec_path": metadata.get("spec_path") or "",
                "spec_name": Path(str(metadata.get("spec_path") or "")).name or "n/a",
                "model": metadata.get("model") or "default",
                "search": bool(metadata.get("search")),
                "phase": vars_payload.get("phase") or "",
                "depth": vars_payload.get("depth") or "",
                "status": normalize_status(metadata, final_report),
                "exit_code": metadata.get("exit_code"),
                "started_at": metadata.get("started_at"),
                "finished_at": metadata.get("finished_at"),
                "duration_seconds": duration_seconds,
                "duration_label": format_duration(duration_seconds),
                "table_rows": report_stats["table_rows"],
                "not_confirmed_rows": report_stats["not_confirmed_rows"],
                "category_headings": report_stats["category_headings"],
                "final_report": str(final_report),
                "final_report_rel": shorten_path(final_report),
                "log_path": metadata.get("log_path") or "",
            }
        )

    runs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    return runs


def collect_datasets(dataset_root: Path) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    if not dataset_root.exists():
        return datasets

    for manifest_path in sorted(dataset_root.glob("*/manifest.json")):
        dataset_dir = manifest_path.parent
        manifest = load_json(manifest_path)
        csv_path = dataset_dir / "heavy_construction_equipment_dataset.csv"
        row_count, csv_unconfirmed, per_phase = count_csv_rows(csv_path)
        final_unconfirmed = manifest.get("final_unconfirmed_image_count")
        if final_unconfirmed is None:
            final_unconfirmed = manifest.get("unconfirmed_image_count")
        if final_unconfirmed is None:
            final_unconfirmed = csv_unconfirmed
        if not row_count:
            row_count = int(manifest.get("row_count") or 0)
        coverage_pct = round(((row_count - final_unconfirmed) / row_count) * 100, 1) if row_count else 0.0
        phase_runs = manifest.get("phase_runs") or []
        completed_phases = sum(1 for run in phase_runs if run.get("row_count") is not None and not run.get("error"))
        failed_phases = sum(1 for run in phase_runs if run.get("error"))
        datasets.append(
            {
                "folder_name": dataset_dir.name,
                "dataset_dir": str(dataset_dir),
                "dataset_dir_rel": shorten_path(dataset_dir),
                "created_at": manifest.get("created_at"),
                "created_label": format_timestamp(manifest.get("created_at")),
                "row_count": row_count,
                "unconfirmed_image_count": int(final_unconfirmed or 0),
                "initial_unconfirmed_image_count": manifest.get("initial_unconfirmed_image_count"),
                "coverage_pct": coverage_pct,
                "completed_phases": completed_phases or len(per_phase),
                "failed_phases": failed_phases,
                "phase_rows": dict(sorted(per_phase.items())),
                "csv_path": str(csv_path),
                "csv_path_rel": shorten_path(csv_path),
                "json_path_rel": shorten_path(dataset_dir / "heavy_construction_equipment_dataset.json"),
                "manifest_path_rel": shorten_path(manifest_path),
                "combined_report_rel": shorten_path(
                    dataset_dir / ("combined_reports.md" if (dataset_dir / "combined_reports.md").exists() else "combined_report.md")
                ),
                "image_followup_enabled": bool(manifest.get("image_followup_enabled")),
                "image_followup_runs": len(manifest.get("image_followup_runs") or []),
                "source_report": manifest.get("source_report") or "",
            }
        )

    datasets.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return datasets


def build_dashboard_model(results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    runs = collect_runs(results_root)
    datasets = collect_datasets(results_root / "datasets")
    completed_runs = [run for run in runs if run["status"] == "completed"]
    failed_runs = [run for run in runs if run["status"] == "failed"]
    incomplete_runs = [run for run in runs if run["status"] not in {"completed", "failed"}]
    latest_dataset = datasets[0] if datasets else None
    latest_run = runs[0] if runs else None

    model = {
        "generated_at": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "results_root": str(results_root),
        "overview": {
            "run_count": len(runs),
            "completed_run_count": len(completed_runs),
            "failed_run_count": len(failed_runs),
            "incomplete_run_count": len(incomplete_runs),
            "dataset_count": len(datasets),
            "latest_run_status": latest_run["status"] if latest_run else "none",
            "latest_dataset_rows": latest_dataset["row_count"] if latest_dataset else 0,
            "latest_dataset_unconfirmed": latest_dataset["unconfirmed_image_count"] if latest_dataset else 0,
        },
        "runs": runs,
        "datasets": datasets,
    }
    return model


def render_text_summary(model: dict[str, Any], *, limit_runs: int, limit_datasets: int) -> str:
    lines: list[str] = []
    overview = model["overview"]
    lines.append("Autoresearch Dashboard")
    lines.append(f"Generated: {format_timestamp(model['generated_at'])}")
    lines.append(
        "Runs: {run_count} total, {completed_run_count} completed, {failed_run_count} failed, {incomplete_run_count} incomplete".format(
            **overview
        )
    )
    lines.append(
        "Datasets: {dataset_count} total, latest rows={latest_dataset_rows}, latest unresolved images={latest_dataset_unconfirmed}".format(
            **overview
        )
    )

    lines.append("")
    lines.append("Recent Runs")
    for run in model["runs"][:limit_runs]:
        lines.append(
            "- {folder_name} [{status}] model={model} phase={phase} depth={depth} rows={table_rows} unresolved={not_confirmed_rows} duration={duration_label}".format(
                **run
            )
        )
        lines.append(f"  {run['output_dir_rel']}")

    lines.append("")
    lines.append("Recent Datasets")
    for dataset in model["datasets"][:limit_datasets]:
        phase_summary = ", ".join(f"{phase}:{count}" for phase, count in dataset["phase_rows"].items()) or "n/a"
        lines.append(
            "- {folder_name} rows={row_count} unresolved={unconfirmed_image_count} coverage={coverage_pct}% phases={completed_phases} followup_runs={image_followup_runs}".format(
                **dataset
            )
        )
        lines.append(f"  {dataset['dataset_dir_rel']} phase_rows={phase_summary}")

    return "\n".join(lines)


def html_table_rows(rows: list[str]) -> str:
    return "\n".join(rows)


def render_dashboard_html(model: dict[str, Any], output_path: Path) -> str:
    overview = model["overview"]

    run_rows: list[str] = []
    for run in model["runs"]:
        report_link = ""
        final_report = Path(run["final_report"])
        if final_report.exists():
            report_link = f'<a href="{html.escape(relative_link(output_path, final_report))}">report</a>'
        output_link = f'<a href="{html.escape(relative_link(output_path, Path(run["output_dir"])))}">{html.escape(run["folder_name"])}</a>'
        status_class = html.escape(run["status"])
        run_rows.append(
            "<tr>"
            f"<td><span class='status {status_class}'>{html.escape(run['status'])}</span></td>"
            f"<td>{output_link}</td>"
            f"<td>{html.escape(run['model'])}</td>"
            f"<td>{html.escape(run['phase'] or 'n/a')}</td>"
            f"<td>{html.escape(run['depth'] or 'n/a')}</td>"
            f"<td>{run['table_rows']}</td>"
            f"<td>{run['not_confirmed_rows']}</td>"
            f"<td>{html.escape(run['duration_label'])}</td>"
            f"<td>{html.escape(format_timestamp(run['started_at']))}</td>"
            f"<td>{report_link}</td>"
            "</tr>"
        )

    dataset_rows: list[str] = []
    for dataset in model["datasets"]:
        csv_link = ""
        csv_path = Path(dataset["csv_path"])
        if csv_path.exists():
            csv_link = f'<a href="{html.escape(relative_link(output_path, csv_path))}">csv</a>'
        report_path = REPO_ROOT / dataset["combined_report_rel"]
        report_link = ""
        if report_path.exists():
            report_link = f'<a href="{html.escape(relative_link(output_path, report_path))}">report</a>'
        phase_summary = ", ".join(f"{phase}:{count}" for phase, count in dataset["phase_rows"].items()) or "n/a"
        coverage_pct = max(0.0, min(100.0, float(dataset["coverage_pct"])))
        dataset_rows.append(
            "<tr>"
            f"<td><a href='{html.escape(relative_link(output_path, Path(dataset['dataset_dir'])))}'>{html.escape(dataset['folder_name'])}</a></td>"
            f"<td>{dataset['row_count']}</td>"
            f"<td>{dataset['unconfirmed_image_count']}</td>"
            f"<td><div class='bar'><span style='width:{coverage_pct:.1f}%'></span></div><div class='bar-label'>{coverage_pct:.1f}%</div></td>"
            f"<td>{dataset['completed_phases']}</td>"
            f"<td>{dataset['image_followup_runs']}</td>"
            f"<td>{html.escape(phase_summary)}</td>"
            f"<td>{html.escape(dataset['created_label'])}</td>"
            f"<td>{csv_link} {report_link}</td>"
            "</tr>"
        )

    generated_label = format_timestamp(model["generated_at"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autoresearch Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1d1d1b;
      --muted: #5d5a52;
      --line: #d5ccbc;
      --accent: #946846;
      --accent-soft: #ead9c6;
      --success: #2f7d4a;
      --warn: #b16a12;
      --danger: #a83232;
      --incomplete: #7d5d8f;
      --shadow: 0 16px 40px rgba(59, 43, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(148, 104, 70, 0.12), transparent 34%),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(148, 104, 70, 0.15), rgba(255, 255, 255, 0.9));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(30px, 4vw, 52px);
      letter-spacing: -0.03em;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .card .label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
      margin-top: 18px;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .panel-header h2 {{
      margin: 0;
      font-size: 22px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .status {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .status.completed {{ background: rgba(47, 125, 74, 0.13); color: var(--success); }}
    .status.failed {{ background: rgba(168, 50, 50, 0.13); color: var(--danger); }}
    .status.incomplete,
    .status.partial,
    .status.missing-report {{ background: rgba(125, 93, 143, 0.12); color: var(--incomplete); }}
    .bar {{
      width: 120px;
      height: 10px;
      background: var(--accent-soft);
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 4px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #d39b5d);
    }}
    .bar-label {{
      font-size: 12px;
      color: var(--muted);
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
      padding: 8px 0 0;
    }}
    @media (max-width: 900px) {{
      main {{ width: min(100vw - 16px, 1400px); margin-top: 16px; }}
      .hero, .panel {{ padding: 16px; border-radius: 16px; }}
      table, thead, tbody, tr, td, th {{ display: block; }}
      thead {{ display: none; }}
      tr {{
        border-top: 1px solid var(--line);
        padding: 10px 0;
      }}
      td {{
        border-top: none;
        padding: 4px 0;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Autoresearch Dashboard</h1>
      <p>Repo: {html.escape(str(REPO_ROOT))}</p>
      <p>Generated: {html.escape(generated_label)}</p>
    </section>

    <section class="cards">
      <article class="card"><span class="label">Runs</span><div class="value">{overview["run_count"]}</div></article>
      <article class="card"><span class="label">Completed Runs</span><div class="value">{overview["completed_run_count"]}</div></article>
      <article class="card"><span class="label">Failed Runs</span><div class="value">{overview["failed_run_count"]}</div></article>
      <article class="card"><span class="label">Incomplete Runs</span><div class="value">{overview["incomplete_run_count"]}</div></article>
      <article class="card"><span class="label">Datasets</span><div class="value">{overview["dataset_count"]}</div></article>
      <article class="card"><span class="label">Latest Dataset Rows</span><div class="value">{overview["latest_dataset_rows"]}</div></article>
      <article class="card"><span class="label">Latest Unresolved Images</span><div class="value">{overview["latest_dataset_unconfirmed"]}</div></article>
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Recent Runs</h2>
        <p>Derived from `results/*/run.json` plus each saved report.</p>
      </div>
      {"<table><thead><tr><th>Status</th><th>Run</th><th>Model</th><th>Phase</th><th>Depth</th><th>Rows</th><th>Unresolved</th><th>Duration</th><th>Started</th><th>Artifacts</th></tr></thead><tbody>" + html_table_rows(run_rows) + "</tbody></table>" if run_rows else "<div class='empty'>No runs found under results/.</div>"}
    </section>

    <section class="panel">
      <div class="panel-header">
        <h2>Datasets</h2>
        <p>Coverage is computed from the exported CSV when needed.</p>
      </div>
      {"<table><thead><tr><th>Dataset</th><th>Rows</th><th>Unresolved</th><th>Coverage</th><th>Completed Phases</th><th>Follow-up Runs</th><th>Phase Rows</th><th>Created</th><th>Artifacts</th></tr></thead><tbody>" + html_table_rows(dataset_rows) + "</tbody></table>" if dataset_rows else "<div class='empty'>No datasets found under results/datasets/.</div>"}
    </section>
  </main>
</body>
</html>
"""


def write_dashboard_html(output_path: Path, model: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dashboard_html(model, output_path), encoding="utf-8")
    return output_path


def serve_dashboard(output_path: Path, host: str, port: int, open_browser: bool) -> None:
    model = build_dashboard_model()
    dashboard_path = write_dashboard_html(output_path, model)
    handler = partial(SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    url_path = relative_link(REPO_ROOT / "index.html", dashboard_path).replace("\\", "/")
    url = f"http://{host}:{port}/{url_path}"
    print(f"dashboard: {dashboard_path}")
    print(f"url: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping server")
    finally:
        server.server_close()


def main() -> int:
    args = parse_args()
    model = build_dashboard_model()

    if args.command == "summary":
        print(render_text_summary(model, limit_runs=args.limit_runs, limit_datasets=args.limit_datasets))
        return 0

    if args.command == "export-html":
        output_path = write_dashboard_html(args.output.resolve(), model)
        print(f"dashboard: {output_path}")
        return 0

    if args.command == "serve":
        serve_dashboard(args.output.resolve(), args.host, args.port, open_browser=not args.no_open)
        return 0

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
