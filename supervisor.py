"""
Autoresearch Supervisor — nightly batch coordinator that runs a bounded
experiment session, detects plateaus, rotates experiment branches, and
optionally traces each cycle to LangSmith.

Designed to run once per night (via cron or systemd timer), NOT as a
persistent daemon. Reads/writes state.json for cross-run persistence.

Usage:
    python3 supervisor.py                         # single nightly batch
    python3 supervisor.py --batch-size 6          # smaller batch
    python3 supervisor.py --plateau-threshold 8   # reset after 8 consecutive discards

Environment:
    AUTORESEARCH_API_URL    default http://localhost:8300
    LANGCHAIN_TRACING_V2    set "true" to enable LangSmith tracing
    LANGCHAIN_API_KEY       LangSmith API key (required if tracing)
    LANGCHAIN_PROJECT       LangSmith project name (default: autoresearch)
"""

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from report import ReportGenerator
from research_director import ResearchDirector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("AUTORESEARCH_API_URL", "http://localhost:8300").rstrip("/")
POLL_INTERVAL = int(os.getenv("SUPERVISOR_POLL_SECONDS", "300"))  # 5 min
PLATEAU_THRESHOLD = int(os.getenv("SUPERVISOR_PLATEAU_THRESHOLD", "10"))
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

# ---------------------------------------------------------------------------
# LangSmith tracing (optional)
# ---------------------------------------------------------------------------

_langsmith_client = None


def _init_langsmith():
    """Lazy-init LangSmith client if tracing is enabled."""
    global _langsmith_client
    if not LANGSMITH_ENABLED:
        return None
    try:
        from langsmith import Client

        project = os.getenv("LANGCHAIN_PROJECT", "autoresearch")
        _langsmith_client = Client()
        log(f"LangSmith tracing enabled (project={project})")
        return _langsmith_client
    except ImportError:
        log("WARNING: LANGCHAIN_TRACING_V2=true but langsmith not installed")
        return None
    except Exception as e:
        log(f"WARNING: LangSmith init failed: {e}")
        return None


def _trace_cycle(round_num: int, tag: str, status: dict, results: list, action: str):
    """Log a supervisor cycle to LangSmith as a run."""
    if not _langsmith_client:
        return
    try:
        project = os.getenv("LANGCHAIN_PROJECT", "autoresearch")
        _langsmith_client.create_run(
            name=f"supervisor-cycle-r{round_num}",
            run_type="chain",
            project_name=project,
            inputs={
                "round": round_num,
                "tag": tag,
                "orchestrator_status": status.get("status"),
                "experiment_num": status.get("experiment_num"),
                "best_val_bpb": status.get("best_val_bpb"),
            },
            outputs={
                "action": action,
                "total_experiments": len(results),
                "kept": sum(1 for r in results if r.get("status") == "keep"),
                "discarded": sum(1 for r in results if r.get("status") == "discard"),
                "crashed": sum(1 for r in results if r.get("status") == "crash"),
            },
        )
    except Exception as e:
        log(f"LangSmith trace error (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str):
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def api_get(path: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"API GET {path} failed: {e}")
        return None


def api_post(path: str, body: dict | None = None) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=body or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"API POST {path} failed: {e}")
        return None


def generate_tag(round_num: int) -> str:
    """Generate a unique tag for each round: apr09-r1, apr09-r2, ..."""
    date_part = datetime.now(UTC).strftime("%b%d").lower()
    return f"{date_part}-r{round_num}"


def count_trailing_discards(results: list) -> int:
    """Count consecutive discards from the end of results."""
    count = 0
    for r in reversed(results):
        if r.get("status") == "discard":
            count += 1
        else:
            break
    return count


def detect_plateau(results: list, threshold: int) -> bool:
    """True if the last `threshold` experiments were all discards (no improvement)."""
    if len(results) < threshold + 1:  # need at least 1 keep + threshold discards
        return False
    return count_trailing_discards(results) >= threshold


# ---------------------------------------------------------------------------
# Main supervisor — single nightly invocation
# ---------------------------------------------------------------------------


def run_supervisor(batch_size: int = 12, plateau_threshold: int = PLATEAU_THRESHOLD):
    """Run a single nightly batch: consult director, run experiments, report."""
    log("=== Autoresearch Supervisor (nightly batch) ===")
    log(f"API: {API_URL}")
    log(f"Batch size: {batch_size}")
    log(f"Plateau threshold: {plateau_threshold} consecutive discards")

    _init_langsmith()

    director = ResearchDirector()
    reporter = ReportGenerator()

    # --- Pre-flight: check API ---
    status = api_get("/status")
    if status is None:
        log("Cannot reach orchestrator API. Is api.py running?")
        sys.exit(1)

    if status.get("status") == "running":
        log("Orchestrator already running — skipping this nightly run.")
        return

    # --- Plateau detection on existing results ---
    results = api_get("/results") or []
    if detect_plateau(results, plateau_threshold):
        trailing = count_trailing_discards(results)
        log(f"PLATEAU detected: {trailing} consecutive discards.")
        log("Generating report and exiting. Manual intervention recommended.")
        _generate_report(reporter, 0, status.get("tag", "?"), [])
        _trace_cycle(0, status.get("tag", "?"), status, results, "plateau-skip")
        return

    # --- Research Director: decide what to test ---
    tag = generate_tag(1)
    directive = _consult_director(director, 1)
    round_directives = [directive] if directive else []

    # --- Start batch via API ---
    log(f"Starting batch: tag={tag}, batch_size={batch_size}")
    resp = api_post(
        "/start",
        {
            "tag": tag,
            "batch_size": batch_size,
        },
    )
    if not resp or not resp.get("started"):
        log(f"Failed to start batch: {resp}")
        sys.exit(1)

    _trace_cycle(1, tag, status, results, "start")

    # --- Poll until batch completes ---
    log("Waiting for batch to complete...")
    while True:
        time.sleep(POLL_INTERVAL)
        status = api_get("/status")
        if status is None:
            log("Lost contact with API — waiting...")
            continue

        orch_status = status.get("status", "unknown")
        if orch_status in ("idle", "stopped"):
            log("Batch complete.")
            break

        exp_num = status.get("experiment_num", 0)
        best = status.get("best_val_bpb")
        desc = status.get("current_description", "")
        log(f"Running: exp#{exp_num} best={best} current='{desc}'")

        # Mid-run plateau check
        mid_results = api_get("/results") or []
        if detect_plateau(mid_results, plateau_threshold):
            log("PLATEAU mid-batch. Stopping.")
            api_post("/stop")
            time.sleep(30)
            break

    # --- Report ---
    final_results = api_get("/results") or []
    _generate_report(reporter, 1, tag, round_directives)
    _trace_cycle(1, tag, status or {}, final_results, "complete")

    best_bpb = None
    for r in reversed(final_results):
        if r.get("status") == "keep":
            best_bpb = r.get("val_bpb")
            break

    log(f"Best val_bpb: {best_bpb}")
    log("Supervisor done. Exiting.")


def _consult_director(director: ResearchDirector, round_num: int) -> dict | None:
    """Ask the Research Director what to test next. Returns directive or None."""
    try:
        train_py = Path("train.py").read_text() if Path("train.py").exists() else ""
        results_tsv = Path("results.tsv").read_text() if Path("results.tsv").exists() else ""

        if not train_py:
            log("Director: no train.py found, skipping directive")
            return None

        log("Director: analyzing history and deciding next experiment...")
        directive = director.next_directive(train_py, results_tsv)

        log(
            f"Director directive: [{directive.get('category')}] "
            f"{directive.get('hypothesis', '?')[:80]}"
        )
        log(f"  Priority={directive.get('priority')} Risk={directive.get('risk')}")
        log(f"  Rationale: {directive.get('rationale', 'N/A')[:100]}")

        # Write directive to disk so orchestrator can pick it up
        directive_file = Path("current_directive.json")
        directive_file.write_text(json.dumps(directive, indent=2))
        log(f"Director: wrote directive to {directive_file}")

        return directive
    except Exception as e:
        log(f"Director error (non-fatal, orchestrator will self-direct): {e}")
        return None


def _generate_report(
    reporter: ReportGenerator,
    round_num: int,
    tag: str | None,
    directives: list[dict],
) -> None:
    """Generate and log a round report."""
    try:
        results_tsv = Path("results.tsv").read_text() if Path("results.tsv").exists() else ""
        if not results_tsv.strip():
            return
        path = reporter.generate(results_tsv, directives, round_num, tag or "")
        log(f"Report generated: {path}")
    except Exception as e:
        log(f"Report generation failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autoresearch nightly supervisor")
    parser.add_argument(
        "--batch-size", type=int, default=12, help="Max experiments per nightly batch (default: 12)"
    )
    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=PLATEAU_THRESHOLD,
        help=f"Consecutive discards before plateau (default: {PLATEAU_THRESHOLD})",
    )
    args = parser.parse_args()

    try:
        run_supervisor(
            batch_size=args.batch_size,
            plateau_threshold=args.plateau_threshold,
        )
    except KeyboardInterrupt:
        log("Supervisor interrupted. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
