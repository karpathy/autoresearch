"""
Research memory for autoresearch.
Append-only JSONL storage with file locking for concurrent multi-agent access.
"""

import fcntl
import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Callable

from config import get_results_dir

RESULTS_DIR = get_results_dir()
EXPERIMENTS_FILE = os.path.join(RESULTS_DIR, "experiments.jsonl")
LESSONS_FILE = os.path.join(RESULTS_DIR, "lessons.jsonl")
JOURNAL_FILE = os.path.join(RESULTS_DIR, "journal.md")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(filepath: str, data: dict):
    """Append a single JSON line with exclusive file locking."""
    _ensure_results_dir()
    line = json.dumps(data, separators=(",", ":")) + "\n"
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_jsonl(filepath: str) -> list[dict]:
    """Read all JSON lines, skipping malformed ones."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    timestamp: str = ""
    agent_role: str = "solo"
    agent_id: str = "solo-0"
    branch: str = ""
    commit: str = ""
    scale: str = "standard"
    time_budget: int = 300
    val_bpb: float = 0.0
    peak_vram_mb: float = 0.0
    mfu_percent: float = 0.0
    num_params_M: float = 0.0
    depth: int = 0
    total_batch_size: int = 0
    matrix_lr: float = 0.0
    loss_trajectory: str = ""
    status: str = "crash"
    description: str = ""
    parent_commit: Optional[str] = None
    escalated_from: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()


@dataclass
class LessonRecord:
    timestamp: str = ""
    agent_role: str = "solo"
    agent_id: str = "solo-0"
    category: str = ""  # "architecture", "hyperparameter", "failure_mode", "insight"
    lesson: str = ""
    evidence_commits: list = field(default_factory=list)
    confidence: str = "medium"  # "high", "medium", "low"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _now_iso()


# ---------------------------------------------------------------------------
# Write operations (with locking)
# ---------------------------------------------------------------------------

def append_experiment(record: ExperimentRecord) -> None:
    _append_jsonl(EXPERIMENTS_FILE, asdict(record))


def append_lesson(record: LessonRecord) -> None:
    _append_jsonl(LESSONS_FILE, asdict(record))


def append_journal(agent_id: str, entry: str) -> None:
    """Append a timestamped markdown entry to the research journal."""
    _ensure_results_dir()
    ts = _now_iso()
    text = f"\n### [{ts}] {agent_id}\n\n{entry}\n"
    fd = os.open(JOURNAL_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, text.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def load_experiments(filter_fn: Optional[Callable[[dict], bool]] = None) -> list[dict]:
    records = _read_jsonl(EXPERIMENTS_FILE)
    if filter_fn:
        records = [r for r in records if filter_fn(r)]
    return records


def load_lessons() -> list[dict]:
    return _read_jsonl(LESSONS_FILE)


def get_best_result() -> Optional[dict]:
    """Return experiment with lowest val_bpb among status='keep'."""
    kept = load_experiments(lambda r: r.get("status") == "keep" and r.get("val_bpb", 0) > 0)
    if not kept:
        return None
    return min(kept, key=lambda r: r["val_bpb"])


def get_agent_experiments(agent_id: str) -> list[dict]:
    return load_experiments(lambda r: r.get("agent_id") == agent_id)


# ---------------------------------------------------------------------------
# Research briefing
# ---------------------------------------------------------------------------

def build_research_briefing(max_recent: int = 20) -> str:
    """Generate a markdown summary for agents to read before each experiment."""
    lines = ["# Research Briefing\n"]

    # Best result
    best = get_best_result()
    if best:
        lines.append(f"## Current Best")
        lines.append(f"- **val_bpb**: {best['val_bpb']:.6f}")
        lines.append(f"- **config**: depth={best.get('depth')}, batch={best.get('total_batch_size')}, matrix_lr={best.get('matrix_lr')}")
        lines.append(f"- **commit**: {best.get('commit')}")
        lines.append(f"- **description**: {best.get('description')}")
        lines.append("")

    # Recent experiments
    experiments = load_experiments()
    if experiments:
        lines.append(f"## Recent Experiments (last {min(max_recent, len(experiments))} of {len(experiments)} total)")
        lines.append("")
        lines.append("| # | Agent | Scale | val_bpb | Status | Description |")
        lines.append("|---|-------|-------|---------|--------|-------------|")
        for i, exp in enumerate(experiments[-max_recent:]):
            idx = len(experiments) - max_recent + i
            bpb = f"{exp['val_bpb']:.6f}" if exp.get("val_bpb", 0) > 0 else "crash"
            lines.append(f"| {idx} | {exp.get('agent_id', '?')} | {exp.get('scale', '?')} | {bpb} | {exp.get('status', '?')} | {exp.get('description', '')[:50]} |")
        lines.append("")

    # Stats
    if experiments:
        kept = [e for e in experiments if e.get("status") == "keep"]
        crashed = [e for e in experiments if e.get("status") == "crash"]
        lines.append(f"## Stats")
        lines.append(f"- Total experiments: {len(experiments)}")
        lines.append(f"- Kept: {len(kept)}, Discarded: {len(experiments) - len(kept) - len(crashed)}, Crashed: {len(crashed)}")
        if kept:
            bpbs = [e["val_bpb"] for e in kept]
            lines.append(f"- Best val_bpb: {min(bpbs):.6f}, Worst kept: {max(bpbs):.6f}")
        lines.append("")

    # Lessons
    lessons = load_lessons()
    if lessons:
        lines.append(f"## Lessons Learned ({len(lessons)} total)")
        lines.append("")
        for lesson in lessons[-15:]:
            conf = lesson.get("confidence", "?")
            lines.append(f"- [{conf}] **{lesson.get('category', '?')}**: {lesson.get('lesson', '')}")
        lines.append("")

    # Journal (last section)
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r") as f:
            journal = f.read().strip()
        if journal:
            # Show last ~2000 chars of journal
            if len(journal) > 2000:
                journal = "...\n" + journal[-2000:]
            lines.append("## Research Journal (recent)")
            lines.append("")
            lines.append(journal)
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy TSV compatibility
# ---------------------------------------------------------------------------

def sync_to_legacy_tsv(tsv_path: str = "results.tsv") -> None:
    """Write a results.tsv file for backward compat with analysis.ipynb."""
    experiments = load_experiments()
    header = "commit\tval_bpb\tmemory_gb\tmfu\tnum_params_M\tdepth\ttotal_batch_size\tmatrix_lr\tstatus\tdescription"
    lines = [header]
    for exp in experiments:
        mem_gb = exp.get("peak_vram_mb", 0) / 1024
        lines.append(
            f"{exp.get('commit', '?')}\t"
            f"{exp.get('val_bpb', 0):.6f}\t"
            f"{mem_gb:.1f}\t"
            f"{exp.get('mfu_percent', 0):.2f}\t"
            f"{exp.get('num_params_M', 0):.1f}\t"
            f"{exp.get('depth', 0)}\t"
            f"{exp.get('total_batch_size', 0)}\t"
            f"{exp.get('matrix_lr', 0):.2f}\t"
            f"{exp.get('status', 'crash')}\t"
            f"{exp.get('description', '')}"
        )
    with open(tsv_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "briefing":
        print(build_research_briefing())
    elif len(sys.argv) > 1 and sys.argv[1] == "sync-tsv":
        path = sys.argv[2] if len(sys.argv) > 2 else "results.tsv"
        sync_to_legacy_tsv(path)
        print(f"Synced to {path}")
    else:
        print("Usage: python knowledge.py [briefing|sync-tsv [path]]")
