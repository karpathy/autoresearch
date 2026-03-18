from __future__ import annotations

from autosaas.state import SliceRun


def format_slice_run(run: SliceRun) -> str:
    """
    Human-readable reporting for a slice run.

    Placeholder for Task 2: enough structure for downstream callers without
    implementing orchestration/reporting logic yet.
    """
    return f"{run.slice_name}: {run.status} ({len(run.gate_results)} gates)"

