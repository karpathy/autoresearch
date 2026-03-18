from __future__ import annotations

from pathlib import Path


RESULTS_TSV_HEADER = (
    "timestamp\trun_id\tbranch\tslice\tgate\tpassed\tsummary\tduration_s\n"
)


def ensure_results_file(results_path: Path) -> None:
    """
    Ensure the results TSV exists with the expected header.

    Intentionally narrow: only header creation is implemented for Task 2.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        return
    results_path.write_text(RESULTS_TSV_HEADER)

