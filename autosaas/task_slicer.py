from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TaskSlice:
    name: str
    summary: str
    success_criteria: Tuple[str, ...]
    allowed_file_patterns: Tuple[str, ...]


def choose_next_slice(task_description: str, repo_context: dict | None = None) -> TaskSlice:
    """Produce a single focused slice for the incoming request."""

    normalized = task_description.strip().rstrip(".")
    lowered = normalized.lower()

    if "billing" in lowered:
        topic = "billing"
    else:
        topic = lowered.split()[0] if lowered else "slice"

    slice_name = f"{topic} slice: {normalized}"
    summary = (
        "Reduce the request to a single development slice with clear boundaries "
        "and measurable success criteria."
    )

    criteria = (
        f"Describe success for '{normalized}' in terms of observable behavior.",
        "Restrict touched files to the listed allowed patterns.",
    )

    allowed_patterns = ("autosaas/**", "tests/**")

    return TaskSlice(
        name=slice_name,
        summary=summary,
        success_criteria=criteria,
        allowed_file_patterns=allowed_patterns,
    )
