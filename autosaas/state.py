from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    name: str
    passed: bool
    summary: str
    duration_s: float


@dataclass
class SliceRun:
    slice_name: str
    branch: str
    base_commit: str
    head_commit: str = ""
    touched_files: list[str] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    status: str = "blocked"

