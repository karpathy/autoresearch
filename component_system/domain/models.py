from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SeedStatus(str, Enum):
    draft = "draft"
    queued = "queued"
    planning = "planning"
    generated = "generated"
    dca_queued = "dca_queued"
    adapting = "adapting"
    running = "running"
    failed = "failed"
    passed = "passed"
    promoted = "promoted"


class StageName(str, Enum):
    p = "p"
    dca = "dca"
    direct = "direct"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class PlanIdea(BaseModel):
    title: str = ""
    target_component: str = "model"
    description: str = ""
    source_refs: list[str] = Field(default_factory=list)
    commit_sha: str | None = None


class StageRun(BaseModel):
    run_id: str
    seed_id: str
    stage: StageName
    status: RunStatus
    task_id: str
    created_at: float
    updated_at: float
    log_path: str | None = None
    stderr_log_path: str | None = None
    prompt_path: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    signal: str | None = None
    error: str | None = None


class SeedRecord(BaseModel):
    seed_id: str
    prompt: str
    status: SeedStatus = SeedStatus.draft
    created_at: float
    updated_at: float
    baseline_branch: str = "baseline"
    worktree_path: str | None = None
    latest_run_id: str | None = None
    ralph_loop_enabled: bool = False
    latest_signal: str | None = None
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    plan: PlanIdea | None = None
    last_error: str | None = None
    """Baseline val_bpb at sync-before-P time; used for positive/negative/neutral judgement in DCA."""
    former_val_bpb: float | None = None


class DashboardColumn(BaseModel):
    id: str
    title: str
    description: str
    seeds: list[SeedRecord]


class DashboardViewModel(BaseModel):
    setup_error: str | None = None
    baseline_metrics_by_branch: dict[str, dict[str, object]] = Field(default_factory=dict)
    default_baseline_branch: str = "master"
    available_branches: list[str] = Field(default_factory=list)
    seed_count: int
    columns: list[DashboardColumn]
    selected_seed: SeedRecord | None = None
    daemon_status: str = "stopped"  # "running" | "stopped"
