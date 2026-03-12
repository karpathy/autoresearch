from __future__ import annotations

from typing import Any

from component_system.domain.models import SeedRecord, StageRun
from component_system.task import (
    append_event,
    list_runs,
    list_seeds,
    load_baseline_branch_map,
    load_baseline_metrics,
    load_events,
    load_run,
    load_seed,
    save_baseline_branch_map,
    save_baseline_metrics,
    save_run,
    save_seed,
)


class BaselineBranchMapRepository:
    """Per-seed baseline branch mapping (seed_id -> baseline_branch)."""

    def set_branch_for_seed(self, seed_id: str, branch: str) -> None:
        m = load_baseline_branch_map()
        m[seed_id] = branch
        save_baseline_branch_map(m)


class BaselineMetricsRepository:
    """Per-baseline-branch metrics (last_val_bpb, promoted_*, commit_sha, etc.)."""

    def get_all(self) -> dict[str, dict[str, Any]]:
        return load_baseline_metrics()

    def get_for_branch(self, branch: str) -> dict[str, Any] | None:
        return load_baseline_metrics().get(branch)

    def update_for_branch(self, branch: str, metrics: dict[str, Any]) -> None:
        data = load_baseline_metrics()
        data[branch] = {**data.get(branch, {}), **metrics}
        save_baseline_metrics(data)


class SeedRepository:
    def list(self) -> list[SeedRecord]:
        return [SeedRecord.model_validate(seed) for seed in list_seeds()]

    def get(self, seed_id: str) -> SeedRecord | None:
        data = load_seed(seed_id)
        return SeedRecord.model_validate(data) if data else None

    def save(self, seed: SeedRecord) -> SeedRecord:
        save_seed(seed.model_dump(mode="json"))
        return seed

    def append_event(self, seed_id: str, kind: str, message: str, **payload: Any) -> list[dict[str, Any]]:
        return append_event(seed_id, {"kind": kind, "message": message, **payload})

    def events(self, seed_id: str) -> list[dict[str, Any]]:
        return load_events(seed_id)


class RunRepository:
    def list(self, seed_id: str | None = None) -> list[StageRun]:
        return [StageRun.model_validate(run) for run in list_runs(seed_id)]

    def get(self, run_id: str) -> StageRun | None:
        data = load_run(run_id)
        return StageRun.model_validate(data) if data else None

    def save(self, run: StageRun) -> StageRun:
        save_run(run.model_dump(mode="json"))
        return run
