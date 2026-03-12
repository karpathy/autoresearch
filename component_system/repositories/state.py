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


def _branch_metrics_view(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Build view with best_val_bpb (min over history) and promoted_* from the record that achieved it."""
    if not history:
        return {"best_val_bpb": None, "history": []}
    vals = [r["val_bpb"] for r in history if r.get("val_bpb") is not None]
    best_val_bpb = min(vals) if vals else None
    best_record = next((r for r in history if r.get("val_bpb") == best_val_bpb), history[-1])
    view: dict[str, Any] = {
        "best_val_bpb": best_val_bpb,
        "history": history,
    }
    if best_record.get("promoted_branch") is not None:
        view["promoted_branch"] = best_record["promoted_branch"]
    if best_record.get("promoted_idea") is not None:
        view["promoted_idea"] = best_record["promoted_idea"]
    if best_record.get("promoted_at") is not None:
        view["promoted_at"] = best_record["promoted_at"]
    if best_record.get("commit_sha") is not None:
        view["commit_sha"] = best_record["commit_sha"]
    return view


class BaselineMetricsRepository:
    """Per-baseline-branch metrics: list of records per branch (val_bpb, promoted_*, etc.)."""

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return branch -> view (best_val_bpb, promoted_branch, commit_sha, history) for dashboard."""
        data = load_baseline_metrics()
        return {branch: _branch_metrics_view(hist) for branch, hist in data.items()}

    def get_for_branch(self, branch: str) -> dict[str, Any] | None:
        """Return view for one branch (best_val_bpb, history, promoted_branch?, commit_sha?)."""
        data = load_baseline_metrics()
        hist = data.get(branch)
        if hist is None:
            return None
        return _branch_metrics_view(hist)

    def append_promotion_for_branch(self, branch: str, record: dict[str, Any]) -> None:
        """Append a promotion record: val_bpb, promoted_branch, promoted_idea, promoted_at, commit_sha."""
        data = load_baseline_metrics()
        data.setdefault(branch, []).append(dict(record))
        save_baseline_metrics(data)

    def append_baseline_run(self, branch: str, val_bpb: float) -> None:
        """Append a baseline measurement (no promotion)."""
        data = load_baseline_metrics()
        data.setdefault(branch, []).append({"val_bpb": val_bpb})
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
