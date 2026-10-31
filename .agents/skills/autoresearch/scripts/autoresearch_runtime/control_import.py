"""Import control-run reports into state_dir/controls (never under runs/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULT_TYPES = ("baseline", "experiment", "reproducibility_control", "provider_check")


def _report_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    runs = report.get("runs")
    if isinstance(runs, list):
        return [item for item in runs if isinstance(item, dict)]
    if isinstance(report.get("metrics"), dict):
        return [report]
    return []


def _next_control_id(state_dir: Path) -> str:
    from .store import _sqlite_document

    target = _sqlite_document(state_dir / "lab.json")
    if target is not None:
        store, lab_id, *_ = target
        numbers = []
        offset = 0
        while True:
            page = store.list_runs(lab_id, limit=100, offset=offset)
            numbers.extend(int(item["alias"][8:]) for item in page["items"]
                           if item.get("kind") == "probe" and item.get("alias", "").startswith("control-")
                           and item["alias"][8:].isdigit())
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
        return f"control-{max(numbers, default=0) + 1:04d}"

    controls_root = state_dir / "controls"
    controls_root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in controls_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("control-") and len(name) == 12 and name[8:].isdigit():
            numbers.append(int(name[8:]))
    return f"control-{max(numbers, default=0) + 1:04d}"


def _primary_metric_name(report: dict[str, Any], runs: list[dict[str, Any]]) -> str | None:
    report_primary = report.get("primary")
    if isinstance(report_primary, str) and report_primary:
        return report_primary
    if not runs:
        return None
    metrics = runs[0].get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    if "score" in metrics:
        return "score"
    return next(iter(metrics))


def import_control_report(
    state_dir: Path,
    source: Path,
    *,
    result_type: str,
    label: str,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if result_type not in RESULT_TYPES:
        raise ValueError(f"unknown result_type: {result_type!r}")

    state_dir = state_dir.resolve()
    source = source.resolve()

    report: dict[str, Any] = {}
    if entries is None:
        try:
            report = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                report = {}
        except Exception:
            report = {}
        extracted = _report_entries(report)
    else:
        extracted = entries

    runs = [{"metrics": item["metrics"]} for item in extracted if isinstance(item.get("metrics"), dict)]
    if not runs:
        raise ValueError("report has no runs with metrics")

    control_id = _next_control_id(state_dir)
    manifest: dict[str, Any] = {
        "id": control_id,
        "result_type": result_type,
        "title": label,
        "status": "non_adoptable",
        "adoptable": False,
        "runs": runs,
    }
    primary = _primary_metric_name(report, runs)
    if primary:
        manifest["primary"] = primary

    from .store import _sqlite_document

    target = _sqlite_document(state_dir / "lab.json")
    if target is None:
        control_dir = state_dir / "controls" / control_id
        control_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = control_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "accepted"
        manifest.pop("adoptable", None)
        return manifest

    store, lab_id, *_ = target
    state = store.load_state(lab_id)
    run_id = store.create_run(lab_id, "probe", control_id, int(state.get("current_epoch") or 1), {
        "candidate_commit": state.get("best_commit"), "result_type": result_type,
    })
    for index, result in enumerate(runs, 1):
        store.append_result(lab_id, run_id, index, {
            "status": "ok", "metrics": result["metrics"],
            "details": {"source": str(source), "result_type": result_type},
        })
    manifest["source"] = str(source)
    store.save_manifest(lab_id, run_id, manifest)
    return manifest


def promote_control_to_baseline(state_dir: Path, control_id: str) -> dict[str, Any]:
    """Promote an imported control run to be the official baseline of the lab."""
    from .errors import LabError

    state_dir = state_dir.resolve()
    from .store import _sqlite_document
    if _sqlite_document(state_dir / "lab.json") is not None:
        raise LabError("Imported SQLite probes are non-adoptable; baseline promotion is unsupported")
    control_manifest_path = state_dir / "controls" / control_id / "manifest.json"
    if not control_manifest_path.is_file():
        raise LabError(f"Control run '{control_id}' manifest not found at {control_manifest_path}")

    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    state_file = state_dir / "state.json"
    if not state_file.is_file():
        raise LabError(f"Lab state file not found at {state_file}")

    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["baseline_id"] = control_id
    state["baseline_verified"] = True
    if state.get("best_id") is None:
        state["best_id"] = control_id
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "promoted_id": control_id,
        "baseline_verified": True,
        "manifest": control_manifest,
    }


def load_control_manifests(state_dir: Path) -> list[dict[str, Any]]:
    from .store import _sqlite_document
    target = _sqlite_document(state_dir.resolve() / "lab.json")
    if target is not None:
        store, lab_id, *_ = target
        manifests = []
        offset = 0
        while True:
            page = store.list_runs(lab_id, limit=100, offset=offset)
            for item in page["items"]:
                if item.get("kind") == "probe":
                    manifest = store.load_manifest(lab_id, item["run_id"])
                    if isinstance(manifest, dict):
                        manifests.append(manifest)
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
        return sorted(manifests, key=lambda item: str(item.get("id", "")))

    controls_root = state_dir / "controls"
    if not controls_root.is_dir():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(controls_root.glob("*/manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            manifests.append(payload)
    return sorted(manifests, key=lambda item: str(item.get("id", "")))
