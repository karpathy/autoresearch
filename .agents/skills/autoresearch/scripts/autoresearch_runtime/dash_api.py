"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .catalog import aggregate_dashboard_status, resolve_lab_path
from .control_import import load_control_manifests
from .paths import home_root_id, lab_storage_roots, resolve_home, resolve_root_by_id, root_id_for, storage_root_for_path
from .dashboard import (
    _primary_score,
    build_problems,
    build_verdict,
    chart_series,
    metric_as_percent,
    metric_label,
    presentation_status,
    reason_label,
    render_unified_dashboard,
)

from .dash_io import _merged_usage, _read_json
from . import dash_model
from .dash_model import _historical_rechecks, _lab_replay_rechecks, _lab_summary


def dashboard_status(home: Path) -> dict[str, Any]:
    """Dashboard summary across AUTORESEARCH_HOME and every registered storage root."""
    return aggregate_dashboard_status(home)


dash_model.dashboard_status = dashboard_status


def _storage_roots(home: Path, root_id: str | None = None) -> list[Path]:
    home = home.resolve()
    if root_id:
        resolved = resolve_root_by_id(root_id, home)
        return [resolved] if resolved is not None else []
    return lab_storage_roots(home)


class AmbiguousLab(Exception):
    """The same project and lab id exists in more than one storage root."""

    def __init__(self, roots: list[str]) -> None:
        self.roots = roots
        super().__init__(", ".join(roots))


def _unique_hit(hits: list[Any], root_id: str | None, home: Path) -> Any | None:
    if not hits:
        return None
    if len(hits) > 1 and not root_id:
        raise AmbiguousLab([root_id_for(hit[0], home) for hit in hits])
    return hits[0]


def _find_active_lab(home: Path, project_id: str, lab_id: str,
                     root_id: str | None = None) -> tuple[Path, dict[str, Any], dict[str, Any], Path] | None:
    hits = []
    for storage in _storage_roots(home, root_id):
        found = _safe_lab_dir(storage, project_id, lab_id)
        if found is not None:
            project, item, state_dir = found
            hits.append((storage, project, item, state_dir))
    return _unique_hit(hits, root_id, home)


def _find_archived_lab(home: Path, project_id: str, lab_id: str,
                       root_id: str | None = None) -> tuple[Path, Path] | None:
    hits = []
    for storage in _storage_roots(home, root_id):
        archive_dir = _safe_archive_dir(storage, project_id, lab_id)
        if archive_dir is not None:
            hits.append((storage, archive_dir))
    return _unique_hit(hits, root_id, home)


def _safe_lab_dir(root: Path, project_id: str, lab_id: str) -> tuple[dict[str, Any], dict[str, Any], Path] | None:
    if any(part in {"", ".", ".."} or "/" in part or "\\" in part for part in (project_id, lab_id)):
        return None
    catalog = _read_json(root / "catalog.json", {"projects": []}) or {"projects": []}
    for project in catalog.get("projects", []) if isinstance(catalog.get("projects"), list) else []:
        if not isinstance(project, dict) or str(project.get("id", "")) != project_id:
            continue
        for item in project.get("labs", []) if isinstance(project.get("labs"), list) else []:
            if not isinstance(item, dict):
                continue
            meta_candidate = resolve_lab_path(root, project, str(item.get("path", "")))
            if (str(item.get("id", "")) == lab_id or meta_candidate.name == lab_id) and meta_candidate.is_dir():
                return project, item, meta_candidate
    # Read old/grouped labs absent from a damaged catalog without writing to it.
    config_paths = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    if (root / "lab.json").is_file():
        config_paths.append(root / "lab.json")
    for config_path in config_paths:
        config = _read_json(config_path, {}) or {}
        configured_state = config.get("state_dir")
        if (config_path.parent / "state.json").is_file():
            state_dir = config_path.parent.resolve()
        elif isinstance(configured_state, str) and configured_state.strip():
            configured_path = Path(configured_state)
            state_dir = (configured_path if configured_path.is_absolute()
                         else config_path.parent / configured_path).resolve()
        else:
            state_dir = config_path.parent.resolve()
        if not state_dir.is_relative_to(root.resolve()) or "archive" in state_dir.relative_to(root).parts:
            continue
        meta = _read_json(state_dir / "lab-meta.json", {}) or {}
        if state_dir.parent.name == project_id and lab_id in {state_dir.name, str(meta.get("contract_id", ""))}:
            project = {"id": project_id, "name": meta.get("project_name", state_dir.parent.name),
                       "source_root": config.get("source_root", ""), "labs": []}
            item = {"id": meta.get("contract_id", state_dir.name), "path": state_dir.relative_to(root).as_posix(),
                    "name": meta.get("goal", state_dir.name), "goal": meta.get("goal", "")}
            return project, item, state_dir
    return None

def _safe_diff(state_dir: Path, run_id: str) -> str | None:
    if not (run_id.startswith("trial-") and run_id[6:].isdigit() and len(run_id) == 10):
        return None
    path = (state_dir / "runs" / run_id / "change.patch").resolve()
    if not path.is_relative_to((state_dir / "runs").resolve()) or not path.is_file() or path.stat().st_size > 512 * 1024:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None

def _open_lab(home: Path, project_id: str, lab_id: str,
              root_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any], Path, bool] | None:
    """Resolve an active lab, or the same id under archive/ for read-only display."""
    home = home.resolve()
    located = _find_active_lab(home, project_id, lab_id, root_id=root_id)
    if located is not None:
        _storage, project, item, state_dir = located
        return project, item, state_dir, False
    archived = _find_archived_lab(home, project_id, lab_id, root_id=root_id)
    if archived is None:
        return None
    storage, archive_dir = archived
    meta = _read_json(archive_dir / "lab-meta.json", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    config = _read_json(archive_dir / "lab.json", {}) or {}
    if not isinstance(config, dict):
        config = {}
    project = {"id": project_id, "name": meta.get("project_name") or project_id,
               "source_root": config.get("source_root", ""), "labs": []}
    item = {"id": lab_id, "path": archive_dir.relative_to(storage).as_posix(),
            "name": meta.get("goal") or archive_dir.name, "goal": meta.get("goal") or ""}
    return project, item, archive_dir, True

def dashboard_details(home: Path, project_id: str, lab_id: str, *, include_diffs: bool = False,
                      root_id: str | None = None) -> dict[str, Any] | None:
    """Return a bounded, presentation-safe selected-lab model without evaluator configs or logs."""
    home = home.resolve()
    try:
        opened = _open_lab(home, project_id, lab_id, root_id=root_id)
    except AmbiguousLab as error:
        return {"error": "ambiguous", "roots": error.roots}
    if opened is None:
        return None
    project, item, state_dir, archived = opened
    root = storage_root_for_path(state_dir, home) or home
    config = _read_json(state_dir / "lab.json", {}) or {}
    state = _read_json(state_dir / "state.json", {}) or {}
    meta = _read_json(state_dir / "lab-meta.json", {}) or {}
    manifests = [_read_json(path, {}) for path in sorted((state_dir / "runs").glob("*/manifest.json"))]
    rechecks = _historical_rechecks(root, state_dir, manifests, config, state)
    for trial_id, replay in _lab_replay_rechecks(state_dir, manifests).items():
        rechecks.setdefault(trial_id, replay)
    view_manifests = [rechecks.get(str(manifest.get("id")), manifest) for manifest in manifests]
    summary_item = dict(item)
    summary_item["path"] = state_dir.relative_to(root).as_posix()
    summary = _lab_summary(root, project, summary_item)
    summary["project_id"] = project_id
    summary["comparison_status"] = item.get("comparison_status")
    comparison_note = item.get("comparison_note", "")
    metric = summary.get("metric", "")
    direction = config.get("metrics", {}).get("direction", "max")
    guard_names = [entry.get("name") for entry in config.get("metrics", {}).get("guards", []) if isinstance(entry, dict)]
    baseline = next((manifest for manifest in manifests if manifest.get("id") == state.get("baseline_id")), None)
    best = next((manifest for manifest in manifests if manifest.get("id") == state.get("best_id")), baseline)

    def metric_record(manifest: dict[str, Any] | None) -> dict[str, Any]:
        output = {}
        for name in [metric, *guard_names]:
            values = [run.get("metrics", {}).get(name) for run in (manifest or {}).get("runs", [])]
            values = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if values:
                sense = direction if name == metric else next(
                    (entry.get("direction", "max") for entry in config.get("metrics", {}).get("guards", [])
                     if isinstance(entry, dict) and entry.get("name") == name), "max")
                output[name] = {"worst": min(values) if sense == "max" else max(values),
                                "low": min(values), "high": max(values)}
        primary = output.get(metric, {})
        primary_values = [run.get("metrics", {}).get(metric) for run in (manifest or {}).get("runs", [])]
        primary_values = [float(value) for value in primary_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        score = _primary_score(primary_values, direction, config) if primary_values else None
        return {"id": (manifest or {}).get("id"), "score": score,
                "low": primary.get("low"), "high": primary.get("high"), "metrics": output}

    hypotheses = []
    historical = []
    history = _read_json(state_dir / "history.json", {"hypotheses": []}) or {"hypotheses": []}
    current_inputs = {entry.get("path"): entry.get("sha256") for entry in state.get("inputs", [])
                      if isinstance(entry, dict)}
    for old in history.get("hypotheses", []) if isinstance(history.get("hypotheses"), list) else []:
        if not isinstance(old, dict):
            continue
        old_runs = old.get("runs", []) if isinstance(old.get("runs"), list) else []
        old_values = [run.get("metrics", {}).get(metric) for run in old_runs
                      if isinstance(run, dict) and isinstance(run.get("metrics"), dict)]
        if not old_values and isinstance(old.get("metrics"), dict):
            old_values = [old["metrics"].get(metric)]
        old_values = [float(value) for value in old_values
                      if isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(float(value))]
        if not old_values:
            score_text = "—"
            comparison = "Неполная запись" if not old.get("metrics") else "Другая метрика"
        elif len(old_values) == 1:
            score_text = f"{old_values[0]:.8g}"
            comparison = "Условия не подтверждены"
        else:
            score_text = f"{min(old_values):.8g}–{max(old_values):.8g} ({len(old_values)} повт.)"
            comparison = "Условия не подтверждены"
        benchmark = old.get("benchmark", {}) if isinstance(old.get("benchmark"), dict) else {}
        input_hashes = benchmark.get("input_hashes", {})
        if isinstance(input_hashes, dict):
            overlap = set(input_hashes) & set(current_inputs)
            if any(input_hashes[key] != current_inputs[key] for key in overlap):
                comparison = "Другие входы"
        context = [f"модель: {benchmark['model']}" for _ in [0] if benchmark.get("model")]
        if benchmark.get("classes"):
            context.append(f"классы: {benchmark['classes']}")
        linked = [f"{manifest.get('id')}: {manifest.get('status')}" for manifest in manifests
                  if manifest.get("legacy_id") == old.get("id")]
        historical.append({
            "id": str(old.get("id", "")), "title": str(old.get("title", "")),
            "source": str(old.get("source", "")),
            "artifact": old.get("artifact", "") if isinstance(old.get("artifact"), str) else "",
            "score": score_text, "conclusion": str(old.get("conclusion", "")),
            "context": ", ".join(context) or "—", "comparison": comparison,
            "recheck": ", ".join(linked) or "Не перепроверено",
        })
    current = None
    for manifest in view_manifests:
        run_id = str(manifest.get("id", ""))
        is_baseline = run_id.startswith("baseline")
        if manifest.get("status") == "running":
            current = {"id": run_id, "title": manifest.get("title", run_id)}
        if not is_baseline and not run_id.startswith("trial-"):
            continue
        runs = manifest.get("runs", [])
        values = [run.get("metrics", {}).get(metric) for run in runs]
        values = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        requests = cost = 0.0
        guards = []
        for index, run in enumerate(runs, start=1):
            run_root = Path(manifest["recheck_dir"]) if manifest.get("recheck_dir") else state_dir / "runs" / run_id
            usage = _merged_usage(run, run_root / f"repeat_{index:02d}" / "usage.json")
            requests += float(usage.get("requests", 0) or 0)
            cost += float(usage.get("cost_usd", 0) or 0)
        for name in guard_names:
            guard_values = [run.get("metrics", {}).get(name) for run in runs]
            guard_values = [value for value in guard_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if guard_values:
                guards.append({"name": name, "value": f"{min(guard_values):.6g}–{max(guard_values):.6g}"})
        status = str(manifest.get("status", "unknown"))
        reason = str(manifest.get("reason") or "")
        kind, outcome = presentation_status(status, reason)
        record = {
            "id": run_id, "title": manifest.get("title", run_id), "observation": manifest.get("observation", ""),
            "expected_effect": manifest.get("expected_effect", ""), "status": status,
            "status_kind": kind, "outcome": outcome, "reason": reason,
            "reason_label": reason_label(reason),
            "primary_range": (f"{min(values):.8g} – {max(values):.8g}" if values else "—"),
            "primary_score": _primary_score(values, direction, config) if values else None,
            "guard_summary": guards, "requests": int(requests), "cost_usd": round(cost, 10),
            "diff_available": (not is_baseline and (state_dir / "runs" / run_id / "change.patch").is_file()),
            "diff": _safe_diff(state_dir, run_id) if include_diffs else None,
            "recheck_source": manifest.get("recheck_source"),
            "cost_status": manifest.get("cost_status"),
            "unknown_cost_attempts": manifest.get("unknown_cost_attempts", 0),
            "prior_unknown_cost_attempts": manifest.get("prior_unknown_cost_attempts", 0),
            "original_status": manifest.get("original_status"),
            "original_reason": manifest.get("original_reason"),
            "original_outcome": presentation_status(
                str(manifest.get("original_status") or ""),
                str(manifest.get("original_reason") or ""),
            )[1] if manifest.get("original_status") else None,
        }
        hypotheses.append(record)
    guard_pairs = [(entry.get("name"), str(entry.get("direction") or "max"))
                   for entry in config.get("metrics", {}).get("guards", [])
                   if isinstance(entry, dict) and isinstance(entry.get("name"), str)]
    chart_manifests = [manifest for manifest in view_manifests
                       if str(manifest.get("id", "")).startswith(("baseline", "trial-"))]
    chart_manifests = chart_manifests + load_control_manifests(state_dir)
    series = chart_series(chart_manifests, str(metric or ""), direction,
                          baseline_id=state.get("baseline_id"), guards=guard_pairs,
                          config=config)
    for row in series:
        if row["id"] in rechecks:
            row["recheck_source"] = rechecks[row["id"]]["recheck_source"]
            row["original_status"] = rechecks[row["id"]]["original_status"]
    by_id = {row["id"]: row for row in series}
    baseline_row = by_id.get(state.get("baseline_id")) or next(
        (row for row in series if str(row["id"]).startswith("baseline") and row["has_value"]), None)
    base_guards = (baseline_row or {}).get("guards") or {}
    for record in hypotheses:
        row = by_id.get(record["id"])
        if row is None:
            continue
        record["delta_vs_baseline"] = row["delta_vs_baseline"]
        record["delta_vs_previous"] = row["delta_vs_previous"]
        record["improved_best"] = row["improved_best"]
        record["has_value"] = row["has_value"]
        failures = []
        for name, sense in guard_pairs:
            trial_guard = (row.get("guards") or {}).get(name) or {}
            base_guard = base_guards.get(name) or {}
            if not trial_guard.get("has_value") or not base_guard.get("has_value"):
                continue
            regressed = (trial_guard["worst"] < base_guard["worst"] - 1e-12 if sense == "max"
                         else trial_guard["worst"] > base_guard["worst"] + 1e-12)
            if regressed and record["id"] != (baseline_row or {}).get("id"):
                failures.append(name)
        record["guard_failures"] = failures
        record["guards_ok"] = not failures
    safe_conditions = {
        "direction": direction, "repeats": config.get("repeats"),
        "timeout_seconds": config.get("timeout_seconds"),
        "input_fingerprint": hashlib.sha256(json.dumps(
            [{"name": entry.get("name"), "sha256": entry.get("sha256")} for entry in state.get("inputs", [])],
            sort_keys=True).encode("utf-8")).hexdigest()[:12] if state.get("inputs") else "",
    }
    summary_stat = getattr(config, "statistic", None)
    if summary_stat is None and isinstance(config, dict):
        acceptance = config.get("acceptance") if isinstance(config.get("acceptance"), dict) else {}
        summary_stat = acceptance.get("statistic") or config.get("statistic")
    summary["statistic"] = summary_stat if summary_stat in {"worst", "mean", "median"} else "worst"
    summary.update({"comparison_status": item.get("comparison_status"), "comparison_note": comparison_note,
                    "conditions": safe_conditions, "archived": archived})
    baseline_record = metric_record(baseline)
    best_record = metric_record(best)
    samples = [row["worst"] for row in series if row.get("has_value")]
    metrics_meta = {
        "primary": {"name": metric, "label": metric_label(str(metric or "")),
                    "as_percent": metric_as_percent(str(metric or ""), samples), "direction": direction},
        "guards": [{"name": name, "label": metric_label(name), "direction": sense,
                    "as_percent": metric_as_percent(name)} for name, sense in guard_pairs],
    }
    verdict = build_verdict(metric_name=str(metric or ""), direction=direction,
                            baseline=baseline_record.get("score"), best=best_record.get("score"),
                            hypotheses=hypotheses, samples=samples)
    budget = {
        "requests": {"used": summary.get("requests")},
        "cost": {"used": summary.get("cost_usd")},
        "duration_seconds": summary.get("duration_seconds"),
    }
    return {"lab": summary, "conditions": safe_conditions, "baseline": baseline_record,
            "best": best_record, "hypotheses": hypotheses,
            "historical_hypotheses": historical, "current": current,
            "series": series, "metrics": metrics_meta, "verdict": verdict,
            "budget": budget, "problems": build_problems(summary),
            "version": summary.get("detail_version")}

def _path_segment(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value

def _directory_running(state_dir: Path) -> bool:
    for manifest_path in (state_dir / "runs").glob("*/manifest.json"):
        manifest = _read_json(manifest_path, {}) or {}
        if isinstance(manifest, dict) and manifest.get("status") == "running":
            return True
    return False

def _safe_archive_dir(root: Path, project_id: str, lab_id: str) -> Path | None:
    if not _path_segment(project_id) or not _path_segment(lab_id):
        return None
    root = root.resolve()
    archive_root = (root / "archive").resolve()
    candidate = (archive_root / project_id / lab_id).resolve()
    if candidate.is_symlink() or not candidate.is_dir() or not candidate.is_relative_to(archive_root):
        return None
    if candidate.parent.name != project_id or candidate.name != lab_id:
        return None
    return candidate

def _is_deletable_lab(root: Path, target: Path) -> bool:
    """A lab folder may sit in projects/, archive/, or directly in the labs root."""
    root = root.resolve()
    if target.is_symlink() or not target.is_dir() or target == root or not target.is_relative_to(root):
        return False
    parts = target.relative_to(root).parts
    if not parts or any(part in {".git", "internal"} for part in parts):
        return False
    if parts[0] in {"projects", "archive"}:
        return len(parts) >= 3
    return len(parts) == 1 and (target / "lab.json").is_file()

def _sqlite_lab_action_block(lab_id: str) -> tuple[int, dict[str, Any]] | None:
    """Refuse legacy file moves for labs whose paths are virtual SQLite projections."""
    from .sqlite_store import AmbiguousAliasError, EntityNotFoundError, SQLiteStore, StoreError

    store = SQLiteStore(resolve_home())
    if not store.is_initialized():
        return None
    try:
        store.get_lab(lab_id)
    except EntityNotFoundError:
        return None
    except AmbiguousAliasError:
        return 409, {"error": "ambiguous", "message": "Lab reference is ambiguous in SQLite"}
    except StoreError:
        return 503, {"error": "store_error", "message": "SQLite lab metadata could not be read"}
    return 409, {
        "error": "unsupported_db_lab_action",
        "message": "Archive, restore, and delete are not available for SQLite-backed labs until database relocation is implemented.",
    }


def _legacy_lab_action_block() -> tuple[int, dict[str, Any]]:
    """Return the read-only contract for legacy dashboard lifecycle actions."""
    return 409, {
        "error": "legacy_read_only",
        "code": "E_LEGACY_READ_ONLY",
        "message": "Legacy labs are read-only; archive export preserves their files without migrating history.",
        "hint": "Export with `--config <legacy-lab.json> storage export --output archive.zip`, then prepare a new lab.",
    }

def archive_lab(home: Path, project_id: str, lab_id: str, *, root_id: str | None = None) -> tuple[int, dict[str, Any]]:
    """Refuse physical archive moves for legacy labs."""
    if not _path_segment(project_id) or not _path_segment(lab_id):
        return 404, {"error": "not_found"}
    blocked = _sqlite_lab_action_block(lab_id)
    if blocked is not None:
        return blocked
    home = home.resolve()
    try:
        located = _find_active_lab(home, project_id, lab_id, root_id=root_id)
    except AmbiguousLab as error:
        return 409, {"error": "ambiguous", "roots": error.roots}
    if located is None:
        return 404, {"error": "not_found"}
    root, _project, _item, target = located
    if not _is_deletable_lab(root, target):
        return 404, {"error": "not_found"}
    if _directory_running(target):
        return 409, {"error": "running"}
    return _legacy_lab_action_block()

def restore_lab(home: Path, project_id: str, lab_id: str, *, root_id: str | None = None) -> tuple[int, dict[str, Any]]:
    """Refuse physical restore moves for legacy labs."""
    if not _path_segment(project_id) or not _path_segment(lab_id):
        return 404, {"error": "not_found"}
    blocked = _sqlite_lab_action_block(lab_id)
    if blocked is not None:
        return blocked
    home = home.resolve()
    try:
        archived = _find_archived_lab(home, project_id, lab_id, root_id=root_id)
    except AmbiguousLab as error:
        return 409, {"error": "ambiguous", "roots": error.roots}
    if archived is None:
        return 404, {"error": "not_found"}
    root, archive_dir = archived
    if not _is_deletable_lab(root, archive_dir):
        return 404, {"error": "not_found"}
    if _directory_running(archive_dir):
        return 409, {"error": "running"}
    return _legacy_lab_action_block()

def delete_lab(home: Path, project_id: str, lab_id: str, *, root_id: str | None = None) -> tuple[int, dict[str, Any]]:
    """Refuse physical deletion of legacy labs."""
    if not _path_segment(project_id) or not _path_segment(lab_id):
        return 404, {"error": "not_found"}
    blocked = _sqlite_lab_action_block(lab_id)
    if blocked is not None:
        return blocked
    home = home.resolve()
    try:
        located = _find_active_lab(home, project_id, lab_id, root_id=root_id)
        archive = None if located is not None else _find_archived_lab(home, project_id, lab_id, root_id=root_id)
    except AmbiguousLab as error:
        return 409, {"error": "ambiguous", "roots": error.roots}
    if located is None and archive is None:
        return 404, {"error": "not_found"}
    root, target = (located[0], located[3]) if located is not None else archive
    if not _is_deletable_lab(root, target):
        return 404, {"error": "not_found"}
    if _directory_running(target):
        return 409, {"error": "running"}
    return _legacy_lab_action_block()

def render_live_page(home: Path) -> str:
    home = home.resolve()
    return render_unified_dashboard(dashboard_status(home), {}, live=True, root_id=home_root_id(home))
