"""Loopback-only live dashboard. Mutations are delete, archive, and restore of one chosen lab."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

from .store import _sha256
from .catalog import _human_goal, lab_lifecycle, lab_status, resolve_lab_path
from .dashboard import STATUS_LABELS, assign_short_labels

from .dash_io import _merged_usage, _read_json


def _historical_rechecks(root: Path, state_dir: Path, manifests: list[dict[str, Any]],
                         config: dict[str, Any], state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Overlay measured rechecks in the chart without changing original trial records."""
    original = {str(item.get("id")): item for item in manifests if isinstance(item, dict)}
    baseline = original.get(str(state.get("baseline_id")), {})
    baseline_runs = baseline.get("runs", [])
    baseline_details = baseline_runs[0].get("details", {}) if baseline_runs else {}
    evaluator = config.get("evaluator", [])
    repeats = config.get("repeats")
    if (not isinstance(evaluator, list) or len(evaluator) < 2
            or not isinstance(evaluator[1], str) or not isinstance(repeats, int) or repeats < 1):
        return {}
    evaluator_path = Path(evaluator[1])
    if not evaluator_path.is_file():
        return {}
    evaluator_sha = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    expected_configuration = str(baseline_details.get("configuration") or "")
    rechecks: dict[str, dict[str, Any]] = {}
    for proof_path in sorted((root / "internal" / "historical-gap-recovery").glob("*/verification.json")):
        proof = _read_json(proof_path, {}) or {}
        if not isinstance(proof, dict):
            continue
        trial_id = str(proof.get("trial_id") or "")
        source = original.get(trial_id)
        if (proof.get("status") != "complete" or proof.get("lab_id") != state_dir.name
                or not source or not trial_id.startswith("trial-")
                or proof.get("baseline_commit") != source.get("base_commit")
                or proof.get("candidate_hashes") != source.get("candidate_hashes")
                or proof.get("input_hashes") != source.get("input_hashes")
                or proof.get("evaluator_sha256") != evaluator_sha
                or proof.get("repeats") != repeats
                or proof.get("provider") != baseline_details.get("provider")
                or proof.get("seed") != baseline_details.get("seed")
                or proof.get("model") != expected_configuration.split(";", 1)[0]
                or proof.get("fallback") is not False
                or proof.get("cost_status") not in {"confirmed", "partial"}
                or any(not isinstance(proof.get(name, 0), int) or isinstance(proof.get(name, 0), bool)
                       or proof.get(name, 0) < 0
                       for name in ("unknown_cost_attempts", "prior_unknown_cost_attempts"))
                or (proof.get("cost_status") == "confirmed"
                    and (proof.get("unknown_cost_attempts", 0) or proof.get("prior_unknown_cost_attempts", 0)))):
            continue
        patch = state_dir / "runs" / trial_id / "change.patch"
        if not patch.is_file() or hashlib.sha256(patch.read_bytes()).hexdigest() != proof.get("patch_sha256"):
            continue
        runs = []
        for index in range(1, repeats + 1):
            repeat = proof_path.parent / f"repeat_{index:02d}"
            result = _read_json(repeat / "result.json", {}) or {}
            if not isinstance(result, dict):
                break
            metrics = result.get("metrics", {})
            usage = result.get("usage", {})
            details = result.get("details", {})
            metadata = _read_json(repeat / "provider-metadata.json", [])
            payloads = repeat / "payloads.jsonl"
            try:
                with payloads.open(encoding="utf-8") as stream:
                    payload_count = sum(1 for _ in stream)
            except (OSError, UnicodeError):
                payload_count = 0
            if (not isinstance(metrics, dict) or not isinstance(usage, dict) or not isinstance(details, dict)
                    or any(not isinstance(metrics.get(name), (int, float)) or isinstance(metrics.get(name), bool)
                           or not math.isfinite(float(metrics[name]))
                           for name in ("accuracy", "cases"))
                    or metrics["cases"] != 39
                    or not isinstance(usage.get("requests"), int) or usage["requests"] < 39
                    or not isinstance(usage.get("cost_usd"), (int, float))
                    or not math.isfinite(float(usage["cost_usd"]))
                    or details.get("configuration") != expected_configuration
                    or details.get("provider") != proof.get("provider")
                    or details.get("seed") != proof.get("seed")
                    or not isinstance(metadata, list) or len(metadata) != 39
                    or any(not isinstance(row, dict) or str(row.get("provider", "")).casefold()
                           != str(proof["provider"]).casefold() for row in metadata)
                    or payload_count != 39):
                break
            runs.append({"metrics": metrics, "usage": usage, "details": details})
        if len(runs) != repeats:
            continue
        rechecked = dict(source)
        rechecked.update({"status": "rechecked", "reason": "", "runs": runs,
                          "original_status": source.get("status"),
                          "original_reason": source.get("reason"),
                          "cost_status": proof.get("cost_status"),
                          "unknown_cost_attempts": proof.get("unknown_cost_attempts", 0),
                          "prior_unknown_cost_attempts": proof.get("prior_unknown_cost_attempts", 0),
                          "recheck_dir": str(proof_path.parent),
                          "recheck_source": proof_path.relative_to(root).as_posix()})
        rechecks[trial_id] = rechecked
    return rechecks


def _probe_comparison(state_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[Path]]:
    metrics = config.get("metrics", {}) if isinstance(config.get("metrics"), dict) else {}
    primary = str(metrics.get("primary") or "")
    direction = str(metrics.get("direction") or "max")
    groups: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    files: list[Path] = []
    probes_dir = state_dir / "probes"
    probe_files = sorted(probes_dir.glob("**/summary.json")) if probes_dir.is_dir() else []
    for summary_path in probe_files:
        summary = _read_json(summary_path, {}) or {}
        if not isinstance(summary, dict) or summary.get("status") != "complete":
            continue
        probe_id = str(summary.get("id") or summary_path.parent.name)
        if probe_id in seen_ids:
            continue
        seen_ids.add(probe_id)
        files.append(summary_path)
        group_id = str(summary.get("group") or "default")
        usage = {"requests": 0, "cost_usd": 0.0}
        for run in summary.get("runs", []):
            if not isinstance(run, dict):
                continue
            row_usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
            usage["requests"] += int(row_usage.get("requests") or 0)
            usage["cost_usd"] += float(row_usage.get("cost_usd") or 0)
        groups.setdefault(group_id, []).append({
            "id": probe_id,
            "label": str(summary.get("label") or summary.get("id") or "probe"),
            "commit": summary.get("commit"),
            "import_source": summary.get("import_source"),
            "worst": summary.get("worst"),
            "mean": summary.get("mean"),
            "spread": summary.get("spread"),
            "repeats": summary.get("runs", []),
            "usage": usage,
            "source": summary_path.relative_to(state_dir).as_posix(),
        })
    if not groups:
        return None, files
    payload_groups: list[dict[str, Any]] = []
    for group_id, entries in groups.items():
        ranked = sorted(
            entries,
            key=lambda item: float(item.get("worst") if item.get("worst") is not None else float("-inf")),
            reverse=direction != "min",
        )
        payload_groups.append({
            "id": group_id,
            "label": group_id,
            "baseline_label": ranked[0]["label"] if ranked else "",
            "entries": ranked,
        })
    return {"primary": primary, "direction": direction, "groups": payload_groups}, files


def _lab_replay_rechecks(state_dir: Path, manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Overlay exact replays stored under runs/<trial>/rechecks without editing the trial."""
    original = {str(item.get("id")): item for item in manifests if isinstance(item, dict)}
    found: dict[str, dict[str, Any]] = {}
    for proof_path in sorted((state_dir / "runs").glob("*/rechecks/*/verification.json")):
        proof = _read_json(proof_path, {}) or {}
        if not isinstance(proof, dict) or proof.get("status") != "complete":
            continue
        trial_id = str(proof.get("trial_id") or proof_path.parents[2].name)
        source = original.get(trial_id)
        if not source or proof.get("recheck_kind") != "exact":
            continue
        if proof.get("candidate_hashes") != source.get("candidate_hashes"):
            continue
        if proof.get("input_hashes") != source.get("input_hashes"):
            continue
        patch = state_dir / "runs" / trial_id / "change.patch"
        if patch.is_file() and proof.get("patch_sha256") and hashlib.sha256(patch.read_bytes()).hexdigest() != proof.get("patch_sha256"):
            continue
        repeats = proof.get("repeats")
        if not isinstance(repeats, int) or repeats < 1:
            continue
        runs = []
        for index in range(1, repeats + 1):
            result = _read_json(proof_path.parent / f"repeat_{index:02d}" / "result.json", {}) or {}
            metrics = result.get("metrics") if isinstance(result, dict) else None
            if not isinstance(metrics, dict) or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in metrics.values()
            ):
                break
            runs.append({"metrics": metrics, "usage": result.get("usage") or {}, "details": result.get("details") or {}})
        if len(runs) != repeats:
            continue
        rechecked = dict(source)
        rechecked.update({
            "status": "rechecked", "reason": "", "runs": runs,
            "original_status": source.get("status"), "original_reason": source.get("reason"),
            "recheck_kind": "exact",
            "unknown_cost_attempts": proof.get("unknown_cost_attempts", 0),
            "recheck_dir": str(proof_path.parent),
            "recheck_source": proof_path.relative_to(state_dir).as_posix(),
        })
        found[trial_id] = rechecked
    return found

def _usage_totals(state_dir: Path) -> dict[str, Any]:
    requests = 0.0
    cost = 0.0
    unknown_requests = 0.0
    unknown_cost = 0.0
    seen: set[Path] = set()
    for manifest_path in (state_dir / "runs").glob("*/manifest.json"):
        manifest = _read_json(manifest_path, {}) or {}
        run_id = str(manifest.get("id", manifest_path.parent.name))
        for index, run in enumerate(manifest.get("runs", []), start=1):
            usage_path = state_dir / "runs" / run_id / f"repeat_{index:02d}" / "usage.json"
            seen.add(usage_path.resolve())
            usage = _merged_usage(run, usage_path)
            requests += float(usage.get("requests", 0) or 0)
            cost += float(usage.get("cost_usd", 0) or 0)
            unknown_requests += float(usage.get("unknown_requests", 0) or 0)
            unknown_cost += float(usage.get("unknown_cost_usd", 0) or 0)
        for usage_path in (state_dir / "runs" / run_id).glob("repeat_*/usage.json"):
            if usage_path.resolve() in seen:
                continue
            usage = _read_json(usage_path, {}) or {}
            requests += float(usage.get("requests", 0) or 0)
            cost += float(usage.get("cost_usd", 0) or 0)
            unknown_requests += float(usage.get("unknown_requests", 0) or 0)
            unknown_cost += float(usage.get("unknown_cost_usd", 0) or 0)
    return {"requests": int(requests), "cost_usd": round(cost, 10),
            "unknown_requests": int(unknown_requests), "unknown_cost_usd": round(unknown_cost, 10)}

def _lab_summary(root: Path, project: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()  # state_dir below is resolved, and later paths are taken relative to root
    state_dir = resolve_lab_path(root, project, str(item["path"]))
    config = _read_json(state_dir / "lab.json", {})
    state = _read_json(state_dir / "state.json", {})
    config_copy_mismatch = False
    expected_copy_sha = state.get("config_copy_sha256")
    lab_config_path = state_dir / "lab.json"
    if isinstance(expected_copy_sha, str) and expected_copy_sha:
        if not lab_config_path.is_file() or _sha256(lab_config_path) != expected_copy_sha:
            config_copy_mismatch = True
    metrics_config = {} if config_copy_mismatch else config
    meta = _read_json(state_dir / "lab-meta.json", {})
    manifests = [_read_json(path, {}) for path in sorted((state_dir / "runs").glob("*/manifest.json"))]
    requests = 0
    cost = 0.0
    duration = 0.0
    current_run = None
    current_repeat = None
    latest_trial = None
    latest_error = None
    unknown_requests = 0
    unknown_cost = 0.0
    rate_limited = 0
    retry_wait = 0.0
    completed_count = 0
    observed_usage: set[Path] = set()
    usage_files = []
    for manifest in manifests:
        runs = manifest.get("runs", [])
        completed_count += len(runs)
        run_id = str(manifest.get("id", ""))
        run_root = state_dir / "runs" / run_id
        for index, run in enumerate(runs, start=1):
            usage_path = run_root / f"repeat_{index:02d}" / "usage.json"
            usage_files.append(usage_path)
            observed_usage.add(usage_path.resolve())
            usage = _merged_usage(run, usage_path)
            requests += int(usage.get("requests", 0) or 0)
            cost += float(usage.get("cost_usd", 0) or 0)
            if usage.get("unknown_requests"):
                unknown_requests += int(usage.get("unknown_requests", 0) or 0)
                unknown_cost += float(usage.get("unknown_cost_usd", 0) or 0)
            stats = _read_json(usage_path.with_name("net-stats.json"), {}) or {}
            if isinstance(stats, dict):
                rate_limited += int(stats.get("rate_limited_responses") or 0)
                retry_wait += float(stats.get("retry_wait_seconds") or 0)
            duration += float(run.get("duration_seconds", 0) or 0)
        if manifest.get("status") == "running":
            current_run = run_id
            repeat_dirs = sorted(run_root.glob("repeat_*"))
            for index, repeat_dir in enumerate(repeat_dirs):
                usage = _read_json(repeat_dir / "usage.json", {})
                if index == len(runs) and current_repeat is None:
                    current_repeat = repeat_dir.name
                usage_files.append(repeat_dir / "usage.json")
                if (repeat_dir / "usage.json").resolve() in observed_usage:
                    continue
                requests += int(usage.get("requests", 0) or 0)
                cost += float(usage.get("cost_usd", 0) or 0)
                if usage.get("unknown_requests"):
                    unknown_requests += int(usage.get("unknown_requests", 0) or 0)
                    unknown_cost += float(usage.get("unknown_cost_usd", 0) or 0)
                stats = _read_json(repeat_dir / "net-stats.json", {}) or {}
                if isinstance(stats, dict):
                    rate_limited += int(stats.get("rate_limited_responses") or 0)
                    retry_wait += float(stats.get("retry_wait_seconds") or 0)
        if str(manifest.get("id", "")).startswith("trial-"):
            latest_trial = manifest
        if manifest.get("status") in {"invalid", "crash", "timeout", "budget", "interrupted"} and manifest.get("reason"):
            latest_error = str(manifest["reason"])
    best = next((manifest for manifest in manifests if manifest.get("id") == state.get("best_id")), None)
    metric = meta.get("primary_metric") or metrics_config.get("metrics", {}).get("primary", "")
    values = [run.get("metrics", {}).get(metric) for run in (best or {}).get("runs", [])]
    values = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    baseline = next((manifest for manifest in manifests if manifest.get("id") == state.get("baseline_id")), None)
    baseline_values = [run.get("metrics", {}).get(metric) for run in (baseline or {}).get("runs", [])]
    baseline_values = [value for value in baseline_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    status = "running" if current_run else lab_lifecycle(state_dir)
    stage = {"running": "Сейчас идёт", "ready": "Готова к опытам",
             "completed": "Завершена", "setup_pending": "Подготовка"}.get(status, status)
    activity = f"{current_run} / {current_repeat or 'оценка'}" if current_run else ""
    direction = metrics_config.get("metrics", {}).get("direction", "max")
    sparkline: list[float | None] = []
    for manifest in manifests:
        run_id = str(manifest.get("id", ""))
        if not (run_id.startswith("baseline") or run_id.startswith("trial-")):
            continue
        samples = [run.get("metrics", {}).get(metric) for run in manifest.get("runs", []) if isinstance(run, dict)]
        samples = [float(value) for value in samples if isinstance(value, (int, float)) and not isinstance(value, bool)]
        sparkline.append((min(samples) if direction == "max" else max(samples)) if samples else None)
    probe_comparison, probe_report_files = _probe_comparison(state_dir, metrics_config)
    response_repro = None
    case_repro_path = state_dir / "case-repro.json"
    if case_repro_path.is_file():
        case_repro = _read_json(case_repro_path, None)
        if isinstance(case_repro, dict):
            mismatched = case_repro.get("mismatched")
            total = case_repro.get("total")
            if isinstance(mismatched, int) and isinstance(total, int):
                response_repro = {"mismatched": mismatched, "total": total}
    recheck_root = root / "internal" / "historical-gap-recovery"
    recheck_files = list(recheck_root.glob("*/verification.json")) + list(recheck_root.glob("*/repeat_*/result.json"))
    recheck_files += list((state_dir / "runs").glob("*/rechecks/*/verification.json"))
    recheck_files += list((state_dir / "runs").glob("*/rechecks/*/repeat_*/result.json"))
    detail_files = [state_dir / "state.json", state_dir / "lab-meta.json", state_dir / "history.json",
                    *usage_files, *probe_report_files, *recheck_files,
                    *((state_dir / "runs").glob("*/manifest.json")),
                    *((state_dir / "runs").glob("*/change.patch"))]
    return {
        "id": item.get("id", state_dir.name), "project_id": project.get("id", ""),
        "name": _human_goal(state_dir, meta, item, str(project.get("name", "Project")),
                            str(metrics_config.get("metrics", {}).get("primary") or item.get("primary_metric") or "")),
        "goal": _human_goal(state_dir, meta, item, str(project.get("name", "Project")),
                             str(metrics_config.get("metrics", {}).get("primary") or item.get("primary_metric") or "")),
        "status": status, "stage": stage, "activity": activity,
        "latest_trial_id": latest_trial.get("id") if latest_trial else None,
        "latest_trial_status": latest_trial.get("status") if latest_trial else None,
        "latest_trial_label": STATUS_LABELS.get(str(latest_trial.get("status")), "") if latest_trial else "",
        "latest_trial_reason": latest_trial.get("reason") if latest_trial else None,
        "comparison_note": item.get("comparison_note", ""),
        "comparison_status": item.get("comparison_status"),
        "metric": metric,
        "best_score": (min(values) if metrics_config.get("metrics", {}).get("direction") == "max" else max(values)) if values else None,
        "best_range": [min(values), max(values)] if values else None,
        "baseline_score": ((min(baseline_values) if metrics_config.get("metrics", {}).get("direction") == "max"
                            else max(baseline_values)) if baseline_values else None),
        "guards": [item.get("name") for item in metrics_config.get("metrics", {}).get("guards", [])
                   if isinstance(item, dict) and isinstance(item.get("name"), str)],
        "repeats": metrics_config.get("repeats"),
        "config_copy_mismatch": config_copy_mismatch,
        "experiments": sum(str(manifest.get("id", "")).startswith("trial-") for manifest in manifests),
        "requests": requests,
        "cost_usd": round(cost, 10),
        "unknown_requests": unknown_requests, "unknown_cost_usd": round(unknown_cost, 10),
        "rate_limited_responses": rate_limited, "retry_wait_seconds": round(retry_wait, 3),
        "duration_seconds": round(duration, 2), "current_run": current_run,
        "current_repeat": current_repeat, "repeats_completed": completed_count,
        "latest_error": latest_error, "sparkline": sparkline,
        "probe_comparison": probe_comparison,
        **({"response_repro": response_repro} if response_repro is not None else {}),
        "detail_version": hashlib.sha256("|".join(
            f"{path.relative_to(root).as_posix()}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
            for path in sorted(set(detail_files)) if path.is_file()
        ).encode("utf-8")).hexdigest()[:16],
    }

def _compute_dashboard_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    catalog = _read_json(root / "catalog.json", {"projects": [], "archived": []})
    catalog = catalog if isinstance(catalog, dict) else {"projects": [], "archived": []}
    raw_projects = catalog.get("projects", [])
    catalog_projects = [item for item in raw_projects if isinstance(item, dict)] if isinstance(raw_projects, list) else []
    projects_by_source = {str(item.get("source_root", "")): item for item in catalog_projects}
    indexed = set()
    for project in catalog_projects:
        project_labs = project.get("labs", [])
        for lab in project_labs if isinstance(project_labs, list) else []:
            if isinstance(lab, dict) and isinstance(lab.get("path"), str):
                indexed.add(str(resolve_lab_path(root, project, lab["path"])))
    config_paths = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    if (root / "lab.json").is_file():
        config_paths.append(root / "lab.json")
    for config_path in config_paths:
        raw = _read_json(config_path, {}) or {}
        configured_state = raw.get("state_dir")
        if (config_path.parent / "state.json").is_file():
            state_dir = config_path.parent.resolve()
        elif isinstance(configured_state, str) and configured_state.strip():
            configured_path = Path(configured_state)
            state_dir = (configured_path if configured_path.is_absolute()
                         else config_path.parent / configured_path).resolve()
        else:
            state_dir = config_path.parent.resolve()
        if not state_dir.is_relative_to(root):
            continue
        if "archive" in state_dir.relative_to(root).parts or str(state_dir) in indexed:
            continue
        source = str(raw.get("source_root", "Project"))
        project = projects_by_source.get(source)
        if project is None:
            project = {"id": state_dir.parent.name, "name": Path(source).name,
                       "source_root": source, "labs": []}
            catalog.setdefault("projects", []).append(project)
            catalog_projects.append(project)
            projects_by_source[source] = project
        meta = _read_json(state_dir / "lab-meta.json", {}) or {}
        project.setdefault("labs", []).append({
            "id": state_dir.name, "path": state_dir.relative_to(root).as_posix(),
            "name": meta.get("goal") or project.get("name") or state_dir.name, "goal": meta.get("goal", ""),
            "status": lab_status(state_dir),
        })
    projects = []
    for project in catalog_projects:
        labs = []
        project_labs = project.get("labs", [])
        for item in project_labs if isinstance(project_labs, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            state_dir = resolve_lab_path(root, project, item["path"])
            if state_dir.is_relative_to(root) and state_dir.is_dir():
                labs.append(_lab_summary(root, project, item))
        assign_short_labels(labs)
        projects.append({"id": project.get("id", ""), "name": project.get("name", "Project"), "labs": labs})
    raw_archived = catalog.get("archived", [])
    archived = []
    if isinstance(raw_archived, list):
        for item in raw_archived:
            if not isinstance(item, dict):
                continue
            value = item.get("archived_path", item.get("path", "Archived lab"))
            path = Path(value)
            archive_dir = path.resolve() if path.is_absolute() else (root / path).resolve()
            if not archive_dir.is_dir():
                archive_dir = next((candidate for candidate in (root / "archive").glob("*/*")
                                    if candidate.name == Path(value).name and candidate.is_dir()), archive_dir)
            usage = _usage_totals(archive_dir) if archive_dir.is_dir() else {"requests": 0, "cost_usd": 0.0}
            experiments = sum(1 for _ in (archive_dir / "runs").glob("trial-*/manifest.json")) if archive_dir.is_dir() else 0
            entry = {"project": item.get("project", "Project"),
                     "project_id": archive_dir.parent.name if archive_dir.is_dir() else "",
                     "id": archive_dir.name if archive_dir.is_dir() else Path(value).name,
                     "name": item.get("name") or Path(value).name,
                     "status": item.get("status", "archived"),
                     "experiments": experiments,
                     **usage}
            entry["hidden"] = (entry["status"] == "setup_pending" and not entry.get("requests")
                               and not entry.get("cost_usd"))
            if archive_dir.is_dir():
                try:
                    summary = _lab_summary(root, {
                        "id": entry["project_id"] or "archive",
                        "name": entry["project"],
                        "labs": [],
                    }, {
                        "id": entry["id"],
                        "path": archive_dir.relative_to(root).as_posix(),
                        "name": entry["name"],
                        "goal": entry["name"],
                    })
                except (OSError, ValueError):
                    summary = {}
                # Keep the catalog status and the usage totals already counted above.
                for key in ("metric", "best_score", "stage", "sparkline", "latest_trial_label", "detail_version"):
                    if key in summary:
                        entry[key] = summary[key]
                if summary.get("name"):
                    entry["name"] = summary["name"]
            archived.append(entry)
    assign_short_labels(archived)
    archived_totals = {
        "requests": sum(int(item.get("requests", 0) or 0) for item in archived),
        "cost_usd": round(math.fsum(float(item.get("cost_usd", 0) or 0) for item in archived), 10),
        "unknown_requests": sum(int(item.get("unknown_requests", 0) or 0) for item in archived),
        "unknown_cost_usd": round(math.fsum(float(item.get("unknown_cost_usd", 0) or 0) for item in archived), 10),
    }
    return {"projects": projects, "archived": archived, "archived_totals": archived_totals,
            "updated_at": time.time()}

_STATUS_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}

_STATUS_CACHE_LOCK = threading.Lock()

_STATUS_NAMES = {
    "catalog.json", "lab.json", "state.json", "lab-meta.json", "history.json",
    "openrouter_budget.json", "manifest.json", "summary.json", "comparison.json",
    "usage.json", "change.patch", "verification.json", "result.json",
}

_STATUS_SKIP = {".git", "__pycache__", ".venv", "node_modules"}

def _status_signature(root: Path) -> str:
    """Fingerprint the files the summary reads, so idle polls skip the JSON parse."""
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _STATUS_SKIP]
        for name in filenames:
            if name not in _STATUS_NAMES:
                continue
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(f"{path.relative_to(root).as_posix()}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return digest.hexdigest()

def dashboard_status(root: Path) -> dict[str, Any]:
    root = root.resolve()
    signature = _status_signature(root)
    key = str(root)
    with _STATUS_CACHE_LOCK:
        cached = _STATUS_CACHE.get(key)
        if cached and cached[0] == signature:
            return cached[1]
    payload = _compute_dashboard_status(root)
    payload["revision"] = signature[:16]
    with _STATUS_CACHE_LOCK:
        if _status_signature(root) == signature:
            _STATUS_CACHE[key] = (signature, payload)
    return payload
