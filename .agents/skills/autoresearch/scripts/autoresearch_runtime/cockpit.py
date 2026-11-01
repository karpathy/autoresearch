"""Compact, bounded agent guidance for AutoResearch labs."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import LabConfig
from .errors import LabError
from .signals import build_research_signals, evaluate_goal_status


def _metric_summary(run: dict[str, Any] | None, metric: str, direction: str) -> dict[str, Any]:
    values: list[float] = []
    if isinstance(run, dict):
        records = run.get("runs")
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict) or record.get("status", "ok") != "ok":
                    continue
                metrics = record.get("metrics")
                raw = metrics.get(metric) if isinstance(metrics, dict) else None
                if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
                    values.append(float(raw))
        if not values:
            metrics = run.get("metrics")
            raw = metrics.get(metric) if isinstance(metrics, dict) else None
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
                values.append(float(raw))
    if not values:
        return {"mean": None, "decision_worst": None, "min": None, "max": None, "completed_repeats": 0}
    return {
        "mean": sum(values) / len(values),
        "decision_worst": min(values) if direction == "max" else max(values),
        "min": min(values), "max": max(values),
        "completed_repeats": len(values),
    }


def _load_manifest(config: LabConfig, run_id: str | None) -> dict[str, Any] | None:
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        return None
    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    if store.is_initialized():
        try:
            manifest = store.load_manifest(config.lab_id, run_id)
            if manifest is not None:
                return manifest
        except Exception:
            pass
    from .store import _read_json

    for parent in (config.state_dir / "runs", config.state_dir / "controls"):
        manifest = _read_json(parent / run_id / "manifest.json", None)
        if isinstance(manifest, dict):
            return manifest
    return None


def _manifest_epoch(manifest: dict[str, Any] | None, current_epoch: int) -> int:
    try:
        value = (manifest or {}).get("epoch")
        return int(value) if value is not None else 1
    except (TypeError, ValueError):
        return 1


def _extract_manifest_error_tail(manifest: dict[str, Any] | None) -> str | None:
    if not isinstance(manifest, dict):
        return None
    runs = manifest.get("runs")
    if isinstance(runs, list) and runs:
        last = runs[-1]
        if isinstance(last, dict) and last.get("error_tail"):
            return str(last["error_tail"])
    if manifest.get("error_tail"):
        return str(manifest["error_tail"])
    return None


def _build_dead_ends_digest(
    manifests: list[dict[str, Any]],
    current_epoch: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    seen_keys: set[str] = set()
    digest: list[dict[str, Any]] = []

    for m in manifests:
        if not isinstance(m, dict):
            continue
        try:
            m_epoch = int(m.get("epoch", 0) if m.get("epoch") is not None else 0)
        except (TypeError, ValueError):
            m_epoch = 0
        if m_epoch != current_epoch:
            continue
        status = str(m.get("status") or "")
        if status not in {"rejected", "crash", "invalid", "timeout"}:
            continue

        hyp = m.get("hypothesis") if isinstance(m.get("hypothesis"), dict) else {}
        key = str(m.get("candidate_hash") or hyp.get("title") or m.get("title") or m.get("id") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        title = str(hyp.get("title") or m.get("title") or m.get("id") or "")[:80]
        decision = m.get("decision") if isinstance(m.get("decision"), dict) else {}
        reason = str(decision.get("reason") or m.get("reason") or status or "")[:100]
        entry = {
            "id": str(m.get("id") or ""),
            "status": status,
            "title": title,
            "reason": reason,
        }
        digest.append(entry)
        if len(digest) >= limit:
            break

    return digest


def _recent_manifests(
    config: Any,
    epoch_or_state: Any = 1,
    limit: int = 100,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    from .store import _read_json

    if "current_epoch" in kwargs:
        current_epoch = int(kwargs["current_epoch"])
    elif isinstance(epoch_or_state, dict):
        current_epoch = int(epoch_or_state.get("current_epoch") or epoch_or_state.get("epoch") or 1)
    elif epoch_or_state is not None:
        try:
            current_epoch = int(epoch_or_state)
        except (TypeError, ValueError):
            current_epoch = 1
    else:
        current_epoch = 1

    if "limit" in kwargs:
        limit = kwargs["limit"]

    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    if store.is_initialized():
        try:
            return store.list_recent_manifests(config.lab_id, current_epoch, limit=limit)
        except Exception:
            pass
        try:
            page = store.list_runs(config.lab_id, limit=max(1, min(limit, 100)), epoch_id=current_epoch)
            return [manifest for item in page["items"]
                    if (manifest := store.load_manifest(config.lab_id, item["run_id"])) is not None]
        except Exception:
            pass

    run_root = config.state_dir / "runs"
    newest: list[tuple[float, str, dict[str, Any]]] = []
    cap = max(0, min(limit, 100))
    for path in run_root.glob("*/manifest.json"):
        manifest = _read_json(path, None)
        if not isinstance(manifest, dict) or _manifest_epoch(manifest, current_epoch) != current_epoch:
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        entry = (stamp, str(path), manifest)
        if cap and len(newest) < cap:
            heapq.heappush(newest, entry)
        elif cap and entry > newest[0]:
            heapq.heapreplace(newest, entry)
    return [entry[2] for entry in sorted(newest, reverse=True)]


def _comparable_manifests(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    if _manifest_epoch(candidate, 1) != _manifest_epoch(parent, 1):
        return False
    for key in ("config_sha256", "input_hashes", "evaluator"):
        left, right = candidate.get(key), parent.get(key)
        if left is None or right is None or left != right:
            return False
    return True


def _current_stage(config: LabConfig, state: dict[str, Any]) -> str:
    if state.get("setup_pending"):
        return "SETUP_PENDING"
    if not state.get("baseline_id") or (config.baseline_validation is not None and not state.get("baseline_verified")):
        return "BASELINE_NEEDED"
    return "ITERATING"


def _fingerprint_summary(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(manifest, dict):
        return None
    inputs = manifest.get("input_hashes")
    evaluator = manifest.get("evaluator")
    config_hash = manifest.get("config_sha256")
    if inputs is None and evaluator is None and config_hash is None:
        return None
    input_blob = json.dumps(inputs or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    evaluator_blob = json.dumps(evaluator or [], separators=(",", ":"), ensure_ascii=False)
    return {
        "config": str(config_hash)[:12] if isinstance(config_hash, str) else None,
        "inputs": hashlib.sha256(input_blob.encode("utf-8")).hexdigest()[:12],
        "input_count": len(inputs) if isinstance(inputs, dict) else 0,
        "evaluator": hashlib.sha256(evaluator_blob.encode("utf-8")).hexdigest()[:12],
    }


def _origin_summary(manifest: dict[str, Any]) -> str:
    origins: set[str] = set()
    for record in manifest.get("runs", []) if isinstance(manifest.get("runs"), list) else []:
        if not isinstance(record, dict) or record.get("status", "ok") != "ok" or not isinstance(record.get("metrics"), dict):
            continue
        details = record.get("details")
        origin = details.get("evidence_origin") if isinstance(details, dict) else None
        origins.add(origin if isinstance(origin, str) else "unknown")
    if not origins:
        return "unknown"
    return next(iter(origins)) if len(origins) == 1 else "mixed"


def _evidence_item(config: LabConfig, manifest: dict[str, Any], roles: list[str]) -> dict[str, Any]:
    run_id = str(manifest.get("id") or "unknown")
    primary = _metric_summary(manifest, config.primary, config.direction)
    expected = int(getattr(config, "repeats", 0) or 0)
    from .sqlite_store import SQLiteStore
    run_uuid = SQLiteStore().resolve_run_id(config.lab_id, run_id)
    source = f"/api/v1/labs/{config.lab_id}/runs/{run_uuid}"
    source_kind = "api"
    return {
        "id": run_id,
        "roles": roles,
        "status": manifest.get("status", "unknown"),
        "epoch": _manifest_epoch(manifest, 1),
        "primary": primary["mean"],
        "decision_worst": primary["decision_worst"],
        "completed_repeats": primary["completed_repeats"],
        "expected_repeats": expected or None,
        "evidence_origin": _origin_summary(manifest),
        "source": source,
        "source_kind": source_kind,
        "metrics": {metric: {"direction": direction, **_metric_summary(manifest, metric, direction)}
                    for metric, direction in ((config.primary, config.direction), *config.guards)},
        "fingerprint": _fingerprint_summary(manifest),
    }


def _evidence(config: LabConfig, manifests: list[tuple[str, dict[str, Any] | None]]) -> list[dict[str, Any]]:
    merged: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for role, manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        run_id = str(manifest.get("id") or "unknown")
        if run_id not in merged:
            merged[run_id] = (_evidence_item(config, manifest, []), [])
        item, roles = merged[run_id]
        if role not in roles:
            roles.append(role)
    result = []
    for item, roles in merged.values():
        item["roles"] = roles
        result.append(item)
    # The guide already carries best/baseline in state; two linked runs keep
    # the evidence useful while staying within the agent output budget.
    return result[:2]


def _argv(config: LabConfig, *args: str) -> list[str]:
    scripts = Path(__file__).resolve().parents[1]
    return [sys.executable, str(scripts / "lab.py"), "--lab", str(config.lab_id), *args]


def _command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def get_state_card(config: LabConfig, state: dict[str, Any]) -> dict[str, Any]:
    baseline_id = state.get("baseline_id")
    baseline_verified = bool(state.get("baseline_verified"))
    best_id = state.get("best_id")
    best_commit = state.get("best_commit")
    current_epoch = int(state.get("current_epoch") or 1)

    stage = _current_stage(config, state)

    baseline_manifest = _load_manifest(config, baseline_id)
    best_manifest = _load_manifest(config, best_id) if best_id else None
    baseline_stats = _metric_summary(baseline_manifest, config.primary, config.direction)
    best_stats = _metric_summary(best_manifest, config.primary, config.direction)
    if best_manifest is None and baseline_manifest is not None:
        best_manifest = baseline_manifest
        best_stats = baseline_stats

    from .sqlite_store import SQLiteStore
    spent = SQLiteStore().lab_usage_totals(config.lab_id)
    spent["confirmed_cost_usd"] = float(spent.get("cost_usd", 0.0) or 0.0)
    if spent.get("unknown_requests") or spent.get("unknown_costs"):
        spent["cost_usd"] = None
    budget_card = {"time_remaining_seconds": None}
    budget = config.budget
    if budget:
        max_seconds = budget.get("max_seconds") if isinstance(budget, dict) else getattr(budget, "max_seconds", None)
        if max_seconds is not None:
            budget_card["time_remaining_seconds"] = max(0.0, max_seconds - spent["seconds"])

    def score_card(run_id: str | None, manifest: dict[str, Any] | None, stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": run_id,
            "score": stats["decision_worst"],  # compatibility score follows the acceptance decision
            "mean": stats["mean"],
            "decision_worst": stats["decision_worst"],
            "completed_repeats": stats["completed_repeats"],
            "expected_repeats": int(getattr(config, "repeats", 0) or 0) or None,
            "evidence_origin": _origin_summary(manifest) if manifest else "unknown",
        }

    return {
        "stage": stage,
        "epoch": current_epoch,
        "primary_metric": config.primary,
        "direction": config.direction,
        "unit": getattr(config, "unit", "") or "",
        "baseline": {
            **score_card(baseline_id, baseline_manifest, baseline_stats),
            "verified": baseline_verified,
        },
        "best": {
            **score_card(best_id, best_manifest, best_stats),
            "commit": best_commit,
        },
        "budget": budget_card,
        "spent": spent,
        "lab_id": config.state_dir.name,
        "paths": {"config": str(config.path.resolve()), "state": str(config.state_dir.resolve()),
                  "source": str(config.source_root.resolve()), "candidate": str((config.state_dir / "candidate").resolve())},
    }


def get_impact_analysis(
    config: LabConfig,
    state: dict[str, Any],
    last_run: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(last_run, dict):
        return None
    run_id = str(last_run.get("id") or "current")
    status = str(last_run.get("status") or "unknown")
    score_stats = _metric_summary(last_run, config.primary, config.direction)
    score = score_stats["decision_worst"]
    mean_score = score_stats["mean"]

    parent_id = last_run.get("parent")
    parent_manifest = _load_manifest(config, str(parent_id)) if isinstance(parent_id, str) else None
    if parent_manifest is None and run_id == state.get("baseline_id"):
        parent_manifest = last_run
    parent_stats = _metric_summary(parent_manifest, config.primary, config.direction)
    baseline_manifest = _load_manifest(config, state.get("baseline_id"))
    baseline_stats = _metric_summary(baseline_manifest, config.primary, config.direction)
    if run_id == state.get("baseline_id"):
        delta_vs_baseline = 0.0 if score is not None else None
        delta_vs_baseline_mean = 0.0 if mean_score is not None else None
    elif baseline_manifest and _comparable_manifests(last_run, baseline_manifest):
        delta_vs_baseline = score - baseline_stats["decision_worst"] if score is not None and baseline_stats["decision_worst"] is not None else None
        delta_vs_baseline_mean = mean_score - baseline_stats["mean"] if mean_score is not None and baseline_stats["mean"] is not None else None
    else:
        delta_vs_baseline = None
        delta_vs_baseline_mean = None
    parent_comparable = bool(parent_manifest and _comparable_manifests(last_run, parent_manifest))
    delta_vs_best = score - parent_stats["decision_worst"] if parent_comparable and score is not None and parent_stats["decision_worst"] is not None else None
    delta_vs_best_mean = mean_score - parent_stats["mean"] if parent_comparable and mean_score is not None and parent_stats["mean"] is not None else None

    is_new_best = False
    if status == "accepted":
        if run_id == state.get("best_id") or run_id == state.get("baseline_id"):
            is_new_best = True
        elif parent_comparable and parent_stats["decision_worst"] is not None and score_stats["decision_worst"] is not None:
            if config.direction == "min":
                is_new_best = score_stats["decision_worst"] < parent_stats["decision_worst"]
            else:
                is_new_best = score_stats["decision_worst"] > parent_stats["decision_worst"]

    unit = getattr(config, "unit", "") or ""
    unit_str = f" {unit}" if unit else ""
    if status == "accepted":
        if is_new_best and score is not None:
            mean_str = f", mean {mean_score:.4g}{unit_str}" if mean_score is not None else ""
            delta_str = f" (decision delta vs parent: {delta_vs_best:+.4g})" if delta_vs_best is not None else ""
            verdict = f"УСПЕХ: Результат принят; best: {config.primary} worst {score:.4g}{unit_str}{mean_str}{delta_str}."
        else:
            verdict = f"Принято: {config.primary} worst {score:.4g}{unit_str}." if score is not None else "Принято."
    elif status == "rejected":
        verdict = f"ОТКЛОНЕНО: {last_run.get('reason', 'регресс метрики')}."
    else:
        verdict = f"Статус: {status}. {last_run.get('reason', '')}".strip()

    impact_data = {
        "run_id": run_id,
        "status": status,
        "score": score,
        "mean": mean_score,
        "decision_worst": score_stats["decision_worst"],
        "completed_repeats": score_stats["completed_repeats"],
        "delta_vs_baseline": delta_vs_baseline,
        "delta_vs_baseline_mean": delta_vs_baseline_mean,
        "delta_vs_best": delta_vs_best,
        "delta_vs_best_mean": delta_vs_best_mean,
        "comparison_run_id": parent_manifest.get("id") if parent_manifest else None,
        "comparison_status": "comparable" if parent_comparable else "incomparable_or_unavailable",
        "is_new_best": is_new_best,
        "verdict": verdict,
    }
    error_tail = _extract_manifest_error_tail(last_run)
    if error_tail:
        impact_data["error_tail"] = error_tail
    return impact_data


def get_recommended_actions(
    config: LabConfig,
    state: dict[str, Any],
    last_run: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage = _current_stage(config, state)
    status = last_run.get("status") if isinstance(last_run, dict) else None
    incomplete = status in {"running", "queued", "finalizing", "recovery_required", "crash", "timeout",
                            "failed", "invalid", "interrupted", "budget", "incomparable", "input_tampered"}
    if stage == "SETUP_PENDING" and not incomplete:
        if config.evaluator and config.primary:
            argv = _argv(config, "baseline")
            return [{
                "priority": 1, "action": "RUN_BASELINE", "argv": argv, "command": _command(argv),
                "reason": "The evaluator and primary metric are configured; finalize setup with a baseline run.",
            }]
        return [{
            "priority": 1, "action": "CONFIG_SETUP", "argv": [], "command": "",
            "reason": "Add an evaluator and primary metric to the lab configuration before measuring.",
        }]
    if stage == "BASELINE_NEEDED" and not incomplete:
        argv = _argv(config, "baseline")
        return [{
            "priority": 1, "action": "RUN_BASELINE", "argv": argv, "command": _command(argv),
            "reason": "Record the baseline before comparing candidate trials.",
        }]

    if status == "recovery_required":
        argv = _argv(config, "recover")
        return [{
            "priority": 1, "action": "RECOVER", "argv": argv, "command": _command(argv),
            "run_id": last_run.get("id"),
            "reason": "The last worker ended with an uncertain outcome; reconcile it before starting another experiment.",
        }]
    if status in {"queued", "running", "finalizing"}:
        if (status == "queued" or not last_run.get("id")
                or str(last_run.get("id", "")).startswith("baseline")):
            argv = _argv(config, "status")
        else:
            argv = _argv(config, "trial", "wait", "--timeout", "30")
        return [{
            "priority": 1, "action": "WAIT", "argv": argv, "command": _command(argv),
            "run_id": last_run.get("id"),
            "reason": "A trial is still running; wait for its recorded result before choosing another action.",
        }]
    if incomplete:
        run_id = str(last_run.get("id") or "")
        if run_id:
            from .sqlite_store import SQLiteStore
            run_id = str(SQLiteStore().resolve_run_id(config.lab_id, run_id))
            argv = _argv(config, "guide", "--json", "--run", run_id)
        else:
            argv = _argv(config, "guide", "--run", run_id, "--json") if run_id else _argv(config, "guide", "--json")
        return [{
            "priority": 1, "action": "INSPECT_RUN", "argv": argv, "command": _command(argv),
            "run_id": run_id or None,
            "reason": f"Latest run ended as {status}; inspect its evidence before retrying or replaying.",
        }]
    if status == "accepted" and not str(last_run.get("id", "")).startswith("baseline"):
        action, reason = "ITERATE", "Accepted run: inspect its evidence, then write a distinct hypothesis and candidate change."
    elif status == "rejected":
        action, reason = "REFINE_OR_RETRY", "Rejected run: use the recorded tradeoffs to form a revised hypothesis."
    else:
        action, reason = "RUN_TRIAL", "Write an explicit hypothesis and candidate change before running a trial."
    hypothesis_path = str((config.path.parent / "hypothesis.json").resolve())
    argv = _argv(config, "trial", "--hypothesis", hypothesis_path)
    return [{
        "priority": 1,
        "action": action,
        "argv": argv,
        "command": _command(argv),
        "hypothesis_file": hypothesis_path,
        "reason": reason,
    }]


def build_cockpit(
    config: LabConfig,
    state: dict[str, Any],
    last_run: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    state_card = get_state_card(config, state)
    current_epoch = state_card["epoch"]
    recent = _recent_manifests(config, current_epoch, limit=100)
    from .locking import _pid_alive
    from .worker import read_worker

    active_lifecycle = None
    from .sqlite_store import SQLiteStore
    active_lifecycle = SQLiteStore().get_active_run(config.lab_id)
    worker = read_worker(config.state_dir)
    if isinstance(worker, dict):
        operation_status = worker.get("status")
        if operation_status == "recovery_required":
            active_lifecycle = {"id": None, "status": "recovery_required",
                                "operation_id": str(worker.get("operation_id") or "")}
        elif operation_status in {"queued", "running", "finalizing"}:
            active_lifecycle = {"id": None, "status": operation_status,
                                "operation_id": str(worker.get("operation_id") or "")}
    if active_lifecycle is None:
        if isinstance(worker, dict) and worker.get("status") == "running":
            try:
                worker_pid = int(worker.get("pid") or 0)
            except (TypeError, ValueError):
                worker_pid = 0
            if _pid_alive(worker_pid):
                worker_run_id = worker.get("trial_id")
                active_lifecycle = {
                    "id": worker_run_id if isinstance(worker_run_id, str) and worker_run_id else None,
                    "status": "running",
                }
    selected = _load_manifest(config, run_id) if run_id is not None else None
    if run_id is not None and selected is None:
        raise LabError(f"Run not found: {run_id}")
    if selected is None:
        selected = last_run if isinstance(last_run, dict) else (recent[0] if recent else None)
    latest_lifecycle = last_run if isinstance(last_run, dict) else (recent[0] if recent else None)
    selected_id = str(selected.get("id")) if isinstance(selected, dict) and selected.get("id") else None
    if selected_id is not None and _manifest_epoch(selected, current_epoch) != current_epoch:
        # Keep explicit historical evidence visible, but never compare it to current-epoch data.
        parent_manifest = None
    else:
        parent_id = selected.get("parent") if isinstance(selected, dict) else None
        parent_manifest = _load_manifest(config, parent_id) if isinstance(parent_id, str) else None

    baseline_manifest = _load_manifest(config, state.get("baseline_id"))
    best_manifest = _load_manifest(config, state.get("best_id"))
    if best_manifest is None:
        best_manifest = baseline_manifest
    impact = get_impact_analysis(config, state, selected)
    next_steps = get_recommended_actions(config, state, active_lifecycle or latest_lifecycle)
    signals = build_research_signals(config, selected, parent_manifest, history=recent)
    goal = evaluate_goal_status(config, best_manifest)

    epoch_record = next((item for item in state.get("epochs", [])
                         if isinstance(item, dict) and item.get("id") == current_epoch), None)
    fingerprint_source = epoch_record if isinstance(epoch_record, dict) else best_manifest
    evidence_pool = [selected, parent_manifest, *recent, baseline_manifest, best_manifest]
    by_id = {str(item.get("id")): item for item in evidence_pool
             if isinstance(item, dict) and isinstance(item.get("id"), str)}
    evidence_sources: list[tuple[str, dict[str, Any] | None]] = [
        ("selected", selected), ("parent", parent_manifest),
    ]
    for signal in signals:
        for evidence_id in signal.get("evidence", []):
            manifest = by_id.get(str(evidence_id))
            if manifest is not None:
                evidence_sources.append(("signal", manifest))
    if parent_manifest is not None or baseline_manifest is not None or best_manifest is not None:
        evidence_sources.extend([
            ("best", best_manifest), ("baseline", baseline_manifest),
        ])
    elif not selected:
        evidence_sources.extend([*(('recent', item) for item in recent[:2])])
    if isinstance(impact, dict) and "error_tail" not in impact:
        error_tail = _extract_manifest_error_tail(selected)
        if error_tail:
            impact["error_tail"] = error_tail
    summary = {
        "state": state_card,
        "impact": impact,
        "next_steps": next_steps[:1],
        "selected_run_id": selected_id,
        "current_epoch": current_epoch,
        "active_run": ({"id": active_lifecycle.get("id"), "status": "running"}
                       if isinstance(active_lifecycle, dict) else None),
        "fingerprints": _fingerprint_summary(fingerprint_source),
        "evidence": _evidence(config, evidence_sources),
        "signals": signals[:3],
        "provider_usage": {
            "requests": int(state_card["spent"].get("requests", 0) or 0),
            "unknown_requests": int(state_card["spent"].get("unknown_requests", 0) or 0),
            "confirmed_cost_usd": float(state_card["spent"].get(
                "confirmed_cost_usd", state_card["spent"].get("cost_usd", 0.0),
            ) or 0.0),
            "cost_usd": state_card["spent"].get("cost_usd"),
        },
        "provider_retry": {
            "runtime": "disabled",
            "external_evaluator": "unverified",
            "note": "Runtime hooks do not retry.",
        },
        "goal_status": goal["status"],
        "goal": goal,
        "dead_ends": _build_dead_ends_digest([m for m in [selected, *recent] if isinstance(m, dict)], current_epoch),
    }
    if run_id is not None and isinstance(selected, dict):
        run_root = config.state_dir / ("controls" if (config.state_dir / "controls" / run_id).is_dir() else "runs") / run_id
        from .sqlite_store import SQLiteStore
        run_source = f"/api/v1/labs/{config.lab_id}/runs/{SQLiteStore().resolve_run_id(config.lab_id, run_id)}"
        hypothesis_limits = {
            "title": 120, "observation": 180, "expected_effect": 180,
            "falsification": 140, "rationale": 140,
        }
        hypothesis = {key: selected[key][:limit] for key, limit in hypothesis_limits.items()
                      if isinstance(selected.get(key), str)}
        if isinstance(selected.get("target_files"), list):
            hypothesis["target_files"] = [str(v)[:80] for v in selected["target_files"][:5] if isinstance(v, str)]
        evidence = selected.get("evidence")
        if isinstance(evidence, str):
            hypothesis["evidence"] = evidence[:200]
        elif isinstance(evidence, list):
            hypothesis["evidence"] = [value[:80] for value in evidence[:5] if isinstance(value, str)]
        detail = {
            "id": run_id, "parent": selected.get("parent"), "epoch": _manifest_epoch(selected, 1),
            "status": selected.get("status"), "reason": str(selected.get("reason") or "")[:180],
            "hypothesis": hypothesis, "manifest": run_source,
            "patch": str((run_root / "change.patch").resolve()) if (run_root / "change.patch").is_file() else None,
        }
        tail = _extract_manifest_error_tail(selected)
        if tail:
            detail["error_tail"] = tail
        summary["run_detail"] = detail

    def _encoded_len(obj: dict[str, Any]) -> int:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    if _encoded_len(summary) > 4500:
        # Step 1: middle-trim error_tail (preserving head traceback and tail exception)
        tail = summary.get("impact", {}).get("error_tail") or summary.get("run_detail", {}).get("error_tail")
        s_tail = str(tail or "").strip()
        if s_tail and len(s_tail) > 140:
            lines = [ln.strip() for ln in s_tail.splitlines() if ln.strip()]
            if len(lines) >= 2:
                compact_tail = lines[0][:70] + "\n...\n" + lines[-1][-100:]
            else:
                compact_tail = s_tail[:60].rstrip() + "\n...\n" + s_tail[-90:].lstrip()
            if isinstance(summary.get("impact"), dict) and "error_tail" in summary["impact"]:
                summary["impact"]["error_tail"] = compact_tail
            if isinstance(summary.get("run_detail"), dict) and "error_tail" in summary["run_detail"]:
                summary["run_detail"]["error_tail"] = compact_tail

        # Step 2: compact dead_ends text fields
        if _encoded_len(summary) > 4500 and isinstance(summary.get("dead_ends"), list):
            for de in summary["dead_ends"]:
                if isinstance(de, dict):
                    de["title"] = str(de.get("title") or "")[:18]
                    de["reason"] = str(de.get("reason") or "")[:20]

        # Step 3: compact signals note and case_samples error preview
        if _encoded_len(summary) > 4500 and isinstance(summary.get("signals"), list):
            for s in summary["signals"]:
                if isinstance(s, dict):
                    if "note" in s:
                        s["note"] = str(s["note"])[:36]
                    if isinstance(s.get("case_samples"), list):
                        for cs in s["case_samples"]:
                            if isinstance(cs, dict) and "error" in cs:
                                cs["error"] = str(cs["error"])[:28]

        # Step 4: compact evidence list to 1 item when under heavy load
        if _encoded_len(summary) > 4500 and len(summary.get("evidence", [])) > 1:
            summary["evidence"] = summary["evidence"][:1]

        # Step 5: compact human-prose verdict/reason/command (never touch executable argv)
        if _encoded_len(summary) > 4500:
            if isinstance(summary.get("impact"), dict) and "verdict" in summary["impact"]:
                summary["impact"]["verdict"] = str(summary["impact"]["verdict"])[:70]
            if isinstance(summary.get("next_steps"), list) and summary["next_steps"]:
                step = summary["next_steps"][0]
                if isinstance(step, dict):
                    if "reason" in step:
                        step["reason"] = str(step["reason"])[:45]
                    if "command" in step and len(str(step["command"])) > 80:
                        step["command"] = str(step["command"])[:80] + "..."
            if isinstance(summary.get("state"), dict):
                if isinstance(summary["state"].get("spent"), dict):
                    spent = summary["state"]["spent"]
                    summary["state"]["spent"] = {
                        k: spent[k] for k in ("seconds", "requests", "cost_usd", "confirmed_cost_usd") if k in spent
                    }
                if isinstance(summary["state"].get("paths"), dict) and len(summary["state"]["paths"]) > 1:
                    cand = summary["state"]["paths"].get("candidate")
                    summary["state"]["paths"] = {"candidate": cand} if cand else summary["state"]["paths"]

        # Step 6: if run_detail is already present and still > 4500, drop redundant evidence
        if _encoded_len(summary) > 4500 and summary.get("run_detail") and summary.get("evidence"):
            summary["evidence"] = []

        # Step 7: pop oldest dead_ends only as a last resort
        while _encoded_len(summary) > 4500 and isinstance(summary.get("dead_ends"), list) and summary["dead_ends"]:
            summary["dead_ends"].pop()
    return summary


def render_ascii_card(cockpit: dict[str, Any]) -> str:
    state = cockpit.get("state", {})
    stage = state.get("stage", "UNKNOWN")
    primary = state.get("primary_metric", "metric")
    unit = f" {state.get('unit')}" if state.get("unit") else ""
    best_score = state.get("best", {}).get("score")
    best_mean = state.get("best", {}).get("mean")
    best_str = f"{best_score:.4g}{unit}" if best_score is not None else "—"
    best_mean_str = f"{best_mean:.4g}{unit}" if best_mean is not None else "—"
    base_score = state.get("baseline", {}).get("score")
    base_mean = state.get("baseline", {}).get("mean")
    base_str = f"{base_score:.4g}{unit}" if base_score is not None else "—"
    base_mean_str = f"{base_mean:.4g}{unit}" if base_mean is not None else "—"
    lines = [
        "╔══════════════════════════ [ AGENT COCKPIT ] ══════════════════════════╗",
        f"║ Фаза: {stage:<16} Эпоха: {state.get('epoch', 1):<4} Метрика: {primary} ({state.get('direction', 'max')})",
        f"║ Baseline worst: {base_str:<12} mean: {base_mean_str} | Best worst: {best_str} mean: {best_mean_str}",
        f"║ Goal: {cockpit.get('goal_status', 'unknown')}",
    ]
    impact = cockpit.get("impact")
    if impact:
        lines.append(f"║ Результат: {impact.get('verdict', '')}")
    for signal in cockpit.get("signals", [])[:3]:
        label = signal.get("metric") or signal.get("failure_kind") or signal.get("kind", "signal")
        value = signal.get("delta")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            detail = f"delta {value:+.4g}"
        elif isinstance(signal.get("spread"), (int, float)):
            detail = f"spread {signal['spread']:.4g}"
        elif isinstance(signal.get("missing_count"), int):
            detail = f"missing {signal['missing_count']} repeats"
        elif isinstance(signal.get("count"), int):
            detail = f"count {signal['count']}"
        else:
            detail = "inspect evidence"
        lines.append(f"║ Signal: {label} {detail} — diagnostic")
    lines.append("╟────────────────────────── [ СЛЕДУЮЩИЙ ШАГ ] ──────────────────────────╢")
    for step in cockpit.get("next_steps", [])[:1]:
        lines.append(f"║ {step.get('action')}: {step.get('reason')}")
        lines.append(f"║     argv: {json.dumps(step.get('argv', []), ensure_ascii=False)}")
    lines.append("╚════════════════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)
