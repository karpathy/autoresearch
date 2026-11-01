"""Compact, evidence-linked research signals for agent guidance."""

from __future__ import annotations

import math
import hashlib
import json
import re
from typing import Any


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_success_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    metrics = record.get("metrics")
    if record.get("status") == "ok":
        return isinstance(metrics, dict) and bool(metrics)
    # Runtime manifests store the evaluator result, while the outcome status is
    # returned alongside it. A metrics-bearing row with no status is that legacy
    # runtime shape; failed rows have no metrics object.
    return record.get("status") is None and isinstance(metrics, dict) and bool(metrics)


def _successful_metrics(manifest: dict[str, Any]) -> list[dict[str, float]]:
    records = manifest.get("runs")
    if not isinstance(records, list):
        return []
    result: list[dict[str, float]] = []
    for record in records:
        if not _is_success_record(record):
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        clean = {key: value for key, raw in metrics.items()
                 if isinstance(key, str) and (value := _finite(raw)) is not None}
        if clean:
            result.append(clean)
    return result


def _means(manifest: dict[str, Any]) -> dict[str, float]:
    records = _successful_metrics(manifest)
    values: dict[str, list[float]] = {}
    for record in records:
        for metric, value in record.items():
            values.setdefault(metric, []).append(value)
    return {metric: sum(items) / len(items) for metric, items in values.items()}


def _current_values(manifest: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for record in _successful_metrics(manifest):
        for metric, value in record.items():
            values.setdefault(metric, []).append(value)
    return values


def _same_contract(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only compare research evidence collected under the same frozen contract."""
    for key in ("config_sha256", "input_hashes", "evaluator"):
        left_value, right_value = left.get(key), right.get(key)
        if left_value is None or right_value is None or left_value != right_value:
            return False
    return True


def _decision_worst(manifest: dict[str, Any], metric: str, direction: str) -> float | None:
    values = _current_values(manifest).get(metric, [])
    if not values or direction not in {"min", "max"}:
        return None
    return min(values) if direction == "max" else max(values)


def _comparable(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    if candidate.get("parent") != parent.get("id"):
        return False
    if candidate.get("epoch") != parent.get("epoch"):
        return False
    return _same_contract(candidate, parent)


def _tokenize_hypothesis(item: dict[str, Any]) -> set[str]:
    hyp = item.get("hypothesis") if isinstance(item.get("hypothesis"), dict) else {}
    title = str(item.get("title") or hyp.get("title") or "")
    observation = str(item.get("observation") or hyp.get("observation") or "")
    text = f"{title} {observation}".lower()
    return {w for w in re.findall(r"[a-z0-9]+", text) if len(w) >= 3}


def _extract_case_outcomes(manifest: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(manifest, dict):
        return {}
    outcomes: dict[str, str] = {}
    sources: list[Any] = []
    for key in ("case_outcomes", "cases", "per_case_summaries", "case_summaries"):
        if key in manifest:
            sources.append(manifest[key])
    for record in manifest.get("runs") or []:
        if isinstance(record, dict):
            for key in ("case_outcomes", "cases", "per_case_summaries", "case_summaries"):
                if key in record:
                    sources.append(record[key])
            details = record.get("details")
            if isinstance(details, dict):
                for key in ("case_outcomes", "cases", "per_case_summaries", "case_summaries"):
                    if key in details:
                        sources.append(details[key])

    for source in sources:
        if isinstance(source, dict):
            for case_id, val in source.items():
                if isinstance(val, str):
                    outcomes[str(case_id)] = val.lower()
                elif isinstance(val, dict):
                    status = val.get("status") or val.get("outcome") or val.get("result")
                    if isinstance(status, str):
                        outcomes[str(case_id)] = status.lower()
                    elif "passed" in val:
                        outcomes[str(case_id)] = "ok" if val["passed"] else "failed"
                elif isinstance(val, bool):
                    outcomes[str(case_id)] = "ok" if val else "failed"
        elif isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    cid = item.get("case_id") or item.get("id") or item.get("name")
                    status = item.get("status") or item.get("outcome") or item.get("result")
                    if cid is not None and isinstance(status, str):
                        outcomes[str(cid)] = status.lower()
                    elif cid is not None and "passed" in item:
                        outcomes[str(cid)] = "ok" if item["passed"] else "failed"
    return outcomes


def _extract_regressed_case_samples(
    candidate: dict[str, Any],
    regressed_ids: list[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict) or not regressed_ids:
        return []

    cases_pool: list[Any] = []
    runs = candidate.get("runs")
    if isinstance(runs, list) and runs:
        for r in reversed(runs):
            if isinstance(r, dict):
                for k in ("cases", "case_summaries", "per_case_summaries"):
                    if k in r:
                        cases_pool.append(r[k])
                details = r.get("details")
                if isinstance(details, dict):
                    for k in ("cases", "case_summaries", "per_case_summaries"):
                        if k in details:
                            cases_pool.append(details[k])
    for k in ("cases", "case_summaries", "per_case_summaries"):
        if k in candidate:
            cases_pool.append(candidate[k])

    case_by_id: dict[str, dict[str, Any]] = {}
    for source in cases_pool:
        if isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    cid = item.get("id") or item.get("case_id") or item.get("name")
                    if cid is not None and str(cid) not in case_by_id:
                        case_by_id[str(cid)] = item
        elif isinstance(source, dict):
            for k, val in source.items():
                cid = str(k)
                if cid not in case_by_id:
                    if isinstance(val, dict):
                        case_by_id[cid] = {"id": cid, **val}
                    elif isinstance(val, str):
                        case_by_id[cid] = {"id": cid, "status": val}
                    elif isinstance(val, bool):
                        case_by_id[cid] = {"id": cid, "status": "ok" if val else "failed"}

    samples: list[dict[str, Any]] = []
    for cid in regressed_ids[:limit]:
        cid_str = str(cid)
        case = case_by_id.get(cid_str, {})
        status = str(case.get("status") or case.get("outcome") or case.get("result") or "unknown")
        sample: dict[str, Any] = {"id": cid_str, "status": status}
        if case.get("error"):
            sample["error"] = str(case["error"])[:120]
        if isinstance(case.get("metrics"), dict) and case["metrics"]:
            scalar_metrics = {}
            for k, v in list(case["metrics"].items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scalar_metrics[str(k)] = v
                if len(scalar_metrics) >= 3:
                    break
            if scalar_metrics:
                sample["metrics"] = scalar_metrics
        samples.append(sample)

    return samples


def _trailing_rejections(window: list[dict[str, Any]], best_id: str = "") -> list[dict[str, Any]]:
    rejections: list[dict[str, Any]] = []
    for item in window:
        if not isinstance(item, dict):
            continue
        if best_id and str(item.get("id") or "") == best_id:
            continue
        if item.get("status") == "rejected" and isinstance(item.get("reason"), str) and item["reason"].strip():
            rejections.append(item)
    return rejections


def build_research_signals(
    config: Any,
    candidate: dict[str, Any] | None,
    parent: dict[str, Any] | None,
    *,
    history: list[dict[str, Any]] | None = None,
    limit: int = 3,
    best: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Summarize repeat patterns without turning diagnostics into decision gates."""
    if not isinstance(candidate, dict):
        return []
    signals: list[dict[str, Any]] = []
    candidate_id = str(candidate.get("id") or "unknown")
    records = candidate.get("runs") if isinstance(candidate.get("runs"), list) else []
    successful = _successful_metrics(candidate)
    expected_raw = getattr(config, "repeats", None) or candidate.get("expected_repeats")
    expected = expected_raw if isinstance(expected_raw, int) and not isinstance(expected_raw, bool) and expected_raw > 0 else None
    failed_count = sum(1 for item in records if not _is_success_record(item))
    missing_count = max(0, (expected or len(records)) - len(successful))
    if missing_count or failed_count:
        signals.append({
            "kind": "missing_repeats",
            "completed_repeats": len(successful),
            "expected_repeats": expected,
            "missing_count": max(missing_count, failed_count),
            "gate": False,
            "evidence": [candidate_id],
            "note": "Some repeats are missing or did not complete successfully.",
        })

    for metric, values in sorted(_current_values(candidate).items()):
        if len(values) > 1 and max(values) != min(values):
            signals.append({
                "kind": "repeat_spread", "metric": metric,
                "min": min(values), "max": max(values), "spread": max(values) - min(values),
                "repeat_count": len(values), "gate": False, "evidence": [candidate_id],
                "note": "Observed repeat variation; inspect the evaluator and case-level evidence.",
            })

    diagnostics: dict[tuple[str, str], list[str]] = {}
    for index, record in enumerate(records, start=1):
        if not _is_success_record(record):
            continue
        details = record.get("details")
        entries = details.get("diagnostics") if isinstance(details, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            case_id, slice_name, failure_kind = (entry.get("case_id"), entry.get("slice"), entry.get("failure_kind"))
            if not all(isinstance(value, str) and value.strip() for value in (case_id, slice_name, failure_kind)):
                continue
            key = (failure_kind.strip()[:100], slice_name.strip()[:100])
            diagnostics.setdefault(key, []).append(case_id.strip()[:100])
    for (failure_kind, slice_name), case_ids in sorted(diagnostics.items(), key=lambda item: (-len(item[1]), item[0])):
        signals.append({
            "kind": "diagnostic_cluster", "failure_kind": failure_kind, "slice": slice_name,
            "count": len(case_ids), "case_ids": list(dict.fromkeys(case_ids))[:3],
            "gate": False, "evidence": [candidate_id],
            "note": "Evaluator diagnostic labels identify cases to inspect; they do not prove cause.",
        })

    if isinstance(parent, dict) and _comparable(candidate, parent):
        candidate_means = _means(candidate)
        parent_means = _means(parent)
        primary = getattr(config, "primary", None)
        primary_direction = getattr(config, "direction", None)
        guarded = dict(getattr(config, "guards", ()) or ())
        parent_primary_mean = parent_means.get(primary)
        candidate_primary_mean = candidate_means.get(primary)
        parent_primary_worst = _decision_worst(parent, primary, primary_direction) if isinstance(primary, str) else None
        candidate_primary_worst = _decision_worst(candidate, primary, primary_direction) if isinstance(primary, str) else None
        primary_improved = (
            parent_primary_worst is not None and candidate_primary_worst is not None
            and primary_direction in {"min", "max"}
            and (candidate_primary_worst > parent_primary_worst if primary_direction == "max"
                 else candidate_primary_worst < parent_primary_worst)
        )
        for metric in sorted((candidate_means.keys() & parent_means.keys()) - {primary}):
            old = parent_means[metric]
            new = candidate_means[metric]
            delta = new - old
            if delta == 0:
                continue
            direction = guarded.get(metric)
            old_worst = _decision_worst(parent, metric, direction) if direction in {"min", "max"} else None
            new_worst = _decision_worst(candidate, metric, direction) if direction in {"min", "max"} else None
            guard_regressed = (
                direction in {"min", "max"} and old_worst is not None and new_worst is not None
                and (new_worst < old_worst if direction == "max" else new_worst > old_worst)
            )
            if primary_improved and guard_regressed:
                effect = "guard_regressed_while_primary_improved"
            elif direction in {"min", "max"}:
                effect = "improved" if (delta < 0 if direction == "min" else delta > 0) else "regressed"
            else:
                effect = "changed"
            signal = {
                "kind": "metric_tradeoff", "metric": metric, "from": old, "to": new,
                "delta": delta, "effect": effect, "gate": False,
                "evidence": [candidate_id, str(parent.get("id") or "unknown")],
                "note": "Observed across matched repeats; diagnostic only.",
                "direction": direction if direction in {"min", "max"} else None,
                "from_mean": old, "to_mean": new, "mean_delta": delta,
                "from_decision_worst": old_worst, "to_decision_worst": new_worst,
                "decision_worst_delta": new_worst - old_worst if old_worst is not None and new_worst is not None else None,
            }
            if effect == "guard_regressed_while_primary_improved":
                signal.update({
                    "primary_metric": primary, "primary_direction": primary_direction,
                    "primary_from_mean": parent_primary_mean, "primary_to_mean": candidate_primary_mean,
                    "primary_mean_delta": candidate_primary_mean - parent_primary_mean,
                    "primary_from_decision_worst": parent_primary_worst,
                    "primary_to_decision_worst": candidate_primary_worst,
                    "primary_decision_worst_delta": candidate_primary_worst - parent_primary_worst,
                    "guard_metric": metric, "guard_direction": direction,
                    "guard_from_mean": old, "guard_to_mean": new, "guard_mean_delta": delta,
                    "guard_from_decision_worst": old_worst, "guard_to_decision_worst": new_worst,
                    "guard_decision_worst_delta": new_worst - old_worst,
                })
            signals.append(signal)

    manifests = [m for m in (history or []) if isinstance(m, dict)]
    if len(manifests) >= 2:
        first_id = str(manifests[0].get("id") or "")
        last_id = str(manifests[-1].get("id") or "")
        if (last_id == candidate_id and first_id != candidate_id) or (
            first_id.startswith("trial-") and last_id.startswith("trial-") and first_id < last_id
        ):
            manifests = list(reversed(manifests))
    try:
        current_epoch = int(candidate.get("epoch", 1) if candidate.get("epoch") is not None else 1)
    except (TypeError, ValueError):
        current_epoch = 1

    best_manifest = best if isinstance(best, dict) else (kwargs.get("best") or parent or {})

    pool = [candidate] if isinstance(candidate, dict) else []
    pool.extend(manifests[:100])
    seen: set[str] = set()
    recent: list[dict[str, Any]] = []
    for item in pool:
        run_id = str(item.get("id") or "")
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        try:
            item_epoch = int(item.get("epoch", 1) if item.get("epoch") is not None else 1)
        except (TypeError, ValueError):
            item_epoch = 1
        if item_epoch == current_epoch and _same_contract(candidate, item):
            recent.append(item)
    rejected = [item for item in recent if item.get("status") == "rejected"
                and isinstance(item.get("reason"), str) and item["reason"].strip()]
    streak_window = recent[:15]
    streak_rejections = _trailing_rejections(streak_window, str(best_manifest.get("id") or ""))
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for item in streak_rejections:
        by_reason.setdefault(item["reason"].strip(), []).append(item)
    for reason, items in sorted(by_reason.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(items) >= 2:
            note = (
                "The same decision reason recurred 3+ times; inspect run evidence or explore external literature approaches."
                if len(items) >= 3
                else "The same decision reason recurred; inspect the hypothesis and run evidence."
            )
            signals.append({
                "kind": "repeated_rejection", "reason": reason[:160], "count": len(items),
                "gate": False, "evidence": [str(item.get("id")) for item in items[:5]],
                "note": note,
            })

    def patch_fingerprint(item: dict[str, Any]) -> str | None:
        digest = item.get("patch_sha256")
        if isinstance(digest, str) and digest:
            return digest
        hashes = item.get("candidate_hashes")
        if not isinstance(hashes, dict) or not hashes:
            return None
        blob = json.dumps(hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    by_patch: dict[str, list[dict[str, Any]]] = {}
    for item in rejected:
        fingerprint = patch_fingerprint(item)
        if fingerprint:
            by_patch.setdefault(fingerprint, []).append(item)
    for fingerprint, items in sorted(by_patch.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(items) >= 2:
            signals.append({
                "kind": "same_candidate_rejected", "candidate_fingerprint": fingerprint[:12],
                "count": len(items), "gate": False,
                "evidence": [str(item.get("id")) for item in items[:5]],
                "note": "The same candidate snapshot was rejected more than once; change the candidate before retrying.",
            })

    if candidate.get("status") == "rejected":
        cand_tokens = _tokenize_hypothesis(candidate)
        if cand_tokens:
            cand_fp = patch_fingerprint(candidate)
            same_patch_flagged = bool(cand_fp and len(by_patch.get(cand_fp, [])) >= 2)
            similar_items = []
            best_sim = 0.0
            for item in rejected:
                if str(item.get("id")) == candidate_id:
                    continue
                if cand_fp and patch_fingerprint(item) == cand_fp:
                    continue
                item_tokens = _tokenize_hypothesis(item)
                if not item_tokens:
                    continue
                intersection = cand_tokens & item_tokens
                union = cand_tokens | item_tokens
                sim = len(intersection) / len(union) if union else 0.0
                if sim >= 0.65:
                    similar_items.append(item)
                    if sim > best_sim:
                        best_sim = sim
            if similar_items and not same_patch_flagged:
                signals.append({
                    "kind": "similar_hypothesis_rejected",
                    "similarity": round(best_sim, 2),
                    "count": len(similar_items),
                    "gate": False,
                    "evidence": [str(item.get("id")) for item in similar_items[:5]],
                    "note": "A semantically similar hypothesis was already rejected; pivot to a distinct mechanism.",
                })

    regressed: list[str] = []
    if isinstance(candidate.get("regressed_cases"), list):
        for c in candidate["regressed_cases"]:
            cid = c.get("case_id") or c.get("id") if isinstance(c, dict) else c
            if cid is not None:
                regressed.append(str(cid))
    elif isinstance(parent, dict):
        parent_cases = _extract_case_outcomes(parent)
        cand_cases = _extract_case_outcomes(candidate)
        if parent_cases and cand_cases:
            pass_statuses = {"ok", "passed", "pass", "success"}
            for cid, p_status in parent_cases.items():
                if p_status in pass_statuses and cid in cand_cases:
                    c_status = cand_cases[cid]
                    if c_status not in pass_statuses:
                        regressed.append(cid)
    if regressed:
        deduped = list(dict.fromkeys(regressed))
        signal_entry = {
            "kind": "case_regression",
            "regressed_case_ids": deduped[:10],
            "count": len(deduped),
            "gate": False,
            "evidence": [candidate_id, str(parent.get("id") or "unknown")] if isinstance(parent, dict) and parent.get("id") else [candidate_id],
            "note": "Candidate regressed on cases that previously passed in parent.",
        }
        samples = _extract_regressed_case_samples(candidate, deduped)
        if samples:
            signal_entry["case_samples"] = samples
        signals.append(signal_entry)

    priority = {
        "repeated_rejection": 1, "same_candidate_rejected": 2,
        "similar_hypothesis_rejected": 2, "case_regression": 2,
        "missing_repeats": 3, "repeat_spread": 5,
        "metric_tradeoff": 4, "diagnostic_cluster": 6,
    }
    signals.sort(key=lambda item: (
        0 if item.get("effect") == "guard_regressed_while_primary_improved"
        else priority.get(item["kind"], 9),
        item.get("metric", ""), item.get("failure_kind", ""),
    ))
    return signals[:max(0, min(int(limit), 3))]


def evaluate_goal_status(config: Any, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate only explicitly configured thresholds and recorded repeat origins."""
    criteria = getattr(config, "goal_criteria", None)
    if not isinstance(criteria, dict):
        return {"status": "unknown", "reason": "No explicit goal_criteria in lab config."}
    thresholds = criteria.get("thresholds")
    confirmation = criteria.get("confirmation_repeats")
    if (not isinstance(thresholds, dict) or not thresholds
            or isinstance(confirmation, bool) or not isinstance(confirmation, int) or confirmation < 1):
        return {"status": "unknown", "reason": "Explicit goal criteria are incomplete."}
    if not isinstance(manifest, dict):
        return {"status": "unknown", "reason": "Best-run evidence is unavailable."}
    if manifest.get("status") != "accepted":
        return {"status": "unknown", "reason": "The selected evidence is not an accepted best run."}

    records = manifest.get("runs")
    if not isinstance(records, list):
        return {"status": "unknown", "reason": "Repeat evidence is unavailable."}
    successful = [record for record in records if _is_success_record(record)]
    matched = 0
    violated: list[str] = []
    missing: list[str] = []
    for record in successful:
        metrics = record.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        for metric, bounds in thresholds.items():
            if not isinstance(metric, str) or not isinstance(bounds, dict):
                return {"status": "unknown", "reason": "Explicit goal criteria are malformed."}
            if set(bounds) - {"min", "max"} or not ({"min", "max"} & bounds.keys()):
                return {"status": "unknown", "reason": "Explicit goal criteria are malformed."}
            value = _finite(metrics.get(metric))
            if value is None:
                missing.append(metric)
                continue
            matched += 1
            low = _finite(bounds["min"]) if "min" in bounds else None
            high = _finite(bounds["max"]) if "max" in bounds else None
            if ("min" in bounds and low is None) or ("max" in bounds and high is None) or (low is not None and high is not None and low > high):
                return {"status": "unknown", "reason": "Explicit goal criteria are malformed."}
            if ((low is not None and value < low) or (high is not None and value > high)):
                violated.append(metric)

    if violated:
        return {
            "status": "not_met", "confirmation_repeats": confirmation,
            "completed_repeats": len(successful), "violated_metrics": sorted(set(violated)),
            "reason": "At least one successful repeat missed an explicit threshold.",
        }
    if missing or len(successful) < confirmation or any(not _is_success_record(record) for record in records):
        return {
            "status": "unconfirmed", "confirmation_repeats": confirmation,
            "completed_repeats": len(successful),
            "reason": "Repeat or threshold evidence is incomplete.",
        }

    origins: list[str | None] = []
    for record in successful:
        details = record.get("details")
        origins.append(details.get("evidence_origin") if isinstance(details, dict) else None)
    if any(origin is None for origin in origins):
        return {
            "status": "unknown", "confirmation_repeats": confirmation,
            "completed_repeats": len(successful),
            "reason": "Repeat evidence has no explicit evidence_origin.",
        }
    if any(origin != "fresh" for origin in origins):
        return {
            "status": "unconfirmed", "confirmation_repeats": confirmation,
            "completed_repeats": len(successful),
            "evidence_origins": sorted({str(origin) for origin in origins}),
            "reason": "Historical, imported, or offline evidence cannot confirm a fresh goal.",
        }

    return {
        "status": "met", "confirmation_repeats": confirmation,
        "completed_repeats": len(successful), "evidence_origin": "fresh",
        "scope": "same candidate within this run; independent model repetitions are not implied",
    }
