"""Authenticated, local-only v1 API for SQLite-backed research labs."""

from __future__ import annotations

import base64
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .auth import AuthError, authenticate
from .commands.common import enqueue_and_start_operation, normalize_trial_hypothesis
from .errors import LabError
from .paths import resolve_home
from .sqlite_store import (
    ActiveOperationError,
    AmbiguousAliasError,
    EntityNotFoundError,
    IdempotencyConflictError,
    SQLiteStore,
    StoreBusyError,
    StoreConflictError,
    StoreError,
    StoreNotInitializedError,
)


MAX_BODY_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_COMMANDS = {"baseline", "trial", "resume", "recover", "epoch", "cancel"}
_FILTERS = {"status", "source_root", "primary_metric"}
_HEADER_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SAFE_PHASE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def _envelope(*, data: Any = None, warnings: list[str] | None = None,
              next_value: Any = None, error: dict[str, str] | None = None) -> dict[str, Any]:
    return {"ok": error is None, "data": data, "warnings": warnings or [], "next": next_value, "error": error}


def _send(handler: Any, status: int, payload: dict[str, Any], *, extra: dict[str, str] | None = None) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    handler._send(status, body, "application/json; charset=utf-8", extra)


def _send_error(handler: Any, status: int, code: str, message: str, hint: str = "") -> None:
    _send(handler, status, _envelope(error={"code": code, "message": message, "hint": hint}))


def _read_json(handler: Any) -> dict[str, Any]:
    if handler.headers.get("Transfer-Encoding"):
        raise ValueError("Transfer-Encoding is not supported")
    raw_length = handler.headers.get("Content-Length", "")
    try:
        length = int(raw_length)
    except (TypeError, ValueError) as error:
        raise ValueError("A valid Content-Length is required") from error
    if length < 0 or length > MAX_BODY_BYTES:
        raise OverflowError("Request body is too large")
    content_type = handler.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        raise ValueError("Content-Type must be application/json")
    try:
        value = json.loads(handler.rfile.read(length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Request body must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object")
    return value


def _single_query(query: str, allowed: set[str]) -> dict[str, str]:
    parsed = parse_qs(query, keep_blank_values=True, strict_parsing=False)
    unknown = set(parsed) - allowed
    if unknown:
        raise ValueError(f"Unsupported query parameter: {sorted(unknown)[0]}")
    if any(len(values) != 1 for values in parsed.values()):
        raise ValueError("Query parameters must appear once")
    return {key: values[0] for key, values in parsed.items()}


def _page_number(value: str | None, default: int, *, zero: bool = False) -> int:
    if value is None:
        return default
    if not value.isdecimal():
        raise ValueError("Pagination values must be non-negative integers")
    parsed = int(value)
    if not zero and parsed == 0:
        raise ValueError("limit must be positive")
    return parsed


def _uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as error:
        raise EntityNotFoundError(f"{label} not found") from error
    if str(parsed) != value:
        raise EntityNotFoundError(f"{label} not found")
    return value


def _job_view(operation: dict[str, Any]) -> dict[str, Any]:
    progress = operation.get("progress")
    safe_progress = None
    if isinstance(progress, dict):
        safe_progress = {}
        phase = progress.get("phase")
        if isinstance(phase, str) and _SAFE_PHASE_RE.fullmatch(phase):
            safe_progress["phase"] = phase
        for key in ("completed", "total"):
            value = progress.get(key)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) and 0 <= value <= 1_000_000_000_000):
                safe_progress[key] = value
        if isinstance(progress.get("cancel_requested"), bool):
            safe_progress["cancel_requested"] = progress["cancel_requested"]
        pending_reason = progress.get("cancel_pending_reason")
        if pending_reason in {
            "worker_identity_unavailable", "worker_identity_mismatch",
            "worker_tree_unavailable", "worker_exit_unconfirmed",
        }:
            safe_progress["cancel_pending_reason"] = pending_reason
        if not safe_progress:
            safe_progress = None
    raw_result = operation.get("result")
    safe_result = None
    if isinstance(raw_result, dict):
        safe_result = {}
        for key in ("id", "run_id"):
            value = raw_result.get(key)
            if isinstance(value, str) and _SAFE_LABEL_RE.fullmatch(value):
                safe_result[key] = value
        status = raw_result.get("status")
        if status in {"created", "running", "accepted", "rejected", "succeeded", "failed", "invalid"}:
            safe_result["status"] = status
        accepted = raw_result.get("accepted")
        if isinstance(accepted, bool):
            safe_result["accepted"] = accepted
        primary_score = raw_result.get("primary_score")
        if (isinstance(primary_score, (int, float)) and not isinstance(primary_score, bool)
                and math.isfinite(float(primary_score))):
            safe_result["primary_score"] = primary_score
        metrics = raw_result.get("metrics")
        if isinstance(metrics, dict):
            safe_result["metrics"] = {
                str(name): value for name, value in metrics.items()
                if isinstance(name, str) and _SAFE_LABEL_RE.fullmatch(name) and isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(float(value))
            }
        if not safe_result:
            safe_result = None
    raw_error = operation.get("error")
    raw_code = str(raw_error.get("code") or "operation_failed") if isinstance(raw_error, dict) else ""
    safe_error = {"code": raw_code if _SAFE_LABEL_RE.fullmatch(raw_code) else "operation_failed",
                  "message": "Operation failed"} if isinstance(raw_error, dict) else None
    operation_id = str(operation["operation_id"])
    return {
        "operation_id": operation_id,
        "lab_id": str(operation["lab_id"]),
        "command": str(operation["kind"]),
        "status": str(operation["status"]),
        "created_at": operation.get("created_at"),
        "updated_at": operation.get("updated_at"),
        "progress": safe_progress,
        "result": safe_result,
        "error": safe_error,
        "poll_url": f"/api/v1/jobs/{operation_id}",
    }


def _metric_summary(run: dict[str, Any]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    repeats = run.get("repeats")
    if isinstance(repeats, list):
        for repeat in repeats:
            metrics = repeat.get("metrics") if isinstance(repeat, dict) else None
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if (isinstance(key, str) and isinstance(value, (int, float))
                        and not isinstance(value, bool) and math.isfinite(float(value))):
                    found.setdefault(key, []).append(float(value))
    if found:
        return found
    manifest = run.get("manifest")
    results = manifest.get("runs") if isinstance(manifest, dict) else None
    if isinstance(results, list):
        for result in results:
            metrics = result.get("metrics") if isinstance(result, dict) else None
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if (isinstance(key, str) and isinstance(value, (int, float))
                        and not isinstance(value, bool) and math.isfinite(float(value))):
                    found.setdefault(key, []).append(float(value))
    return found


def _run_summary(run: dict[str, Any], config: Any | None = None) -> dict[str, Any]:
    manifest = run.get("manifest") if isinstance(run.get("manifest"), dict) else {}
    if config is not None:
        from .cockpit import _metric_summary as _cockpit_metric_summary

        metrics = {
            name: {"direction": direction, **_cockpit_metric_summary(manifest, name, direction)}
            for name, direction in ((config.primary, config.direction), *list(config.guards))
        }
    else:
        metrics = _metric_summary(run)
    return {
        "run_id": str(run.get("run_id") or manifest.get("id") or ""),
        "alias": str(run.get("alias") or manifest.get("id") or ""),
        "kind": run.get("kind"),
        "status": manifest.get("status") or run.get("status"),
        "epoch_id": run.get("epoch_id") or manifest.get("epoch"),
        "created_at": run.get("created_at") or manifest.get("created_at_utc"),
        "parent_run_id": str(run.get("parent_run_id") or manifest.get("parent") or "") or None,
        "title": (manifest.get("hypothesis", {}).get("title") if isinstance(manifest.get("hypothesis"), dict) else None) or manifest.get("title") or "",
        "reason": manifest.get("reason") or "",
        "hypothesis": manifest.get("hypothesis") if isinstance(manifest.get("hypothesis"), dict) else None,
        "statistic": getattr(config, "statistic", "worst") if config is not None else "worst",
        "usage": manifest.get("usage") if isinstance(manifest.get("usage"), dict) else None,
        "regressed_case_ids": manifest.get("regressed_case_ids") if isinstance(manifest.get("regressed_case_ids"), list) else [],
        "metrics": metrics,
    }


def _load_lab_config(lab_id: str) -> Any:
    """Project the authoritative SQLite config into the legacy runtime consumer shape."""
    from .config import load_config

    return load_config(lab_id)


def _comparison_with_config(config: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    from .cockpit import _comparable_manifests, _metric_summary as _cockpit_metric_summary

    left_manifest = left.get("manifest") if isinstance(left.get("manifest"), dict) else {}
    right_manifest = right.get("manifest") if isinstance(right.get("manifest"), dict) else {}
    left_view = _run_summary(left, config)
    right_view = _run_summary(right, config)
    keys = ("config_sha256", "input_hashes", "evaluator")
    fingerprint_complete = all(left_manifest.get(key) is not None and right_manifest.get(key) is not None for key in keys)
    statuses_terminal = left_view["status"] in {"accepted", "rejected"} and right_view["status"] in {"accepted", "rejected"}
    contract_match = _comparable_manifests(left_manifest, right_manifest) if fingerprint_complete and statuses_terminal else None
    left_pins = left_manifest.get("pin_hashes", left_manifest.get("pins"))
    right_pins = right_manifest.get("pin_hashes", right_manifest.get("pins"))
    pins_match = left_pins == right_pins if left_pins is not None and right_pins is not None else None
    comparable = (False if contract_match is False or pins_match is False else
                  True if contract_match is True and pins_match is True else None)

    metric_specs = [(config.primary, config.direction), *list(config.guards)]
    metrics: dict[str, Any] = {}
    for name, direction in metric_specs:
        left_stats = _cockpit_metric_summary(left_manifest, name, direction)
        right_stats = _cockpit_metric_summary(right_manifest, name, direction)
        lm, rm = left_stats["mean"], right_stats["mean"]
        metrics[name] = {
            "direction": direction,
            "left": left_stats,
            "right": right_stats,
            "delta_mean": rm - lm if lm is not None and rm is not None else None,
            "decision_available": (
                left_stats["completed_repeats"] >= config.repeats
                and right_stats["completed_repeats"] >= config.repeats
            ),
        }
    return {
        "left": left_view,
        "right": right_view,
        "comparable": comparable,
        "contract_match": contract_match,
        "pin_fingerprints_match": pins_match,
        "metrics": metrics,
    }


def _bounded_text(value: Any, field: str, *, limit: int = 4000, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise ValueError(f"{field} must be a non-empty string no longer than {limit} characters")
    return value


def _validate_command(body: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    if set(body) not in ({"command", "payload"}, {"command", "payload", "confirmed"}):
        raise ValueError("Command body must contain command, payload, and optional confirmed")
    command = body.get("command")
    payload = body.get("payload")
    if command not in _COMMANDS or not isinstance(payload, dict):
        raise ValueError("command must be one of baseline, trial, resume, recover, epoch, cancel and payload must be an object")
    if command != "cancel" and body.get("confirmed") is not True:
        raise ValueError("confirmed must be the boolean true for evaluator/write operations")

    if command == "baseline":
        if set(payload) - {"supersede", "reason"}:
            raise ValueError("Unsupported baseline payload field")
        params: dict[str, Any] = {}
        if "supersede" in payload:
            if not isinstance(payload["supersede"], bool):
                raise ValueError("supersede must be a boolean")
            params["supersede"] = payload["supersede"]
        if "reason" in payload:
            params["reason"] = _bounded_text(payload["reason"], "reason")
        label = "baseline"
    elif command == "cancel":
        if set(payload) != {"run_id"}:
            raise ValueError("cancel payload must contain run_id")
        params = {"run_id": _bounded_text(payload["run_id"], "run_id", limit=128, required=True)}
        label = "cancel"
    elif command == "trial":
        if set(payload) != {"hypothesis"} or not isinstance(payload.get("hypothesis"), dict):
            raise ValueError("trial payload must contain one hypothesis object")
        params = {"hypothesis": normalize_trial_hypothesis(payload["hypothesis"])}
        label = "trial"
    elif command == "resume":
        if set(payload) != {"run_id"}:
            raise ValueError("resume payload must contain run_id")
        params = {"run_id": _bounded_text(payload["run_id"], "run_id", limit=128, required=True)}
        label = "resume"
    elif command == "recover":
        if set(payload) - {"restore_candidate"}:
            raise ValueError("Unsupported recover payload field")
        restore = payload.get("restore_candidate", True)
        if not isinstance(restore, bool):
            raise ValueError("restore_candidate must be a boolean")
        params = {"restore_candidate": restore}
        label = "recover"
    else:
        if set(payload) - {"action", "spec", "reason"} or payload.get("action") != "open":
            raise ValueError("epoch payload requires action=open and only spec/reason fields")
        if not isinstance(payload.get("spec"), dict):
            raise ValueError("epoch spec must be an object")
        params = {"action": "open", "spec": payload["spec"]}
        if "reason" in payload:
            params["reason"] = _bounded_text(payload["reason"], "reason")
        label = "epoch"
    if command != "cancel":
        params["confirmed"] = True
    return command, params, label


def _handle_get(handler: Any, store: SQLiteStore, parts: list[str], query: str) -> bool:
    if parts == ["api", "v1", "revision"]:
        _single_query(query, set())
        revision = str(store.get_revision())
        if handler.headers.get("If-None-Match", "").strip('"') == revision:
            handler._send(304, b"", "application/json; charset=utf-8", {"ETag": f'"{revision}"'})
            return True
        _send(handler, 200, _envelope(data={"revision": revision, "initialized": store.is_initialized()}),
              extra={"ETag": f'"{revision}"'})
        return True

    if parts == ["api", "v1", "labs"]:
        values = _single_query(query, {"limit", "offset", *_FILTERS, "lifecycle"})
        requested_limit = _page_number(values.get("limit"), 20)
        offset = _page_number(values.get("offset"), 0, zero=True)
        lifecycle = values.get("lifecycle", "active")
        if lifecycle not in {"active", "archived"}:
            raise ValueError("lifecycle must be 'active' or 'archived'")
        filters = {key: value for key, value in values.items() if key in _FILTERS}
        page = store.list_labs(limit=min(requested_limit, 100), offset=offset, filters=filters,
                               lifecycle=lifecycle, include_dashboard_summary=True)
        _send(handler, 200, _envelope(data=page, next_value=page.get("next_cursor")))
        return True

    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "guide":
        values = _single_query(query, {"run"})
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        from .state import load_state
        from .cockpit import build_cockpit

        config = _load_lab_config(lab_id)
        state = load_state(config)
        guide = build_cockpit(config, state, run_id=values.get("run"))
        _send(handler, 200, _envelope(data=guide))
        return True

    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "runs":
        values = _single_query(query, {"limit", "offset", "cursor", "epoch_id"})
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        requested_limit = _page_number(values.get("limit"), 20)
        offset = _page_number(values.get("offset"), 0, zero=True)
        epoch_id = None
        if "epoch_id" in values:
            epoch_id = _page_number(values["epoch_id"], 1)
        page = store.list_runs(lab_id, limit=min(requested_limit, 100), offset=offset,
                               cursor=values.get("cursor"), epoch_id=epoch_id, include_metrics=True)
        _send(handler, 200, _envelope(data=page, next_value=page.get("next_cursor")))
        return True

    if len(parts) == 6 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "runs":
        _single_query(query, set())
        lab_id = _uuid(parts[3], "Lab")
        run_id = _uuid(parts[5], "Run")
        store.get_lab(lab_id)
        run = store.get_run(lab_id, run_id)
        if run is None:
            raise EntityNotFoundError("Run not found")
        _send(handler, 200, _envelope(data=_run_summary(run, _load_lab_config(lab_id))))
        return True

    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "compare":
        values = _single_query(query, {"left", "right"})
        if set(values) != {"left", "right"}:
            raise ValueError("left and right run references are required")
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        left = store.get_run(lab_id, values["left"])
        right = store.get_run(lab_id, values["right"])
        if left is None or right is None:
            raise EntityNotFoundError("Run not found in this lab")
        _send(handler, 200, _envelope(data=_comparison_with_config(_load_lab_config(lab_id), left, right)))
        return True

    if len(parts) == 6 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "evidence":
        values = _single_query(query, set())
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        artifact_id = _uuid(parts[5], "Artifact")
        # Implemented through the repository-owned ownership check; no caller path is accepted.
        artifact = store.get_artifact(lab_id, artifact_id)
        if artifact is None:
            raise EntityNotFoundError("Evidence artifact not found in this lab")
        size = int(artifact.get("size_bytes") or 0)
        if size > MAX_ARTIFACT_BYTES:
            raise OverflowError("Evidence is too large for the bounded JSON endpoint")
        content = store.load_artifact(artifact_id)
        result = {
            key: artifact.get(key)
            for key in ("artifact_id", "lab_id", "run_id", "name", "evidence_key", "media_type", "sha256", "size_bytes", "created_at")
        }
        result["encoding"] = "base64"
        result["content_base64"] = base64.b64encode(content).decode("ascii")
        _send(handler, 200, _envelope(data=result))
        return True

    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "evidence":
        values = _single_query(query, {"limit", "offset", "run"})
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        limit = min(_page_number(values.get("limit"), 20), 100)
        offset = _page_number(values.get("offset"), 0, zero=True)
        records = store.list_artifacts(lab_id, values.get("run"), limit=limit, offset=offset)
        items = [{key: str(item[key]) if key in {"artifact_id", "lab_id", "run_id"} and item.get(key) is not None
                  else item.get(key) for key in (
                      "artifact_id", "lab_id", "run_id", "name", "evidence_key", "media_type",
                      "sha256", "size_bytes", "created_at",
                  )} for item in records]
        page = {"items": items, "limit": limit, "offset": offset,
                "next_offset": offset + limit if len(records) == limit else None}
        _send(handler, 200, _envelope(data=page))
        return True

    if len(parts) == 4 and parts[:3] == ["api", "v1", "jobs"]:
        _single_query(query, set())
        operation_id = _uuid(parts[3], "Job")
        operation = store.get_operation(operation_id)
        _send(handler, 200, _envelope(data=_job_view(operation)))
        return True
    return False


def _handle_post(handler: Any, store: SQLiteStore, parts: list[str]) -> bool:
    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "lifecycle":
        lab_id = _uuid(parts[3], "Lab")
        body = _read_json(handler)
        if set(body) - {"action", "confirm_lab_id"}:
            raise ValueError("Lifecycle request has unsupported fields")
        action = body.get("action")
        if not isinstance(action, str) or action not in {"archive", "restore", "delete"}:
            raise ValueError("action must be 'archive', 'restore', or 'delete'")
        confirm_lab_id = body.get("confirm_lab_id")
        if action == "delete" and confirm_lab_id != lab_id:
            raise ValueError("confirm_lab_id must exactly match the lab UUID for delete")
        if confirm_lab_id is not None and confirm_lab_id != lab_id:
            raise ValueError("confirm_lab_id must match the lab UUID")
        result = store.set_lab_lifecycle(lab_id, action, confirm_lab_id=confirm_lab_id)
        _send(handler, 200, _envelope(data=result))
        return True

    if parts == ["api", "v1", "labs"]:
        body = _read_json(handler)
        if set(body) != {"confirmed", "program", "spec", "passport"}:
            raise ValueError("Lab creation requires confirmed, program, spec, and passport")
        if body.get("confirmed") is not True:
            raise ValueError("confirmed must be true")
        if not isinstance(body.get("program"), str) or not Path(body["program"]).is_absolute():
            raise ValueError("program must be an absolute local file path")
        if not isinstance(body.get("spec"), dict) or not isinstance(body.get("passport"), dict):
            raise ValueError("spec and passport must be objects")
        from .bootstrap import prepare_lab_from_dict

        prepared = prepare_lab_from_dict(
            Path(body["program"]), body["spec"], confirmed=True, passport=body["passport"]
        )
        _send(handler, 201, _envelope(data=prepared))
        return True

    if len(parts) == 5 and parts[:3] == ["api", "v1", "labs"] and parts[4] == "commands":
        lab_id = _uuid(parts[3], "Lab")
        store.get_lab(lab_id)
        idempotency_key = handler.headers.get("Idempotency-Key", "")
        if not _HEADER_TOKEN_RE.fullmatch(idempotency_key):
            raise ValueError("Idempotency-Key must contain 1 to 200 safe characters")
        body = _read_json(handler)
        command, params, _ = _validate_command(body)
        if command == "cancel":
            from .runtimeops import cancel_run

            operation = cancel_run(str(lab_id), params["run_id"])
            _send(handler, 202, _envelope(data=_job_view(operation)))
            return True
        if command == "trial":
            from .runtimeops import safe_operation_context
            from .trial import validate_trial_request

            config = _load_lab_config(lab_id)
            safe_operation_context(config)
            params["hypothesis"] = validate_trial_request(config, params["hypothesis"])
        operation = enqueue_and_start_operation(lab_id, command, params, idempotency_key)
        _send(handler, 202, _envelope(data=_job_view(operation),
                                       warnings=["Worker launch needs explicit recovery; inspect the job status"]
                                       if operation.get("spawn_error") or operation.get("status") == "recovery_required" else []))
        return True
    return False


def dispatch(handler: Any) -> bool:
    """Serve `/api/v1/*`; return False only for a non-v1 path."""
    parsed = urlsplit(handler.path)
    if not parsed.path.startswith("/api/v1/"):
        return False
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    method = str(getattr(handler, "command", "GET")).upper()
    try:
        authenticate(handler)
        store = SQLiteStore(resolve_home())
        if method == "GET":
            if _handle_get(handler, store, parts, parsed.query):
                return True
        elif method == "POST":
            if _handle_post(handler, store, parts):
                return True
        else:
            _send_error(handler, 405, "method_not_allowed", "Method is not allowed")
            return True
        _send_error(handler, 404, "not_found", "API route not found")
    except AuthError as error:
        _send_error(handler, error.status, error.code, str(error))
    except OverflowError as error:
        _send_error(handler, 413, "request_too_large", str(error))
    except (IdempotencyConflictError, ActiveOperationError, StoreConflictError) as error:
        _send_error(handler, 409, getattr(error, "code", "conflict"), str(error))
    except AmbiguousAliasError as error:
        _send_error(handler, 409, error.code, str(error))
    except EntityNotFoundError:
        _send_error(handler, 404, "not_found", "The requested lab, run, job, or artifact was not found")
    except StoreNotInitializedError:
        _send_error(handler, 503, "store_not_initialized", "The lab database is not initialized")
    except StoreBusyError:
        _send_error(handler, 503, "store_busy", "The lab database is busy; retry after the active write finishes")
    except LabError as error:
        status = 409 if getattr(error, "code", "") in {
            "E_LOCKED", "operation_already_active", "operation_not_cancellable", "operation_state_conflict",
        } else 400
        _send_error(handler, status, getattr(error, "code", "invalid_request"), str(error),
                    getattr(error, "hint", ""))
    except ValueError as error:
        _send_error(handler, 400, "invalid_request", str(error))
    except StoreError:
        _send_error(handler, 503, "store_error", "The lab database could not complete the request")
    except Exception:
        # Do not expose filesystem paths, evaluator output, or request secrets.
        _send_error(handler, 500, "internal_error", "The local API could not complete the request")
    return True

