"""Runs the baseline evaluation and finalizes its records and fixture digests."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from .config import LabConfig
from .errors import LabError
from .evaluator import _budget_overrun, _evaluate_once, _remaining
from .gitutil import _restore_candidate
from .state import _check_candidate_unchanged, _check_frozen, _finalize_setup, _fingerprints, _runs, _save_manifest, _verify_passport_pins, load_state
from .store import _write_json


def _body_field_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _request_identity_digest(config: LabConfig, baseline_id: str, repeat_count: int) -> str | None:
    chunks: list[str] = []
    from .store import _sqlite_document
    for index in range(repeat_count):
        log_path = config.state_dir / "runs" / baseline_id / f"repeat_{index + 1:02d}" / "sent-requests.jsonl"
        target = _sqlite_document(log_path)
        if target is not None and target[2] == "wire":
            store, lab_id, _, run_id, repeat_index_db = target
            # SQLite exposes attempt journals newest-first; the digest is order-sensitive.
            records = list(reversed(store.load_wire_events(lab_id, run_id, repeat_index=repeat_index_db)))
        elif log_path.is_file():
            records = []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
        else:
            return None
        sequences = [record.get("wire_sequence") for record in records]
        if (records
                and all(type(sequence) is int and sequence > 0 for sequence in sequences)
                and len(set(sequences)) == len(records)):
            records = sorted(records, key=lambda record: record["wire_sequence"])
        for record in records:
            # Повтор 429/503 пишет то же тело ещё раз. В отпечаток входит только исходный запрос.
            if record.get("status") in (429, 503):
                continue
            if "body" not in record:
                return None
            chunks.append(_body_field_text(record["body"]))
    return hashlib.sha256("".join(chunks).encode("utf-8")).hexdigest()


def _finalize_baseline_records(config: LabConfig, state: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Commit baseline outcome and state pointers together for SQLite labs."""
    from .sqlite_store import SQLiteStore
    from .runtimeops import mark_operation_finalizing
    mark_operation_finalizing()
    SQLiteStore().finalize_run(config.lab_id, manifest["id"], state, manifest)
def _evaluator_fixture_digest(spec: dict[str, Any]) -> tuple[str | None, str | None]:
    fixture = spec.get("evaluator_fixture")
    if not isinstance(fixture, dict):
        return None, None
    path_value = fixture.get("path")
    expected = fixture.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        return None, "evaluator_fixture requires path and sha256"
    path = Path(path_value)
    if not path.is_file():
        return None, "evaluator_fixture file is missing"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        return None, f"evaluator_fixture sha256 mismatch: expected {expected}, got {actual}"
    return actual, None


def _finalize_baseline(
    config: LabConfig, state: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    from .measurement import record_noise_floor

    _verify_passport_pins(config, manifest)
    baseline_id = manifest["id"]
    overrun = _budget_overrun(config)
    if overrun:
        manifest.update(status="budget", reason=overrun)
        _save_manifest(config, manifest)
        raise LabError(f"Baseline failed: {overrun}")
    validation = config.baseline_validation
    if validation is not None:
        mode = validation["mode"]
        checks: list[dict[str, Any]] = []
        mismatch = None
        if mode == "historical":
            tolerance = float(validation.get("tolerance", 1e-9))
            for index, (expected, actual_run) in enumerate(zip(validation["runs"], manifest["runs"])):
                for metric, expected_value in expected["metrics"].items():
                    actual_value = actual_run["metrics"].get(metric)
                    matches = (isinstance(actual_value, (int, float)) and not isinstance(actual_value, bool)
                               and math.isfinite(actual_value)
                               and abs(float(actual_value) - float(expected_value)) <= tolerance)
                    checks.append({"repeat": index + 1, "reference_run": expected["name"],
                                   "metric": metric, "expected": expected_value,
                                   "actual": actual_value, "matches": matches})
                    if not matches and mismatch is None:
                        mismatch = (f"repeat {index + 1} metric {metric}: expected {expected_value}, "
                                    f"got {actual_value}")
        model_repro = validation.get("model_repro")
        model_repro_status: str | None = None
        historical_reference: list[dict[str, Any]] | None = None
        if mode == "historical" and model_repro is not None:
            model_repro_status = "match" if mismatch is None else "drift"
            historical_reference = list(validation["runs"])
            if not model_repro.get("blocking", True):
                mismatch = None

        block_reason: str | None = None
        if mode == "historical":
            if "evaluator_fixture" in validation:
                block_reason = _evaluator_fixture_digest(validation)[1]
            request_spec = validation.get("request_identity")
            if block_reason is None and isinstance(request_spec, dict):
                expected_hash = request_spec.get("sha256")
                if isinstance(expected_hash, str):
                    actual_hash = _request_identity_digest(config, baseline_id, config.repeats)
                    if actual_hash is None:
                        block_reason = (
                            "request_identity: sent-requests.jsonl or captured request bodies are missing; "
                            "set AUTORESEARCH_WIRE_LOG_BODY=1 and rerun"
                        )
                    elif actual_hash != expected_hash:
                        block_reason = (f"request_identity sha256 mismatch: expected {expected_hash}, "
                                        f"got {actual_hash}")
        if block_reason is None and mismatch is not None:
            block_reason = f"historical baseline mismatch: {mismatch}"
        if block_reason is not None:
            manifest.update(status="incomparable", reason=block_reason,
                            comparison_status="incomparable",
                            baseline_validation={"mode": mode, "status": "incomparable",
                                                 "tolerance": validation.get("tolerance", 1e-9),
                                                 "checks": checks,
                                                 **({"model_repro_status": model_repro_status,
                                                     "historical_reference": historical_reference}
                                                    if model_repro_status is not None else {})})
            state["baseline_verified"] = False
            state["baseline_id"] = None
            state["best_id"] = None
            _finalize_baseline_records(config, state, manifest)
            _restore_candidate(config, state["best_commit"])
            raise LabError(f"Baseline is incomparable: {block_reason}")
        manifest["baseline_validation"] = {
            "mode": mode, "status": "verified",
            **({"known_case": validation["known_case"]} if mode == "fresh" else {
                "tolerance": validation.get("tolerance", 1e-9), "checks": checks,
                **({"model_repro_status": model_repro_status,
                    "historical_reference": historical_reference}
                   if model_repro_status is not None else {}),
            }),
        }
    manifest["status"] = "accepted"
    record_noise_floor(state, config, manifest["runs"])
    state["baseline_id"] = baseline_id
    state["best_id"] = baseline_id
    if validation is not None:
        state["baseline_verified"] = True
    state["setup_pending"] = False
    _finalize_baseline_records(config, state, manifest)
    return manifest


def run_baseline(config: LabConfig, *, supersede: bool = False, reason: str | None = None) -> dict[str, Any]:
    from .runtimeops import require_durable_operation_context
    require_durable_operation_context(config, "baseline")
    from .measurement import ensure_epoch, migrate_state, open_epoch

    state = load_state(config)
    migrate_state(state)
    if state["baseline_id"] is not None and not supersede:
        raise LabError("Baseline already exists")
    if supersede:
        if state["baseline_id"] is None:
            raise LabError("No baseline to supersede")
        if not isinstance(reason, str) or not reason.strip():
            raise LabError("supersede requires --reason")
        open_epoch(state, config, reason=reason.strip())
        state["baseline_id"] = None
        state["best_id"] = None
        state["baseline_verified"] = False
    epoch_id = ensure_epoch(state, config)
    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    raw_config_bytes = store.load_config_bytes(config.lab_id)
    store.save_config_and_state(
        config.lab_id,
        json.loads(raw_config_bytes.decode("utf-8")),
        state,
        raw_config_bytes=raw_config_bytes,
    )
    _finalize_setup(config, state)
    _check_frozen(config, state)
    previous = [item for item in _runs(config) if item["id"] == "baseline" or item["id"].startswith("baseline-")]
    if any(item["status"] == "running" for item in previous):
        from .trial import recover_lab

        recover_lab(config, restore_candidate=True)
        state = load_state(config)
        previous = [item for item in _runs(config) if item["id"] == "baseline" or item["id"].startswith("baseline-")]
    baseline_id = "baseline" if not previous else f"baseline-{len(previous) + 1:04d}"
    manifest: dict[str, Any] = {
        "id": baseline_id, "title": "Baseline", "status": "running", "runs": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": state["best_commit"], "epoch": epoch_id,
        "evaluator": list(config.evaluator),
        **_fingerprints(config, state),
    }
    _save_manifest(config, manifest)
    for index in range(config.repeats):
        try:
            timeout = min(config.timeout_seconds, _remaining(config))
        except LabError as error:
            manifest.update(status="budget", reason=str(error))
            _save_manifest(config, manifest)
            _restore_candidate(config, state["best_commit"])
            raise LabError(f"Baseline failed: {error}") from error
        status, result, reason = _evaluate_once(config, config.state_dir / "runs" / baseline_id / f"repeat_{index + 1:02d}", timeout)
        manifest["runs"].append(result)
        _save_manifest(config, manifest)
        if status != "ok":
            manifest.update(status=status, reason=reason)
            _save_manifest(config, manifest)
            _restore_candidate(config, state["best_commit"])
            raise LabError(f"Baseline failed: {reason}")
        try:
            _check_candidate_unchanged(config, manifest["candidate_hashes"])
            _check_frozen(config, state)
        except LabError as error:
            manifest.update(status="invalid", reason=str(error))
            _save_manifest(config, manifest)
            _restore_candidate(config, state["best_commit"])
            raise LabError(f"Baseline failed: {error}") from error
    return _finalize_baseline(config, state, manifest)
