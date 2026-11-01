"""Loads and validates the lab configuration (LabConfig and load_config)."""

from __future__ import annotations

import fnmatch
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


from .errors import LabError
from .store import _sha256, _write_json


@dataclass(frozen=True)
class LabConfig:
    path: Path
    source_root: Path
    state_dir: Path
    source_files: tuple[str, ...]
    editable_files: frozenset[str] | None
    input_files: tuple[Path, ...]
    evaluator: tuple[str, ...]
    primary: str
    direction: str
    guards: tuple[tuple[str, str], ...]
    repeats: int
    timeout_seconds: float
    budget: dict[str, float | None]
    baseline_validation: dict[str, Any] | None = None
    delegation_required: bool = False
    large_candidate_files: frozenset[str] = frozenset()
    pending: bool = False
    network_retry: bool = True
    exclude_globs: tuple[str, ...] = ()
    expected_attestations: dict[str, str] | None = None
    acceptance: dict[str, float | int] | None = None
    repeat_policy: str = "stop_on_failure"
    unit: str | None = None
    target_threshold: float | None = None
    guard_thresholds: dict[str, dict[str, float]] | None = None
    goal_criteria: dict[str, Any] | None = None
    evaluator_capabilities: dict[str, Any] = field(default_factory=dict)
    storage_mode: str = "legacy"
    lab_id: str | None = None
    config_bytes: bytes | None = None
    passport: dict[str, Any] | None = None


SERVICE_PARTS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", "results", "outputs",
    ".lab", ".labs", ".autoresearch", ".ssh", ".aws",
})

SECRET_BASENAMES = frozenset({".npmrc", ".netrc", "credentials.json", "secrets.json"})

MAX_UNCLASSIFIED_CANDIDATE_BYTES = 25 * 1024 * 1024

def _service_file(name: str) -> bool:
    parts = PurePosixPath(name).parts
    base = parts[-1]
    base_lower = base.lower()
    return (any(part.lower() in SERVICE_PARTS for part in parts)
            or base_lower == ".env" or base_lower.startswith(".env.")
            or base_lower in SECRET_BASENAMES
            or base_lower.startswith("id_rsa")
            or base_lower.endswith((".pem", ".key", ".pfx", ".kdbx")))


def _matches_exclude_glob(name: str, globs: tuple[str, ...]) -> bool:
    if not globs:
        return False
    normalized = name.replace("\\", "/")
    for pattern in globs:
        if not isinstance(pattern, str) or not pattern:
            continue
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(normalized.split("/")[-1], pattern):
            return True
    return False

def _relative_file(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabError("Paths must be non-empty strings")
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ":" in path.parts[0] or any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(f"Unsafe relative path: {value}")
    if ".git" in path.parts:
        raise LabError("Git internals cannot be snapshotted")
    return path.as_posix()

def _validate_fresh_known_case(path: Path, source_root: Path, state_dir: Path,
                               inputs: tuple[Path, ...], evaluator: tuple[str, ...],
                               primary: str, guards: tuple[tuple[str, str], ...],
                               known_case: Any) -> None:
    if not isinstance(known_case, dict) or not isinstance(known_case.get("path"), str):
        raise LabError("Fresh baseline requires a saved known-case validation record")
    record_path = Path(known_case["path"])
    if not record_path.is_absolute():
        record_path = (path.parent / record_path).resolve()
    else:
        record_path = record_path.resolve()
    if (not record_path.is_file() or record_path.is_relative_to(source_root)
            or record_path.is_relative_to(state_dir / "candidate")):
        raise LabError("Known-case record must exist outside the candidate and source project")
    expected_record_hash = known_case.get("sha256")
    if not isinstance(expected_record_hash, str) or _sha256(record_path) != expected_record_hash.lower():
        raise LabError("Known-case validation record hash is missing or mismatched")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabError("Known-case validation record is invalid") from error
    if not isinstance(record, dict) or record.get("schema") != 1 or not record.get("id") or record.get("reproducible") is not True:
        raise LabError("Known-case validation record must identify a reproducible known case")
    uses_python = evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python")
    script_arg = evaluator[1] if uses_python and len(evaluator) > 1 else evaluator[0]
    script_path = Path(script_arg)
    if not script_path.is_absolute():
        script_path = path.parent / script_path
    if not script_path.is_file() or record.get("evaluator_sha256") != _sha256(script_path):
        raise LabError("Known-case record does not match the frozen evaluator")
    expected_inputs = {str(item.resolve()): _sha256(item) for item in inputs if item.resolve() != script_path.resolve()}
    recorded_inputs = record.get("input_hashes")
    if not isinstance(recorded_inputs, dict):
        raise LabError("Known-case record must include hashes of its inputs")
    normalized_inputs = {}
    for name, digest in recorded_inputs.items():
        input_path = Path(name)
        input_path = (record_path.parent / input_path).resolve() if not input_path.is_absolute() else input_path.resolve()
        normalized_inputs[str(input_path)] = digest
    if normalized_inputs != expected_inputs:
        raise LabError("Known-case input hashes must exactly match the protected evaluator inputs")
    expected_metrics = record.get("expected_metrics")
    observed_metrics = record.get("observed_metrics")
    required = {primary, *(name for name, _ in guards)}
    tolerance = record.get("tolerance", 1e-9)
    if (not isinstance(expected_metrics, dict) or not isinstance(observed_metrics, dict)
            or not required.issubset(expected_metrics) or not required.issubset(observed_metrics)
            or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance < 0):
        raise LabError("Known-case record is missing expected/observed metric values")
    for name in required:
        expected = expected_metrics.get(name)
        observed = observed_metrics.get(name)
        if (isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed)
                or abs(float(expected) - float(observed)) > tolerance):
            raise LabError(f"Known-case metric {name} does not reproduce its expected value")

def _research_fields(raw: dict[str, Any], path: Path, source: Path, state_dir: Path,
                     evaluator: tuple[str, ...], inputs: list[Path]) -> tuple[dict | None, dict]:
    """Optional research metadata never changes the trial acceptance rules."""
    goal = raw.get("goal_criteria")
    if goal is not None:
        if not isinstance(goal, dict) or set(goal) - {"thresholds", "confirmation_repeats"}:
            raise LabError("goal_criteria needs thresholds and confirmation_repeats")
        thresholds = goal.get("thresholds")
        count = goal.get("confirmation_repeats", 1)
        if not isinstance(thresholds, dict) or not thresholds:
            raise LabError("goal_criteria.thresholds must be a non-empty object")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise LabError("goal_criteria.confirmation_repeats must be positive")
        for name, bounds in thresholds.items():
            if not isinstance(name, str) or not name or not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max"}:
                raise LabError("Goal thresholds need a metric name and min/max bounds")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds.values()):
                raise LabError("Goal thresholds must be finite numbers")
            if bounds.get("min", -math.inf) > bounds.get("max", math.inf):
                raise LabError("Goal threshold min exceeds max")
        goal = {"thresholds": thresholds, "confirmation_repeats": count}

    capabilities = raw.get("evaluator_capabilities", {})
    if not isinstance(capabilities, dict) or set(capabilities) - {"case_resume", "offline_verified", "resume_fixture"}:
        raise LabError("Invalid evaluator_capabilities")
    capabilities = dict(capabilities)
    for name in ("case_resume", "offline_verified"):
        value = capabilities.setdefault(name, False)
        if not isinstance(value, bool):
            raise LabError(f"evaluator_capabilities.{name} must be boolean")
    fixture = capabilities.get("resume_fixture")
    if capabilities["offline_verified"] and fixture is None:
        raise LabError("offline_verified requires a hash-pinned resume_fixture")
    if fixture is not None:
        if not isinstance(fixture, dict) or not isinstance(fixture.get("path"), str) or not isinstance(fixture.get("sha256"), str):
            raise LabError("resume_fixture requires path and sha256")
        fixture_path = (path.parent / fixture["path"]).resolve()
        if fixture_path.is_relative_to(source) or fixture_path.is_relative_to(state_dir / "candidate"):
            raise LabError("resume_fixture must stay outside source and candidate")
        if not fixture_path.is_file() or _sha256(fixture_path) != fixture["sha256"]:
            raise LabError("resume_fixture hash mismatch")
        report = json.loads(fixture_path.read_text(encoding="utf-8"))
        python = evaluator and (evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python"))
        script_name = evaluator[1] if python and len(evaluator) > 1 else (evaluator[0] if evaluator else "")
        script = (path.parent / script_name).resolve()
        if (not isinstance(report, dict) or report.get("schema") != 1 or report.get("offline") is not True
                or report.get("case_resume") is not True or not script.is_file()
                or report.get("evaluator_sha256") != _sha256(script)):
            raise LabError("resume_fixture does not verify this evaluator")
        capabilities["resume_fixture"] = {"path": str(fixture_path), "sha256": fixture["sha256"]}
        inputs.append(fixture_path)
    return goal, capabilities


def _load_config(path: Path | str, *, allow_pending: bool = False, read_only: bool = False) -> LabConfig:
    from .snapshot import _discover_source
    raw_ref = str(path)
    try:
        canonical_ref = str(uuid.UUID(raw_ref))
    except (ValueError, TypeError, AttributeError):
        canonical_ref = None
    path = Path(path).expanduser().resolve()
    storage_mode = "legacy"
    lab_id: str | None = None
    config_bytes: bytes | None = None
    passport: dict[str, Any] | None = None
    try:
        from .sqlite_store import EntityNotFoundError, SQLiteStore
        store = SQLiteStore()
        if store.is_initialized():
            if canonical_ref is not None:
                try:
                    resolved_id = store.resolve_lab_id(canonical_ref)
                except EntityNotFoundError as error:
                    raise LabError(
                        f"Unknown SQLite lab: {canonical_ref}", code="E_LAB_NOT_FOUND",
                        hint="Pass a valid --lab reference or a known `lab.json` path with --config.",
                    ) from error
            else:
                try:
                    resolved_id = store.resolve_lab_id(path)
                except EntityNotFoundError:
                    resolved_id = None
            if resolved_id is not None:
                row = store.get_lab(resolved_id)
                path = Path(row["config_path"])
                config_bytes = store.load_config_bytes(resolved_id)
                raw_projection = store.load_projection(resolved_id, "config")
                passport = raw_projection.get("passport") if isinstance(raw_projection.get("passport"), dict) else None
                lab_id = str(resolved_id)
                storage_mode = "sqlite"
    except ImportError:
        pass
    try:
        if config_bytes is None:
            config_bytes = path.read_bytes()
        raw = json.loads(config_bytes.decode("utf-8"))
        if (any(key in raw and not isinstance(raw[key], list) for key in ("source_paths", "editable_files", "large_candidate_files"))
                or not isinstance(raw["evaluator"], list)):
            raise LabError("File lists and evaluator must be arrays")
        if not isinstance(raw.get("input_files", []), list):
            raise LabError("input_files must be an array")
        source_root = (path.parent / raw.get("source_root", ".")).resolve()
        state_dir = (path.parent / raw.get("state_dir", "../AutoResearch-state")).resolve()
        if not source_root.is_dir() or state_dir == source_root or source_root in state_dir.parents:
            raise LabError("Source must exist and state must be outside it")
        state_path = state_dir / "state.json"
        lab_config_path = state_dir / "lab.json"
        if storage_mode == "legacy" and not read_only and state_path.is_file() and lab_config_path.is_file():
            try:
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                if state_data.get("setup_pending"):
                    actual_sha = _sha256(lab_config_path)
                    if state_data.get("config_copy_sha256") != actual_sha:
                        state_data["config_copy_sha256"] = actual_sha
                        if path.resolve() == lab_config_path.resolve():
                            state_data["config_sha256"] = actual_sha
                        _write_json(state_path, state_data)
            except Exception:
                pass
        exclude_raw = raw.get("exclude_globs", [])
        if exclude_raw is None:
            exclude_raw = []
        if not isinstance(exclude_raw, list) or any(not isinstance(item, str) for item in exclude_raw):
            raise LabError("exclude_globs must be an array of glob patterns")
        exclude_globs = tuple(exclude_raw)
        selected: set[str] = set()
        if "source_paths" not in raw:
            selected.update(_discover_source(source_root, exclude_globs))
        else:
            for entry in raw["source_paths"]:
                name = _relative_file(entry)
                target = source_root / name
                if target.is_symlink() or not target.exists() or not target.resolve().is_relative_to(source_root):
                    raise LabError(f"Missing or linked source path: {name}")
                paths = [target] if target.is_file() else target.rglob("*")
                for file in paths:
                    relative = file.relative_to(source_root)
                    if ".git" in relative.parts:
                        raise LabError("Git internals cannot be snapshotted")
                    if file.is_symlink():
                        raise LabError(f"Linked source file: {file}")
                    if not file.resolve().is_relative_to(source_root):
                        raise LabError(f"Source path leaves project: {file}")
                    if file.is_file():
                        selected.add(relative.as_posix())
        if not selected:
            raise LabError("Source snapshot is empty")
        editable = (frozenset(_relative_file(item) for item in raw["editable_files"])
                    if "editable_files" in raw else None)
        if editable is not None and (not editable or not editable.issubset(selected)):
            raise LabError("Editable files must exist in the source snapshot")
        inputs = [(path.parent / name).resolve() for name in raw.get("input_files", [])]
        if any(not item.is_file() for item in inputs):
            raise LabError("Input files must exist")
        protected_sources = {item.relative_to(source_root).as_posix() for item in inputs
                             if item.is_relative_to(source_root)}
        selected.difference_update(protected_sources)
        if not selected:
            raise LabError("Source snapshot is empty after excluding protected input files")
        if editable is None and any((state_dir / "candidate") in item.parents for item in inputs):
            raise LabError("Reference inputs must stay outside the editable candidate")
        if editable is not None and editable & protected_sources:
            raise LabError("Protected input files cannot be editable")
        large_candidate_files = frozenset(_relative_file(item) for item in raw.get("large_candidate_files", []))
        if not large_candidate_files.issubset(selected):
            raise LabError("large_candidate_files must name files in the candidate snapshot")
        for name in sorted(selected):
            file = source_root / name
            if file.stat().st_size > MAX_UNCLASSIFIED_CANDIDATE_BYTES and name not in large_candidate_files:
                raise LabError(
                    f"Unclassified large candidate file: {name}; register it in input_files "
                    "or explicitly allow it with large_candidate_files"
                )
        metrics = raw["metrics"]
        primary = metrics["primary"]
        direction = metrics["direction"]
        guards = tuple((item["name"], item["direction"]) for item in metrics.get("guards", []))
        unit = metrics.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            raise LabError("metrics.unit must be a non-empty string when present")
        target_threshold = metrics.get("target_threshold")
        if target_threshold is not None:
            if isinstance(target_threshold, bool) or not isinstance(target_threshold, (int, float)) or not math.isfinite(target_threshold):
                raise LabError("metrics.target_threshold must be a finite number")
            target_threshold = float(target_threshold)
        guard_thresholds: dict[str, dict[str, float]] = {}
        for item in metrics.get("guards", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            thresh: dict[str, float] = {}
            for t_key in ("min_threshold", "max_threshold"):
                if t_key in item and item[t_key] is not None:
                    t_val = item[t_key]
                    if isinstance(t_val, bool) or not isinstance(t_val, (int, float)) or not math.isfinite(t_val):
                        raise LabError(f"{t_key} for guard {name} must be a finite number")
                    thresh[t_key] = float(t_val)
            if thresh:
                guard_thresholds[name] = thresh
        pending = raw.get("setup_pending", False)
        if not isinstance(pending, bool):
            raise LabError("setup_pending must be a boolean")
        if (not isinstance(primary, str) or (not primary and not (pending and allow_pending))
                or direction not in {"max", "min"}):
            raise LabError("Invalid primary metric")
        if any(not isinstance(name, str) or not name or sense not in {"max", "min"} for name, sense in guards):
            raise LabError("Invalid guard metric")
        repeats = raw["repeats"]
        repeat_policy = raw.get("repeat_policy", "stop_on_failure")
        if repeat_policy not in {"stop_on_failure", "complete_all"}:
            raise LabError("repeat_policy must be stop_on_failure or complete_all")

        timeout = raw["timeout_seconds"]
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise LabError("repeats must be positive")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise LabError("timeout_seconds must be positive")
        evaluator = tuple(raw["evaluator"])
        if (not evaluator and not (pending and allow_pending)) or any(not isinstance(arg, str) or not arg for arg in evaluator):
            raise LabError("evaluator must be an argument list")
        if evaluator:
            uses_python = evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python")
            script = evaluator[1] if uses_python and len(evaluator) > 1 else evaluator[0]
            if script not in {"{candidate}", "{run_dir}", "{config_dir}"}:
                script_path = Path(script)
                external = script_path if script_path.is_absolute() else path.parent / script_path
                if external.is_file():
                    external = external.resolve()
                    if editable is None and (state_dir / "candidate") in external.parents:
                        raise LabError("Evaluator must run outside the editable candidate")
                    if (state_dir / "candidate") not in external.parents:
                        inputs.append(external)
                elif editable is None and (source_root / script_path).is_file():
                    raise LabError("Evaluator must run outside the editable candidate")
        goal_criteria, evaluator_capabilities = _research_fields(raw, path, source_root, state_dir, evaluator, inputs)
        inputs = tuple(dict.fromkeys(inputs))
        if any(not item.is_file() for item in inputs):
            raise LabError("Input files must exist")
        budget = raw["budget"]
        if not isinstance(budget, dict):
            raise LabError("Invalid budget")
        # Retired spend keys (max_requests, max_cost_usd) may remain in stored labs.
        # They are never validated or enforced.
        for key in ("max_seconds", "max_trials"):
            value = budget.get(key)
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise LabError(f"Invalid budget: {key}")
            if key == "max_trials" and not isinstance(value, int):
                raise LabError("Time and trial budgets must be positive; counts must be integers")
        retry_raw = raw.get("network_retry", {"enabled": True})
        if not isinstance(retry_raw, dict) or not isinstance(retry_raw.get("enabled", True), bool):
            raise LabError("network_retry.enabled must be a boolean")
        network_retry = bool(retry_raw.get("enabled", True))
        expected_attestations = raw.get("expected_attestations")
        if expected_attestations is not None:
            if not isinstance(expected_attestations, dict) or any(
                not isinstance(key, str) or not key or not isinstance(value, str)
                for key, value in expected_attestations.items()
            ):
                raise LabError("expected_attestations must be an object of string keys and values")
        acceptance = raw.get("acceptance")
        if acceptance is not None:
            if not isinstance(acceptance, dict):
                raise LabError("acceptance must be an object when present")
            min_gain = acceptance.get("min_gain")
            confirm_repeats = acceptance.get("confirm_repeats", 1)
            if (min_gain is not None and (
                isinstance(min_gain, bool) or not isinstance(min_gain, (int, float))
                or not math.isfinite(min_gain) or min_gain < 0
            )):
                raise LabError("acceptance.min_gain must be a finite non-negative number")
            if (isinstance(confirm_repeats, bool) or not isinstance(confirm_repeats, int)
                    or confirm_repeats < 1):
                raise LabError("acceptance.confirm_repeats must be a positive integer")
        baseline_validation = raw.get("baseline_validation")
        if baseline_validation is not None:
            if not isinstance(baseline_validation, dict) or baseline_validation.get("mode") not in {"fresh", "historical"}:
                raise LabError("baseline_validation.mode must be fresh or historical")
            if baseline_validation["mode"] == "fresh":
                _validate_fresh_known_case(path, source_root, state_dir, inputs, evaluator, primary, guards,
                                           baseline_validation.get("known_case"))
            else:
                runs = baseline_validation.get("runs")
                tolerance = baseline_validation.get("tolerance", 1e-9)
                input_hashes = baseline_validation.get("input_hashes")
                if not isinstance(runs, list) or len(runs) != repeats or not isinstance(input_hashes, dict) or not input_hashes:
                    raise LabError("Historical baseline validation needs one expected run and frozen input hashes per repeat")
                if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                        or not math.isfinite(tolerance) or tolerance < 0):
                    raise LabError("Historical baseline tolerance must be finite and non-negative")
                expected_metrics = {primary, *(name for name, _ in guards)}
                names: set[str] = set()
                known_inputs = {str(item): item for item in inputs}
                for name, expected_hash in input_hashes.items():
                    resolved = str(Path(name).resolve() if Path(name).is_absolute() else (path.parent / name).resolve())
                    input_file = known_inputs.get(resolved)
                    if (input_file is None or not isinstance(expected_hash, str)
                            or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash.lower())
                            or _sha256(input_file) != expected_hash.lower()):
                        raise LabError(f"Historical baseline input hash is missing or mismatched: {name}")
                for index, item in enumerate(runs):
                    if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                        raise LabError("Historical baseline runs need unique names and metric values")
                    if item["name"] in names:
                        raise LabError("Historical baseline run names must be unique")
                    names.add(item["name"])
                    expected = item.get("metrics")
                    if not isinstance(expected, dict) or not expected_metrics.issubset(expected):
                        raise LabError(f"Historical baseline repeat {index + 1} is missing expected metrics")
                    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in expected.values()):
                        raise LabError("Historical baseline metrics must be finite numbers")
        delegation = raw.get("delegation", {})
        if delegation is None:
            delegation = {}
        if not isinstance(delegation, dict):
            raise LabError("delegation must be an object when present")
        # Old labs stored a required subagent. It is kept on disk and ignored.
        delegation_required = False
        strict_contract = raw.get("strict_contract", False)
        if not isinstance(strict_contract, bool):
            raise LabError("strict_contract must be a boolean")
        if strict_contract:
            if baseline_validation is None:
                raise LabError("Strict labs require baseline validation")
            declared_budget = raw.get("budget_declared", [])
            budget_sources = raw.get("budget_sources", {})
            if (not isinstance(declared_budget, list) or not isinstance(budget_sources, dict)
                    or any(key not in {"max_seconds", "max_trials", "max_requests", "max_cost_usd"}
                           for key in declared_budget)):
                raise LabError("Invalid strict budget provenance")
            try:
                brief = (path.parent / "program.snapshot.md").read_text(encoding="utf-8")
            except OSError as error:
                raise LabError("Strict lab is missing program.snapshot.md") from error
            for key in declared_budget:
                # Retired spend keys need no quote and are not checked; loading ignores them.
                if key in {"max_requests", "max_cost_usd"} or budget.get(key) is None:
                    continue
                quote = budget_sources.get(key)
                if not isinstance(quote, str) or not quote.strip() or quote not in brief:
                    raise LabError(f"Budget source for {key} is not an exact quote from program.md")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise LabError(f"Invalid lab config: {error}") from error
    return LabConfig(
        path=path, source_root=source_root, state_dir=state_dir,
        source_files=tuple(sorted(selected)), editable_files=editable, input_files=inputs,
        evaluator=evaluator, primary=primary, direction=direction, guards=guards,
        repeats=repeats, repeat_policy=repeat_policy, timeout_seconds=float(timeout), budget=budget,
        baseline_validation=baseline_validation, delegation_required=delegation_required,
        large_candidate_files=large_candidate_files, pending=pending,
        network_retry=network_retry, exclude_globs=exclude_globs,
        expected_attestations=expected_attestations, acceptance=acceptance,
        unit=unit, target_threshold=target_threshold,
        guard_thresholds=guard_thresholds or None,
        goal_criteria=goal_criteria, evaluator_capabilities=evaluator_capabilities,
        storage_mode=storage_mode, lab_id=lab_id, config_bytes=config_bytes, passport=passport,
    )


def load_config(path: Path | str, *, allow_pending: bool = False, read_only: bool = False) -> LabConfig:
    """Load a runtime lab config and normalize malformed inputs for CLI callers."""
    try:
        return _load_config(path, allow_pending=allow_pending, read_only=read_only)
    except LabError as error:
        if error.code in {"E_CONFIG_INVALID", "E_LAB_NOT_FOUND", "E_LAB_AMBIGUOUS"}:
            raise
        raise LabError(str(error), code="E_CONFIG_INVALID", hint=error.hint) from error

