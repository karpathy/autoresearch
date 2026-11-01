"""Pinned baseline contracts. A pin blocks a fresh baseline that would bypass it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_home
from .store import _sha256


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _pin_paths() -> list[Path]:
    paths: list[Path] = []
    home_pins = resolve_home() / "pins"
    if home_pins.is_dir():
        paths.extend(sorted(home_pins.glob("*.json")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _load_pin(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_pin_contract(
    reference: dict[str, Any], program: Path, project: Path, spec: dict[str, Any], inputs: list[Path],
) -> None:
    project_name = reference.get("project_name")
    if not isinstance(project_name, str) or not project_name.strip():
        return
    if project.name.casefold() != project_name.casefold():
        return
    validation = spec.get("baseline_validation")
    if not isinstance(validation, dict) or validation.get("mode") != "historical":
        raise ValueError(
            f"{project_name or 'Registered project'} is pinned; fresh baseline cannot bypass historical validation"
        )
    metrics = spec.get("metrics", {})
    guards = {item.get("name"): item.get("direction") for item in metrics.get("guards", [])
              if isinstance(item, dict)}
    primary = metrics.get("primary")
    required_metrics = {primary, *guards} if primary else set()
    if reference.get("metrics"):
        ref_metrics = reference["metrics"]
        if isinstance(ref_metrics, dict):
            if metrics.get("primary") != ref_metrics.get("primary"):
                raise ValueError("Pinned contract primary metric mismatch")
            ref_guards = ref_metrics.get("guards", [])
            if isinstance(ref_guards, list):
                for item in ref_guards:
                    if isinstance(item, dict) and guards.get(item.get("name")) != item.get("direction"):
                        raise ValueError("Pinned contract guard metrics mismatch")
    expected_runs = reference.get("runs", [])
    tolerance = float(reference.get("tolerance", 1e-9))
    if (spec.get("repeats", 2) != len(expected_runs)
            or abs(float(validation.get("tolerance", tolerance)) - tolerance) > 0
            or not isinstance(validation.get("runs"), list)
            or len(validation["runs"]) != len(expected_runs)):
        raise ValueError("Pinned contract requires matching historical repeats and tolerance")
    for expected, actual in zip(expected_runs, validation["runs"]):
        if (actual.get("name") != expected.get("name")
                or not required_metrics.issubset(actual.get("metrics", {}))
                or any(abs(float(actual["metrics"][name]) - float(expected["metrics"][name])) > tolerance
                       for name in required_metrics)):
            raise ValueError("Baseline repeats do not match the pinned metric contract")
    expected_hashes: dict[str, str] = {}
    for item in reference.get("inputs_relative_to_program_md", []):
        path = _path(program.parent, item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"Pinned reference input is missing or changed: {item['path']}")
        expected_hashes[str(path)] = item["sha256"]
    actual_hashes = validation.get("input_hashes")
    if not isinstance(actual_hashes, dict):
        raise ValueError("Pinned contract requires frozen reference input hashes")
    normalized_hashes = {str(_path(program.parent, name)): value for name, value in actual_hashes.items()}
    if any(normalized_hashes.get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("Pinned reference input hashes differ from the registered contract")
    input_paths = {str(path.resolve()) for path in inputs}
    if expected_hashes and not set(expected_hashes).issubset(input_paths):
        raise ValueError("Pinned reference files must all be registered as protected input_files")


def validate(program: Path, project: Path, spec: dict[str, Any], inputs: list[Path]) -> None:
    """Apply every pin that matches this project (AUTORESEARCH_HOME/pins).

    A damaged or unknown pin raises instead of being skipped, because skipping it would silently drop its guard.
    """
    for path in _pin_paths():
        try:
            reference = _load_pin(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Pin file cannot be read: {path.name}") from error
        if not isinstance(reference, dict):
            raise ValueError(f"Pin file is not a JSON object: {path.name}")
        if reference.get("schema") not in (None, 1):
            raise ValueError(f"Pin file has an unknown schema: {path.name}")
        _validate_pin_contract(reference, program, project, spec, inputs)


def _has_comparable_history(root: Path, project: Path, metric: str, inputs: list[Path],
                            evaluator_sha256: str, *, exclude_state_dir: Path | None = None) -> bool:
    current_hashes = {str(path.resolve()): _sha256(path) for path in inputs}
    excluded = exclude_state_dir.resolve() if exclude_state_dir is not None else None
    from .sqlite_store import SQLiteStore
    store = SQLiteStore()
    if store.is_initialized():
        offset = 0
        while True:
            page = store.list_labs(limit=100, offset=offset, filters={"source_root": project})
            for row in page["items"]:
                if excluded is not None and Path(row["state_dir"]).resolve() == excluded:
                    continue
                lab_id = row["lab_id"]
                try:
                    raw = json.loads(store.load_config_bytes(lab_id).decode("utf-8"))
                    if (Path(raw.get("source_root", "")).resolve() != project.resolve()
                            or raw.get("metrics", {}).get("primary") != metric):
                        continue
                    evaluator = raw.get("evaluator", [])
                    if not isinstance(evaluator, list) or not evaluator:
                        continue
                    script_index = 1 if evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python") else 0
                    old_script = Path(evaluator[script_index])
                    if not old_script.is_absolute():
                        old_script = Path(row["state_dir"]) / old_script
                    if not old_script.is_file() or _sha256(old_script) != evaluator_sha256:
                        continue
                    state = store.load_state(lab_id)
                    # Imported history may be visible as results, but only a
                    # verified baseline can prevent a fresh run from bypassing history.
                    if state.get("baseline_verified") is not True:
                        continue
                    baseline_id = state.get("baseline_id")
                    baseline = store.load_manifest(lab_id, baseline_id) if baseline_id else None
                    if (not isinstance(baseline, dict) or baseline.get("status") != "accepted"
                            or not str(baseline_id).startswith("baseline")):
                        continue
                    old_hashes: dict[str, str] = {}
                    declared_inputs = raw.get("input_files", [])
                    if not isinstance(declared_inputs, list):
                        continue
                    inputs_valid = True
                    config_dir = Path(row["config_path"]).resolve().parent
                    for name in declared_inputs:
                        if not isinstance(name, str):
                            inputs_valid = False
                            break
                        path = Path(name)
                        path = (path if path.is_absolute() else config_dir / path).resolve()
                        if not path.is_file():
                            inputs_valid = False
                            break
                        old_hashes[str(path)] = _sha256(path)
                    if inputs_valid and old_hashes == current_hashes:
                        return True
                except (OSError, KeyError, TypeError, ValueError, UnicodeError):
                    continue
            offset += len(page["items"])
            if offset >= page["total"] or not page["items"]:
                break
    roots = list((root / "projects").glob("*/*/lab.json")) + list(root.glob("*/lab.json"))
    roots += list((root / "archive").glob("*/*/lab.json"))
    for config_path in roots:
        if excluded is not None and config_path.parent.resolve() == excluded:
            continue
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            source_root = Path(raw.get("source_root", "")).resolve()
            metrics = raw.get("metrics", {})
            if source_root != project.resolve() or metrics.get("primary") != metric:
                continue
            lab_dir = config_path.parent
            old_evaluator = raw.get("evaluator", [])
            if not isinstance(old_evaluator, list) or not old_evaluator:
                continue
            old_script_index = 1 if old_evaluator[0] == "{python}" or Path(old_evaluator[0]).name.lower().startswith("python") else 0
            old_script = Path(old_evaluator[old_script_index])
            if not old_script.is_absolute():
                old_script = config_path.parent / old_script
            if not old_script.is_file() or _sha256(old_script) != evaluator_sha256:
                continue
            state = json.loads((lab_dir / "state.json").read_text(encoding="utf-8"))
            manifests = [json.loads(path.read_text(encoding="utf-8"))
                         for path in (lab_dir / "runs").glob("baseline*/manifest.json")]
            if not (state.get("baseline_id") or any(item.get("status") == "accepted" for item in manifests)):
                continue
            old_hashes = {}
            for name in raw.get("input_files", []):
                path = Path(name)
                path = path if path.is_absolute() else (config_path.parent / path)
                path = path.resolve()
                if path.is_file():
                    old_hashes[str(path)] = _sha256(path)
            if old_hashes == current_hashes:
                return True
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return False
