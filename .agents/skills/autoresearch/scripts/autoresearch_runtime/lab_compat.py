"""Audit and migrate lab.json format without rewriting run history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .store import _sha256

CURRENT_LAB_FORMAT = 2

_PLACEHOLDER_ARGS = frozenset({"{python}", "{candidate}", "{run_dir}", "{config_dir}"})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_path(config_path: Path, raw: dict[str, Any]) -> Path:
    state_dir = raw.get("state_dir", "../AutoResearch-state")
    path = Path(state_dir)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path / "state.json"


def _evaluator_script_paths(config_path: Path, evaluator: list[Any]) -> list[Path]:
    if not isinstance(evaluator, list):
        return []
    args = [item for item in evaluator if isinstance(item, str) and item]
    if not args:
        return []
    uses_python = args[0] == "{python}" or Path(args[0]).name.lower().startswith("python")
    script = args[1] if uses_python and len(args) > 1 else args[0]
    if script in _PLACEHOLDER_ARGS:
        return []
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = (config_path.parent / script_path).resolve()
    return [script_path]


def _path_like_evaluator_args(config_path: Path, evaluator: list[Any]) -> list[Path]:
    paths: list[Path] = []
    if not isinstance(evaluator, list):
        return paths
    for index, arg in enumerate(evaluator):
        if not isinstance(arg, str) or not arg or arg in _PLACEHOLDER_ARGS:
            continue
        if index == 0 and (arg == "{python}" or Path(arg).name.lower().startswith("python")):
            continue
        if arg.startswith("-"):
            continue
        if "/" in arg or "\\" in arg or arg.endswith(".py"):
            candidate = Path(arg)
            if not candidate.is_absolute():
                candidate = (config_path.parent / candidate).resolve()
            paths.append(candidate)
    return paths


def _openrouter_budget_missing(config_path: Path, raw: dict[str, Any], scripts: list[Path]) -> bool:
    lab_text = config_path.read_text(encoding="utf-8")
    if "openrouter_budget" not in lab_text:
        for script in scripts:
            if script.is_file():
                try:
                    if "openrouter_budget" in script.read_text(encoding="utf-8"):
                        break
                except OSError:
                    continue
        else:
            return False
    state_dir = _state_path(config_path, raw).parent
    search_roots = {config_path.parent, state_dir, *(script.parent for script in scripts)}
    for root in search_roots:
        if (root / "openrouter_budget.py").is_file():
            return False
    return True


def audit_lab(config_path: Path) -> dict[str, Any]:
    """Returns format_version, findings: list[{code, path, message}], migratable: bool."""
    config_path = config_path.resolve()
    findings: list[dict[str, str]] = []
    try:
        raw = _read_json(config_path)
    except (OSError, ValueError) as error:
        return {
            "format_version": None,
            "findings": [{
                "code": "invalid_lab_json",
                "path": str(config_path),
                "message": str(error),
            }],
            "migratable": False,
        }

    declared = raw.get("lab_format_version")
    if declared is None:
        format_version = 1
        findings.append({
            "code": "missing_format_version",
            "path": str(config_path),
            "message": "lab.json has no lab_format_version; treating as format version 1",
        })
    elif not isinstance(declared, int) or isinstance(declared, bool):
        format_version = 1
        findings.append({
            "code": "missing_format_version",
            "path": str(config_path),
            "message": "lab_format_version must be a positive integer",
        })
    else:
        format_version = int(declared)

    evaluator = raw.get("evaluator", [])
    for target in _path_like_evaluator_args(config_path, evaluator):
        if not target.is_file():
            findings.append({
                "code": "evaluator_missing",
                "path": str(target),
                "message": "Evaluator path from lab.json does not exist",
            })

    scripts = _evaluator_script_paths(config_path, evaluator)
    if _openrouter_budget_missing(config_path, raw, scripts):
        findings.append({
            "code": "dependency_missing",
            "path": str(config_path.parent / "openrouter_budget.py"),
            "message": "Configuration references openrouter_budget but openrouter_budget.py is missing",
        })

    state_path = _state_path(config_path, raw)
    epochs_missing_on_disk = False
    hash_drift = False
    if state_path.is_file():
        try:
            state_raw = _read_json(state_path)
        except (OSError, ValueError) as error:
            findings.append({
                "code": "invalid_state_json",
                "path": str(state_path),
                "message": str(error),
            })
            state_raw = {}
        if "epochs" not in state_raw or not isinstance(state_raw.get("epochs"), list):
            epochs_missing_on_disk = True
            findings.append({
                "code": "epochs_memory_only",
                "path": str(state_path),
                "message": "state.json has no persisted epochs list",
            })
        recorded = state_raw.get("config_sha256")
        if isinstance(recorded, str):
            current = _sha256(config_path)
            if recorded != current:
                hash_drift = True
                findings.append({
                    "code": "config_hash_drift",
                    "path": str(state_path),
                    "message": "state.json config_sha256 does not match lab.json",
                })
    else:
        findings.append({
            "code": "state_missing",
            "path": str(state_path),
            "message": "state.json is missing",
        })

    migratable = format_version < CURRENT_LAB_FORMAT or epochs_missing_on_disk or hash_drift
    return {
        "format_version": format_version,
        "findings": findings,
        "migratable": migratable,
    }
