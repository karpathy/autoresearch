"""Turn an agent's interpretation of program.md into a reproducible lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .envelope import emit, fail
from .config import validate_config_file
from .lab import LabError, init_lab, load_config, load_state
from .paths import brief_path, home_root_id, project_directory, project_id, resolve_home
from .pins import _has_comparable_history, validate as pins_validate
from .runtime_env import hidden_process_kwargs


def _path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# AutoResearch does not limit spending. New specs cannot declare the retired spend fields;
# stored labs that still carry them are read with those fields ignored.
SPEND_LIMIT_CODE = "E_CONTRACT_SPEND_LIMIT"
RETIRED_SPEND_KEYS = ("max_requests", "max_cost_usd")


def _reject_spend_limits(spec: dict[str, Any]) -> None:
    declared = [key for key in RETIRED_SPEND_KEYS
                if any(isinstance(spec.get(field), dict) and key in spec[field]
                       for field in ("budget", "budget_sources"))]
    if declared:
        raise LabError(
            "AutoResearch does not limit spending. Remove "
            + " and ".join(f"budget.{key}" for key in declared)
            + " from the spec, and set a limit on the API key that the evaluator uses.",
            code=SPEND_LIMIT_CODE,
            hint="max_seconds and max_trials still apply. Usage and cost are still recorded.",
        )


def _reject_retired_fields(spec: dict[str, Any]) -> None:
    """A declared gate that no longer runs must fail loudly, not pass silently."""
    validation = spec.get("baseline_validation")
    if isinstance(validation, dict) and "request_identity" in validation:
        raise ValueError("baseline_validation.request_identity is no longer supported; remove it from the spec")
    budget = spec.get("budget")
    if isinstance(budget, dict) and "openrouter" in budget:
        raise ValueError("budget.openrouter is no longer supported; remove it from the spec")


def _retired_spend_identity() -> dict[str, Any]:
    """Historical values of the retired spend fields. They feed the lab id and are never stored."""
    return {"max_requests": 0, "max_cost_usd": 0}


def _without_retired_spend(stored: Any) -> Any:
    """A lab stored before 0.8.0 may still carry the retired fields. They are ignored, not compared."""
    budget = stored.get("budget") if isinstance(stored, dict) else None
    if not isinstance(budget, dict) or not any(key in budget for key in RETIRED_SPEND_KEYS):
        return stored
    return {**stored, "budget": {key: value for key, value in budget.items() if key not in RETIRED_SPEND_KEYS}}


def _lab_root(program: Path, project: Path) -> Path:
    root = resolve_home()
    if root == project.resolve() or root.is_relative_to(project.resolve()):
        raise ValueError("AUTORESEARCH_HOME must stay outside the measured project")
    return root


def _brief_goal(program: Path, spec: dict[str, Any]) -> str:
    declared = spec.get("goal")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()[:240]
    try:
        lines = [line.strip() for line in program.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return ""
    for line in lines:
        if line.lower().startswith(("goal:", "цель:")):
            return line.split(":", 1)[1].strip()[:240]
    return (lines[0] if lines else "")[:240]


def _confirmed_goal(program: Path, spec: dict[str, Any]) -> str:
    declared = spec.get("goal")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()[:240]
    try:
        lines = program.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for index, line in enumerate(lines):
        label, separator, value = line.partition(":")
        if separator and label.strip().casefold() in {"goal", "цель"} and value.strip():
            return value.strip()[:240]
        # The passport template in references/brief.md uses a "## Goal" section.
        if line.strip().casefold() in {"## goal", "## цель"}:
            for following in lines[index + 1:]:
                if following.startswith("#"):
                    break
                if following.strip():
                    return following.strip()[:240]
    return ""


def _resume_from_config(program: Path, project: Path, config_path: Path, storage_root: Path) -> dict[str, Any] | None:
    from .sqlite_store import EntityNotFoundError, SQLiteStore
    db_store = SQLiteStore(storage_root)
    if not db_store.is_initialized():
        return None
    try:
        db_lab_id = db_store.resolve_lab_id(config_path)
    except EntityNotFoundError:
        return None
    raw_config = db_store.load_config_bytes(db_lab_id)
    try:
        relative = Path(db_store.get_lab(db_lab_id)["state_dir"]).relative_to(storage_root.resolve())
    except ValueError:
        return None
    if "archive" in relative.parts:
        return None
    saved_config = json.loads(raw_config.decode("utf-8"))
    load_ref = str(db_lab_id)
    if saved_config.get("strict_contract") is not True:
        return None
    try:
        config = load_config(load_ref, allow_pending=True)
    except (LabError, OSError, ValueError):
        return None
    state = load_state(config)
    if state.get("baseline_id") is not None and state.get("baseline_verified") is not True:
        return None
    config_sha256 = hashlib.sha256(config.config_bytes).hexdigest()
    if config.source_root != project or config_sha256 != state["config_sha256"]:
        raise ValueError("Active lab configuration changed")
    snapshot = config.state_dir / "program.snapshot.md"
    if not snapshot.is_file() or _hash(snapshot) != _hash(program):
        return None
    for item in state["inputs"]:
        file = Path(item["path"])
        if not file.is_file() or _hash(file) != item["sha256"]:
            return None
    home = _lab_root(program, project)
    resumed = {"config": str(config.path), "candidate": str(config.state_dir / "candidate"),
               "dashboard": None,
               "dashboard_root": str(home),
               "created": False}
    resumed["lab_id"] = config.lab_id
    return resumed


def resume_lab(program: Path, project: Path) -> dict[str, Any] | None:
    """Find the last lab for an unchanged brief without the staging evaluator."""
    program = program.resolve()
    if not program.is_file():
        raise ValueError(f"program.md is missing: {program}")
    project = (project if project.is_absolute() else program.parent / project).resolve()
    if not project.is_dir():
        raise ValueError(f"Project is missing: {project}")
    home = _lab_root(program, project)
    from .sqlite_store import SQLiteStore
    store = SQLiteStore(home)
    if store.is_initialized():
        if True:
            active_id = store.get_active_lab(home_root_id(home), project_id(project))
            if active_id is not None:
                row = store.get_lab(active_id)
                resumed = _resume_from_config(program, project, Path(row["config_path"]), home)
                if resumed is not None:
                    return resumed
            offset = 0
            while True:
                page = store.list_labs(limit=100, offset=offset, filters={"source_root": project})
                for row in page["items"]:
                    snapshot = Path(row["state_dir"]) / "program.snapshot.md"
                    if not snapshot.is_file() or _hash(snapshot) != _hash(program):
                        continue
                    resumed = _resume_from_config(program, project, Path(row["config_path"]), home)
                    if resumed is not None:
                        return resumed
                offset += len(page["items"])
                if offset >= page["total"] or not page["items"]:
                    break
    return None


def prepare_lab(
    program: Path | None,
    spec_path: Path,
    *,
    dry_run: bool = False,
    confirmed: bool | None = None,
    passport: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate agent-derived settings, preserve the evaluator, and prepare or resume."""
    spec_path = spec_path.resolve()
    has_program = program is not None
    if not has_program and not dry_run:
        raise ValueError("A confirmed brief is required to prepare a lab")
    program = program.resolve() if has_program else spec_path
    if not program.is_file():
        raise ValueError(f"program.md is missing: {program}")
    try:
        brief_bytes = program.read_bytes() if has_program else b""
        brief_text = brief_bytes.decode("utf-8")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("Derived spec must be an object")
        _reject_spend_limits(spec)
        _reject_retired_fields(spec)
        project = _path(program.parent, spec["project"])
        if not project.is_dir():
            raise ValueError(f"Project is missing: {project}")
        if spec_path.is_relative_to(project):
            raise ValueError("Agent-derived spec must stay outside the project")
        evaluator = spec["evaluator"]
        if not isinstance(evaluator, list) or not evaluator or any(not isinstance(arg, str) or not arg for arg in evaluator):
            raise ValueError("evaluator must be a non-empty command array")
        evaluator = evaluator.copy()
        uses_python = evaluator[0] == "{python}" or Path(evaluator[0]).name.lower().startswith("python")
        script_index = 1 if uses_python else 0
        if script_index >= len(evaluator):
            raise ValueError("evaluator script is missing")
        script = _path(program.parent, evaluator[script_index])
        if not script.is_file():
            raise ValueError(f"evaluator script is missing: {script}")
        if spec.get("evaluator_generated", False) and script.is_relative_to(project):
            raise ValueError("Generated evaluator must be outside the project")
        evaluator[script_index] = str(script)
        declared_budget = spec.get("budget", {})
        if not isinstance(declared_budget, dict):
            raise ValueError("budget must be an object")
        budget_sources = spec.get("budget_sources", {})
        if not isinstance(budget_sources, dict):
            raise ValueError("budget_sources must be an object")
        for key_name in ("max_seconds", "max_trials"):
            if key_name in declared_budget and declared_budget[key_name] is not None:
                quote = budget_sources.get(key_name)
                if not isinstance(quote, str) or not quote.strip() or quote not in brief_text:
                    raise ValueError(f"budget source for {key_name} must be an exact quote from program.md")
        metrics = spec["metrics"]
        if not isinstance(metrics, dict) or not metrics.get("primary") or metrics.get("direction") not in {"min", "max"}:
            raise ValueError("metrics needs a primary name and min/max direction")
        inputs = [_path(program.parent, name) for name in spec.get("input_files", [])]
        capabilities = json.loads(json.dumps(spec.get("evaluator_capabilities", {})))
        if isinstance(capabilities, dict) and isinstance(capabilities.get("resume_fixture"), dict):
            fixture = capabilities["resume_fixture"]
            fixture["path"] = str(_path(program.parent, fixture["path"]))
            inputs.append(Path(fixture["path"]))
        inputs = list(dict.fromkeys(inputs))
        if any(not item.is_file() for item in inputs):
            raise ValueError("input_files must name existing files")
        baseline_validation = spec.get("baseline_validation")
        if not isinstance(baseline_validation, dict) or baseline_validation.get("mode") not in {"fresh", "historical"}:
            raise ValueError("baseline_validation is required and must use fresh or historical mode")
        baseline_validation = json.loads(json.dumps(baseline_validation))
        pins_validate(program, project, {
            **spec, "baseline_validation": baseline_validation,
        }, inputs)
        if baseline_validation["mode"] == "historical":
            hashes = baseline_validation.get("input_hashes")
            if not isinstance(hashes, dict):
                raise ValueError("historical baseline requires input_hashes")
            normalized_hashes = {}
            for input_name, digest in hashes.items():
                input_path = _path(program.parent, input_name)
                normalized_hashes[str(input_path)] = digest
            baseline_validation["input_hashes"] = normalized_hashes
        else:
            known_case = baseline_validation.get("known_case")
            if (not isinstance(known_case, dict) or not isinstance(known_case.get("path"), str)
                    or not isinstance(known_case.get("sha256"), str)):
                raise ValueError("fresh baseline requires a known_case record path and SHA-256")
            known_case["path"] = str(_path(program.parent, known_case["path"]))
        delegation = spec.get("delegation", {"required": False})
        if not isinstance(delegation, dict):
            raise ValueError("delegation must be an object when present")
        budget = {"max_seconds": None, "max_trials": None}
        budget.update(declared_budget)
        # The lab id hashes the historical budget shape, so a spec that never declared the
        # retired spend fields keeps its id. Those fields are not written to lab.json.
        identity_budget = {**budget, **_retired_spend_identity()}
        network_retry = spec.get("network_retry")
        if network_retry is not None and (
            not isinstance(network_retry, dict) or not isinstance(network_retry.get("enabled", True), bool)
        ):
            raise ValueError("network_retry.enabled must be a boolean")
        contract = {
            "contract_version": 2, "project": str(project),
            "evaluator": evaluator[:script_index] + [{"sha256": _hash(script),
                "path": None if spec.get("evaluator_generated", False) else str(script)}] + evaluator[script_index + 1:],
            "inputs": [{"path": str(item), "sha256": _hash(item)} for item in inputs],
            "metrics": metrics, "repeats": spec.get("repeats", 2),
            "timeout_seconds": spec.get("timeout_seconds", 120), "budget": identity_budget,
            "budget_sources": budget_sources,
            "baseline_validation": baseline_validation,
            "delegation": delegation,
            "source_paths": spec.get("source_paths"), "editable_files": spec.get("editable_files"),
            "large_candidate_files": spec.get("large_candidate_files"),
            "exclude_globs": spec.get("exclude_globs"),
        }
        if network_retry is not None:
            contract["network_retry"] = {"enabled": network_retry.get("enabled", True)}
        # Omitted fields preserve legacy contract identifiers exactly.
        for key_name in ("goal_criteria", "evaluator_capabilities", "expected_attestations"):
            if key_name in spec:
                contract[key_name] = capabilities if key_name == "evaluator_capabilities" else spec[key_name]
        key = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        root = _lab_root(program, project)
        state_dir = project_directory(root, project) / key
        config_path = state_dir / "lab.json"
        if (baseline_validation["mode"] == "fresh" and not config_path.exists()
                and _has_comparable_history(root, project, metrics["primary"], inputs, _hash(script),
                                            exclude_state_dir=state_dir)):
            raise ValueError("Comparable baseline history exists; fresh mode cannot bypass it")
        if spec.get("evaluator_generated", False):
            evaluator[script_index] = str(state_dir / "inputs" / script.name)
        raw = {
            "lab_format_version": 2,
            "source_root": str(project), "state_dir": str(state_dir),
            "evaluator": evaluator, "input_files": [str(item) for item in inputs],
            "metrics": metrics, "repeats": spec.get("repeats", 2),
            "timeout_seconds": spec.get("timeout_seconds", 120), "budget": budget,
            "baseline_validation": baseline_validation,
            "delegation": delegation,
            "budget_sources": budget_sources,
            "budget_declared": [key for key in ("max_seconds", "max_trials") if key in declared_budget],
            "strict_contract": True,
            "setup_pending": True,
        }
        if network_retry is not None:
            raw["network_retry"] = {"enabled": network_retry.get("enabled", True)}
        if "repeat_policy" in spec:
            raw["repeat_policy"] = spec["repeat_policy"]
        for key_name in ("goal_criteria", "evaluator_capabilities", "expected_attestations"):
            if key_name in spec:
                raw[key_name] = capabilities if key_name == "evaluator_capabilities" else spec[key_name]
        for key_name in ("source_paths", "editable_files", "large_candidate_files", "exclude_globs"):
            if key_name in spec:
                raw[key_name] = spec[key_name]
        goal = _confirmed_goal(program, spec) if has_program else ""
        # Validate everything before creating the real state directory. In particular,
        # reject unclassified large files without leaving a half-created lab behind.
        with tempfile.TemporaryDirectory(prefix="autoresearch-preflight-") as temporary:
            preflight_path = Path(temporary) / "lab.json"
            (Path(temporary) / "program.snapshot.md").write_bytes(brief_bytes)
            preflight_raw = dict(raw)
            preflight_evaluator = list(evaluator)
            preflight_script = script
            if spec.get("evaluator_generated", False):
                preflight_script = Path(temporary) / script.name
                shutil.copy2(script, preflight_script)
                preflight_evaluator[script_index] = str(preflight_script)
            preflight_raw["evaluator"] = preflight_evaluator
            preflight_path.write_text(json.dumps(preflight_raw, ensure_ascii=False), encoding="utf-8")
            checked = validate_config_file(preflight_path, allow_pending=True)
            if dry_run:
                return {"contract_valid": True, "config_preview": preflight_raw,
                        "config": str(config_path), "candidate": str(state_dir / "candidate"),
                        "input_hashes": {str(item): _hash(item) for item in checked.input_files},
                        "integration_status": "unverified", "new_requests": 0}
        raw_config_bytes = (json.dumps(raw, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        passport_pins = [
            {"name": "brief", "path": str(program), "sha256": hashlib.sha256(brief_bytes).hexdigest()},
            {"name": "evaluator", "path": str(script), "sha256": _hash(script)},
            *[{"name": f"input:{index}", "path": str(item), "sha256": _hash(item)}
              for index, item in enumerate(inputs)],
            {"name": "baseline_validation", "payload": baseline_validation,
             "sha256": hashlib.sha256(json.dumps(baseline_validation, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()},
            {"name": "candidate_snapshot", "files": [
                {"path": name, "sha256": _hash(project / name)} for name in checked.source_files
            ]},
        ]
        passport_payload = {
            **(passport or {}),
            "confirmed": True,
            "goal": goal or _brief_goal(program, spec),
            "project": str(project),
            "metrics": metrics,
            "constraints": budget,
            "limits": budget,
            "brief_sha256": hashlib.sha256(brief_bytes).hexdigest(),
            "contract_id": key,
            "pins": passport_pins,
        }
        existing_db_id = None
        from .sqlite_store import EntityNotFoundError, SQLiteStore
        database = SQLiteStore(root)
        if database.is_initialized():
            try:
                existing_db_id = database.resolve_lab_id(config_path)
            except EntityNotFoundError:
                existing_db_id = None
        if existing_db_id is not None:
            existing_bytes = database.load_config_bytes(existing_db_id)
            if _without_retired_spend(json.loads(existing_bytes.decode("utf-8"))) != raw:
                raise ValueError(f"Lab configuration differs from the prepared contract: {config_path}")
            existing_config = load_config(config_path, allow_pending=True)
            if hashlib.sha256(existing_bytes).hexdigest() != load_state(existing_config)["config_sha256"]:
                raise ValueError(f"Lab configuration changed after preparation: {config_path}")
            row = database.get_lab(existing_db_id)
            database.set_active_lab(home_root_id(root), project_id(project), existing_db_id)
            return {"lab_id": str(existing_db_id), "config": str(config_path),
                    "candidate": str(state_dir / "candidate"),
                    "dashboard": None,
                    "dashboard_root": str(root), "created": False,
                    "goal": row["goal"], "primary_metric": row["primary_metric"]}
        if confirmed is not True or not goal:
            raise ValueError("A new confirmed lab requires an explicit confirmation and non-empty goal")
        if state_dir.exists():
            raise ValueError(f"Incomplete lab directory needs inspection: {state_dir}")
        state_dir.mkdir(parents=True)
        if spec.get("evaluator_generated", False):
            target = state_dir / "inputs" / script.name
            target.parent.mkdir()
            shutil.copy2(script, target)
        (state_dir / "program.snapshot.md").write_bytes(program.read_bytes())
        validation_path = state_dir / ".lab-config.validate.json"
        validation_path.write_bytes(raw_config_bytes)
        try:
            validated = validate_config_file(validation_path, allow_pending=True)
        finally:
            validation_path.unlink(missing_ok=True)
        config = replace(validated, path=config_path, config_bytes=raw_config_bytes, passport=passport_payload)
        init_lab(config)
        lab_id = database.resolve_lab_id(config_path)
        database.set_active_lab(home_root_id(root), project_id(project), lab_id)
        return {"lab_id": str(lab_id), "config": str(config_path),
                "candidate": str(state_dir / "candidate"),
                "dashboard": None, "dashboard_root": str(root),
                "created": True, "goal": passport_payload["goal"],
                "primary_metric": metrics["primary"]}
    except LabError as error:
        if error.code == SPEND_LIMIT_CODE:
            raise
        raise ValueError(str(error)) from error
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid agent-derived spec: {error}") from error


def prepare_lab_from_dict(
    program: Path | str,
    spec: dict[str, Any],
    *,
    confirmed: bool,
    passport: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validated API entry point for a confirmed passport and local brief file."""
    if confirmed is not True:
        raise ValueError("Lab creation requires explicit confirmation")
    if not isinstance(spec, dict):
        raise ValueError("Derived spec must be an object")
    program_path = Path(program).expanduser().resolve()
    if not program_path.is_file():
        raise ValueError("A local program.md file is required")
    validated_passport = dict(passport or {})
    supplied_goal = validated_passport.get("goal")
    if isinstance(supplied_goal, str) and supplied_goal.strip() and not spec.get("goal"):
        spec = {**spec, "goal": supplied_goal.strip()}
    with tempfile.TemporaryDirectory(prefix="autoresearch-api-spec-") as temporary:
        spec_path = Path(temporary) / "derived.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        prepared = prepare_lab(program_path, spec_path, confirmed=True, passport=validated_passport)
    if prepared.get("lab_id"):
        config = load_config(Path(prepared["config"]), allow_pending=True)
        state = load_state(config)
        from .cockpit import build_cockpit
        prepared["guide"] = build_cockpit(config, state)
    return prepared


def _write_brief(project: Path, brief_text: str) -> Path:
    if brief_text == "-":
        brief_text = sys.stdin.read()
    if not isinstance(brief_text, str) or not brief_text.strip():
        raise ValueError("Brief text is empty; pass --brief-text or pipe content to stdin with '-'")
    path = brief_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief_text, encoding="utf-8")
    return path.resolve()


def _resolve_program(args: argparse.Namespace, spec_path: Path | None = None) -> Path:
    if args.program is not None and args.brief_text is not None:
        raise ValueError("Use either --program or --brief-text, not both")
    if args.brief_text is not None:
        if args.project is not None:
            project = args.project.resolve()
        elif spec_path is not None:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("Derived spec must be an object")
            project = _path(spec_path.parent, spec["project"])
        else:
            raise ValueError("--brief-text requires --project or a --spec with project")
        if not project.is_dir():
            raise ValueError(f"Project is missing: {project}")
        return _write_brief(project, args.brief_text)
    if args.program is not None:
        return args.program.resolve()
    raise ValueError("Provide --program PATH or --brief-text TEXT ('-' reads stdin)")


def _auto_generate_spec(
    project: Path,
    eval_cmd: str,
    metric_arg: str,
    program: Path | None,
    home: Path,
    repeats: int = 1,
) -> tuple[Path, Path]:
    """Execute one evaluation in a temporary run_dir, extract metric, and generate known_case.json and spec.json."""
    import shlex
    import subprocess

    if ":" in metric_arg:
        metric_name, direction = metric_arg.split(":", 1)
    else:
        metric_name, direction = metric_arg, "max"
    metric_name = metric_name.strip()
    direction = direction.strip().lower()
    if direction not in ("min", "max"):
        raise ValueError(f"Invalid metric direction '{direction}', must be 'min' or 'max'")

    eval_parts = [p.strip('\'"') for p in shlex.split(eval_cmd, posix=False) if p.strip()]
    if not eval_parts:
        raise ValueError("Evaluator command cannot be empty")

    first_token_lower = eval_parts[0].lower()
    uses_python = (
        first_token_lower in ("python", "python.exe", "{python}")
        or first_token_lower.endswith(("python.exe", "python"))
        or first_token_lower.startswith("python")
    )
    script_index = 1 if uses_python and len(eval_parts) > 1 else 0
    script_token = eval_parts[script_index]

    script_file = None
    for loc in [
        Path(script_token),
        project / script_token,
        program.parent / script_token if program else None,
    ]:
        if loc is not None and loc.is_file():
            script_file = loc.resolve()
            break
    if script_file is None:
        raise ValueError(f"Evaluator script not found: {script_token}")

    if not uses_python and script_file.suffix == ".py":
        uses_python = True
        eval_parts.insert(0, "{python}")
        script_index = 1

    spec_eval = list(eval_parts)
    if uses_python:
        spec_eval[0] = "{python}"
    spec_eval[script_index] = str(script_file)
    if "{candidate}" not in spec_eval and "{run_dir}" not in spec_eval:
        spec_eval.extend(["{candidate}", "{run_dir}"])

    with tempfile.TemporaryDirectory(prefix="autoresearch-auto-run-") as temp_run_str:
        temp_run = Path(temp_run_str)
        cmd = [
            sys.executable if arg == "{python}" else
            (str(project) if arg == "{candidate}" else
            (str(temp_run) if arg == "{run_dir}" else arg))
            for arg in spec_eval
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            cmd,
            cwd=str(project),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            env=env,
            **hidden_process_kwargs(),
        )
        if proc.returncode != 0:
            raise ValueError(
                f"Auto evaluation failed with exit code {proc.returncode}:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        result_path = temp_run / "result.json"
        if not result_path.is_file():
            raise ValueError(
                f"Evaluator did not write result.json in {temp_run}:\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid result.json written by evaluator: {error}") from error

        metrics = result_data.get("metrics")
        if not isinstance(metrics, dict) or metric_name not in metrics:
            raise ValueError(
                f"Metric '{metric_name}' not found in result.json metrics: {metrics}"
            )
        metric_val = float(metrics[metric_name])

    specs_dir = home / "specs" / project_id(project)
    specs_dir.mkdir(parents=True, exist_ok=True)
    known_case_path = specs_dir / "known_case.json"
    spec_path = specs_dir / "spec.json"

    # If evaluator script is inside project, isolate it to specs_dir so it won't conflict with candidate snapshot
    rel_exclude = []
    if script_file.is_relative_to(project):
        rel_exclude.append(script_file.relative_to(project).as_posix())
        isolated_script = specs_dir / script_file.name
        shutil.copy2(script_file, isolated_script)
        script_file = isolated_script
        spec_eval[script_index] = str(isolated_script)

    script_sha = _hash(script_file)
    known_case_data = {
        "schema": 1,
        "id": f"auto-{project_id(project)}-baseline",
        "reproducible": True,
        "evaluator_sha256": script_sha,
        "input_hashes": {},
        "expected_metrics": {metric_name: metric_val},
        "observed_metrics": {metric_name: metric_val},
        "tolerance": 1e-6,
    }
    known_case_path.write_text(json.dumps(known_case_data, indent=2, ensure_ascii=False), encoding="utf-8")
    known_case_sha = _hash(known_case_path)

    spec_data = {
        "project": str(project),
        "evaluator": spec_eval,
        "metrics": {
            "primary": metric_name,
            "direction": direction,
            "guards": [],
        },
        "repeats": repeats,
        "timeout_seconds": 60,
        "baseline_validation": {
            "mode": "fresh",
            "known_case": {
                "path": str(known_case_path),
                "sha256": known_case_sha,
            },
        },
    }
    if rel_exclude:
        spec_data["exclude_globs"] = rel_exclude
    spec_path.write_text(json.dumps(spec_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return spec_path, known_case_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare an AutoResearch lab from program.md")
    parser.add_argument("--program", type=Path, default=None)
    parser.add_argument("--brief-text", dest="brief_text", default=None)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--confirmed", action="store_true",
                        help="Confirm creation of a new SQLite lab from the supplied brief and spec")
    parser.add_argument("--auto", action="store_true",
                        help="Deprecated: prepare with --spec, then run a separate confirmed baseline")
    parser.add_argument("--eval", dest="eval_cmd", type=str, default=None, help="Evaluator command for auto-spec")
    parser.add_argument("--metric", dest="metric", type=str, default=None, help="Metric in name:direction format (e.g. accuracy:max)")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats for evaluation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--no-dashboard", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--open-browser", action="store_true",
                        help="Open a browser tab for the dashboard")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser tab")
    args = parser.parse_args(argv)
    if args.auto:
        return fail(
            "prepare",
            LabError(
                "--auto no longer runs an evaluator during preparation",
                code="E_AUTO_MIGRATION_REQUIRED",
                hint=(
                    "Prepare a non-evaluating lab with --program <program.md> --spec <spec.json>, "
                    "then start a separate baseline with python scripts/lab.py baseline --confirmed "
                    "--config <lab.json> after configuring an approved harness profile."
                ),
            ),
        )
    if args.resume and args.project is None and args.brief_text is None:
        parser.error("--resume requires --project or --brief-text")
    if not args.resume and args.spec is None and not args.auto:
        parser.error("preparation requires --spec (or --auto)")
    if args.auto and args.spec is None:
        if not args.eval_cmd or not args.metric:
            parser.error("--auto requires --eval <cmd> and --metric <name>:<dir> when --spec is omitted")
    try:
        project = args.project.resolve() if args.project is not None else None
        if args.auto and args.spec is None:
            if project is None:
                project = Path.cwd().resolve()
            if args.program is not None:
                program = args.program.resolve()
            elif args.brief_text is not None:
                program = _write_brief(project, args.brief_text)
            elif (project / "program.md").is_file():
                program = (project / "program.md").resolve()
            else:
                m_name = args.metric.split(":", 1)[0]
                program = _write_brief(project, f"Goal: optimize {m_name}\nMetric: {args.metric}\n")
            home = _lab_root(program, project)
            spec_path, _ = _auto_generate_spec(
                project=project,
                eval_cmd=args.eval_cmd,
                metric_arg=args.metric,
                program=program,
                home=home,
                repeats=args.repeats or 1,
            )
            args.spec = spec_path
        else:
            program = _resolve_program(args, args.spec.resolve() if args.spec is not None else None)
            if project is None and args.spec is not None:
                spec = json.loads(args.spec.resolve().read_text(encoding="utf-8"))
                project = _path(args.spec.resolve().parent, spec["project"]).resolve()
        if args.resume and project is None:
            raise ValueError("--resume requires --project or a --spec with project")
        output = (resume_lab(program, project) if args.resume
                  else prepare_lab(program, args.spec, confirmed=args.confirmed))
        if output is not None and not args.no_dashboard and not os.environ.get("AUTORESEARCH_NO_DASHBOARD"):
            from .dashboard_server import start_dashboard
            if project is not None:
                root = _lab_root(program, project.resolve())
                try:
                    open_browser = args.open_browser and not args.no_browser and os.environ.get("AUTORESEARCH_NO_BROWSER") != "1"
                    output["dashboard_server"] = start_dashboard(root, open_browser=open_browser)
                    if not output["dashboard_server"]["browser_opened"]:
                        output["dashboard_url"] = output["dashboard_server"]["url"]
                except RuntimeError as error:
                    output["dashboard_server"] = {"running": False, "error": str(error)}
    except (OSError, ValueError, LabError) as error:
        return fail("prepare", error)
    return emit("prepare", output if output is not None else {"found": False})


if __name__ == "__main__":
    raise SystemExit(main())
