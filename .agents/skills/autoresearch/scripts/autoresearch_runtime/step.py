"""Unified step command for AutoResearch.

Inspects current state of lab and recommends the next action, optionally executing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baseline import run_baseline
from .cockpit import build_cockpit, get_state_card
from .config import LabConfig
from .gitutil import _git, _git_bytes
from .errors import LabError
from .locking import lab_lock
from .state import load_state
from .trial import run_trial


REQUIRED_HYPOTHESIS_FIELDS = ("title", "observation", "expected_effect")


def load_and_validate_hypothesis(hypothesis_input: Path | str | dict[str, Any] | None) -> dict[str, Any]:
    if hypothesis_input is None:
        raise LabError("Hypothesis file required", code="E_CONTRACT_HYPOTHESIS")

    if isinstance(hypothesis_input, dict):
        data = dict(hypothesis_input)
    elif isinstance(hypothesis_input, (str, Path)):
        raw_str = str(hypothesis_input).strip()
        if raw_str.startswith("{"):
            try:
                data = json.loads(raw_str)
            except ValueError as error:
                raise LabError(f"Invalid hypothesis file: {error}", code="E_CONTRACT_HYPOTHESIS") from error
        else:
            p = Path(hypothesis_input)
            if not p.exists():
                raise LabError(f"Invalid hypothesis file: {p} does not exist", code="E_CONTRACT_HYPOTHESIS")
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as error:
                raise LabError(f"Invalid hypothesis file: {error}", code="E_CONTRACT_HYPOTHESIS") from error
    else:
        raise LabError("Invalid hypothesis file: unsupported input type", code="E_CONTRACT_HYPOTHESIS")

    if not isinstance(data, dict):
        raise LabError("Invalid hypothesis file: must be a JSON object", code="E_CONTRACT_HYPOTHESIS")

    for field in REQUIRED_HYPOTHESIS_FIELDS:
        val = data.get(field)
        if not isinstance(val, str) or not val.strip():
            raise LabError(f"Hypothesis requires non-empty {field}", code="E_CONTRACT_HYPOTHESIS")
        data[field] = val.strip()

    if not data.get("rationale") or not str(data["rationale"]).strip():
        data["rationale"] = f"{data['observation']} -> {data['expected_effect']}"
    else:
        data["rationale"] = str(data["rationale"]).strip()

    return data


def resolve_hypothesis_args(args: Any) -> Path | dict[str, Any] | None:
    raw_json = getattr(args, "hypothesis_json", None)
    if raw_json is not None:
        try:
            parsed = json.loads(str(raw_json))
            if not isinstance(parsed, dict):
                raise ValueError("Hypothesis JSON must be an object")
            return parsed
        except ValueError as error:
            raise LabError(f"Hypothesis JSON is invalid: {error}", code="E_BAD_HYPOTHESIS") from error

    title = getattr(args, "title", None)
    observation = getattr(args, "observation", None)
    expected_effect = getattr(args, "expected_effect", None)
    if title is not None or observation is not None or expected_effect is not None:
        payload: dict[str, Any] = {
            "title": (title or "").strip(),
            "observation": (observation or "").strip(),
            "expected_effect": (expected_effect or "").strip(),
        }
        if getattr(args, "rationale", None):
            payload["rationale"] = str(args.rationale).strip()
        if getattr(args, "falsification", None):
            payload["falsification"] = str(args.falsification).strip()
        if getattr(args, "evidence", None):
            payload["evidence"] = list(args.evidence) if isinstance(args.evidence, list) else str(args.evidence).strip()
        if getattr(args, "target_files", None):
            payload["target_files"] = list(args.target_files)
        return payload

    hyp = getattr(args, "hypothesis", None)
    if hyp is not None:
        raw_hyp = str(hyp).strip()
        if raw_hyp.startswith("{"):
            try:
                parsed = json.loads(raw_hyp)
                if not isinstance(parsed, dict):
                    raise ValueError("Hypothesis JSON must be an object")
                return parsed
            except ValueError as error:
                raise LabError(f"Hypothesis JSON is invalid: {error}", code="E_BAD_HYPOTHESIS") from error
        return Path(hyp)

    return None


def execute_step(
    config: Any,
    hypothesis_path: Path | str | dict[str, Any] | None = None,
    dry_run: bool = False,
    run: bool | None = None,
) -> dict[str, Any]:
    should_run = run if run is not None else (not dry_run)
    return run_step(config, run=should_run, hypothesis_path=hypothesis_path)


def run_step(
    config: LabConfig,
    *,
    run: bool = False,
    hypothesis_path: Path | str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect current lab state, determine recommended action, and optionally execute it.

    Returns:
        dict: {"stage": current_stage, "recommended_action": action_dict, "command": cmd_str, "status": ...}
    """
    state = load_state(config)
    guide = build_cockpit(config, state)
    card = guide["state"]
    current_stage = card["stage"]
    actions = guide["next_steps"]

    action_dict = actions[0] if actions else None
    cmd_str = action_dict.get("command", "") if action_dict else ""

    result: dict[str, Any] = {
        "stage": current_stage,
        "recommended_action": action_dict,
        "command": cmd_str,
        "argv": action_dict.get("argv", []) if action_dict else [],
        "status": "recommended",
        "current_epoch": guide["current_epoch"],
        "evidence": guide["evidence"],
        "signals": guide["signals"],
        "impact": guide["impact"],
        "goal_status": guide["goal_status"],
        "fingerprints": guide["fingerprints"],
        "provider_usage": guide["provider_usage"],
        "provider_retry": guide["provider_retry"],
        "dead_ends": guide.get("dead_ends", []),
    }

    if not run or not action_dict:
        return result

    action_type = action_dict.get("action")

    if action_type == "WAIT":
        from .worker import read_worker

        worker = read_worker(config.state_dir)
        result["action_taken"] = "wait"
        if isinstance(worker, dict) and worker.get("status") == "finished" and isinstance(worker.get("manifest"), dict):
            result["result"] = worker["manifest"]
            result["status"] = worker["manifest"].get("status", "finished")
        elif isinstance(worker, dict) and worker.get("status") == "failed":
            result["worker"] = {key: worker.get(key) for key in ("status", "error", "trial_id")}
            result["status"] = "worker_failed"
        else:
            result["worker"] = {
                key: worker.get(key) for key in ("status", "pid", "heartbeat", "trial_id")
            } if isinstance(worker, dict) else {"status": "unknown"}
            result["status"] = "waiting_for_trial"
        return result

    if action_type == "RUN_BASELINE":
        with lab_lock(config.state_dir):
            baseline_result = run_baseline(config)
        result["status"] = baseline_result.get("status", "accepted")
        result["action_taken"] = "baseline"
        result["result"] = baseline_result
        new_state = load_state(config)
        result["new_stage"] = get_state_card(config, new_state)["stage"]

    elif action_type in ("RUN_TRIAL", "ITERATE", "REFINE_OR_RETRY") or (
        action_type == "INSPECT_RUN" and hypothesis_path is not None
    ):
        if hypothesis_path is None:
            result["status"] = "hypothesis_required"
            result["message"] = (
                "Trial execution needs an explicit JSON hypothesis file with title, observation, "
                "and expected_effect. No rationale is generated by step."
            )
            return result
        hypothesis = load_and_validate_hypothesis(hypothesis_path)
        candidate = config.state_dir / "candidate"
        _git(candidate, "add", "-N", "--", ".")
        patch_bytes = _git_bytes(candidate, "diff", "--binary", "HEAD")
        if patch_bytes:
            with lab_lock(config.state_dir):
                trial_result = run_trial(config, hypothesis)
            result["id"] = trial_result.get("id")
            result["run_id"] = trial_result.get("id")
            result["status"] = trial_result.get("status", "accepted")
            result["action_taken"] = "trial"
            result["result"] = trial_result
            new_state = load_state(config)
            result["new_stage"] = get_state_card(config, new_state)["stage"]
        else:
            result["status"] = "waiting_for_changes"
            result["message"] = (
                "No candidate changes detected in candidate/ workspace. "
                "Apply modifications before running trial."
            )

    elif action_type == "CONFIG_SETUP":
        result["status"] = "setup_pending"
        result["message"] = "Lab configuration setup is pending."

    elif action_type == "INSPECT_RUN":
        result["status"] = "action_not_runnable"
        result["message"] = "Inspect the selected run evidence before retrying it."

    else:
        result["status"] = "action_not_runnable"
        result["message"] = f"Action '{action_type}' cannot be executed automatically."

    return result
