"""`step` subcommand for unified agent step decision and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..gitutil import _git, _git_bytes
from ..state import load_state
from ..cockpit import get_state_card
from ..step import load_and_validate_hypothesis, resolve_hypothesis_args, run_step
from .common import explicit_confirmation, resolve_config, run_durable_operation


def execute_step(
    config: Any,
    hypothesis_path: Path | str | dict[str, Any] | None = None,
    dry_run: bool = False,
    run: bool | None = None,
) -> dict[str, Any]:
    should_run = run if run is not None else (not dry_run)
    return run_step(config, run=should_run, hypothesis_path=hypothesis_path)


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("step")
    parser.add_argument("--run", action="store_true", help="Automatically execute the recommended action")
    parser.add_argument("--hypothesis", type=Path, default=None, help="Explicit JSON hypothesis required when --run executes a trial")
    parser.add_argument("--hypothesis-json", type=str, default=None, help="Inline JSON string for the hypothesis (avoids writing a temporary file)")
    parser.add_argument("--title", type=str, default=None, help="Inline hypothesis title")
    parser.add_argument("--observation", type=str, default=None, help="Inline hypothesis observation")
    parser.add_argument("--expected-effect", type=str, default=None, help="Inline hypothesis expected effect")
    parser.add_argument("--rationale", type=str, default=None, help="Optional inline hypothesis rationale")
    parser.add_argument("--falsification", type=str, default=None, help="Optional inline hypothesis falsification criterion")
    parser.add_argument("--target-files", nargs="*", default=None, help="Optional target files for inline hypothesis")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    config = resolve_config(args, labs_root, pending_ok=True)
    hypothesis_input = resolve_hypothesis_args(args)

    if bool(getattr(args, "run", False)):
        result = run_step(config, run=False, hypothesis_path=hypothesis_input)
        action = result.get("recommended_action") or {}
        action_type = action.get("action")
        if action_type == "RUN_BASELINE":
            baseline = run_durable_operation(config, "baseline", explicit_confirmation(args))
            result.update(status=baseline.get("status", "accepted"), action_taken="baseline", result=baseline)
            result["new_stage"] = get_state_card(config, load_state(config))["stage"]
            return result
        if action_type in {"RUN_TRIAL", "ITERATE", "REFINE_OR_RETRY"} or (
            action_type == "INSPECT_RUN" and hypothesis_input is not None
        ):
            candidate = config.state_dir / "candidate"
            _git(candidate, "add", "-N", "--", ".")
            if hypothesis_input is None or not _git_bytes(candidate, "diff", "--binary", "HEAD"):
                return run_step(config, run=True, hypothesis_path=hypothesis_input)
            hypothesis = load_and_validate_hypothesis(hypothesis_input)
            allowed_durable = {
                "title", "observation", "expected_effect", "evidence",
                "falsification", "rationale", "target_files", "delegation",
            }
            durable_hypothesis = {k: v for k, v in hypothesis.items() if k in allowed_durable}
            trial = run_durable_operation(
                config, "trial", {"hypothesis": durable_hypothesis, **explicit_confirmation(args)}
            )
            result.update(status=trial.get("status", "accepted"), action_taken="trial", result=trial)
            result["new_stage"] = get_state_card(config, load_state(config))["stage"]
            return result
        # WAIT and informational actions only read the operation/state projection.
        return run_step(config, run=True, hypothesis_path=hypothesis_input)
    return execute_step(
        config,
        hypothesis_path=hypothesis_input,
        dry_run=not bool(getattr(args, "run", False)),
    )
