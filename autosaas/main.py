from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping

from autosaas.branch_keeper import decide_keep_or_revert
from autosaas.config import load_target_config
from autosaas.implementation_executor import ImplementationExecutor
from autosaas.privacy_guard import redact_text
from autosaas.repo_context_loader import load_repo_context
from autosaas.reporter import format_slice_run
from autosaas.state import SliceRun
from autosaas.task_slicer import choose_next_slice
from autosaas.validation_pipeline import run_required_gates


@dataclass(frozen=True)
class RunResult:
    slice_name: str
    status: str
    report: str


def _load_target_commands(repo_path: Path) -> Mapping[str, str]:
    config_path = repo_path / "project.autosaas.yaml"
    if not config_path.is_file():
        return {}
    try:
        config = load_target_config(config_path)
    except Exception:
        return {}

    return {
        "lint": config.commands.lint,
        "typecheck": config.commands.typecheck,
        "test": config.commands.test,
        "dev": config.commands.dev,
        "smoke": config.commands.smoke,
    }


def run_once(target_repo: Path | str, request: str, dry_run: bool = False) -> RunResult:
    repo_path = Path(target_repo)
    repo_context = load_repo_context(repo_path)
    context_data = asdict(repo_context)

    slice_def = choose_next_slice(request, repo_context=context_data)
    executor = ImplementationExecutor(slice_def)
    executor.run([])

    run = SliceRun(slice_name=slice_def.name, branch="autosaas/dry-run", base_commit="dry-run")
    run.touched_files.extend(sorted(executor.touched_files))

    if dry_run:
        run.status = "keep"
    else:
        command_map = _load_target_commands(repo_path)
        if not command_map:
            run.status = "blocked"
        else:
            run = run_required_gates(run, list(command_map.keys()), command_map, repo_path)
            gate_outcomes = {gate.name: gate.passed for gate in run.gate_results}
            run.status = decide_keep_or_revert(gate_outcomes)

    report = format_slice_run(run)
    redacted = redact_text(report, literals=context_data.get("sensitive_paths"))
    if "sk_live" in redacted:
        redacted = redact_text(redacted, patterns=("sk_live",))

    return RunResult(slice_name=run.slice_name, status=run.status, report=redacted)
