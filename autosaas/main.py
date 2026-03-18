from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Sequence

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


SUPPORTED_VALIDATION_GATES = ("lint", "typecheck", "test", "app_boot", "smoke")
SK_LIVE_TOKEN_PATTERN = r"sk_live_[A-Za-z0-9]+"


def _load_target_commands(repo_path: Path) -> Mapping[str, str]:
    config_path = repo_path / "project.autosaas.yaml"
    if not config_path.is_file():
        return {}
    config = load_target_config(config_path)

    commands: dict[str, str] = {
        "lint": config.commands.lint,
        "typecheck": config.commands.typecheck,
        "test": config.commands.test,
        "dev": config.commands.dev,
        "smoke": config.commands.smoke,
    }
    if config.app_boot_url:
        commands["app_boot"] = config.app_boot_url
        commands["app_boot_url"] = config.app_boot_url
    return commands


def _sanitize_report(report: str, sensitive_literals: Sequence[str] | None) -> str:
    patterns = (SK_LIVE_TOKEN_PATTERN, "sk_live")
    return redact_text(report, literals=sensitive_literals, patterns=patterns)


def run_once(target_repo: Path | str, request: str, dry_run: bool = False) -> RunResult:
    repo_path = Path(target_repo)
    sensitive_literals: Sequence[str] | None = ()
    try:
        repo_context = load_repo_context(repo_path)
        context_data = asdict(repo_context)
        sensitive_literals = context_data.get("sensitive_paths")

        slice_def = choose_next_slice(request, repo_context=context_data)
        executor = ImplementationExecutor(slice_def)
        executor.run([])

        run = SliceRun(slice_name=slice_def.name, branch="autosaas/dry-run", base_commit="dry-run")
        run.touched_files.extend(sorted(executor.touched_files))

        if dry_run:
            run.status = "keep"
        else:
            try:
                command_map = _load_target_commands(repo_path)
            except (ValueError, KeyError) as exc:
                run.status = "blocked"
                blocked_report = _sanitize_report(f"blocked: {exc}", sensitive_literals)
                return RunResult(slice_name=run.slice_name, status=run.status, report=blocked_report)

            if not command_map:
                run.status = "blocked"
            else:
                supported_gates = [gate for gate in SUPPORTED_VALIDATION_GATES if gate in command_map]
                if not supported_gates:
                    run.status = "blocked"
                else:
                    run = run_required_gates(run, supported_gates, command_map, repo_path)
                    gate_outcomes = {gate.name: gate.passed for gate in run.gate_results}
                    run.status = decide_keep_or_revert(gate_outcomes)

        report = format_slice_run(run)
        redacted = _sanitize_report(report, sensitive_literals)
        return RunResult(slice_name=run.slice_name, status=run.status, report=redacted)
    except Exception as exc:
        fallback_report = f"crash: {exc}"
        redacted = _sanitize_report(fallback_report, sensitive_literals)
        fallback_slice = request.strip() or "slice"
        return RunResult(slice_name=fallback_slice, status="crash", report=redacted)
