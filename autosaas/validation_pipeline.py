from __future__ import annotations

from pathlib import Path

from autosaas.state import GateResult, SliceRun
from autosaas.utils import run_command, wait_for_app_ready

_GATE_ORDER = ["lint", "typecheck", "test", "app_boot", "smoke"]


def run_required_gates(run: SliceRun, required_gates: list[str], command_map: dict[str, str], cwd: Path | str):
    """Run every gate we care about and stop at the first failure."""
    gate_results: list[GateResult] = []
    seen: set[str] = set()

    for gate in _GATE_ORDER:
        if gate not in required_gates:
            continue
        seen.add(gate)
        passed, summary, duration = _execute_gate(gate, command_map, cwd)
        gate_results.append(GateResult(name=gate, passed=passed, summary=summary, duration_s=duration))
        if not passed:
            run.status = "revert"
            run.gate_results = gate_results
            return run

    for gate in required_gates:
        if gate in seen:
            continue
        gate_results.append(GateResult(name=gate, passed=False, summary="unsupported gate", duration_s=0.0))
        run.status = "revert"
        run.gate_results = gate_results
        return run

    run.status = "pass"
    run.gate_results = gate_results
    return run


def _execute_gate(gate: str, command_map: dict[str, str], cwd: Path | str) -> tuple[bool, str, float]:
    if gate == "app_boot":
        url = command_map.get("app_boot_url") or command_map.get("app_boot")
        return wait_for_app_ready(url)

    command = command_map.get(gate)
    if not command:
        return False, f"no command configured for {gate}", 0.0

    return run_command(command, cwd)
