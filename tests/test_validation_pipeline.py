import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosaas.validation_pipeline import run_required_gates
from autosaas.state import SliceRun

def test_run_required_gates_returns_fail_on_first_required_error(tmp_path):
    run = SliceRun(slice_name="demo", branch="autosaas/demo", base_commit="abc1234")
    lint_cmd = f"{sys.executable} -c 'import sys; sys.exit(1)'"
    typecheck_cmd = f"{sys.executable} -c 'print(1)'"

    result = run_required_gates(
        run,
        required_gates=["lint", "typecheck"],
        command_map={"lint": lint_cmd, "typecheck": typecheck_cmd},
        cwd=tmp_path,
    )

    assert result.status == "revert"
    assert any(g.name == "lint" and not g.passed for g in result.gate_results)
    assert not any(g.name == "typecheck" for g in result.gate_results)


def test_run_required_gates_returns_keep_when_all_gates_pass(tmp_path):
    run = SliceRun(slice_name="demo", branch="autosaas/demo", base_commit="abc1234")
    lint_cmd = f"{sys.executable} -c 'print(0)'"
    typecheck_cmd = f"{sys.executable} -c 'print(1)'"

    result = run_required_gates(
        run,
        required_gates=["lint", "typecheck"],
        command_map={"lint": lint_cmd, "typecheck": typecheck_cmd},
        cwd=tmp_path,
    )

    assert result.status == "keep"
    assert len(result.gate_results) == 2
