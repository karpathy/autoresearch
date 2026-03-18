import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosaas.validation_pipeline import run_required_gates
from autosaas.state import SliceRun


def test_run_required_gates_returns_fail_on_first_required_error(tmp_path):
    run = SliceRun(slice_name="demo", branch="autosaas/demo", base_commit="abc1234")
    result = run_required_gates(
        run,
        required_gates=["lint", "typecheck"],
        command_map={"lint": "python -c 'raise SystemExit(1)'", "typecheck": "python -c 'print(1)'"},
        cwd=tmp_path,
    )
    assert result.status == "revert"
    assert any(g.name == "lint" and not g.passed for g in result.gate_results)
