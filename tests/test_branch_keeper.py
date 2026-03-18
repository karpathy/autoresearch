import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autosaas.branch_keeper import decide_keep_or_revert


def test_decide_keep_or_revert_requires_all_required_gates():
    status = decide_keep_or_revert(required_gate_results={"lint": True, "typecheck": False})
    assert status == "revert"


def test_decide_keep_or_revert_returns_keep_when_all_gates_pass():
    status = decide_keep_or_revert(required_gate_results={"lint": True, "test": True})
    assert status == "keep"


def test_decide_keep_or_revert_returns_revert_when_no_gates():
    status = decide_keep_or_revert(required_gate_results={})
    assert status == "revert"
