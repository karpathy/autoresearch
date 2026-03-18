from autosaas.branch_keeper import decide_keep_or_revert


def test_decide_keep_or_revert_requires_all_required_gates():
    status = decide_keep_or_revert(required_gate_results={"lint": True, "typecheck": False})
    assert status == "revert"
