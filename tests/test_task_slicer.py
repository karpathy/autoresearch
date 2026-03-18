from autosaas.task_slicer import choose_next_slice


def test_choose_next_slice_reduces_large_request_to_one_slice():
    slice_def = choose_next_slice("Build a full billing system", repo_context=None)

    assert slice_def.name
    assert "billing" in slice_def.name.lower()
    assert len(slice_def.success_criteria) >= 1
    assert len(slice_def.allowed_file_patterns) >= 1
