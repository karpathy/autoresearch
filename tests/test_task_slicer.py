import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autosaas.implementation_executor import ImplementationExecutor
from autosaas.task_slicer import TaskSlice, choose_next_slice


def test_choose_next_slice_reduces_large_request_to_one_slice():
    slice_def = choose_next_slice("Build a full billing system", repo_context=None)

    assert slice_def.name
    assert "billing" in slice_def.name.lower()
    assert len(slice_def.success_criteria) >= 1
    assert len(slice_def.allowed_file_patterns) >= 1


def test_implementation_executor_respects_slice_boundaries():
    slice_def = TaskSlice(
        name="billing slice: Build a billing system",
        summary="Focus on the billing slice.",
        success_criteria=("criteria",),
        allowed_file_patterns=("autosaas/**",),
    )
    executor = ImplementationExecutor(slice_def)

    executor.run(["autosaas/task_slicer.py"])
    assert "autosaas/task_slicer.py" in executor.touched_files

    with pytest.raises(ValueError) as excinfo:
        executor.run(["README.md"])
    assert "outside allowed patterns" in str(excinfo.value)
