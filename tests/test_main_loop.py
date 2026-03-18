import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autosaas.main import run_once


def test_run_once_emits_redacted_report_for_single_slice(tmp_path):
    result = run_once(target_repo=tmp_path, request="Add billing status badge", dry_run=True)
    assert result.slice_name
    assert result.status in {"keep", "revert", "blocked", "crash"}
    assert "sk_live" not in result.report
