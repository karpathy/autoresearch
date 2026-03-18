import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autosaas.main as main_module
from autosaas.main import run_once


def test_run_once_emits_redacted_report_for_single_slice(tmp_path):
    result = run_once(target_repo=tmp_path, request="Add billing status badge", dry_run=True)
    assert result.slice_name
    assert result.status in {"keep", "revert", "blocked", "crash"}
    assert "sk_live" not in result.report


def test_run_once_filters_unsupported_gates(tmp_path, monkeypatch):
    config_path = tmp_path / "project.autosaas.yaml"
    config_path.write_text(
        """
commands:
  lint: echo lint
  typecheck: echo typecheck
  test: echo test
  dev: echo dev
  smoke: echo smoke
"""
    )

    captured = {"called": False, "gates": ()}

    def fake_run_required_gates(run, required_gates, command_map, cwd):
        captured["called"] = True
        captured["gates"] = tuple(required_gates)
        run.status = "keep"
        run.gate_results = []
        return run

    monkeypatch.setattr(main_module, "run_required_gates", fake_run_required_gates)

    run_once(target_repo=tmp_path, request="Add billing status badge", dry_run=False)

    assert captured["called"]
    assert captured["gates"] == ("lint", "typecheck", "test", "smoke")


def test_run_once_redacts_full_sk_live_token(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "format_slice_run", lambda run: "status sk_live_ABC123 secret")

    result = run_once(target_repo=tmp_path, request="Add billing status badge", dry_run=True)

    assert "sk_live" not in result.report
    assert "ABC123" not in result.report


def test_run_once_handles_exceptions_and_returns_crash(tmp_path, monkeypatch):
    def raise_error(repo_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "load_repo_context", raise_error)
    result = run_once(target_repo=tmp_path, request="recover", dry_run=True)

    assert result.status == "crash"
    assert "boom" in result.report
