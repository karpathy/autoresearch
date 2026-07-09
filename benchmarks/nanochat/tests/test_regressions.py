"""Focused regressions owned by the optional NanoChat fixture."""

import ast
import hashlib
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import asep_loopgraph_benchmark as asep
import run_agent
from agents.editor import apply_hunks
from evals.nanochat_bpb import metrics
from researchfs.db import initialize_db
from researchfs.retrieve import get_experiment_brief
from seed import EXPERIMENT_BRIEF_NAME, seed_initial_data


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def _memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    return conn


def _attr_path(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_is_fill_one(call):
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "fill_"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 1.0
    )


def test_train_init_weights_explicitly_initializes_all_rmsnorm_scales_without_importing_training():
    tree = ast.parse((BENCHMARK_ROOT / "train.py").read_text(encoding="utf-8"))
    gpt_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPT"
    )
    init_weights = next(
        node
        for node in gpt_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_weights"
    )
    filled_paths = {
        _attr_path(call.func.value)
        for call in ast.walk(init_weights)
        if _call_is_fill_one(call)
    }
    assert {
        "block.attn.q_norm.scale",
        "block.attn.k_norm.scale",
        "block.norm1.scale",
        "block.norm2.scale",
        "self.ln_f.scale",
    } <= filled_paths


def _insert_run(conn, mode):
    run_id = conn.execute(
        "INSERT INTO agent_runs (mode, model) VALUES (?, 'fixture-test')",
        (mode,),
    ).lastrowid
    conn.execute(
        """
        INSERT INTO experiments (agent_run_id, commit_hash, val_bpb, memory_gb, status, description)
        VALUES (?, 'baseline', 2.500000, 1.0, 'keep', 'baseline')
        """,
        (run_id,),
    )
    conn.commit()
    return run_id


def test_missing_token_usage_renders_unknown_not_zero_savings():
    conn = _memory_db()
    baseline_id = _insert_run(conn, "baseline")
    researchfs_id = _insert_run(conn, "loopgraph")
    comparison = metrics.compare_runs(baseline_id, researchfs_id, conn=conn)
    assert comparison["baseline"]["has_token_usage"] is False
    assert comparison["researchfs"]["has_token_usage"] is False
    assert comparison["token_reduction_ratio"] is None
    rendered = metrics.format_comparison(comparison)
    assert "| Total tokens | N/A | N/A |" in rendered
    assert "fewer tokens" not in rendered
    assert "more tokens" not in rendered


def test_codex_dirty_guard_allows_preexisting_non_train_diff_but_rejects_new_outside_edits(
    tmp_path, monkeypatch
):
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Regression Tester",
            "GIT_AUTHOR_EMAIL": "regression@example.invalid",
            "GIT_COMMITTER_NAME": "Regression Tester",
            "GIT_COMMITTER_EMAIL": "regression@example.invalid",
        }
    )
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, env=env)
    (tmp_path / "train.py").write_text("print('base')\n")
    (tmp_path / "notes.txt").write_text("base notes\n")
    (tmp_path / ".gitignore").write_text("results.tsv\nresults/\n")
    subprocess.run(
        ["git", "add", "train.py", "notes.txt", ".gitignore"], cwd=tmp_path, check=True, env=env
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=env,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("pre-existing user note\n")
    preexisting_untracked = tmp_path / "preexisting-untracked.txt"
    preexisting_untracked.write_text("pre-existing untracked user note\n")
    ignored_results = tmp_path / "results.tsv"
    ignored_results_content = "step\tval_bpb\n1\t2.5\n"
    ignored_results.write_text(ignored_results_content)
    ignored_dir = tmp_path / "results"
    ignored_dir.mkdir()
    ignored_result = ignored_dir / "seed.tsv"
    ignored_result_content = "seed\tval_bpb\n1\t2.6\n"
    ignored_result.write_text(ignored_result_content)

    before = run_agent._snapshot_tracked_non_train_diff(env)
    path, size, digest = before["untracked"].splitlines()[0].split("\t")
    assert path == "preexisting-untracked.txt"
    assert int(size) == len("pre-existing untracked user note\n")
    assert len(digest) == 64
    ignored_entries = {
        path: (size, digest)
        for path, size, digest in (line.split("\t") for line in before["ignored"].splitlines())
    }
    assert ignored_entries["results.tsv"] == (
        str(len(ignored_results_content)),
        hashlib.sha256(ignored_results_content.encode()).hexdigest(),
    )

    (tmp_path / "train.py").write_text("print('codex experiment')\n")
    assert run_agent._ensure_codex_only_changed_train_py(before, env) == before
    ignored_result.write_text("seed\tval_bpb\n1\t9.9\n")
    with pytest.raises(RuntimeError, match=r"outside train\.py .*ignored"):
        run_agent._ensure_codex_only_changed_train_py(before, env)


def test_asep_branch_selection_deletes_single_candidate_but_refuses_multiple_new_candidates():
    before = {"autoresearch/exp_existing"}
    after_one = before | {"autoresearch/exp_new"}
    assert (
        asep._temporary_branch_from_this_run(
            "autoresearch/exp_new", "main", before, branches_after=after_one
        )
        == "autoresearch/exp_new"
    )
    after_two = before | {"autoresearch/exp_new", "autoresearch/exp_other"}
    assert (
        asep._temporary_branch_from_this_run(
            "autoresearch/exp_new", "main", before, branches_after=after_two
        )
        is None
    )


def test_asep_cleanup_checks_out_original_branch_before_deleting_single_generated_branch(monkeypatch):
    calls = []
    monkeypatch.setattr(asep, "current_git_branch", lambda: "autoresearch/exp_new")

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd == ["git", "branch", "--list", "autoresearch/exp_*"]:
            return SimpleNamespace(
                returncode=0,
                stdout="  autoresearch/exp_existing\n* autoresearch/exp_new\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(asep.subprocess, "run", fake_run)
    asep.cleanup_git_state("main", {"autoresearch/exp_existing"})
    checkout_call = ["git", "checkout", "main"]
    delete_call = ["git", "branch", "-D", "autoresearch/exp_new"]
    assert checkout_call in calls
    assert delete_call in calls
    assert calls.index(checkout_call) < calls.index(delete_call)


def test_empty_search_replace_replacement_deletes_matched_text():
    content = "alpha\nremove me\nomega"
    response = "<<<<<<< SEARCH\nremove me\n=======\n>>>>>>>"
    assert apply_hunks(content, response) == "alpha\n\nomega"


def test_seed_bundle_installs_fixture_methods_and_experiment_brief():
    conn = _memory_db()
    seed_initial_data(conn)
    assert conn.execute("SELECT COUNT(*) FROM methods").fetchone()[0] >= 10
    assert "validation bits per byte" in get_experiment_brief(EXPERIMENT_BRIEF_NAME, conn)
