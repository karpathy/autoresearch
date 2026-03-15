"""Shared fixtures for the autoanything test suite."""

import os
import sqlite3
import tempfile
import textwrap

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """A temporary directory that acts as a problem root."""
    return tmp_path


@pytest.fixture
def minimal_problem_yaml():
    """The smallest valid problem.yaml — only required fields."""
    return textwrap.dedent("""\
        name: my-problem
        description: Minimize the cost function.
        score:
          direction: minimize
    """)


@pytest.fixture
def full_problem_yaml():
    """A complete problem.yaml with every field populated."""
    return textwrap.dedent("""\
        name: test-problem
        description: >
          A fully specified test problem for validation.

        score:
          name: cost
          direction: minimize
          description: "Total cost"
          timeout: 300
          bounded: true

        git:
          base_branch: main
          proposal_pattern: "proposals/*"
    """)


@pytest.fixture
def problem_dir(tmp_path, full_problem_yaml):
    """A fully populated problem directory matching the planned structure."""
    # problem.yaml
    (tmp_path / "problem.yaml").write_text(full_problem_yaml)

    # state/
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "solution.py").write_text("x = [0.0] * 10\n")
    (state_dir / "config.py").write_text("learning_rate = 0.01\n")

    # context/
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "background.py").write_text("# background info\n")

    # scoring/
    scoring_dir = tmp_path / "scoring"
    scoring_dir.mkdir()
    score_py = scoring_dir / "score.py"
    score_py.write_text(textwrap.dedent("""\
        def score():
            return {"cost": 42.5, "iterations": 100}
    """))

    # .gitignore
    (tmp_path / ".gitignore").write_text("scoring/\n.autoanything/\n")

    # .autoanything/
    (tmp_path / ".autoanything").mkdir()

    return tmp_path


@pytest.fixture
def history_db(tmp_path):
    """An initialized history database (in-memory style, but on disk for subprocess compat)."""
    from autoanything.history import init_db

    db_path = tmp_path / ".autoanything" / "history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(str(db_path))
    yield conn, str(db_path)
    conn.close()
