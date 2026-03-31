"""
FluxScore autoresearch — pytest test suite.
Run from repo root: pytest fluxscore-research/tests/

5 required tests (per CEO review):
1. prepare.py output: files exist, 18 feature columns, row count > 0
2. experiment.py happy path: prints auc: X.XXXXXX, exits 0
3. experiment.py missing data: prints ERROR, exits 1
4. experiment.py crash model: prints auc: 0.000000, exits 1
5. Manual integration: covered by happy-path + discard cycle
"""

import os
import subprocess
import sys
import shutil
import textwrap
from pathlib import Path

import pytest

RESEARCH_DIR = Path(__file__).parent.parent
PREPARE = RESEARCH_DIR / "prepare.py"
EXPERIMENT = RESEARCH_DIR / "experiment.py"


# ---------------------------------------------------------------------------
# Test 1: prepare.py generates valid output
# ---------------------------------------------------------------------------
class TestPrepare:
    def test_output_files_exist(self, tmp_path, monkeypatch):
        """prepare.py must create train.parquet and holdout.parquet."""
        import importlib.util, sys

        # Run prepare.py in tmp_path
        result = subprocess.run(
            [sys.executable, str(PREPARE)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        # prepare.py writes to the directory of the script, not cwd
        # Re-run with a copy in tmp_path
        shutil.copy(PREPARE, tmp_path / "prepare.py")
        result = subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, f"prepare.py failed:\n{result.stderr}"
        assert (tmp_path / "train.parquet").exists(), "train.parquet not created"
        assert (tmp_path / "holdout.parquet").exists(), "holdout.parquet not created"

    def test_feature_count(self, tmp_path):
        """Output must have exactly 18 feature columns (plus 'default' label)."""
        import pandas as pd

        shutil.copy(PREPARE, tmp_path / "prepare.py")
        subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            cwd=str(tmp_path),
        )
        train = pd.read_parquet(tmp_path / "train.parquet")
        feature_cols = [c for c in train.columns if c != "default"]
        assert len(feature_cols) == 18, (
            f"Expected 18 features, got {len(feature_cols)}: {feature_cols}"
        )

    def test_row_count(self, tmp_path):
        """Train must have > 0 rows, holdout must have > 0 rows."""
        import pandas as pd

        shutil.copy(PREPARE, tmp_path / "prepare.py")
        subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            cwd=str(tmp_path),
        )
        train = pd.read_parquet(tmp_path / "train.parquet")
        holdout = pd.read_parquet(tmp_path / "holdout.parquet")
        assert len(train) > 0, "train.parquet is empty"
        assert len(holdout) > 0, "holdout.parquet is empty"

    def test_default_column_exists(self, tmp_path):
        """'default' binary label must be present."""
        import pandas as pd

        shutil.copy(PREPARE, tmp_path / "prepare.py")
        subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            cwd=str(tmp_path),
        )
        train = pd.read_parquet(tmp_path / "train.parquet")
        assert "default" in train.columns
        assert set(train["default"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test 2: experiment.py happy path
# ---------------------------------------------------------------------------
class TestExperimentHappyPath:
    @pytest.fixture(autouse=True)
    def setup_data(self, tmp_path):
        """Generate data once, copy experiment.py into tmp_path."""
        shutil.copy(PREPARE, tmp_path / "prepare.py")
        shutil.copy(EXPERIMENT, tmp_path / "experiment.py")
        subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            cwd=str(tmp_path),
        )
        self.tmp_path = tmp_path

    def test_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(self.tmp_path),
        )
        assert result.returncode == 0, (
            f"experiment.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_auc_line_present(self):
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(self.tmp_path),
        )
        auc_lines = [l for l in result.stdout.splitlines() if l.startswith("auc:")]
        assert len(auc_lines) == 1, f"Expected 1 'auc:' line, got {auc_lines}\nfull output:\n{result.stdout}"

    def test_auc_value_is_valid(self):
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(self.tmp_path),
        )
        auc_line = next(l for l in result.stdout.splitlines() if l.startswith("auc:"))
        auc_val = float(auc_line.split(":")[1].strip())
        assert 0.5 < auc_val < 1.0, f"AUC {auc_val} outside expected range (0.5, 1.0)"

    def test_auc_format_greppable(self):
        """Output must be grep-parseable: 'auc: X.XXXXXX'"""
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(self.tmp_path),
        )
        import re
        auc_lines = [l for l in result.stdout.splitlines() if l.startswith("auc:")]
        assert auc_lines, "No 'auc:' line found"
        assert re.match(r"^auc:\s+\d+\.\d{6}$", auc_lines[0]), (
            f"auc line format wrong: '{auc_lines[0]}'"
        )


# ---------------------------------------------------------------------------
# Test 3: experiment.py missing data → exits 1 with clear error
# ---------------------------------------------------------------------------
class TestExperimentMissingData:
    def test_exits_one_when_no_parquet(self, tmp_path):
        shutil.copy(EXPERIMENT, tmp_path / "experiment.py")
        # No prepare.py run — no parquet files
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1, (
            f"Expected exit 1 when data missing, got {result.returncode}"
        )

    def test_error_message_mentions_prepare(self, tmp_path):
        shutil.copy(EXPERIMENT, tmp_path / "experiment.py")
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        combined = result.stdout + result.stderr
        assert "prepare.py" in combined, (
            f"Error message should mention 'prepare.py'. Got:\n{combined}"
        )


# ---------------------------------------------------------------------------
# Test 4: experiment.py crash model → auc: 0.000000, exits 1
# ---------------------------------------------------------------------------
class TestExperimentCrashModel:
    def test_crash_outputs_zero_auc(self, tmp_path):
        """When agent writes broken model code, output must be auc: 0.000000 and exit 1."""
        shutil.copy(PREPARE, tmp_path / "prepare.py")
        subprocess.run(
            [sys.executable, "prepare.py"],
            capture_output=True,
            cwd=str(tmp_path),
        )

        # Write a version of experiment.py with a crash in the agent zone
        exp_src = EXPERIMENT.read_text()
        # Inject a crash after the AGENT ZONE marker
        crash_injection = textwrap.dedent("""\
            # Injected crash for testing
            raise ValueError("Intentional crash from test_fluxscore_research")
        """)
        marker = "# AGENT ZONE — modify freely below this line"
        assert marker in exp_src, "AGENT ZONE marker not found in experiment.py"
        # Insert after everything that sets up the try block in agent zone
        # Replace the try block contents with a crash
        bad_exp = exp_src.replace(
            "    from sklearn.linear_model import LogisticRegression",
            "    raise ValueError('Intentional crash from test_fluxscore_research')\n    from sklearn.linear_model import LogisticRegression",
        )
        (tmp_path / "experiment.py").write_text(bad_exp)

        result = subprocess.run(
            [sys.executable, "experiment.py"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 1, (
            f"Expected exit 1 on crash, got {result.returncode}\n{result.stdout}"
        )
        auc_lines = [l for l in result.stdout.splitlines() if l.startswith("auc:")]
        assert auc_lines, f"No 'auc:' line on crash. stdout:\n{result.stdout}"
        assert auc_lines[0] == "auc: 0.000000", (
            f"Crash should output 'auc: 0.000000', got '{auc_lines[0]}'"
        )
