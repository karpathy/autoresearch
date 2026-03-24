"""Tests for experiment infrastructure (Phase 2 INFRA requirements).

Validates metrics.json contract, results.tsv format, crash detection,
VRAM tracking, epoch budget enforcement, and gitignore entries.
All tests are pure schema validation, AST checks, and fixture assertions
-- no GPU, no training, no network required.
"""
import ast
import json
import os


class TestMetricsJsonSchema:
    """INFRA-02, INFRA-05, INFRA-06: Validate metrics.json output contract."""

    def test_success_schema_has_all_required_fields(self, success_metrics):
        """Every successful run must include all sub-metrics."""
        required = {
            "status", "combined_metric", "recall_at_1", "recall_at_5",
            "mean_cosine", "distill_loss", "arc_loss", "vat_loss", "sep_loss",
            "peak_vram_mb", "epochs", "elapsed_seconds",
        }
        assert required.issubset(success_metrics.keys()), (
            f"Missing fields: {required - success_metrics.keys()}"
        )

    def test_success_status_is_success(self, success_metrics):
        assert success_metrics["status"] == "success"

    def test_success_metrics_are_numeric(self, success_metrics):
        """All metric values must be numbers (float or int)."""
        numeric_fields = [
            "combined_metric", "recall_at_1", "recall_at_5", "mean_cosine",
            "distill_loss", "arc_loss", "vat_loss", "sep_loss",
            "peak_vram_mb", "epochs", "elapsed_seconds",
        ]
        for field in numeric_fields:
            assert isinstance(success_metrics[field], (int, float)), (
                f"{field} is {type(success_metrics[field])}, expected number"
            )

    def test_oom_schema_has_required_fields(self, oom_metrics):
        """OOM crash must include status, peak_vram_mb, error."""
        required = {"status", "peak_vram_mb", "error"}
        assert required.issubset(oom_metrics.keys())
        assert oom_metrics["status"] == "oom"

    def test_crash_schema_has_required_fields(self, crash_metrics):
        """General crash must include status, peak_vram_mb, error."""
        required = {"status", "peak_vram_mb", "error"}
        assert required.issubset(crash_metrics.keys())
        assert crash_metrics["status"] == "crash"

    def test_vram_always_present(self, success_metrics, oom_metrics, crash_metrics):
        """INFRA-05: peak_vram_mb must be in every metrics.json variant."""
        for m in [success_metrics, oom_metrics, crash_metrics]:
            assert "peak_vram_mb" in m
            assert isinstance(m["peak_vram_mb"], (int, float))

    def test_metrics_json_is_valid_json(self, success_metrics, metrics_json_path):
        """metrics.json must be valid JSON when written to disk."""
        with open(metrics_json_path, "w") as f:
            json.dump(success_metrics, f, indent=2)
        with open(metrics_json_path) as f:
            loaded = json.load(f)
        assert loaded == success_metrics


class TestResultsTsvFormat:
    """INFRA-02: Validate results.tsv format contract."""

    def test_header_has_7_columns(self, sample_results_tsv):
        header = sample_results_tsv.strip().split("\n")[0]
        columns = header.split("\t")
        assert len(columns) == 7, f"Expected 7 columns, got {len(columns)}: {columns}"

    def test_header_column_names(self, sample_results_tsv):
        header = sample_results_tsv.strip().split("\n")[0]
        expected = ["commit", "combined_metric", "recall_at_1", "mean_cosine",
                    "peak_vram_mb", "status", "description"]
        assert header.split("\t") == expected

    def test_data_rows_have_7_columns(self, sample_results_tsv):
        lines = sample_results_tsv.strip().split("\n")
        for i, line in enumerate(lines[1:], start=2):
            cols = line.split("\t")
            assert len(cols) == 7, f"Row {i} has {len(cols)} columns: {line}"

    def test_status_values_are_valid(self, sample_results_tsv):
        valid_statuses = {"keep", "discard", "crash"}
        lines = sample_results_tsv.strip().split("\n")
        for line in lines[1:]:
            status = line.split("\t")[5]
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_numeric_columns_are_parseable(self, sample_results_tsv):
        lines = sample_results_tsv.strip().split("\n")
        for line in lines[1:]:
            cols = line.split("\t")
            float(cols[1])  # combined_metric
            float(cols[2])  # recall_at_1
            float(cols[3])  # mean_cosine
            float(cols[4])  # peak_vram_mb


class TestCrashStreakDetection:
    """INFRA-04: Verify 3-consecutive-crash detection is possible from results.tsv."""

    def test_no_crash_streak_in_normal_results(self, sample_results_tsv):
        """Normal results with 1 crash should not trigger streak."""
        lines = sample_results_tsv.strip().split("\n")[1:]  # skip header
        consecutive_crashes = 0
        for line in reversed(lines):
            if line.split("\t")[5] == "crash":
                consecutive_crashes += 1
            else:
                break
        assert consecutive_crashes < 3

    def test_crash_streak_detected(self, crash_streak_results_tsv):
        """3 consecutive crashes at end should be detectable."""
        lines = crash_streak_results_tsv.strip().split("\n")[1:]
        consecutive_crashes = 0
        for line in reversed(lines):
            if line.split("\t")[5] == "crash":
                consecutive_crashes += 1
            else:
                break
        assert consecutive_crashes >= 3


class TestTrainPyContract:
    """INFRA-01, INFRA-03, INFRA-07: Verify train.py structure via AST."""

    def test_train_py_has_main_function(self):
        """train.py must have a main() function."""
        source = open("train.py").read()
        tree = ast.parse(source)
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "main" in func_names, "train.py must define main()"

    def test_train_py_has_main_guard(self):
        """train.py must have if __name__ == '__main__' guard."""
        source = open("train.py").read()
        assert '__name__' in source and '__main__' in source

    def test_train_py_handles_oom(self):
        """INFRA-03: train.py must catch OutOfMemoryError."""
        source = open("train.py").read()
        assert "OutOfMemoryError" in source

    def test_train_py_handles_general_exception(self):
        """INFRA-03: train.py must catch general Exception."""
        source = open("train.py").read()
        # Must have both OOM and general except
        assert source.count("except") >= 2

    def test_train_py_writes_metrics_json(self):
        """train.py must write metrics.json in at least 3 places (success, oom, crash)."""
        source = open("train.py").read()
        assert source.count("metrics.json") >= 3

    def test_train_py_imports_epochs_from_prepare(self):
        """INFRA-07: EPOCHS must be imported from prepare.py."""
        source = open("train.py").read()
        assert "from prepare import" in source or "import prepare" in source
        # Verify EPOCHS is used
        assert "EPOCHS" in source

    def test_epochs_value_is_10(self):
        """INFRA-07: EPOCHS constant in prepare.py must equal 10."""
        source = open("prepare.py").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "EPOCHS":
                        assert isinstance(node.value, ast.Constant)
                        assert node.value.value == 10, (
                            f"EPOCHS must be 10, got {node.value.value}"
                        )
                        return
        raise AssertionError("EPOCHS not found as a top-level assignment in prepare.py")

    def test_no_early_stopping(self):
        """INFRA-07: No early stopping or patience logic allowed."""
        source = open("train.py").read().lower()
        assert "early_stop" not in source, "train.py must not have early stopping"
        assert "patience" not in source, "train.py must not have patience-based stopping"

    def test_train_py_tracks_vram(self):
        """INFRA-05: peak VRAM must be tracked."""
        source = open("train.py").read()
        assert "max_memory_allocated" in source

    def test_train_py_prints_summary(self):
        """INFRA-06: Greppable summary block with --- separator."""
        source = open("train.py").read()
        assert '---' in source
        # Must print key metrics
        for metric in ["combined_metric", "recall@1", "mean_cosine", "peak_vram_mb"]:
            assert metric in source, f"Summary must include {metric}"


class TestGitIgnore:
    """Verify experiment artifacts are git-ignored."""

    def test_metrics_json_in_gitignore(self):
        content = open(".gitignore").read()
        assert "metrics.json" in content

    def test_run_log_in_gitignore(self):
        content = open(".gitignore").read()
        assert "run.log" in content

    def test_results_tsv_in_gitignore(self):
        content = open(".gitignore").read()
        assert "results.tsv" in content
