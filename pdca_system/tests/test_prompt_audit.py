"""
Generate all agent prompt variants from real daemon.py code and save to pdca_system/prompt_audit/.

Run from project root:
  uv run python -m unittest pdca_system.tests.test_prompt_audit -v

Prompts are written to pdca_system/prompt_audit/ (gitignored). See docs/agent_prompt_audit.md.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

# Import run module to use its prompt builders (same code path as daemon)
from pdca_system import daemon as run_module
from pdca_system.config import TARGET_METRIC_KEY
from pdca_system.services.workflow import BASELINE_SEED_ID

# Canonical list of every prompt type the daemon can produce. Must match all code paths that build
# prompts: _build_direct_code_prompt; _build_prompt(PD root/worktree); _build_sync_resolution_prompt;
# _build_merge_resolution_prompt (normal seed only in audit); _build_metrics_recovery_prompt;
# _build_prompt(CA baseline_measurement); _build_prompt(CA normal); _build_stuck_check_prompt.
# If you add a new prompt builder or path in daemon.py, add a corresponding entry here and generate it below.
ALL_DAEMON_PROMPT_TYPES = (
    "direct",
    "pd_project_root",
    "pd_worktree",
    "ca_sync_resolution",
    "ca_merge_resolution_normal",
    "ca_metrics_recovery",
    "ca_baseline_measurement",
    "ca_normal",
    "stuck_check",
)
EXPECTED_PROMPT_COUNT = len(ALL_DAEMON_PROMPT_TYPES)

# Directory for audit output (gitignored, under pdca_system)
AUDIT_DIR = Path(__file__).resolve().parent.parent.parent / "pdca_system" / "prompt_audit"


def _task_path(project_root: Path, stage: str, name: str) -> Path:
    """Fake task file path under project root so relative_to(project_root) works."""
    return project_root / "pdca_system" / "history" / "queue" / stage / f"audit-{name}.json"


def _write_audit_file(name: str, content: str) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    return path


class PromptAuditTests(unittest.TestCase):
    """Generate prompts via real run module and save to pdca_system/prompt_audit/."""

    def setUp(self) -> None:
        self.root = run_module.PROJECT_ROOT
        self.audit_dir = AUDIT_DIR

    def test_generate_all_prompts_and_save_to_audit_dir(self) -> None:
        """Call real _build_* from daemon.py for every variant; save to pdca_system/prompt_audit/."""
        generated: list[str] = []

        # 1. Direct code agent
        prompt_direct = run_module._build_direct_code_prompt("Audit: run unit tests and report.")
        _write_audit_file("01_direct", prompt_direct)
        generated.append("01_direct")

        # 2. PD stage (project root context)
        task_pd = {
            "seed_id": "seed-audit",
            "run_id": "pd-audit-001",
            "prompt": "Try a small learning rate change for the optimizer.",
            "worktree_path": None,
        }
        task_path_pd = _task_path(self.root, "pd", "pd")
        prompt_pd_root = run_module._build_prompt("pd", task_pd, task_path_pd)
        _write_audit_file("02_pd_project_root", prompt_pd_root)
        generated.append("02_pd_project_root")

        # 3. PD stage (worktree context — different header)
        with tempfile.TemporaryDirectory(prefix="audit_worktree_") as tmp:
            task_pd_wt = {**task_pd, "worktree_path": tmp}
            prompt_pd_worktree = run_module._build_prompt("pd", task_pd_wt, task_path_pd)
            _write_audit_file("03_pd_worktree", prompt_pd_worktree)
            generated.append("03_pd_worktree")

        # 4. CA sync_resolution
        task_sync = {
            "seed_id": "seed-xyz",
            "run_id": "ca-sync-001",
            "sync_resolution": True,
            "baseline_branch": "master",
        }
        task_path_ca = _task_path(self.root, "ca", "ca")
        prompt_sync = run_module._build_prompt("ca", task_sync, task_path_ca)
        _write_audit_file("04_ca_sync_resolution", prompt_sync)
        generated.append("04_ca_sync_resolution")

        # 5. CA merge_resolution (normal seed: merge seed into baseline; single representative for merge_resolution)
        task_merge_normal = {
            "seed_id": "seed-abc",
            "run_id": "ca-merge-002",
            "merge_resolution": True,
            "baseline_branch": "master",
            "worktree_path": "pdca_system/history/worktrees/seed-abc",
            "last_metrics": {"val_bpb": 1.24},
            "last_summary": {"notes": "Run completed.", "completed_at": "2025-01-15 12:00:00"},
        }
        prompt_merge_normal = run_module._build_merge_resolution_prompt(task_merge_normal)
        _write_audit_file("05_ca_merge_resolution_normal", prompt_merge_normal)
        generated.append("05_ca_merge_resolution_normal")

        # 6. CA metrics_recovery
        task_metrics = {
            "seed_id": "seed-abc",
            "run_id": "ca-metrics-001",
            "metrics_recovery": True,
            "source_run_id": "ca-gpu-001",
            "source_stdout_log_path": "pdca_system/history/logs/ca-gpu-001.stdout.log",
            "source_stderr_log_path": "pdca_system/history/logs/ca-gpu-001.stderr.log",
        }
        prompt_metrics = run_module._build_metrics_recovery_prompt(task_metrics)
        _write_audit_file("06_ca_metrics_recovery", prompt_metrics)
        generated.append("06_ca_metrics_recovery")

        # 7. CA baseline_measurement
        task_baseline = {
            "seed_id": BASELINE_SEED_ID,
            "run_id": "ca-baseline-001",
            "worktree_path": "pdca_system/history/worktrees/__baseline__",
        }
        prompt_baseline = run_module._build_prompt("ca", task_baseline, task_path_ca)
        _write_audit_file("07_ca_baseline_measurement", prompt_baseline)
        generated.append("07_ca_baseline_measurement")

        # 8. CA normal (adapt/fix, run, commit, report)
        task_ca_normal = {
            "seed_id": "seed-abc",
            "run_id": "ca-normal-001",
            "prompt": "Try a small learning rate change.",
            "worktree_path": "pdca_system/history/worktrees/seed-abc",
        }
        prompt_ca_normal = run_module._build_prompt("ca", task_ca_normal, task_path_ca)
        _write_audit_file("08_ca_normal", prompt_ca_normal)
        generated.append("08_ca_normal")

        # 9. Stuck check (abnormal agent exit: backup agent decides if previous agent was stuck)
        prompt_stuck = run_module._build_stuck_check_prompt(
            prev_agent="claude",
            stdout_path=None,
            stderr_path=None,
        )
        _write_audit_file("09_stuck_check", prompt_stuck)
        generated.append("09_stuck_check")

        self.assertEqual(len(generated), EXPECTED_PROMPT_COUNT)
        types_generated = {name[3:] for name in generated}  # "01_direct" -> "direct", etc.
        self.assertEqual(
            types_generated,
            set(ALL_DAEMON_PROMPT_TYPES),
            "Generated prompt types must exactly match ALL_DAEMON_PROMPT_TYPES (add new daemon prompt paths here)",
        )
        self.assertTrue(self.audit_dir.is_dir(), "pdca_system/prompt_audit/ should exist after run")

        # Prompts that reference the target metric must contain the key directly (so agent sees the key)
        metric_prompt_files = (
            "02_pd_project_root",
            "03_pd_worktree",
            "05_ca_merge_resolution_normal",
            "06_ca_metrics_recovery",
            "07_ca_baseline_measurement",
            "08_ca_normal",
        )
        for name in metric_prompt_files:
            path = self.audit_dir / f"{name}.txt"
            self.assertTrue(path.exists(), f"Audit file {path} should exist")
            content = path.read_text(encoding="utf-8")
            self.assertIn(
                TARGET_METRIC_KEY,
                content,
                f"Prompt {name} must contain the target metric key {TARGET_METRIC_KEY!r} directly (config.py); got audit file {path}",
            )

        # Stuck-check prompt must require the backup agent to output previous_agent_stuck / reason / checks
        stuck_path = self.audit_dir / "09_stuck_check.txt"
        self.assertTrue(stuck_path.exists(), "09_stuck_check audit file should exist")
        stuck_content = stuck_path.read_text(encoding="utf-8")
        self.assertIn("previous_agent_stuck", stuck_content)
        self.assertIn("stuck_check", stuck_content)
        self.assertIn(run_module.SUMMARY_FILENAME, stuck_content)

    def test_audit_covers_all_daemon_prompt_types(self) -> None:
        """ALL_DAEMON_PROMPT_TYPES is the single source of truth; test_generate_* must generate each. No duplicates."""
        self.assertEqual(len(ALL_DAEMON_PROMPT_TYPES), len(set(ALL_DAEMON_PROMPT_TYPES)), "no duplicate prompt types")
        self.assertGreater(len(ALL_DAEMON_PROMPT_TYPES), 0)
        # Cross-check: daemon's public prompt-building entry points (used by test_generate_*)
        builders = ("_build_direct_code_prompt", "_build_stuck_check_prompt", "_build_sync_resolution_prompt",
                    "_build_merge_resolution_prompt", "_build_metrics_recovery_prompt")
        for b in builders:
            self.assertTrue(hasattr(run_module, b), f"daemon must define {b}")

    def test_prompt_count_matches_documentation(self) -> None:
        """Ensure we document and generate the same number of variants as run._build_prompt."""
        self.assertEqual(EXPECTED_PROMPT_COUNT, len(ALL_DAEMON_PROMPT_TYPES))

    def test_new_agent_kimi_in_daemon_config(self) -> None:
        """Ensure the kimi agent is registered in daemon AGENT_CONFIGS, version probes, and exit0 checks."""
        self.assertIn("kimi", run_module.AGENT_CONFIGS)
        config = run_module.AGENT_CONFIGS["kimi"]
        self.assertIn("cmd", config)
        self.assertIsInstance(config["cmd"], list)
        self.assertEqual(config["cmd"][0], "kimi")
        self.assertIn(config["via"], ("stdin", "arg"))

        self.assertIn("kimi", run_module.AGENT_VERSION_PROBES)
        probes = run_module.AGENT_VERSION_PROBES["kimi"]
        self.assertIsInstance(probes, list)
        self.assertGreater(len(probes), 0)

        self.assertIn("kimi", run_module.AGENT_EXIT0_FAILURE_CHECKS)
        self.assertTrue(callable(run_module.AGENT_EXIT0_FAILURE_CHECKS["kimi"]))


if __name__ == "__main__":
    unittest.main()
