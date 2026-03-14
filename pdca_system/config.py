"""Static configuration for the PDCA system. No dynamic or per-run values."""
from __future__ import annotations

from pathlib import Path

PDCA_SYSTEM_ROOT = Path(__file__).resolve().parent

# Promotion threshold: improve target metric (see TARGET_METRIC_KEY) by at least this much to promote
PROMOTION_THRESHOLD = 0.001

# Target metric: key reported by run (e.g. val_bpb, val_accuracy), direction, and display label
TARGET_METRIC_KEY = "val_bpb"
TARGET_METRIC_LOWER_IS_BETTER = True  # True = minimize (e.g. bpb, loss), False = maximize (e.g. accuracy)
TARGET_METRIC_LABEL = "Validation BPB"


def best_target_metric_key() -> str:
    """Key for best value in branch metrics view (e.g. best_val_bpb)."""
    return f"best_{TARGET_METRIC_KEY}"


def former_target_metric_key() -> str:
    """Key for former baseline value in run summary / seed context (e.g. former_val_bpb)."""
    return f"former_{TARGET_METRIC_KEY}"

# Worktree root relative to project
WORKTREE_ROOT = "pdca_system/history/worktrees"

# Default branch name suggested in UI when no branches exist (not a global baseline)
DEFAULT_BASELINE_BRANCH = "master"
