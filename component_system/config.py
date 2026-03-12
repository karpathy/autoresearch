"""Static configuration for the component system. No dynamic or per-run values."""
from __future__ import annotations

from pathlib import Path

COMPONENT_SYSTEM_ROOT = Path(__file__).resolve().parent

# Module import paths for training (used by mainline assembler)
MODEL_MODULE = "component_system.components.model"
OPTIMIZER_MODULE = "component_system.components.optimizer"
TRAINING_STEP_MODULE = "component_system.components.trainer"

# Promotion threshold: improve val_bpb by at least this much to promote
PROMOTION_THRESHOLD = 0.001

# Worktree root relative to project
WORKTREE_ROOT = "component_system/history/worktrees"

# Default branch name suggested in UI when no branches exist (not a global baseline)
DEFAULT_BASELINE_BRANCH = "master"


def get_training_binding() -> dict[str, str | float]:
    """Return a static dict used by training mainline/trainer (no baseline_version)."""
    return {
        "model_module": MODEL_MODULE,
        "optimizer_module": OPTIMIZER_MODULE,
        "training_step_module": TRAINING_STEP_MODULE,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "worktree_root": WORKTREE_ROOT,
    }
