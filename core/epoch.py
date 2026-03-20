"""Epoch management: eval counter, holdout rotation, salt.

The eval counter tracks how many evaluations have been run. The epoch
(counter // epoch_length) determines which window is held out from scoring.
The holdout index is derived from a SHA-256 hash of the epoch and an
environment-variable salt, making the mapping unpredictable to the agent.
"""

import hashlib
import os
from pathlib import Path


def read_and_increment_eval_count(eval_count_path: Path) -> int:
    """Atomically read and increment the evaluation counter.

    Returns the count BEFORE incrementing (i.e., this evaluation's index).
    """
    eval_count_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if eval_count_path.exists():
        try:
            count = int(eval_count_path.read_text().strip())
        except (ValueError, FileNotFoundError):
            count = 0
    eval_count_path.write_text(str(count + 1))
    return count


def peek_eval_count(eval_count_path: Path) -> int:
    """Read the eval counter without incrementing. For diagnostics only."""
    if eval_count_path.exists():
        try:
            return int(eval_count_path.read_text().strip())
        except (ValueError, FileNotFoundError):
            return 0
    return 0


def get_holdout_idx(eval_count: int, n_windows: int, epoch_length: int,
                    salt_env_var: str = "AUTOTRADER_HOLDOUT_SALT") -> int:
    """Determine which window to hold out based on current epoch.

    Uses an environment variable salt to make the epoch-to-window mapping
    unpredictable. Without the salt, falls back to sequential rotation.
    """
    epoch = eval_count // epoch_length
    salt = os.environ.get(salt_env_var, "")
    if salt:
        h = hashlib.sha256(f"{epoch}:{salt}".encode()).hexdigest()
        return int(h, 16) % n_windows
    return epoch % n_windows
