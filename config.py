"""
Centralized configuration for autoresearch.
All constants are overridable via environment variables (AR_ prefix).
Without env vars, defaults match the original hardcoded values.
"""

import os

TIME_BUDGETS = {
    "probe": 30,
    "quick": 120,
    "standard": 300,
    "long": 900,
    "deep": 1800,
}

SCALE_PRIORITY = {"probe": 0, "quick": 1, "standard": 2, "long": 3, "deep": 4}


def get_scale() -> str:
    return os.environ.get("AR_SCALE", "standard")


def get_time_budget() -> int:
    explicit = os.environ.get("AR_TIME_BUDGET")
    if explicit is not None:
        return int(explicit)
    return TIME_BUDGETS.get(get_scale(), 300)


def get_max_seq_len() -> int:
    return int(os.environ.get("AR_MAX_SEQ_LEN", "2048"))


def get_eval_tokens() -> int:
    return int(os.environ.get("AR_EVAL_TOKENS", str(160 * 524288)))


def get_checkpoint_dir() -> str:
    return os.environ.get("AR_CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))


def get_results_dir() -> str:
    return os.environ.get("AR_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))


def get_queue_dir() -> str:
    return os.environ.get("AR_QUEUE_DIR", os.path.join(os.path.dirname(__file__), "queue"))
