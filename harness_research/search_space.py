"""Harness design search space definition and exploration tracking.

The total combinatorial space of harness configs we are searching:
  cost_gate_usd:     8 values  [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20]
  thinking_budget:   7 values  [512, 768, 1024, 1536, 2048, 3072, 4096]
  context_budget:    7 values  [4096, 6144, 8192, 12288, 16384, 24576, 32768]
  max_iterations:    7 values  [3, 4, 5, 6, 8, 10, 12]
  drift_detection:   2 values  [True, False]
  retry_on_error:    2 values  [True, False]
  pattern:           3 values  [single_agent_supervisor, initializer_executor, multi_agent]
  memory:            4 values  [wm/truncate, wm/drop_oldest, fs/truncate, fs/drop_oldest]
  security:          2 values  [no guardrail, audit guardrail]

Total: 8 × 7 × 7 × 7 × 2 × 2 × 3 × 4 × 2 = 263,424
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SEARCH_SPACE_AXES: dict[str, list] = {
    "cost_gate_usd": [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20],
    "thinking_budget": [512, 768, 1024, 1536, 2048, 3072, 4096],
    "context_budget": [4096, 6144, 8192, 12288, 16384, 24576, 32768],
    "max_iterations": [3, 4, 5, 6, 8, 10, 12],
    "drift_detection": [True, False],
    "retry_on_error": [True, False],
    "pattern": ["single_agent_supervisor", "initializer_executor", "multi_agent"],
    "memory": [
        "working_memory/truncate",
        "working_memory/drop_oldest",
        "filesystem/truncate",
        "filesystem/drop_oldest",
    ],
    "security": ["none", "audit"],
}

SEARCH_SPACE_SIZE: int = 1
for _vals in SEARCH_SPACE_AXES.values():
    SEARCH_SPACE_SIZE *= len(_vals)
# 8*7*7*7*2*2*3*4*2 = 263,424

_OPS_REPO = Path(os.environ.get("OPENCASTOR_OPS_DIR", Path.home() / "opencastor-ops"))
_STATE_FILE = _OPS_REPO / "harness-research" / "search_space_state.json"


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text()) if _STATE_FILE.exists() else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as exc:
        log.warning("Could not save search space state: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def explored_count(firestore_client=None) -> int:
    """Return total number of distinct candidate configs evaluated so far.

    Tries Firestore first (harness_eval_results collection count), falls back
    to local state file, then returns 0 gracefully.
    """
    if firestore_client is not None:
        try:
            docs = firestore_client.collection("harness_eval_results").get()
            count = len(docs)
            _save_state({**_load_state(), "explored_count": count})
            return count
        except Exception as exc:
            log.warning("Firestore count failed, using local state: %s", exc)

    # Local state file fallback
    state = _load_state()
    if "explored_count" in state:
        return int(state["explored_count"])

    # Last resort: count candidates in ops repo
    try:
        candidates_dir = _OPS_REPO / "harness-research" / "candidates"
        if candidates_dir.exists():
            return len(list(candidates_dir.glob("*.yaml")))
    except Exception:
        pass

    return 0


def explored_pct(firestore_client=None) -> float:
    """Return fraction of search space explored (0.0–1.0)."""
    count = explored_count(firestore_client=firestore_client)
    return round(count / SEARCH_SPACE_SIZE, 6)


def record_evaluation(candidate_id: str, count_delta: int = 1) -> None:
    """Increment the local explored count after an evaluation batch."""
    state = _load_state()
    current = int(state.get("explored_count", 0))
    state["explored_count"] = current + count_delta
    _save_state(state)


def status_dict(firestore_client=None) -> dict:
    """Return a structured status dict for CLI / API output."""
    count = explored_count(firestore_client=firestore_client)
    pct = round(count / SEARCH_SPACE_SIZE * 100, 4)

    # Load champion from ops repo
    champion: dict = {}
    try:
        import yaml  # type: ignore[import-untyped]
        champion_path = _OPS_REPO / "harness-research" / "champion.yaml"
        if champion_path.exists():
            data = yaml.safe_load(champion_path.read_text()) or {}
            champion = {
                "candidate_id": data.get("id", ""),
                "score": data.get("score", 0.0),
            }
    except Exception:
        pass

    return {
        "search_space_size": SEARCH_SPACE_SIZE,
        "explored": count,
        "explored_pct": pct,
        "axes": {k: len(v) for k, v in SEARCH_SPACE_AXES.items()},
        "champion": champion,
    }
