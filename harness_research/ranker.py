"""Rank evaluated harness candidates and compare against champion."""

import logging
from pathlib import Path

import yaml

from .evaluator import EvalResults

import os

log = logging.getLogger(__name__)

# Allow CI to override the ops repo path via env var
_OPS_DIR = Path(os.environ.get("OPENCASTOR_OPS_DIR", Path.home() / "opencastor-ops"))
CHAMPION_PATH = _OPS_DIR / "harness-research" / "champion.yaml"

IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement required


def compute_score(result: EvalResults) -> float:
    """Compute composite score for a candidate.

    score = (success_rate * 0.50) + (p66_rate * 0.25)
          + (token_efficiency * 0.15) + (latency_score * 0.10)
    """
    return (
        result.success_rate * 0.50
        + result.p66_rate * 0.25
        + result.token_efficiency * 0.15
        + result.latency_score * 0.10
    )


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model_id for use in file paths (replace / with _)."""
    return model_id.replace("/", "_")


def load_champion_score(
    hardware_tier: str | None = None,
    model_id: str | None = None,
) -> float:
    """Load the current champion score from opencastor-ops.

    If hardware_tier + model_id: loads from profiles/{tier}/{model_id_sanitized}.yaml.
    If hardware_tier only: loads from profiles/{tier}.yaml (old format).
    Otherwise: loads generic champion.yaml.
    """
    if hardware_tier and model_id:
        profile_path = (
            _OPS_DIR / "harness-research" / "profiles"
            / hardware_tier / f"{_sanitize_model_id(model_id)}.yaml"
        )
        if profile_path.exists():
            data = yaml.safe_load(profile_path.read_text())
            if data and isinstance(data, dict):
                return float(data.get("score", 0.0))
        return 0.0

    if hardware_tier:
        profile_path = _OPS_DIR / "harness-research" / "profiles" / f"{hardware_tier}.yaml"
        if profile_path.exists():
            data = yaml.safe_load(profile_path.read_text())
            if data and isinstance(data, dict):
                return float(data.get("score", 0.0))
        return 0.0  # No profile champion yet — any winner is an improvement

    if CHAMPION_PATH.exists():
        data = yaml.safe_load(CHAMPION_PATH.read_text())
        if data and isinstance(data, dict):
            return float(data.get("score", 0.0))
    return 0.0


def rank_candidates(results: list[EvalResults]) -> list[tuple[EvalResults, float]]:
    """Rank candidates by composite score, descending.

    Returns list of (EvalResults, score) tuples.
    """
    scored = [(r, compute_score(r)) for r in results]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def find_winner(
    results: list[EvalResults],
    hardware_tier: str | None = None,
    model_id: str | None = None,
) -> tuple[list[tuple[EvalResults, float]], EvalResults | None, float, float]:
    """Rank candidates and determine if any beats the champion.

    Returns:
        ranked: sorted list of (EvalResults, score)
        winner: the winning EvalResults or None
        champion_score: current champion score
        best_score: score of the top candidate
    """
    champion_score = load_champion_score(hardware_tier=hardware_tier, model_id=model_id)
    ranked = rank_candidates(results)

    if not ranked:
        return [], None, champion_score, 0.0

    best_result, best_score = ranked[0]

    log.info("Champion score: %.4f | Best candidate: %.4f", champion_score, best_score)

    if best_score > champion_score + IMPROVEMENT_THRESHOLD:
        log.info(
            "Winner found: '%s' beats champion by %.4f (>%.4f threshold)",
            best_result.candidate_id,
            best_score - champion_score,
            IMPROVEMENT_THRESHOLD,
        )
        return ranked, best_result, champion_score, best_score

    log.info("No candidate beats champion by >%.0f%%", IMPROVEMENT_THRESHOLD * 100)
    return ranked, None, champion_score, best_score
