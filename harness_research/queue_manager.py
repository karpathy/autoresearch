"""Push harness candidate configs to Firestore for fleet evaluation.

Robots running `castor contribute` with harness_research pull from these queues,
evaluate locally, and submit scores back to Firestore for aggregation by
contribute_eval.py.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_FIRESTORE_PROJECT = "opencastor"


def _get_firestore_client():
    """Create Firestore client using service account or ADC."""
    from google.cloud import firestore as _firestore  # type: ignore[import-untyped]

    creds_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(Path.home() / ".config/opencastor/firebase-sa-key.json"),
    )
    try:
        from google.oauth2 import service_account  # type: ignore[import-untyped]

        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=[
                "https://www.googleapis.com/auth/datastore",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )
        return _firestore.Client(project=_FIRESTORE_PROJECT, credentials=creds)
    except Exception:
        import google.auth  # type: ignore[import-untyped]

        creds, project = google.auth.default()
        return _firestore.Client(project=project or _FIRESTORE_PROJECT, credentials=creds)


def push_candidates_to_queue(
    candidates: list[dict],
    hardware_tier: str,
    max_evaluations: int = 5,
    model_id: str | None = None,
) -> int:
    """Push candidates to Firestore harness_eval_queue/{tier}/candidates/.

    Each document:
        {candidate_id, config, description, hardware_tier, model_id,
         status="pending", created_at, max_evaluations, evaluation_count=0}

    Args:
        candidates: List of candidate dicts (each may carry a ``model_id`` field).
        hardware_tier: Target hardware tier for the queue.
        max_evaluations: Max evaluations per candidate.
        model_id: Override model_id for all candidates. If None, uses each
            candidate's own ``model_id`` field, falling back to "default".

    Returns:
        Number of candidates pushed.
    """
    db = _get_firestore_client()
    pushed = 0
    queue_ref = (
        db.collection("harness_eval_queue")
        .document(hardware_tier)
        .collection("candidates")
    )

    for candidate in candidates:
        candidate_id = candidate.get("id") or candidate.get("candidate_id")
        if not candidate_id:
            log.warning("Skipping candidate with no id: %s", candidate)
            continue

        effective_model_id = model_id or candidate.get("model_id") or "default"
        doc_data = {
            "candidate_id": candidate_id,
            "config": candidate.get("config", {}),
            "description": candidate.get("description", ""),
            "hardware_tier": hardware_tier,
            "model_id": effective_model_id,
            "status": "pending",
            "created_at": int(time.time()),
            "max_evaluations": max_evaluations,
            "evaluation_count": 0,
        }

        queue_ref.document(candidate_id).set(doc_data)
        pushed += 1
        log.debug("Pushed candidate %s to queue (tier=%s)", candidate_id, hardware_tier)

    log.info("Pushed %d candidate(s) to harness_eval_queue/%s/candidates/", pushed, hardware_tier)
    return pushed


def get_queue_status(hardware_tier: str | None = None) -> dict:
    """Return status of all/one tier's queue: pending/assigned/completed counts.

    Returns:
        Dict mapping hardware_tier → {pending, assigned, completed, total}.
        If hardware_tier is given, returns only that tier's counts.
    """
    from .generator import HARDWARE_TIERS

    db = _get_firestore_client()
    tiers = [hardware_tier] if hardware_tier else HARDWARE_TIERS
    result: dict[str, dict] = {}

    for tier in tiers:
        queue_ref = (
            db.collection("harness_eval_queue")
            .document(tier)
            .collection("candidates")
        )
        docs = list(queue_ref.stream())
        counts: dict[str, int] = {"pending": 0, "assigned": 0, "completed": 0, "total": len(docs)}
        for doc in docs:
            data = doc.to_dict() or {}
            status = data.get("status", "pending")
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1
        result[tier] = counts
        log.debug(
            "Queue status for %s: pending=%d assigned=%d completed=%d total=%d",
            tier,
            counts["pending"],
            counts["assigned"],
            counts["completed"],
            counts["total"],
        )

    return result


def cleanup_completed_candidates(hardware_tier: str, max_age_days: int = 7) -> int:
    """Remove completed candidates older than max_age_days.

    Returns:
        Number of documents deleted.
    """
    db = _get_firestore_client()
    queue_ref = (
        db.collection("harness_eval_queue")
        .document(hardware_tier)
        .collection("candidates")
    )
    cutoff_ts = int((datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)).timestamp())

    completed = list(queue_ref.where("status", "==", "completed").stream())
    deleted = 0
    for doc in completed:
        data = doc.to_dict() or {}
        completed_at = data.get("completed_at", 0)
        if isinstance(completed_at, int) and completed_at < cutoff_ts:
            doc.reference.delete()
            deleted += 1
            log.debug("Deleted completed candidate %s (tier=%s)", doc.id, hardware_tier)

    log.info(
        "Cleaned up %d completed candidate(s) older than %d days from tier %s",
        deleted,
        max_age_days,
        hardware_tier,
    )
    return deleted
