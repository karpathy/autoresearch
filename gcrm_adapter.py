"""GCRM adapter for autoresearch — bridges the nightly batch to formal GPU leasing.

Replaces the legacy ``rag_gpu_reservation.sh`` shell script approach with
proper GCRM compute reservations.  Falls back to the shell script when the
GCRM daemon is unreachable (e.g. running outside the Docker Compose stack).

Usage from the orchestrator::

    from gcrm_adapter import acquire_gpu, release_gpu

    request_id = acquire_gpu()      # blocks until lease granted or falls back
    try:
        ... run experiments ...
    finally:
        release_gpu(request_id)

Usage from the daemon scheduler (async)::

    from autoresearch.gcrm_adapter import acquire_gpu_async, release_gpu_async

    request_id = await acquire_gpu_async(gcrm_client)
    try:
        ... spawn nightly.sh ...
    finally:
        await release_gpu_async(gcrm_client, request_id)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

try:
    from cotton.interfaces.compute import IGCRMClient as _IGCRMClient
except ImportError:
    _IGCRMClient = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GCRM_ENABLED: bool = os.getenv("AUTORESEARCH_GCRM_ENABLED", "true").lower() in ("1", "true", "yes")
GCRM_REDIS_URL: str = os.getenv("REDIS_URL", "redis://crsai-redis:6379")
GCRM_PG_URL: str = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:postgres@crsai-postgre:5432/crsai"
)
GCRM_GRANT_TIMEOUT: float = float(os.getenv("AUTORESEARCH_GCRM_GRANT_TIMEOUT", "600"))
RAG_RESERVATION_SCRIPT: str = os.getenv(
    "AUTORESEARCH_RAG_RESERVATION_SCRIPT",
    "/workspace/scripts/rag_gpu_reservation.sh",
)

# Estimated values for autoresearch batch
_ESTIMATED_VRAM_MB: int = int(os.getenv("AUTORESEARCH_ESTIMATED_VRAM_MB", "9800"))
_ESTIMATED_DURATION_SEC: int = int(os.getenv("AUTORESEARCH_ESTIMATED_DURATION_SEC", "7200"))


# ---------------------------------------------------------------------------
# Async API (for daemon scheduler)
# ---------------------------------------------------------------------------


async def acquire_gpu_async(
    gcrm_client: _IGCRMClient | None = None,
    *,
    grant_timeout: float = GCRM_GRANT_TIMEOUT,
) -> str | None:
    """Acquire a GCRM GPU lease for the autoresearch nightly batch.

    Args:
        gcrm_client: An existing ``GCRMClient`` instance.
            If ``None`` and GCRM is enabled, one is created using env vars.
        grant_timeout: Max seconds to wait for the GCRM daemon to grant the lease.

    Returns:
        The ``request_id`` string if a lease was acquired, or ``None`` if the
        legacy shell-script fallback was used instead.
    """
    if not GCRM_ENABLED:
        logger.info("GCRM disabled — falling back to shell script reservation")
        _run_rag_reservation_script()
        return None

    try:
        from cotton.utils.gcrm.client import GCRMClient
        from cotton.utils.gcrm.models import (
            ComputeRequest,
            DeviceTarget,
            Priority,
            WorkloadType,
        )
    except ImportError:
        logger.warning("GCRM modules not importable — falling back to shell script")
        _run_rag_reservation_script()
        return None

    # Create client if not provided
    if gcrm_client is None:
        try:
            import asyncpg
            import redis.asyncio as aioredis

            redis_conn = aioredis.from_url(GCRM_REDIS_URL)
            pg_pool = await asyncpg.create_pool(GCRM_PG_URL, min_size=1, max_size=2)
            gcrm_client = GCRMClient(redis_conn, pg_pool)
        except Exception as exc:
            logger.warning("Failed to connect to GCRM — falling back to shell script: %s", exc)
            _run_rag_reservation_script()
            return None

    req = ComputeRequest(
        agent_id="autoresearch-nightly",
        workload_type=WorkloadType.ML_TRAINING,
        priority=Priority.ML_TRAINING,
        device=DeviceTarget.GPU_0,
        estimated_vram_mb=_ESTIMATED_VRAM_MB,
        estimated_duration_sec=_ESTIMATED_DURATION_SEC,
        preemptible=False,  # nightly batch should not be preempted
        checkpoint_capable=False,  # autoresearch manages its own state.json
    )

    try:
        reservation = await gcrm_client.request_compute(req)
        logger.info(
            "GCRM: autoresearch compute request queued request_id=%s status=%s",
            reservation.request_id,
            reservation.status,
        )
        granted = await gcrm_client.await_grant(
            reservation.request_id,
            timeout=grant_timeout,
        )
        logger.info(
            "GCRM: autoresearch GPU lease granted request_id=%s",
            granted.request_id,
        )
        # Also force MCP-RAG to CPU — GCRM handles vLLM lifecycle but RAG
        # embed device is managed separately
        _run_rag_reservation_script_nonblocking()
        return granted.request_id
    except TimeoutError:
        logger.error(
            "GCRM: grant timeout after %.0fs for autoresearch — falling back to shell script",
            grant_timeout,
        )
        _run_rag_reservation_script()
        return None
    except Exception as exc:
        logger.error("GCRM: acquisition failed for autoresearch: %s — falling back", exc)
        _run_rag_reservation_script()
        return None


async def release_gpu_async(
    gcrm_client: _IGCRMClient | None = None,
    request_id: str | None = None,
) -> None:
    """Release the GCRM GPU lease after the nightly batch completes."""
    if request_id is None:
        return
    try:
        if gcrm_client is not None:
            await gcrm_client.release(request_id)
            logger.info("GCRM: autoresearch lease released request_id=%s", request_id)
    except Exception as exc:
        logger.warning("GCRM: failed to release lease %s: %s", request_id, exc)


# ---------------------------------------------------------------------------
# Sync API (for orchestrator.py subprocess context)
# ---------------------------------------------------------------------------


def acquire_gpu() -> str | None:
    """Synchronous wrapper for ``acquire_gpu_async`` — for use in orchestrator.py.

    Returns the ``request_id`` or ``None`` if using the legacy fallback.
    """
    if not GCRM_ENABLED:
        _run_rag_reservation_script()
        return None

    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(acquire_gpu_async())
    except Exception as exc:
        logger.warning("GCRM sync acquire failed: %s — falling back", exc)
        _run_rag_reservation_script()
        return None
    finally:
        loop.close()


def release_gpu(request_id: str | None = None) -> None:
    """Synchronous wrapper for ``release_gpu_async``."""
    if request_id is None:
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(release_gpu_async(request_id=request_id))
    except Exception as exc:
        logger.warning("GCRM sync release failed: %s", exc)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Legacy shell script fallback
# ---------------------------------------------------------------------------


def _run_rag_reservation_script() -> None:
    """Run the legacy rag_gpu_reservation.sh cpu script (blocking)."""
    try:
        proc = subprocess.run(
            ["bash", RAG_RESERVATION_SCRIPT, "cpu"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            logger.error("Legacy GPU reservation failed: %s", output[-2000:])
            raise RuntimeError(f"GPU reservation script failed: {output[-2000:]}")
        logger.info("Legacy GPU reservation applied (MCP-RAG → CPU)")
    except subprocess.TimeoutExpired:
        raise RuntimeError("GPU reservation script timed out after 180s")


def _run_rag_reservation_script_nonblocking() -> None:
    """Best-effort RAG → CPU switch, non-fatal."""
    try:
        _run_rag_reservation_script()
    except Exception as exc:
        logger.warning("Non-fatal: RAG CPU switch failed: %s", exc)
