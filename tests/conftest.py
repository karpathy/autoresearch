"""Shared fixtures for autoresearch tests."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Infrastructure test fixtures (Phase 2, Plan 02)
# ---------------------------------------------------------------------------


@pytest.fixture
def success_metrics():
    """Valid metrics.json for a successful experiment run."""
    return {
        "status": "success",
        "combined_metric": 0.6234,
        "recall_at_1": 0.4821,
        "recall_at_5": 0.7123,
        "mean_cosine": 0.7647,
        "distill_loss": 0.3210,
        "arc_loss": 0.1540,
        "vat_loss": 0.0200,
        "sep_loss": 0.0150,
        "peak_vram_mb": 18432.5,
        "epochs": 10,
        "elapsed_seconds": 245.3,
    }


@pytest.fixture
def oom_metrics():
    """Valid metrics.json for an OOM crash."""
    return {
        "status": "oom",
        "peak_vram_mb": 23800.2,
        "error": "CUDA out of memory",
    }


@pytest.fixture
def crash_metrics():
    """Valid metrics.json for a general crash."""
    return {
        "status": "crash",
        "peak_vram_mb": 8200.0,
        "error": "RuntimeError: mat1 and mat2 shapes cannot be multiplied",
    }


@pytest.fixture
def sample_results_tsv():
    """Sample results.tsv content with header and mixed statuses."""
    return (
        "commit\tcombined_metric\trecall_at_1\tmean_cosine\tpeak_vram_mb\tstatus\tdescription\n"
        "a1b2c3d\t0.623400\t0.482100\t0.764700\t18432.5\tkeep\tbaseline\n"
        "b2c3d4e\t0.635200\t0.501000\t0.769400\t18440.2\tkeep\tincrease LR to 0.2\n"
        "c3d4e5f\t0.610000\t0.460000\t0.760000\t18435.0\tdiscard\tswitch to GeLU\n"
        "d4e5f6g\t0.000000\t0.000000\t0.000000\t22100.5\tcrash\tdouble batch size\n"
    )


@pytest.fixture
def crash_streak_results_tsv():
    """results.tsv with 3 consecutive crashes at the end."""
    return (
        "commit\tcombined_metric\trecall_at_1\tmean_cosine\tpeak_vram_mb\tstatus\tdescription\n"
        "a1b2c3d\t0.623400\t0.482100\t0.764700\t18432.5\tkeep\tbaseline\n"
        "b2c3d4e\t0.000000\t0.000000\t0.000000\t22100.5\tcrash\tbig batch\n"
        "c3d4e5f\t0.000000\t0.000000\t0.000000\t22300.1\tcrash\tbig batch v2\n"
        "d4e5f6g\t0.000000\t0.000000\t0.000000\t22500.0\tcrash\tbig batch v3\n"
    )


@pytest.fixture
def metrics_json_path(tmp_path):
    """Provides a temporary path for metrics.json testing."""
    return tmp_path / "metrics.json"
