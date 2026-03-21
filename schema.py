"""
Experiment memory schema for the autoresearch loop.

ExperimentRecord stores the core, fixed fields of every experiment.
VerdictSnapshot is a lightweight, separate record that captures val_bpb
at the exact moment an experiment is accepted or rejected — kept outside
the main schema because val_bpb is a varying quantity during training.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class ExperimentRecord(BaseModel):
    """Core, immutable-ish record for one experiment."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    hyperparameters: dict

    confidence: Optional[float] = None

    llm_used: bool = False

    # Current verdict for this experiment
    last_verdict: Optional[Verdict] = None


class VerdictSnapshot(BaseModel):
    """
    Captured at the moment an experiment is accepted or rejected.

    Stored *outside* ExperimentRecord because val_bpb is a moving target
    during training.  Each snapshot pins the metric to a specific verdict
    event so the memory system can look up "what was the bpb when we
    decided to keep / drop this config?"
    """

    experiment_id: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verdict: Verdict
    val_bpb: float
