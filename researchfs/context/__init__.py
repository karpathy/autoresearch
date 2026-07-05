"""Context routing and evaluation helpers for ResearchFS."""

from .metrics import (
    citation_f1,
    citation_precision,
    citation_recall,
    context_efficiency,
    false_stop_rate,
    gold_recall_at_k,
    recovery_rate,
    recovery_success,
    supported_answer_rate,
)
from .researchfs import (
    log_context_routing_audit,
    paper_memories_to_context_candidates,
    paper_memory_to_context_candidate,
    research_brief_to_context_candidates,
    selection_audit_payload,
)
from .routing import (
    ContextBudget,
    ContextCandidate,
    ContextSelection,
    EvidenceResult,
    route_top_k_context,
)

__all__ = [
    "ContextBudget",
    "ContextCandidate",
    "ContextSelection",
    "EvidenceResult",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "context_efficiency",
    "false_stop_rate",
    "gold_recall_at_k",
    "recovery_rate",
    "recovery_success",
    "log_context_routing_audit",
    "paper_memories_to_context_candidates",
    "paper_memory_to_context_candidate",
    "research_brief_to_context_candidates",
    "selection_audit_payload",
    "route_top_k_context",
    "supported_answer_rate",
]
