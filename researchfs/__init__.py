"""ResearchFS: local research memory with durable SQLite provenance."""

from .db import get_connection, initialize_db
from .codegraph import (
    get_code_map,
    replace_method_code_targets,
    refresh_code_map,
    search_code_symbols,
)
from .retrieve import (
    finish_agent_run,
    get_graph_context,
    get_research_brief,
    get_method_history,
    get_experiment_brief,
    get_method_pack,
    get_paper_memory,
    log_experiment,
    link_experiment_methods,
    log_lesson,
    log_retrieval_event,
    log_token_usage,
    recommend_next_experiments,
    search_methods,
    start_agent_run,
    update_experiment,
)
from .client import ResearchFSClient

__all__ = [
    "CORPUS_BUNDLE_KIND",
    "ContextBudget",
    "ContextCandidate",
    "ContextSelection",
    "EvidenceResult",
    "ReaderFn",
    "VerifierFn",
    "LabAPIResponse",
    "ResearchFSLabAPI",
    "ResearchFSClient",
    "ResearchFSLabClient",
    "citation_verifier",
    "default_reader",
    "ResearchFSLabClientError",
    "export_paper_memory_bundle",
    "export_corpus_memory_bundle",
    "finish_agent_run",
    "get_connection",
    "get_code_map",
    "get_graph_context",
    "get_research_brief",
    "get_method_history",
    "get_experiment_brief",
    "get_method_pack",
    "get_paper_memory",
    "import_paper_memory_bundle",
    "import_corpus_memory_bundle",
    "import_memory_bundle",
    "initialize_db",
    "log_experiment",
    "link_experiment_methods",
    "replace_method_code_targets",
    "log_lesson",
    "log_retrieval_event",
    "log_token_usage",
    "ROUTING_TRACE_SCHEMA_VERSION",
    "DEFAULT_ROUTING_TRACE_EVENT_TYPES",
    "export_routing_traces",
    "run_routed_ask",
    "search_methods",
    "recommend_next_experiments",
    "refresh_code_map",
    "search_code_symbols",
    "start_agent_run",
    "update_experiment",
    "citation_f1",
    "citation_precision",
    "citation_recall",
    "context_efficiency",
    "dispatch_request",
    "false_stop_rate",
    "gold_recall_at_k",
    "log_context_routing_audit",
    "paper_memories_to_context_candidates",
    "paper_memory_to_context_candidate",
    "recovery_rate",
    "recovery_success",
    "research_brief_to_context_candidates",
    "route_top_k_context",
    "iter_routing_traces",
    "selection_audit_payload",
    "supported_answer_rate",
    "write_routing_traces_jsonl",
]

_ASK_EXPORTS = {
    "ReaderFn",
    "VerifierFn",
    "citation_verifier",
    "default_reader",
    "run_routed_ask",
}

_ROUTING_TRACE_EXPORTS = {
    "ROUTING_TRACE_SCHEMA_VERSION",
    "DEFAULT_ROUTING_TRACE_EVENT_TYPES",
    "iter_routing_traces",
    "export_routing_traces",
    "write_routing_traces_jsonl",
}

_CONTEXT_EXPORTS = {
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
    "log_context_routing_audit",
    "paper_memories_to_context_candidates",
    "paper_memory_to_context_candidate",
    "recovery_rate",
    "recovery_success",
    "research_brief_to_context_candidates",
    "route_top_k_context",
    "selection_audit_payload",
    "supported_answer_rate",
}


def __getattr__(name):
    if name in _ASK_EXPORTS:
        from . import ask as ask_module

        return getattr(ask_module, name)
    if name in _ROUTING_TRACE_EXPORTS:
        from . import routing_traces as routing_traces_module

        return getattr(routing_traces_module, name)
    if name in _CONTEXT_EXPORTS:
        from . import context as context_module

        return getattr(context_module, name)
    if name in {"LabAPIResponse", "ResearchFSLabAPI", "dispatch_request"}:
        from .api import LabAPIResponse, ResearchFSLabAPI, dispatch_request

        exports = {
            "LabAPIResponse": LabAPIResponse,
            "ResearchFSLabAPI": ResearchFSLabAPI,
            "dispatch_request": dispatch_request,
        }
        return exports[name]
    if name in {"ResearchFSLabClient", "ResearchFSLabClientError"}:
        from .lab_client import ResearchFSLabClient, ResearchFSLabClientError

        exports = {
            "ResearchFSLabClient": ResearchFSLabClient,
            "ResearchFSLabClientError": ResearchFSLabClientError,
        }
        return exports[name]
    if name in {
        "CORPUS_BUNDLE_KIND",
        "export_corpus_memory_bundle",
        "export_paper_memory_bundle",
        "import_corpus_memory_bundle",
        "import_memory_bundle",
        "import_paper_memory_bundle",
    }:
        from .bundle import (
            CORPUS_BUNDLE_KIND,
            export_corpus_memory_bundle,
            export_paper_memory_bundle,
            import_corpus_memory_bundle,
            import_memory_bundle,
            import_paper_memory_bundle,
        )

        exports = {
            "CORPUS_BUNDLE_KIND": CORPUS_BUNDLE_KIND,
            "export_corpus_memory_bundle": export_corpus_memory_bundle,
            "export_paper_memory_bundle": export_paper_memory_bundle,
            "import_corpus_memory_bundle": import_corpus_memory_bundle,
            "import_memory_bundle": import_memory_bundle,
            "import_paper_memory_bundle": import_paper_memory_bundle,
        }
        return exports[name]
    raise AttributeError(f"module 'researchfs' has no attribute {name!r}")
