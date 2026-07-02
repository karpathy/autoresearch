"""ResearchFS v0: lightweight literature retrieval and experiment logging package."""

from .db import get_connection, initialize_db
from .codegraph import (
    get_code_map,
    link_methods_to_code,
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
    "LabAPIResponse",
    "ResearchFSLabAPI",
    "ResearchFSClient",
    "ResearchFSLabClient",
    "ResearchFSLabClientError",
    "ensure_seed",
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
    "link_methods_to_code",
    "log_lesson",
    "log_retrieval_event",
    "log_token_usage",
    "search_methods",
    "recommend_next_experiments",
    "refresh_code_map",
    "search_code_symbols",
    "seed_initial_data",
    "start_agent_run",
    "update_experiment",
    "dispatch_request",
]


def __getattr__(name):
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
    if name in {"ensure_seed", "seed_initial_data"}:
        from .ingest import ensure_seed, seed_initial_data

        exports = {
            "ensure_seed": ensure_seed,
            "seed_initial_data": seed_initial_data,
        }
        return exports[name]
    raise AttributeError(f"module 'researchfs' has no attribute {name!r}")
