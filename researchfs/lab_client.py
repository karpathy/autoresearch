"""Convenience SDK wrapper for the local ResearchFS lab API."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlencode
from typing import Any, Callable, Mapping

from researchfs.api import ArxivFetcher, CodeRepoSearcher, JSONDict, LLMFn, RepoFetcher, ResearchFSLabAPI


class ResearchFSLabClientError(Exception):
    """Raised when the local lab API returns a non-2xx response."""

    def __init__(self, status: int, code: str, message: str, body: JSONDict) -> None:
        super().__init__(f"ResearchFS lab API error {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.body = body


class ResearchFSLabClient:
    """Typed-ish stdlib client over :class:`ResearchFSLabAPI`.

    The client is intentionally thin: each convenience method builds the JSON
    payload for the existing in-process API dispatcher and returns the response
    body unchanged for successful responses.
    """

    def __init__(
        self,
        *,
        api: ResearchFSLabAPI | None = None,
        conn: Any | None = None,
        db_path: str | Path | None = None,
        llm_fn: LLMFn | None = None,
        arxiv_fetcher: ArxivFetcher | None = None,
        repo_fetcher: RepoFetcher | None = None,
        code_repo_searcher: CodeRepoSearcher | None = None,
        reader_fn: Callable[[str, list[Any]], Mapping[str, Any] | str] | None = None,
        verifier_fn: Callable[[Mapping[str, Any], list[Any]], Mapping[str, Any] | bool] | None = None,
    ) -> None:
        if api is not None and any(
            value is not None
            for value in (
                conn,
                db_path,
                llm_fn,
                arxiv_fetcher,
                repo_fetcher,
                code_repo_searcher,
                reader_fn,
                verifier_fn,
            )
        ):
            raise ValueError("Pass either api or API construction options, not both")
        self._api = api or ResearchFSLabAPI(
            conn=conn,
            db_path=str(db_path) if isinstance(db_path, Path) else db_path,
            llm_fn=llm_fn,
            arxiv_fetcher=arxiv_fetcher,
            repo_fetcher=repo_fetcher,
            code_repo_searcher=code_repo_searcher,
            reader_fn=reader_fn,
            verifier_fn=verifier_fn,
        )

    def dispatch(self, method: str, path: str, body: Any = None) -> JSONDict:
        """Dispatch a local API request and return its JSON body.

        Raises:
            ResearchFSLabClientError: if the API response status is not 2xx.
        """
        response = self._api.dispatch(method, path, body)
        if 200 <= response.status < 300:
            return response.body
        error = response.body.get("error") if isinstance(response.body, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "api_error")
            message = str(error.get("message") or f"HTTP {response.status}")
        else:
            code = "api_error"
            message = f"HTTP {response.status}"
        raise ResearchFSLabClientError(response.status, code, message, response.body)

    def list_papers(self) -> JSONDict:
        """Return all papers and extraction status summaries."""
        return self.dispatch("GET", "/papers")

    def get_paper(self, paper_id: int) -> JSONDict:
        """Return one paper with chunks, methods, and provenance."""
        return self.dispatch("GET", f"/paper/{paper_id}")

    def get_arxiv_paper(self, arxiv_id: str) -> JSONDict:
        """Return one PaperStream arXiv paper state."""
        return self.dispatch("GET", f"/arxiv/papers/{quote(arxiv_id, safe='')}")

    def search_arxiv_papers(
        self,
        query: str,
        limit: int | None = None,
        max_chunk_chars: int | None = None,
        include_chunks: bool | None = None,
    ) -> JSONDict:
        """Search PaperStream arXiv metadata and chunks."""
        params: JSONDict = {"query": query}
        if limit is not None:
            params["limit"] = limit
        if max_chunk_chars is not None:
            params["max_chunk_chars"] = max_chunk_chars
        if include_chunks is not None:
            params["include_chunks"] = include_chunks
        return self.dispatch("GET", "/arxiv/search?" + urlencode(params))

    def get_arxiv_paper_chunks(
        self,
        arxiv_id: str,
        version: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
        max_chars: int | None = None,
    ) -> JSONDict:
        """Return bounded PaperStream chunks for one arXiv paper."""
        params: JSONDict = {}
        if version is not None:
            params["version"] = version
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if max_chars is not None:
            params["max_chars"] = max_chars
        path = f"/arxiv/papers/{quote(arxiv_id, safe='')}/chunks"
        if params:
            path += "?" + urlencode(params)
        return self.dispatch("GET", path)

    def list_missing_arxiv_assets(
        self,
        asset_type: str,
        category: str | None = None,
        limit: int | None = None,
    ) -> JSONDict:
        """Return PaperStream arXiv papers missing the requested asset type."""
        query: JSONDict = {"asset_type": asset_type}
        if category is not None:
            query["category"] = category
        if limit is not None:
            query["limit"] = limit
        return self.dispatch("GET", "/arxiv/assets/missing?" + urlencode(query))

    def list_arxiv_jobs(
        self,
        status: str | None = None,
        job_types: str | list[str] | tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> JSONDict:
        """Return PaperStream arXiv queue jobs."""
        query: JSONDict = {}
        if status is not None:
            query["status"] = status
        if job_types is not None:
            if isinstance(job_types, str):
                query["job_types"] = job_types
            else:
                query["job_types"] = ",".join(job_types)
        if limit is not None:
            query["limit"] = limit
        path = "/arxiv/jobs"
        if query:
            path += "?" + urlencode(query)
        return self.dispatch("GET", path)

    def paperstream_sync_status(self) -> JSONDict:
        """Return PaperStream arXiv sync and queue status."""
        return self.dispatch("GET", "/arxiv/sync/status")

    def get_arxiv_sync_status(self) -> JSONDict:
        """Return PaperStream arXiv sync and queue status."""
        return self.paperstream_sync_status()

    def get_paper_memory(self, paper_id: int, max_chars: int | None = None) -> JSONDict:
        """Return one compact paper memory card."""
        path = f"/paper/{paper_id}/memory"
        if max_chars is not None:
            path += f"?max_chars={max_chars}"
        return self.dispatch("GET", path)

    def enrich_paper_code_repositories(self, paper_id: int) -> JSONDict:
        """Run opt-in code repository enrichment for an existing paper."""
        return self.dispatch("POST", f"/paper/{paper_id}/code-repositories/enrich", {})

    def export_paper_memory_bundle(self, paper_id: int, max_chars: int | None = None) -> JSONDict:
        """Export one compressed paper memory bundle."""
        path = f"/paper/{paper_id}/memory/export"
        if max_chars is not None:
            path += f"?max_chars={max_chars}"
        return self.dispatch("GET", path)

    def export_corpus_memory_bundle(
        self,
        paper_ids: list[int] | None = None,
        max_chars: int | None = None,
    ) -> JSONDict:
        """Export a compressed corpus memory bundle."""
        path = "/memory/export"
        query: list[str] = []
        if paper_ids is not None:
            query.append("paper_ids=" + ",".join(str(paper_id) for paper_id in paper_ids))
        if max_chars is not None:
            query.append(f"max_chars={max_chars}")
        if query:
            path += "?" + "&".join(query)
        return self.dispatch("GET", path)

    def import_memory_bundle(self, bundle: JSONDict) -> JSONDict:
        """Import a compressed paper or corpus memory bundle."""
        return self.dispatch("POST", "/memory/import", {"bundle": bundle})

    def import_corpus_memory_bundle(self, bundle: JSONDict) -> JSONDict:
        """Import one compressed corpus memory bundle."""
        return self.import_memory_bundle(bundle)


    def import_paper_memory_bundle(self, bundle: JSONDict) -> JSONDict:
        """Import one compressed paper memory bundle."""
        return self.import_memory_bundle(bundle)

    def get_method(self, method_id: int) -> JSONDict:
        """Return one method with provenance chunks and feedback summary."""
        return self.dispatch("GET", f"/method/{method_id}")

    def stats(self) -> JSONDict:
        """Return aggregate ResearchFS lab API statistics."""
        return self.dispatch("GET", "/stats")

    def ingest(
        self,
        paper: JSONDict | None = None,
        methods: list[JSONDict] | None = None,
        **fields: Any,
    ) -> JSONDict:
        """Ingest a paper payload, optionally with direct method records.

        When called with only a positional mapping, treat it as the full
        top-level `/ingest` payload so `methods` inside that mapping are used.
        If keyword fields or the `methods` argument are supplied, `paper` is sent
        as the nested paper metadata object.
        """
        payload = dict(fields)
        if paper is not None:
            if fields or methods is not None:
                payload["paper"] = paper
            else:
                payload.update(paper)
        if methods is not None:
            payload["methods"] = methods
        return self.dispatch("POST", "/ingest", payload)

    def ingest_file(self, path: str | Path, **overrides: Any) -> JSONDict:
        """Ingest a supported local text, markdown, TeX, or PDF file."""
        payload = {"path": str(path), **overrides}
        return self.dispatch("POST", "/ingest/file", payload)

    def ingest_arxiv(self, arxiv_id: str, **fields: Any) -> JSONDict:
        """Fetch and ingest an arXiv paper through the configured fetcher."""
        payload = dict(fields)
        payload["id"] = arxiv_id
        return self.dispatch("POST", "/ingest/arxiv", payload)

    def ingest_repo(self, repo: str, **fields: Any) -> JSONDict:
        """Fetch and ingest a GitHub repository README through the configured fetcher."""
        payload = dict(fields)
        payload["repo"] = repo
        return self.dispatch("POST", "/ingest/repo", payload)

    def retrieve(
        self,
        query: str,
        max_chars: int | None = None,
        max_papers: int | None = None,
        mode: str = "method",
        log: bool = False,
        include_scores: bool = False,
        event_type: str | None = None,
        agent_run_id: int | None = None,
        experiment_id: int | None = None,
        source_types: str | list[str] | tuple[str, ...] | None = None,
        source_type: str | None = None,
    ) -> JSONDict:
        """Retrieve method, graph, or compressed paper memory context for a query."""
        payload: JSONDict = {
            "query": query,
            "mode": mode,
            "log": log,
            "include_scores": include_scores,
        }
        if max_chars is not None:
            payload["max_chars"] = max_chars
        if max_papers is not None:
            payload["max_papers"] = max_papers
        if source_types is not None:
            payload["source_types"] = source_types
        elif source_type is not None:
            payload["source_type"] = source_type
        if event_type is not None:
            payload["event_type"] = event_type
        if agent_run_id is not None:
            payload["agent_run_id"] = agent_run_id
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id
        return self.dispatch("POST", "/retrieve", payload)

    def research_brief(
        self,
        query: str,
        max_chars: int | None = None,
        max_papers: int | None = None,
        include_scores: bool = False,
        log: bool = False,
        event_type: str | None = None,
        agent_run_id: int | None = None,
        experiment_id: int | None = None,
        source_types: str | list[str] | tuple[str, ...] | None = None,
        source_type: str | None = None,
    ) -> JSONDict:
        """Retrieve a bounded compressed research brief for one query."""
        payload: JSONDict = {
            "query": query,
            "include_scores": include_scores,
            "log": log,
        }
        if max_chars is not None:
            payload["max_chars"] = max_chars
        if max_papers is not None:
            payload["max_papers"] = max_papers
        if source_types is not None:
            payload["source_types"] = source_types
        elif source_type is not None:
            payload["source_type"] = source_type
        if event_type is not None:
            payload["event_type"] = event_type
        if agent_run_id is not None:
            payload["agent_run_id"] = agent_run_id
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id
        return self.dispatch("POST", "/brief", payload)

    def ask(
        self,
        query: str,
        max_chars: int | None = None,
        max_papers: int | None = None,
        k: int | None = None,
        max_reader_tokens: int | None = None,
        min_score: float | None = None,
        overflow_policy: str | None = None,
        max_retries: int | None = None,
        include_scores: bool = False,
        log: bool = False,
        event_type: str | None = None,
        agent_run_id: int | None = None,
        experiment_id: int | None = None,
        source_types: str | list[str] | tuple[str, ...] | None = None,
        source_type: str | None = None,
    ) -> JSONDict:
        """Run a routed local ask workflow over compressed ResearchFS memory."""
        payload: JSONDict = {
            "query": query,
            "include_scores": include_scores,
            "log": log,
        }
        if max_chars is not None:
            payload["max_chars"] = max_chars
        if max_papers is not None:
            payload["max_papers"] = max_papers
        if k is not None:
            payload["k"] = k
        if max_reader_tokens is not None:
            payload["max_reader_tokens"] = max_reader_tokens
        if min_score is not None:
            payload["min_score"] = min_score
        if overflow_policy is not None:
            payload["overflow_policy"] = overflow_policy
        if max_retries is not None:
            payload["max_retries"] = max_retries
        if event_type is not None:
            payload["event_type"] = event_type
        if agent_run_id is not None:
            payload["agent_run_id"] = agent_run_id
        if experiment_id is not None:
            payload["experiment_id"] = experiment_id
        if source_types is not None:
            payload["source_types"] = source_types
        elif source_type is not None:
            payload["source_type"] = source_type
        return self.dispatch("POST", "/ask", payload)

    def feedback(
        self,
        method_id: int,
        rating: float,
        event_id: int | None = None,
        label: str | None = None,
        note: str | None = None,
        source: str | None = None,
    ) -> JSONDict:
        """Submit bounded human feedback for a method."""
        payload: JSONDict = {"method_id": method_id, "rating": rating}
        if event_id is not None:
            payload["event_id"] = event_id
        if label is not None:
            payload["label"] = label
        if note is not None:
            payload["note"] = note
        if source is not None:
            payload["source"] = source
        return self.dispatch("POST", "/feedback", payload)

    def evaluate_extraction(
        self,
        paper_id: int,
        expected_methods: list[JSONDict],
        **options: Any,
    ) -> JSONDict:
        """Evaluate extracted methods for a paper against expected methods."""
        payload = {"paper_id": paper_id, "expected_methods": expected_methods, **options}
        return self.dispatch("POST", "/evaluate/extraction", payload)

    def close(self) -> None:
        """Close the underlying API if it exposes a close hook."""
        close = getattr(self._api, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "ResearchFSLabClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
