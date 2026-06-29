"""ResearchFS client SDK for auto-researcher agents.

This module exposes a thin, agent-agnostic wrapper around the ResearchFS
retrieval database.  Any auto-research agent can instantiate a
`ResearchFSClient`, start a session, and ask it to suggest queries,
cache context, and log outcomes.  The goal is to centralise token- and
time-saving RAG logic so that it can be reused across different agent
setups without copying prompt templates or SQL queries.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple

import researchfs.db as db_module
from researchfs import retrieve as rfs


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English/code text."""
    return max(1, len(text) // 4)


def _clean_query(query: str) -> str:
    """Normalize a query string for cache lookup."""
    return " ".join(query.lower().split())


def _extract_json_block(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = raw.strip()

    # Drop markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx : end_idx + 1])
        except Exception:
            pass

    return None


def _assemble_retrieval_prompt(
    results_content: str, experiment_brief: str, query_history: Optional[str] = None
) -> str:
    """Assemble prompt to ask the agent for a search query decision."""
    prompt = (
        "You are an expert machine learning researcher agent.\n"
        "Your goal is to optimize the training script `train.py` to minimize the "
        "validation bits per byte (val_bpb) on a pretraining task.\n\n"
        f"Experiment Brief:\n{experiment_brief}\n\n"
        f"Here is the history of previous experiments:\n{results_content}\n\n"
    )
    if query_history:
        prompt += (
            "Here is the history of prior search queries and their experiment "
            f"outcomes:\n{query_history}\n\n"
        )

    prompt += (
        "Based on the results, decide what optimization area you want to explore "
        "next (e.g. attention, MLP, optimizer learning rate, precision, regularization).\n"
        "You must decide whether you want to search the methods database or skip searching.\n\n"
        "Please respond with a structured JSON object containing the following keys:\n"
        '  - "search": a boolean indicating whether you want to search (true) or skip search (false).\n'
        '  - "query": a string specifying the search query (1-2 keywords, e.g. \'MLP SwiGLU\', '
        "'GQA attention', or 'Muon LR') if you want to search, or empty string \"\" if you want to skip search.\n"
        '  - "reason": a string explaining your decision.\n\n'
        "Response format:\n"
        "{\n"
        '  "search": true,\n'
        '  "query": "MLP SwiGLU",\n'
        '  "reason": "We want to try SwiGLU activation for the feedforward network."\n'
        "}"
    )
    return prompt


class ResearchFSSession:
    """A single research run session backed by ResearchFS.

    The session tracks query decisions, caches retrieved method packs, and logs
    experiment outcomes.  It is deliberately lightweight so that any
    auto-research agent can hold one without importing the rest of LoopGraph.
    """

    def __init__(
        self,
        client: "ResearchFSClient",
        agent_run_id: int,
        baseline_metric: Optional[float] = None,
    ) -> None:
        self.client = client
        self.agent_run_id = agent_run_id
        self.baseline_metric = baseline_metric
        self.search_cache: Dict[str, Tuple[str, List[int]]] = {}
        self.prompt_tokens_accum = 0
        self.completion_tokens_accum = 0

    # ------------------------------------------------------------------
    # Query reuse and history helpers
    # ------------------------------------------------------------------
    def get_reusable_query(self) -> Optional[str]:
        """Return the query from the most recent kept experiment that improved over baseline."""
        if self.baseline_metric is None:
            return None
        conn = self.client.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT re.query
                FROM experiments e
                JOIN retrieval_events re ON e.experiment_id = re.experiment_id
                WHERE e.agent_run_id = ?
                  AND e.status = 'keep'
                  AND e.val_bpb < ?
                  AND re.event_type IN ('agent_search', 'agent_search_reuse')
                ORDER BY e.experiment_id DESC
                LIMIT 1
                """,
                (self.agent_run_id, self.baseline_metric),
            )
            row = cur.fetchone()
            return row["query"] if row else None
        finally:
            pass

    def get_query_history(self) -> str:
        """Format history of prior queries and their outcomes."""
        conn = self.client.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT re.query, e.status, e.val_bpb, e.description
                FROM retrieval_events re
                LEFT JOIN experiments e ON re.experiment_id = e.experiment_id
                WHERE re.agent_run_id = ?
                  AND re.event_type IN ('agent_search', 'agent_search_reuse')
                ORDER BY re.event_id ASC
                """,
                (self.agent_run_id,),
            )
            rows = cur.fetchall()
            history_lines = []
            for row in rows:
                query_val = row["query"]
                status_val = row["status"]
                val_bpb_val = row["val_bpb"]

                if status_val == "keep":
                    outcome = (
                        f"Kept (improved BPB to {val_bpb_val:.6f})"
                        if val_bpb_val is not None
                        else "Kept"
                    )
                elif status_val == "discard":
                    outcome = (
                        f"Discarded (BPB: {val_bpb_val:.6f})"
                        if val_bpb_val is not None
                        else "Discarded"
                    )
                elif status_val == "crash":
                    outcome = "Crashed"
                elif status_val == "pending":
                    outcome = "Pending"
                elif status_val is None:
                    outcome = "Unknown"
                else:
                    outcome = status_val.capitalize()

                history_lines.append(f"Query: '{query_val}' -> Outcome: {outcome}")
            return "\n".join(history_lines)
        finally:
            pass

    # ------------------------------------------------------------------
    # Main RAG entry points
    # ------------------------------------------------------------------
    def suggest_query(
        self,
        results_content: str,
        experiment_brief: str,
        llm_fn: Callable[[str], str],
    ) -> Dict[str, object]:
        """Decide whether to search and, if so, what query to use.

        Returns a dict with keys:
            search (bool): whether the caller should search ResearchFS
            query (str): the query string (empty when search is False)
            reason (str): the agent's reasoning
            reused (bool): True if a previous successful query was reused
        """
        reused_query = self.get_reusable_query()
        if reused_query:
            return {
                "search": True,
                "query": reused_query,
                "reason": "Reusing prior query that improved over baseline.",
                "reused": True,
            }

        query_history = self.get_query_history()
        retrieval_prompt = _assemble_retrieval_prompt(
            results_content, experiment_brief, query_history
        )
        self.prompt_tokens_accum += _estimate_tokens(retrieval_prompt)

        query_res = llm_fn(retrieval_prompt)
        self.completion_tokens_accum += _estimate_tokens(query_res)

        parsed = _extract_json_block(query_res)
        if parsed is not None and isinstance(parsed, dict):
            return {
                "search": bool(parsed.get("search", True)),
                "query": str(parsed.get("query", "")).strip(),
                "reason": str(parsed.get("reason", "")),
                "reused": False,
            }

        # Fallback: treat the raw response as a plain query.
        return {
            "search": True,
            "query": query_res.strip().strip('"').strip("'"),
            "reason": "JSON parsing failed; treating raw response as query.",
            "reused": False,
        }

    def retrieve_context(
        self, query: str, max_method_chars: Optional[int] = None
    ) -> Tuple[str, Optional[List[int]]]:
        """Fetch a formatted method pack for `query`, using the in-memory cache.

        Returns:
            (method_pack, method_ids)
        """
        if not query:
            return "", None

        cache_key = _clean_query(query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        pack, ids = rfs.search_methods_and_ids(
            query, conn=self.client.get_connection(), max_chars=max_method_chars
        )
        self.search_cache[cache_key] = (pack, ids)
        return pack, ids

    # ------------------------------------------------------------------
    # Outcome logging (used for query reuse and history feedback)
    # ------------------------------------------------------------------
    def log_outcome(
        self,
        val_metric: Optional[float],
        status: str,
        description: str,
        memory_gb: Optional[float] = None,
        commit_hash: str = "pending",
    ) -> int:
        """Log an experiment outcome and return the experiment_id."""
        return rfs.log_experiment(
            commit_hash=commit_hash,
            val_bpb=val_metric,
            memory_gb=memory_gb,
            status=status,
            description=description,
            agent_run_id=self.agent_run_id,
            conn=self.client.get_connection(),
        )

    def log_retrieval(
        self,
        event_type: str,
        query: str,
        retrieved_content: str,
        experiment_id: Optional[int] = None,
    ) -> int:
        """Log a retrieval event associated with this session."""
        return rfs.log_retrieval_event(
            event_type=event_type,
            query=query,
            retrieved_content=retrieved_content,
            agent_run_id=self.agent_run_id,
            experiment_id=experiment_id,
            conn=self.client.get_connection(),
        )


class ResearchFSClient:
    """Lightweight client for ResearchFS-backed auto-research agents.

    Example usage:

        client = ResearchFSClient(db_path="researchfs.db")
        session = client.start_session(run_id=None, baseline_metric=2.18)

        decision = session.suggest_query(history, brief, llm_fn)
        if decision["search"]:
            pack, ids = session.retrieve_context(decision["query"])

        # ... run experiment ...

        exp_id = session.log_outcome(val_metric=2.15, status="keep",
                                     description="Tuned Muon LR")
        session.log_retrieval("agent_search", decision["query"], pack[:2000],
                              experiment_id=exp_id)
        client.close()
    """

    def __init__(self, db_path: str = "researchfs.db") -> None:
        self.db_path = db_path
        self._conn: Optional[object] = None
        self._close_on_done = True

    def get_connection(self):
        """Return a database connection, creating one if necessary."""
        if self._conn is None:
            # Backward compatibility: some test mocks replace get_connection
            # with a zero-argument function. Try with the path first, then
            # fall back to the legacy signature.
            try:
                self._conn = db_module.get_connection(self.db_path)
            except TypeError:
                self._conn = db_module.get_connection()
            db_module.initialize_db(self._conn)
        return self._conn

    def start_session(
        self,
        mode: str = "loopgraph",
        model: str = "unknown",
        baseline_metric: Optional[float] = None,
        run_id: Optional[int] = None,
    ) -> ResearchFSSession:
        """Start a new research session.

        If `run_id` is None, a new agent_run row is created and its id is used.
        """
        if run_id is None:
            run_id = rfs.start_agent_run(mode=mode, model=model)
        return ResearchFSSession(self, run_id, baseline_metric=baseline_metric)

    def close(self) -> None:
        """Close any connection held by the client."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ResearchFSClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
