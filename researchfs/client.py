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
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import researchfs.db as db_module
from researchfs import retrieve as rfs


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English/code text."""
    return max(1, len(text) // 4)


def _clean_query(query: str) -> str:
    """Normalize a query string for cache lookup."""
    return " ".join(query.lower().split())

def _validate_optional_non_negative_int(value: object, name: str) -> Optional[int]:
    """Validate an optional non-negative integer parameter."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be None or a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be None or a non-negative integer")
    return value



def _query_has_content(query: str) -> bool:
    """Return True if the query contains at least one alphanumeric character."""
    return any(c.isalnum() for c in query)


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


def _recent_distinct_queries(query_history: str, count: int) -> List[str]:
    """Return the most recent `count` distinct query strings from `query_history`."""
    if count <= 0:
        return []
    queries: List[str] = []
    seen: set[str] = set()
    for line in reversed(query_history.splitlines()):
        if "Query:" not in line:
            continue
        prefix = line.split("Query:", 1)[1]
        parts = prefix.split("-> Outcome:", 1)
        if not parts:
            continue
        raw = parts[0].strip()
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        cleaned = _clean_query(raw)
        if raw and cleaned and cleaned not in seen:
            seen.add(cleaned)
            queries.append(raw)
            if len(queries) == count:
                break
    queries.reverse()
    return queries


def _assemble_retrieval_prompt(
    session: ResearchFSSession,
    results_content: str,
    experiment_brief: str,
    query_history: Optional[str] = None,
) -> str:
    """Assemble prompt to ask the agent for a search query decision."""
    prompt = (
        "You are an expert machine learning researcher agent.\n"
        f"Your goal is to optimize {session.task_target} to {session.objective} "
        f"on {session.task_name}.\n\n"
        f"Experiment Brief:\n{experiment_brief}\n\n"
        f"Here is the history of previous experiments:\n{results_content}\n\n"
    )
    if query_history:
        prompt += (
            "Here is the history of prior search queries and their experiment "
            f"outcomes:\n{query_history}\n\n"
        )
        recent = _recent_distinct_queries(query_history, session.recent_areas_count)
        if recent:
            prompt += (
                "You have recently explored: "
                + ", ".join(recent)
                + ". Prefer a different optimization area unless one of these "
                "still has clear, unexplored potential.\n\n"
            )

    prompt += (
        "Avoid repeating queries that were recently used or already reused multiple "
        "times in a row. Prefer exploring a different optimization area unless a "
        "prior query has clear, unexplored potential.\n\n"
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
        task_target: str = "the training script `train.py`",
        objective: str = "minimize the validation bits per byte (val_bpb)",
        task_name: str = "pretraining task",
        recent_areas_count: int = 3,
    ) -> None:
        self.client = client
        self.agent_run_id = agent_run_id
        self.baseline_metric = baseline_metric
        self.task_target = task_target
        self.objective = objective
        self.task_name = task_name
        self.recent_areas_count = recent_areas_count
        self.search_cache: Dict[Tuple[str, Optional[int]], Tuple[str, List[int]]] = {}
        self.prompt_tokens_accum = 0
        self.completion_tokens_accum = 0

    def _clear_search_cache(self) -> None:
        """Drop cached method packs whose rankings may now be stale."""
        self.search_cache.clear()


    # ------------------------------------------------------------------
    # Query reuse and history helpers
    # ------------------------------------------------------------------
    def _query_outcomes(self) -> List[Dict[str, object]]:
        """Return per-query empirical outcome statistics for this run."""
        if self.baseline_metric is None:
            return []
        conn = self.client.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT re.query, e.status, e.val_bpb, e.experiment_id
            FROM retrieval_events re
            LEFT JOIN experiments e ON re.experiment_id = e.experiment_id
            WHERE re.agent_run_id = ?
              AND re.event_type IN ('agent_search', 'agent_search_reuse')
            ORDER BY re.event_id ASC
            """,
            (self.agent_run_id,),
        )
        rows = cur.fetchall()

        by_query: Dict[str, Dict[str, object]] = {}
        event_queries: List[str] = []
        for row in rows:
            raw_query = str(row["query"] or "").strip()
            query_key = _clean_query(raw_query)
            if not query_key:
                continue
            status = row["status"]
            val_bpb = row["val_bpb"]
            experiment_id = row["experiment_id"]

            event_queries.append(query_key)
            stats = by_query.setdefault(
                query_key,
                {
                    "query": raw_query,
                    "kept_count": 0,
                    "total_count": 0,
                    "best_delta": 0.0,
                    "avg_delta": 0.0,
                    "last_experiment_id": None,
                    "consecutive_uses": 0,
                    "_deltas": [],
                },
            )
            stats["query"] = raw_query or stats["query"]
            stats["total_count"] += 1
            if experiment_id is not None and (
                stats["last_experiment_id"] is None
                or experiment_id > stats["last_experiment_id"]
            ):
                stats["last_experiment_id"] = experiment_id

            if status == "keep" and val_bpb is not None:
                stats["kept_count"] += 1
                stats["_deltas"].append(self.baseline_metric - val_bpb)

        global_max_exp_id: Optional[int] = None
        for stats in by_query.values():
            last_id = stats["last_experiment_id"]
            if last_id is not None:
                if global_max_exp_id is None or last_id > global_max_exp_id:
                    global_max_exp_id = last_id
            deltas = stats.pop("_deltas", [])
            if deltas:
                stats["best_delta"] = max(deltas)
                stats["avg_delta"] = sum(deltas) / len(deltas)

        if event_queries:
            for query, stats in by_query.items():
                consecutive = 0
                for q in reversed(event_queries):
                    if q == query:
                        consecutive += 1
                    else:
                        break
                stats["consecutive_uses"] = consecutive

        return list(by_query.values())

    @staticmethod
    def _score_outcome(
        outcome: Dict[str, object], global_max_exp_id: Optional[int]
    ) -> float:
        """Compute the empirical reuse score for a single query outcome."""
        if outcome["kept_count"] <= 0 or outcome["best_delta"] <= 0:
            return 0.0
        score = float(outcome["best_delta"]) * (
            outcome["kept_count"] / max(outcome["total_count"], 1)
        )
        if (
            outcome["last_experiment_id"] is not None
            and outcome["last_experiment_id"] == global_max_exp_id
        ):
            score *= 0.5
        if outcome["consecutive_uses"] >= 2:
            score = 0.0
        return score

    def _pick_reusable_query(self) -> Optional[str]:
        """Score candidate queries and return the best reusable query, if any."""
        outcomes = self._query_outcomes()
        if not outcomes:
            return None

        global_max_exp_id: Optional[int] = None
        for outcome in outcomes:
            last_id = outcome["last_experiment_id"]
            if last_id is not None:
                if global_max_exp_id is None or last_id > global_max_exp_id:
                    global_max_exp_id = last_id

        best_query: Optional[str] = None
        best_score = 0.0
        for outcome in outcomes:
            score = self._score_outcome(outcome, global_max_exp_id)
            if score > best_score:
                best_score = score
                best_query = outcome["query"]

        if best_score > 0.001:
            return best_query
        return None

    def get_reusable_query(self) -> Optional[str]:
        """Return the best reusable query according to the empirical scorer."""
        return self._pick_reusable_query()

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
    def _make_reuse_decision(self, reused_query: str) -> Dict[str, object]:
        """Build the decision dict when reusing a prior query."""
        outcomes = self._query_outcomes()
        global_max_exp_id: Optional[int] = None
        for outcome in outcomes:
            last_id = outcome["last_experiment_id"]
            if last_id is not None:
                if global_max_exp_id is None or last_id > global_max_exp_id:
                    global_max_exp_id = last_id
        score = 0.0
        for outcome in outcomes:
            if outcome["query"] == reused_query:
                score = self._score_outcome(outcome, global_max_exp_id)
                break
        return {
            "search": True,
            "query": reused_query,
            "reason": f"Reusing prior query '{reused_query}' with empirical score {score:.6f}.",
            "reused": True,
        }

    def _parse_suggestion_response(self, query_res: str) -> Dict[str, object]:
        """Parse a raw LLM response into a suggestion decision."""
        parsed = _extract_json_block(query_res)
        if parsed is not None and isinstance(parsed, dict):
            search = bool(parsed.get("search", True))
            query = str(parsed.get("query", "")).strip()
            reason = str(parsed.get("reason", ""))
            if not search:
                query = ""
            elif not _query_has_content(query):
                search = False
                query = ""
                reason = "LLM returned an empty or punctuation-only query; skipping search."
            return {
                "search": search,
                "query": query,
                "reason": reason,
                "reused": False,
            }

        # Fallback: treat the raw response as a plain query.
        query = query_res.strip().strip('"').strip("'")
        if not _query_has_content(query):
            return {
                "search": False,
                "query": "",
                "reason": "Fallback query was empty or punctuation-only; skipping search.",
                "reused": False,
            }
        return {
            "search": True,
            "query": query,
            "reason": "JSON parsing failed; treating raw response as query.",
            "reused": False,
        }

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
        reused_query = self._pick_reusable_query()
        if reused_query:
            return self._make_reuse_decision(reused_query)

        query_history = self.get_query_history()
        retrieval_prompt = _assemble_retrieval_prompt(
            self, results_content, experiment_brief, query_history
        )
        self.prompt_tokens_accum += _estimate_tokens(retrieval_prompt)

        query_res = llm_fn(retrieval_prompt)
        self.completion_tokens_accum += _estimate_tokens(query_res)
        return self._parse_suggestion_response(query_res)

    async def suggest_query_async(
        self,
        results_content: str,
        experiment_brief: str,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> Dict[str, object]:
        """Async version of `suggest_query`.

        Mirrors the synchronous method exactly; only the LLM call is awaited.
        """
        reused_query = self._pick_reusable_query()
        if reused_query:
            return self._make_reuse_decision(reused_query)

        query_history = self.get_query_history()
        retrieval_prompt = _assemble_retrieval_prompt(
            self, results_content, experiment_brief, query_history
        )
        self.prompt_tokens_accum += _estimate_tokens(retrieval_prompt)

        query_res = await llm_fn(retrieval_prompt)
        self.completion_tokens_accum += _estimate_tokens(query_res)
        return self._parse_suggestion_response(query_res)

    def retrieve_context(
        self, query: str, max_method_chars: Optional[int] = None
    ) -> Tuple[str, Optional[List[int]]]:
        """Fetch a formatted method pack for `query`, using the in-memory cache.

        Returns:
            (method_pack, method_ids)
        """
        max_method_chars = _validate_optional_non_negative_int(
            max_method_chars, "max_method_chars"
        )
        if not query:
            return "", None
        cache_key = (_clean_query(query), max_method_chars)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        pack, ids = rfs.search_methods_and_ids(
            query,
            conn=self.client.get_connection(),
            max_chars=max_method_chars,
            agent_run_id=self.agent_run_id,
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
        experiment_id = rfs.log_experiment(
            commit_hash=commit_hash,
            val_bpb=val_metric,
            memory_gb=memory_gb,
            status=status,
            description=description,
            agent_run_id=self.agent_run_id,
            conn=self.client.get_connection(),
        )
        self._clear_search_cache()
        return experiment_id

    def log_retrieval(
        self,
        event_type: str,
        query: str,
        retrieved_content: str,
        experiment_id: Optional[int] = None,
        method_ids: Optional[List[int]] = None,
    ) -> int:
        """Log a retrieval event associated with this session."""
        event_id = rfs.log_retrieval_event(
            event_type=event_type,
            query=query,
            retrieved_content=retrieved_content,
            agent_run_id=self.agent_run_id,
            experiment_id=experiment_id,
            conn=self.client.get_connection(),
            method_ids=method_ids,
        )
        self._clear_search_cache()
        return event_id


class ResearchFSClient:
    """Lightweight client for ResearchFS-backed auto-research agents.

    Example usage:

        client = ResearchFSClient()
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

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path
        self._conn: Optional[object] = None
        self._close_on_done = True
        self._conn_initialized = False

    def get_connection(self):
        """Return a database connection, creating one if necessary."""
        if self._conn is None:
            # Backward compatibility: some test mocks replace get_connection
            # with a zero-argument function. Try the configured path first,
            # then fall back to the legacy signature.
            try:
                if self.db_path is None:
                    self._conn = db_module.get_connection()
                else:
                    self._conn = db_module.get_connection(self.db_path)
            except TypeError:
                self._conn = db_module.get_connection()
            try:
                db_module.initialize_db(self._conn)
            except Exception:
                if self._close_on_done and self._conn is not None:
                    self._conn.close()
                self._conn = None
                raise
            self._conn_initialized = True
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
            run_id = rfs.start_agent_run(mode=mode, model=model, conn=self.get_connection())
        return ResearchFSSession(self, run_id, baseline_metric=baseline_metric)

    def close(self) -> None:
        """Close any connection held by the client."""
        if self._conn is not None:
            if self._close_on_done:
                self._conn.close()
            self._conn = None
        self._conn_initialized = False

    def __enter__(self) -> "ResearchFSClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
