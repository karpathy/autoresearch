"""Retrieval and logging API for ResearchFS v0."""

import re
from typing import Optional

from researchfs.db import get_connection


def _fetch_all_methods(cur) -> list[dict]:
    """Fetch every method row in the canonical retrieval shape."""
    cur.execute(
        """
        SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
        FROM methods m
        """
    )
    return [dict(m) for m in cur.fetchall()]


def _fts_phrase_query(text: str | None, joiner: str = " ") -> str:
    """Build an FTS5-safe query from free text terms."""
    terms = re.findall(r"[A-Za-z0-9]+", text or "")
    return joiner.join(f'"{term}"' for term in terms)


def _get_empirical_stats(method_id: int, conn, baselines: dict) -> dict:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT e.experiment_id, e.status, e.val_bpb, e.agent_run_id
        FROM experiments e
        JOIN retrieval_events re ON e.experiment_id = re.experiment_id
        JOIN retrieved_methods rm ON re.event_id = rm.event_id
        WHERE rm.method_id = ? AND e.status IN ('keep', 'discard', 'crash')
        """,
        (method_id,),
    )
    exps = cur.fetchall()

    keeps = 0
    discards = 0
    crashes = 0
    deltas = []

    for exp in exps:
        status = exp["status"]
        if status == "keep":
            keeps += 1
            val_bpb = exp["val_bpb"]
            run_id = exp["agent_run_id"]
            if val_bpb is not None and run_id is not None:
                if run_id not in baselines:
                    cur.execute(
                        """
                        SELECT val_bpb FROM experiments
                        WHERE agent_run_id = ? AND commit_hash = 'baseline' AND val_bpb IS NOT NULL
                        LIMIT 1
                        """,
                        (run_id,),
                    )
                    row = cur.fetchone()
                    baselines[run_id] = row["val_bpb"] if row else None
                base_bpb = baselines[run_id]
                if base_bpb is not None:
                    deltas.append(val_bpb - base_bpb)
        elif status == "discard":
            discards += 1
        elif status == "crash":
            crashes += 1

    total_runs = keeps + discards + crashes
    success_rate = keeps / total_runs if total_runs > 0 else 0.5
    avg_bpb_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return {
        "total_runs": total_runs,
        "keeps": keeps,
        "discards": discards,
        "crashes": crashes,
        "success_rate": success_rate,
        "avg_bpb_delta": avg_bpb_delta,
    }


def _build_method_pack(
    methods: list, conn, header: str, sort_key=None, max_chars: int | None = None
) -> tuple[str, list[int]]:
    """Helper to calculate stats, sort methods, format them, and return (pack_str, method_ids)."""
    baselines = {}
    methods_with_stats = []
    for method in methods:
        stats = _get_empirical_stats(method["method_id"], conn, baselines)
        method_with_stat = dict(method)
        method_with_stat["stats"] = stats
        methods_with_stats.append(method_with_stat)

    # Sorting: Sort methods by success rate descending, then keeps descending, then category, then name
    # (unrun methods have default success rate 0.5 for sorting)
    if sort_key is None:
        def sort_key(m):
            stats = m["stats"]
            success_rate = stats["success_rate"]
            keeps = stats["keeps"]
            category = m["category"] or ""
            name = m["name"] or ""
            return (-success_rate, -keeps, category, name)

    methods_with_stats.sort(key=sort_key)

    # Pre-render each method's contribution so we can enforce a character budget.
    rendered_methods: list[tuple[int, list[str]]] = []
    for m in methods_with_stats:
        method_lines: list[str] = []
        method_lines.append(f"## {m['name']} ({m['category']})")
        method_lines.append(f"**Description:** {m['description']}")
        method_lines.append(f"**Implementation hint:** {m['implementation_hint']}")
        if m["risks"]:
            method_lines.append(f"**Risks:** {m['risks']}")

        # Empirical Swarm Stats
        stats = m["stats"]
        if stats["total_runs"] > 0:
            method_lines.append(
                f"**Empirical Swarm Stats:** {stats['total_runs']} runs, "
                f"{stats['success_rate']:.0%} success ({stats['keeps']} keeps, "
                f"{stats['discards']} discards, {stats['crashes']} crashes), "
                f"avg {stats['avg_bpb_delta']:+.6f} BPB."
            )
        else:
            method_lines.append("**Empirical Swarm Stats:** No runs yet.")

        # Source provenance
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.title, p.url, c.section, c.content
            FROM method_sources ms
            JOIN chunks c ON ms.chunk_id = c.chunk_id
            JOIN papers p ON c.paper_id = p.paper_id
            WHERE ms.method_id = ?
            """,
            (m["method_id"],),
        )
        sources = cur.fetchall()
        if sources:
            method_lines.append("**Sources:**")
            for src in sources:
                method_lines.append(f"- {src['title']} ({src['url']}) [{src['section']}]")
        method_lines.append("")

        rendered = "\n".join(method_lines)
        rendered_methods.append((m["method_id"], rendered))

    lines = [header]
    sorted_ids = []
    current_len = len(header)
    omitted = 0
    sep_len = len("\n")
    budget = max_chars

    for method_id, rendered in rendered_methods:
        if budget is not None:
            add_len = sep_len + len(rendered)
            if current_len + add_len > budget:
                omitted += 1
                continue
            current_len += add_len
        sorted_ids.append(method_id)
        lines.append(rendered)

    if omitted:
        lines.append(f"... and {omitted} more methods omitted")

    return "\n".join(lines), sorted_ids


def get_method_pack(conn=None) -> str:
    """Return a formatted method pack string for agent prompts."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
            FROM methods m
            """
        )
        methods = cur.fetchall()
        method_list = [dict(m) for m in methods]
        pack_str, _ = _build_method_pack(
            method_list,
            conn,
            "# Method Pack: Autoresearch Optimization Catalog\n"
        )
        return pack_str
    finally:
        if close:
            conn.close()


def _search_methods_hybrid(
    query: str, conn
) -> tuple[list[dict], list[int], dict[int, float]]:
    """Run FTS5 hybrid search and return (method_rows, ordered_ids, method_ranks).

    method_ranks maps method_id to the best (lowest) FTS5 rank found either
    directly in methods_fts or indirectly via chunks_fts + method_sources.
    """
    cur = conn.cursor()
    fts_query = _fts_phrase_query(query)
    if not fts_query:
        methods = _fetch_all_methods(cur)
        return methods, [m["method_id"] for m in methods], {}

    method_ranks: dict[int, float] = {}

    # Method-direct matches
    cur.execute(
        "SELECT rowid, rank FROM methods_fts WHERE methods_fts MATCH ?",
        (fts_query,),
    )
    for row in cur.fetchall():
        method_id = row["rowid"]
        rank = row["rank"]
        method_ranks[method_id] = min(method_ranks.get(method_id, float("inf")), rank)

    # Chunk-provenance matches: chunk -> method_sources -> method_id
    cur.execute(
        """
        SELECT chunks_fts.rowid, chunks_fts.rank, ms.method_id
        FROM chunks_fts
        JOIN method_sources ms ON chunks_fts.rowid = ms.chunk_id
        WHERE chunks_fts MATCH ?
        """,
        (fts_query,),
    )
    for row in cur.fetchall():
        method_id = row["method_id"]
        rank = row["rank"]
        method_ranks[method_id] = min(method_ranks.get(method_id, float("inf")), rank)

    if not method_ranks:
        return [], [], {}

    ids = list(method_ranks.keys())
    placeholders = ",".join("?" * len(ids))
    cur.execute(
        f"""
        SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
        FROM methods m
        WHERE m.method_id IN ({placeholders})
        """,
        ids,
    )
    methods = [dict(m) for m in cur.fetchall()]
    return methods, ids, method_ranks


def _get_method_fatigue_scores(agent_run_id: int | None, conn) -> dict[int, float]:
    """Return per-method fatigue multipliers for the current agent run.

    - Methods retrieved for experiments that were kept: 1.1
    - Methods retrieved for experiments that were discarded/crashed: 0.5
    - Default (no run history): 1.0
    - Repeated retrieval beyond the first: additional 0.7^(uses-1) decay
    """
    if agent_run_id is None:
        return {}

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            rm.method_id,
            COUNT(*) AS uses,
            SUM(CASE WHEN e.status = 'keep' THEN 1 ELSE 0 END) AS keeps,
            SUM(CASE WHEN e.status IN ('discard', 'crash') THEN 1 ELSE 0 END) AS bad
        FROM retrieval_events re
        JOIN retrieved_methods rm ON re.event_id = rm.event_id
        LEFT JOIN experiments e ON re.experiment_id = e.experiment_id
        WHERE re.agent_run_id = ?
        GROUP BY rm.method_id
        """,
        (agent_run_id,),
    )

    fatigue: dict[int, float] = {}
    for row in cur.fetchall():
        method_id = row["method_id"]
        uses = row["uses"]
        keeps = row["keeps"] or 0
        bad = row["bad"] or 0

        if keeps > 0:
            multiplier = 1.1
        elif bad > 0:
            multiplier = 0.5
        else:
            multiplier = 1.0

        if uses > 1:
            multiplier *= 0.7 ** (uses - 1)

        fatigue[method_id] = multiplier

    return fatigue


def _score_methods(
    methods: list[dict],
    conn,
    method_ranks: dict[int, float],
    agent_run_id: int | None = None,
) -> tuple[list[dict], list[int]]:
    """Combine FTS5 rank, empirical success rate, BPB delta, and fatigue."""
    baselines = {}
    scored = []
    fatigue = _get_method_fatigue_scores(agent_run_id, conn)

    if method_ranks:
        ranks = list(method_ranks.values())
        min_rank = min(ranks)
        max_rank = max(ranks)
        rank_range = max_rank - min_rank if max_rank > min_rank else 1.0
    else:
        min_rank = max_rank = 0.0
        rank_range = 1.0

    deltas = []
    for method in methods:
        stats = _get_empirical_stats(method["method_id"], conn, baselines)
        method["stats"] = stats
        deltas.append(stats["avg_bpb_delta"])

    min_delta = min(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    delta_range = max_delta - min_delta if max_delta > min_delta else 0.0

    for method in methods:
        stats = method["stats"]
        rank = method_ranks.get(method["method_id"], max_rank)
        normalized_rank = (rank - min_rank) / rank_range if rank_range else 0.0
        rank_score = 1.0 - normalized_rank

        if delta_range:
            delta_score = (max_delta - stats["avg_bpb_delta"]) / delta_range
        else:
            delta_score = 0.5

        composite = (
            0.35 * stats["success_rate"]
            + 0.25 * rank_score
            + 0.4 * delta_score
        )
        fatigue_multiplier = fatigue.get(method["method_id"], 1.0)
        method["_score"] = composite * fatigue_multiplier
        scored.append(method)

    scored.sort(key=lambda m: (-m["_score"], m["category"] or "", m["name"] or ""))
    ids = [m["method_id"] for m in scored]
    return scored, ids


def search_methods_and_ids(
    query: str,
    conn=None,
    max_chars: int | None = None,
    agent_run_id: int | None = None,
) -> tuple[str, list[int]]:
    """Search for methods matching the query keywords and return a tuple of (formatted_method_pack, list_of_method_ids)."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        methods, _, method_ranks = _search_methods_hybrid(query, conn)

        if not methods:
            main_pack_str = f"# Method Pack Search Results\n\nNo methods found matching query: '{query}'"
            main_ids = []
        elif not method_ranks:
            # Empty query fallback: preserve default success-rate ordering.
            main_pack_str, main_ids = _build_method_pack(
                methods,
                conn,
                "# Method Pack: Autoresearch Optimization Catalog\n",
                max_chars=max_chars,
            )
        else:
            ordered_methods, main_ids = _score_methods(
                methods, conn, method_ranks, agent_run_id=agent_run_id
            )
            main_pack_str, main_ids = _build_method_pack(
                ordered_methods,
                conn,
                f"# Method Pack Search Results for: '{query}'\n",
                sort_key=lambda m: (-m["_score"], m["category"] or "", m["name"] or ""),
                max_chars=max_chars,
            )

        if query and query.strip():
            cur = conn.cursor()
            cur.execute(
                """
                SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
                FROM methods m
                WHERE m.category IN ('optimizer', 'regularization')
                """
            )
            general_methods = [dict(row) for row in cur.fetchall()]

            # Remove any general methods that are already present in the search results
            search_result_ids = {m["method_id"] for m in methods}
            general_methods = [m for m in general_methods if m["method_id"] not in search_result_ids]

            if general_methods:
                remaining_max_chars = None
                if max_chars is not None:
                    remaining_max_chars = max(0, max_chars - len(main_pack_str))

                if remaining_max_chars is None or remaining_max_chars > 0:
                    gen_pack_str, gen_ids = _build_method_pack(
                        general_methods,
                        conn,
                        "\n# General Optimization & Hyperparameter Tuning Methods\n",
                        max_chars=remaining_max_chars,
                    )
                    main_pack_str += gen_pack_str
                    for gid in gen_ids:
                        if gid not in main_ids:
                            main_ids.append(gid)

        return main_pack_str, main_ids
    finally:
        if close:
            conn.close()


def search_methods(query: str, conn=None, max_chars: int | None = None) -> str:
    """Search for methods matching the query keywords and return a formatted method pack."""
    pack, _ = search_methods_and_ids(query, conn, max_chars=max_chars)
    return pack
def get_experiment_brief(name: str = "nanochat_autoresearch", conn=None) -> str:
    """Return the named experiment brief content."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM experiment_briefs WHERE name = ?",
            (name,),
        )
        row = cur.fetchone()
        if row is None:
            return ""
        return row["content"]
    finally:
        if close:
            conn.close()


def start_agent_run(mode: str, model: Optional[str] = None, conn=None) -> int:
    """Start a new agent run and return its ID."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO agent_runs (mode, model) VALUES (?, ?)",
            (mode, model),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        if close:
            conn.close()


def finish_agent_run(agent_run_id: int, conn=None) -> None:
    """Mark an agent run as finished."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE agent_runs SET finished_at = CURRENT_TIMESTAMP WHERE agent_run_id = ?",
            (agent_run_id,),
        )
        conn.commit()
    finally:
        if close:
            conn.close()


def log_experiment(
    commit_hash: str,
    val_bpb: Optional[float],
    memory_gb: Optional[float],
    status: str,
    description: str,
    agent_run_id: Optional[int] = None,
    conn=None,
) -> int:
    """Log a ratchet-loop experiment outcome and return the experiment_id."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO experiments
            (agent_run_id, commit_hash, val_bpb, memory_gb, status, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (agent_run_id, commit_hash, val_bpb, memory_gb, status, description),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        if close:
            conn.close()


def update_experiment(
    experiment_id: int,
    val_bpb: Optional[float] = None,
    memory_gb: Optional[float] = None,
    status: Optional[str] = None,
    description: Optional[str] = None,
    conn=None,
) -> None:
    """Update an existing experiment record with final metrics."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        fields = []
        params = []
        if val_bpb is not None:
            fields.append("val_bpb = ?")
            params.append(val_bpb)
        if memory_gb is not None:
            fields.append("memory_gb = ?")
            params.append(memory_gb)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if description is not None:
            fields.append("description = ?")
            params.append(description)
        if not fields:
            return
        params.append(experiment_id)
        cur.execute(
            f"UPDATE experiments SET {', '.join(fields)} WHERE experiment_id = ?",
            params,
        )
        conn.commit()
    finally:
        if close:
            conn.close()

def log_retrieval_event(
    event_type: str,
    query: str,
    retrieved_content: str,
    agent_run_id: Optional[int] = None,
    experiment_id: Optional[int] = None,
    conn=None,
    method_ids: Optional[list[int]] = None,
) -> int:
    """Log a retrieval event for token/cost auditing."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO retrieval_events
            (agent_run_id, experiment_id, event_type, query, retrieved_content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent_run_id, experiment_id, event_type, query, retrieved_content),
        )
        event_id = cur.lastrowid

        if method_ids:
            unique_method_ids = list(dict.fromkeys(method_ids))
            for method_id in unique_method_ids:
                cur.execute(
                    """
                    INSERT INTO retrieved_methods (event_id, method_id)
                    VALUES (?, ?)
                    """,
                    (event_id, method_id),
                )

        conn.commit()
        return event_id
    finally:
        if close:
            conn.close()

def log_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cost: Optional[float],
    experiment_id: Optional[int] = None,
    conn=None,
) -> int:
    """Log token usage for an experiment step."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO token_usage
            (experiment_id, prompt_tokens, completion_tokens, cost)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, prompt_tokens, completion_tokens, cost),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        if close:
            conn.close()


def get_agent_run_summary(agent_run_id: int, conn=None) -> dict:
    """Return aggregate metrics for an agent run."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_experiments,
                SUM(CASE WHEN status = 'keep' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN status = 'crash' THEN 1 ELSE 0 END) AS crashed,
                SUM(CASE WHEN status = 'discard' THEN 1 ELSE 0 END) AS discarded,
                MIN(val_bpb) AS best_bpb
            FROM experiments
            WHERE agent_run_id = ?
            """,
            (agent_run_id,),
        )
        exp_summary = dict(cur.fetchone())

        cur.execute(
            """
            SELECT COUNT(*) AS retrieval_count
            FROM retrieval_events
            WHERE agent_run_id = ?
            """,
            (agent_run_id,),
        )
        retrieval_count = cur.fetchone()["retrieval_count"]

        cur.execute(
            """
            SELECT SUM(prompt_tokens) AS prompt_tokens,
                   SUM(completion_tokens) AS completion_tokens,
                   SUM(cost) AS total_cost
            FROM token_usage
            WHERE experiment_id IN (
                SELECT experiment_id FROM experiments WHERE agent_run_id = ?
            )
            """,
            (agent_run_id,),
        )
        token_summary = dict(cur.fetchone())

        return {
            "agent_run_id": agent_run_id,
            "experiments": exp_summary,
            "retrieval_count": retrieval_count,
            "tokens": token_summary,
        }
    finally:
        if close:
            conn.close()
