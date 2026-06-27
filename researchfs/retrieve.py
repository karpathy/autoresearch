"""Retrieval and logging API for ResearchFS v0."""

import json
from typing import Optional

from researchfs.db import get_connection


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


def _build_method_pack(methods: list, conn, header: str) -> tuple[str, list[int]]:
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
    def sort_key(m):
        stats = m["stats"]
        success_rate = stats["success_rate"]
        keeps = stats["keeps"]
        category = m["category"] or ""
        name = m["name"] or ""
        return (-success_rate, -keeps, category, name)

    methods_with_stats.sort(key=sort_key)

    lines = [header]
    sorted_ids = []
    for m in methods_with_stats:
        sorted_ids.append(m["method_id"])
        lines.append(f"## {m['name']} ({m['category']})")
        lines.append(f"**Description:** {m['description']}")
        lines.append(f"**Implementation hint:** {m['implementation_hint']}")
        if m["risks"]:
            lines.append(f"**Risks:** {m['risks']}")

        # Empirical Swarm Stats
        stats = m["stats"]
        if stats["total_runs"] > 0:
            lines.append(
                f"**Empirical Swarm Stats:** {stats['total_runs']} runs, "
                f"{stats['success_rate']:.0%} success ({stats['keeps']} keeps, "
                f"{stats['discards']} discards, {stats['crashes']} crashes), "
                f"avg {stats['avg_bpb_delta']:+.6f} BPB."
            )
        else:
            lines.append("**Empirical Swarm Stats:** No runs yet.")

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
            lines.append("**Sources:**")
            for src in sources:
                lines.append(f"- {src['title']} ({src['url']}) [{src['section']}]")
        lines.append("")

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


def search_methods_and_ids(query: str, conn=None) -> tuple[str, list[int]]:
    """Search for methods matching the query keywords and return a tuple of (formatted_method_pack, list_of_method_ids)."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            # Fall back to returning all methods
            cur.execute(
                """
                SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
                FROM methods m
                """
            )
            methods = cur.fetchall()
            method_list = [dict(m) for m in methods]
            pack_str, ids = _build_method_pack(
                method_list,
                conn,
                "# Method Pack: Autoresearch Optimization Catalog\n"
            )
            return pack_str, ids

        conditions = []
        params = []
        for kw in keywords:
            conditions.append("(m.name LIKE ? OR m.description LIKE ? OR m.category LIKE ?)")
            like_pattern = f"%{kw}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        sql = f"""
            SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
            FROM methods m
            WHERE {" AND ".join(conditions)}
        """
        cur.execute(sql, params)
        methods = cur.fetchall()

        if not methods:
            return f"# Method Pack Search Results\n\nNo methods found matching query: '{query}'", []

        method_list = [dict(m) for m in methods]
        pack_str, ids = _build_method_pack(
            method_list,
            conn,
            f"# Method Pack Search Results for: '{query}'\n"
        )
        return pack_str, ids
    finally:
        if close:
            conn.close()


def search_methods(query: str, conn=None) -> str:
    """Search for methods matching the query keywords and return a formatted method pack."""
    pack, _ = search_methods_and_ids(query, conn)
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
