"""Retrieval and logging API for ResearchFS v0."""

import json
from typing import Optional

from researchfs.db import get_connection


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
            ORDER BY m.category, m.name
            """
        )
        methods = cur.fetchall()

        lines = ["# Method Pack: Autoresearch Optimization Catalog\n"]
        for method in methods:
            lines.append(f"## {method['name']} ({method['category']})")
            lines.append(f"**Description:** {method['description']}")
            lines.append(f"**Implementation hint:** {method['implementation_hint']}")
            if method["risks"]:
                lines.append(f"**Risks:** {method['risks']}")

            # Source provenance
            cur.execute(
                """
                SELECT p.title, p.url, c.section, c.content
                FROM method_sources ms
                JOIN chunks c ON ms.chunk_id = c.chunk_id
                JOIN papers p ON c.paper_id = p.paper_id
                WHERE ms.method_id = ?
                """,
                (method["method_id"],),
            )
            sources = cur.fetchall()
            if sources:
                lines.append("**Sources:**")
                for src in sources:
                    lines.append(f"- {src['title']} ({src['url']}) [{src['section']}]")
            lines.append("")

        return "\n".join(lines)
    finally:
        if close:
            conn.close()


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
        conn.commit()
        return cur.lastrowid
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
