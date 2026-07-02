"""Retrieval and logging API for ResearchFS v0."""

import math
import re
import sqlite3
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


LESSON_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
}
MAX_LESSON_MATCHES_PER_METHOD = 3
MAX_LESSON_SEARCH_ROWS = 200

MAX_STANDALONE_LESSONS = 5

def _tokenize_lesson_query(query: str) -> list[str]:
    """Return stable, meaningful tokens for deterministic lesson LIKE matching."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]+", query or ""):
        normalized = token.lower()
        if (
            normalized in seen
            or normalized in LESSON_QUERY_STOPWORDS
            or (len(normalized) < 3 and not normalized.isdigit())
        ):
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tokens


def _lesson_match_score(text: str, tokens: list[str], link_weight: float) -> float:
    """Return a bounded deterministic score for a matching lesson text."""
    token_hits = sum(min(text.count(token), 3) for token in tokens)
    return min(25.0, link_weight + 0.2 * token_hits + 0.1 * len(tokens))


def _find_lesson_method_matches(query: str, conn) -> dict[int, dict]:
    """Find method IDs whose linked lessons match every meaningful query token."""
    tokens = _tokenize_lesson_query(query)
    if not tokens:
        return {}

    lesson_text_sql = """
        LOWER(
            COALESCE(l.label, '') || ' ' ||
            COALESCE(l.content, '') || ' ' ||
            COALESCE(l.outcome, '')
        )
    """
    where = " AND ".join(f"{lesson_text_sql} LIKE ?" for _ in tokens)
    params = [f"%{token}%" for token in tokens]

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            l.lesson_id,
            l.method_id AS direct_method_id,
            l.experiment_id,
            l.label,
            l.content,
            l.outcome,
            l.created_at,
            em.method_id AS experiment_method_id,
            rm.method_id AS retrieved_method_id
        FROM lessons l
        LEFT JOIN experiment_methods em
          ON l.experiment_id = em.experiment_id
        LEFT JOIN retrieval_events re
          ON l.experiment_id = re.experiment_id
        LEFT JOIN retrieved_methods rm
          ON re.event_id = rm.event_id
        WHERE {where}
          AND (
              l.method_id IS NOT NULL
              OR em.method_id IS NOT NULL
              OR rm.method_id IS NOT NULL
          )
        ORDER BY l.created_at DESC, l.lesson_id DESC
        LIMIT ?
        """,
        (*params, MAX_LESSON_SEARCH_ROWS),
    )

    matches: dict[int, dict] = {}
    seen_method_lessons: set[tuple[int, int]] = set()
    for row in cur.fetchall():
        text = " ".join(
            str(part or "").lower()
            for part in (row["label"], row["content"], row["outcome"])
        )
        lesson_tokens = set(re.findall(r"[a-z0-9]+", text))
        if any(token not in lesson_tokens for token in tokens):
            continue
        best_candidates: dict[int, tuple[str, float]] = {}
        if row["direct_method_id"] is not None:
            best_candidates[row["direct_method_id"]] = ("direct", 2.0)
        if row["experiment_method_id"] is not None:
            method_id = row["experiment_method_id"]
            best_candidates.setdefault(method_id, ("experiment", 1.25))
        if row["retrieved_method_id"] is not None:
            method_id = row["retrieved_method_id"]
            best_candidates.setdefault(method_id, ("retrieved", 1.0))

        for method_id, (link_type, link_weight) in best_candidates.items():
            method_lesson_key = (method_id, row["lesson_id"])
            score = _lesson_match_score(text, tokens, link_weight)
            method_match = matches.setdefault(
                method_id,
                {"score": 0.0, "lessons": {}, "lesson_count": 0},
            )
            if method_lesson_key not in seen_method_lessons:
                seen_method_lessons.add(method_lesson_key)
                method_match["score"] = min(50.0, method_match["score"] + score)
                method_match["lesson_count"] += 1

            lesson = method_match["lessons"].get(row["lesson_id"])
            lesson_row = {
                "lesson_id": row["lesson_id"],
                "method_id": row["direct_method_id"],
                "experiment_id": row["experiment_id"],
                "label": row["label"],
                "content": row["content"],
                "outcome": row["outcome"],
                "created_at": row["created_at"],
                "match_link": link_type,
                "match_score": _json_safe_float(score),
            }
            if lesson is None or score > lesson["match_score"]:
                method_match["lessons"][row["lesson_id"]] = lesson_row

    for method_match in matches.values():
        lessons = sorted(
            method_match["lessons"].values(),
            key=lambda lesson: (
                lesson["created_at"] or "",
                lesson["lesson_id"] or 0,
            ),
            reverse=True,
        )
        method_match["lessons"] = lessons[:MAX_LESSON_MATCHES_PER_METHOD]
        method_match["score"] = _json_safe_float(method_match["score"])
        method_match["lesson_count"] = int(method_match["lesson_count"])

    return matches

def _standalone_lesson_matches(query: str, conn, limit: int = MAX_STANDALONE_LESSONS) -> list[dict]:
    """Return bounded lab-level lessons that are not tied to a method or experiment."""
    tokens = _tokenize_lesson_query(query)
    if not tokens or limit <= 0:
        return []

    lesson_text_sql = """
        LOWER(
            COALESCE(label, '') || ' ' ||
            COALESCE(content, '') || ' ' ||
            COALESCE(outcome, '')
        )
    """
    where = " AND ".join(f"{lesson_text_sql} LIKE ?" for _ in tokens)
    rows = conn.execute(
        f"""
        SELECT lesson_id, label, content, outcome, created_at
        FROM lessons
        WHERE method_id IS NULL
          AND experiment_id IS NULL
          AND {where}
        ORDER BY created_at DESC, lesson_id DESC
        LIMIT ?
        """,
        (*(f"%{token}%" for token in tokens), MAX_LESSON_SEARCH_ROWS),
    ).fetchall()

    matches: list[dict] = []
    for row in rows:
        text = " ".join(str(part or "").lower() for part in (row["label"], row["content"], row["outcome"]))
        lesson_tokens = set(re.findall(r"[a-z0-9]+", text))
        if any(token not in lesson_tokens for token in tokens):
            continue
        matches.append(dict(row))
        if len(matches) >= limit:
            break
    return matches


def _format_standalone_lessons(lessons: list[dict]) -> str:
    """Render lab-level lessons as bounded compressed memory, not raw paper text."""
    if not lessons:
        return "## Standalone lab lessons\n- None matched."
    lines = ["## Standalone lab lessons"]
    for row in lessons:
        parts = [f"lesson_id={row['lesson_id']}"]
        if row.get("outcome"):
            parts.append(f"outcome={row['outcome']}")
        lines.append(f"- {row['label']} ({', '.join(parts)}): {_truncate_text(row.get('content'), 300)}")
    return "\n".join(lines)



def _lesson_rank(match: dict) -> float:
    """Convert a lesson match score into a synthetic rank; lower remains better."""
    score = float(match.get("score") or 0.0)
    return 1.0 / (1.0 + score)


FEEDBACK_SCORE_WEIGHT = 0.12
MIN_FEEDBACK_MULTIPLIER = 0.88
MAX_FEEDBACK_MULTIPLIER = 1.12


def _feedback_multiplier(feedback_count: int, avg_rating: float | None) -> float:
    """Return a small bounded scoring multiplier from lab feedback."""
    if feedback_count <= 0 or avg_rating is None:
        return 1.0
    clamped_rating = max(-1.0, min(1.0, float(avg_rating)))
    multiplier = 1.0 + (FEEDBACK_SCORE_WEIGHT * clamped_rating)
    return max(MIN_FEEDBACK_MULTIPLIER, min(MAX_FEEDBACK_MULTIPLIER, multiplier))


def get_method_feedback_summary(method_id: int, conn=None) -> dict:
    """Return aggregate lab feedback for one method."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS feedback_count,
                AVG(rating) AS feedback_avg_rating,
                SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS positive_feedback_count,
                SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS negative_feedback_count,
                SUM(CASE WHEN rating = 0 THEN 1 ELSE 0 END) AS neutral_feedback_count,
                MAX(timestamp) AS last_feedback_at
            FROM method_feedback
            WHERE method_id = ?
            """,
            (method_id,),
        )
        row = cur.fetchone()
        count = row["feedback_count"] if row else 0
        avg_rating = row["feedback_avg_rating"] if count else 0.0
        return {
            "method_id": method_id,
            "feedback_count": count,
            "feedback_avg_rating": avg_rating,
            "positive_feedback_count": row["positive_feedback_count"] or 0 if row else 0,
            "negative_feedback_count": row["negative_feedback_count"] or 0 if row else 0,
            "neutral_feedback_count": row["neutral_feedback_count"] or 0 if row else 0,
            "last_feedback_at": row["last_feedback_at"] if row else None,
        }
    finally:
        if close:
            conn.close()


def _get_empirical_stats(method_id: int, conn, baselines: dict) -> dict:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT e.experiment_id, e.status, e.val_bpb, e.agent_run_id
        FROM experiments e
        JOIN (
            SELECT re.experiment_id
            FROM retrieval_events re
            JOIN retrieved_methods rm ON re.event_id = rm.event_id
            WHERE rm.method_id = ? AND re.experiment_id IS NOT NULL
            UNION
            SELECT em.experiment_id
            FROM experiment_methods em
            WHERE em.method_id = ?
        ) linked ON e.experiment_id = linked.experiment_id
        WHERE e.status IN ('keep', 'discard', 'crash')
        """,
        (method_id, method_id),
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
    feedback_summary = get_method_feedback_summary(method_id, conn)

    return {
        "total_runs": total_runs,
        "keeps": keeps,
        "discards": discards,
        "crashes": crashes,
        "success_rate": success_rate,
        "avg_bpb_delta": avg_bpb_delta,
        "feedback_count": feedback_summary["feedback_count"],
        "feedback_avg_rating": feedback_summary["feedback_avg_rating"],
    }


def _fit_rendered_method_to_budget(rendered: str, max_chars: int) -> str:
    """Shed optional method details before truncating a rendered method entry."""
    if len(rendered) <= max_chars:
        return rendered
    lines = rendered.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for marker in ("**Sources:**", "**Matched lessons:**", "**Lab Feedback:**"):
        if marker in lines:
            compact = "\n".join(lines[: lines.index(marker)]).rstrip()
            if compact and len(compact) <= max_chars:
                return compact
            lines = lines[: lines.index(marker)]
    compact = "\n".join(lines).rstrip()
    if compact and len(compact) <= max_chars:
        return compact
    return _limit_text(compact or rendered, max_chars)


def _build_method_pack(
    methods: list,
    conn,
    header: str,
    sort_key=None,
    max_chars: int | None = None,
    source_types: list[str] | None = None,
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
        if stats["feedback_count"] > 0:
            method_lines.append(
                f"**Lab Feedback:** {stats['feedback_count']} ratings, "
                f"avg {stats['feedback_avg_rating']:+.2f}."
            )

        matched_lessons = m.get("_lesson_matches") or []
        if matched_lessons:
            method_lines.append("**Matched lessons:**")
            for lesson in matched_lessons[:MAX_LESSON_MATCHES_PER_METHOD]:
                method_lines.append(_format_lesson_summary(lesson))

        # Source provenance
        cur = conn.cursor()
        source_join_sql = ""
        source_filter_sql, source_filter_params = _source_type_filter_sql(
            source_types, "pes.source_type"
        )
        if source_types:
            source_join_sql = (
                "JOIN paper_extraction_status pes ON pes.paper_id = c.paper_id"
            )
        cur.execute(
            f"""
            SELECT p.title, p.url, c.section, c.content
            FROM method_sources ms
            JOIN chunks c ON ms.chunk_id = c.chunk_id
            JOIN papers p ON c.paper_id = p.paper_id
            {source_join_sql}
            WHERE ms.method_id = ?
            {source_filter_sql}
            """,
            (m["method_id"], *source_filter_params),
        )
        sources = cur.fetchall()
        if sources:
            method_lines.append("**Sources:**")
            for src in sources:
                method_lines.append(f"- {src['title']} ({src['url']}) [{src['section']}]")
        method_lines.append("")

        rendered = "\n".join(method_lines)
        rendered_methods.append((m["method_id"], rendered))

    if max_chars is not None and max_chars <= 0:
        return "", []

    if max_chars is not None and len(header) > max_chars:
        return header[:max_chars], []

    lines = [header]
    sorted_ids = []
    included_add_lens: list[int] = []
    current_len = len(header)
    omitted = 0
    sep_len = len("\n")
    budget = max_chars

    for method_id, rendered in rendered_methods:
        add_len = sep_len + len(rendered)
        if budget is not None and current_len + add_len > budget:
            omitted += 1
            continue
        current_len += add_len
        sorted_ids.append(method_id)
        included_add_lens.append(add_len)
        lines.append(rendered)

    if omitted:
        marker = f"... and {omitted} more methods omitted"
        marker_len = sep_len + len(marker)
        while budget is not None and len(lines) > 2 and len("\n".join([*lines, marker])) > budget:
            lines.pop()
            sorted_ids.pop()
            current_len -= included_add_lens.pop()
            omitted += 1
            marker = f"... and {omitted} more methods omitted"
            marker_len = sep_len + len(marker)
        if budget is not None and len(lines) == 2 and len("\n".join([*lines, marker])) > budget:
            max_render_len = budget - len(lines[0]) - (2 * sep_len) - len(marker)
            if max_render_len > 0:
                lines[1] = _fit_rendered_method_to_budget(lines[1], max_render_len)
            else:
                lines.pop()
                sorted_ids.pop()
                current_len = len(lines[0])
        if budget is None or len("\n".join([*lines, marker])) <= budget:
            lines.append(marker)

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

    method_ranks maps method_id to the best (lowest) rank found directly in
    methods_fts, indirectly via chunks_fts + method_sources, or through
    deterministic lesson text matches linked to methods.
    """
    cur = conn.cursor()
    fts_query = _fts_phrase_query(query)
    if not fts_query:
        methods = _fetch_all_methods(cur)
        return methods, [m["method_id"] for m in methods], {}

    method_ranks: dict[int, float] = {}
    lesson_matches: dict[int, dict] = {}

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

    lesson_matches = _find_lesson_method_matches(query, conn)
    for method_id, match in lesson_matches.items():
        rank = _lesson_rank(match)
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
    for method in methods:
        match = lesson_matches.get(method["method_id"])
        if not match:
            continue
        method["_lesson_matches"] = match["lessons"]
        method["_lesson_match_count"] = min(999, match["lesson_count"])
        method["_lesson_match_score"] = _json_safe_float(match["score"])
    return methods, ids, method_ranks


def _get_method_fatigue_scores(agent_run_id: int | None, conn) -> dict[int, float]:
    """Return per-method fatigue multipliers for the current agent run.

    Only agent_search and agent_search_reuse events affect fatigue:
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
          AND re.event_type IN ('agent_search', 'agent_search_reuse')
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

def _json_safe_float(value, default: float | None = 0.0) -> float | None:
    """Return a finite float for JSON encoding, or a default for missing/non-finite values."""
    if value is None:
        return default
    numeric = float(value)
    if not math.isfinite(numeric):
        return default
    return numeric


def _json_safe_stats(stats: dict) -> dict:
    """Copy empirical stats into a JSON-safe, stable shape for score explanations."""
    return {
        "total_runs": int(stats["total_runs"]),
        "keeps": int(stats["keeps"]),
        "discards": int(stats["discards"]),
        "crashes": int(stats["crashes"]),
        "success_rate": _json_safe_float(stats["success_rate"]),
        "avg_bpb_delta": _json_safe_float(stats["avg_bpb_delta"]),
        "feedback_count": int(stats["feedback_count"]),
        "feedback_avg_rating": _json_safe_float(stats["feedback_avg_rating"]),
    }


def _attach_score_details(
    method: dict,
    stats: dict,
    *,
    base_score: float,
    final_score: float,
    rank_score: float,
    delta_score: float,
    fatigue_multiplier: float,
    feedback_multiplier: float,
    method_rank: float | None,
    normalized_rank: float | None,
) -> None:
    """Attach bounded, JSON-safe retrieval score details to one method row."""
    safe_stats = _json_safe_stats(stats)
    lesson_match_count = min(999, int(method.get("_lesson_match_count") or 0))
    lesson_match_score = min(50.0, float(method.get("_lesson_match_score") or 0.0))
    method["score_details"] = {
        "method_id": int(method["method_id"]),
        "name": method["name"],
        "category": method["category"],
        "final_score": _json_safe_float(final_score),
        "base_score": _json_safe_float(base_score),
        "composite_score": _json_safe_float(base_score),
        "rank_score": _json_safe_float(rank_score),
        "delta_score": _json_safe_float(delta_score),
        "success_rate_contribution": _json_safe_float(0.35 * stats["success_rate"]),
        "fatigue_multiplier": _json_safe_float(fatigue_multiplier),
        "feedback_multiplier": _json_safe_float(feedback_multiplier),
        "method_rank": _json_safe_float(method_rank, None),
        "normalized_rank": _json_safe_float(normalized_rank, None),
        "lesson_match_count": lesson_match_count,
        "lesson_match_score": _json_safe_float(lesson_match_score),
        "feedback_count": int(stats["feedback_count"]),
        "feedback_avg_rating": _json_safe_float(stats["feedback_avg_rating"]),
        "stats": safe_stats,
    }


def _score_rows_for_ids(methods: list[dict], method_ids: list[int]) -> list[dict]:
    """Return score details for method_ids in exact order, skipping omitted methods."""
    methods_by_id = {method["method_id"]: method for method in methods}
    return [
        dict(methods_by_id[method_id]["score_details"])
        for method_id in method_ids
        if method_id in methods_by_id and "score_details" in methods_by_id[method_id]
    ]


def _score_methods(
    methods: list[dict],
    conn,
    method_ranks: dict[int, float],
    agent_run_id: int | None = None,
) -> tuple[list[dict], list[int]]:
    """Combine FTS5 rank, empirical outcomes, fatigue, and bounded feedback."""
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

        base_score = (
            0.35 * stats["success_rate"]
            + 0.25 * rank_score
            + 0.4 * delta_score
        )
        fatigue_multiplier = fatigue.get(method["method_id"], 1.0)
        feedback_multiplier = _feedback_multiplier(
            stats["feedback_count"], stats["feedback_avg_rating"]
        )
        final_score = base_score * fatigue_multiplier * feedback_multiplier
        method["_feedback_multiplier"] = feedback_multiplier
        method["_score"] = final_score
        _attach_score_details(
            method,
            stats,
            base_score=base_score,
            final_score=final_score,
            rank_score=rank_score,
            delta_score=delta_score,
            fatigue_multiplier=fatigue_multiplier,
            feedback_multiplier=feedback_multiplier,
            method_rank=method_ranks.get(method["method_id"]),
            normalized_rank=normalized_rank if method["method_id"] in method_ranks else None,
        )
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
            main_pack_str = _limit_text(
                f"# Method Pack Search Results\n\nNo methods found matching query: '{query}'",
                max_chars,
            )
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

        if methods and query and query.strip():
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
                general_header = "\n# General Optimization & Hyperparameter Tuning Methods\n"
                remaining_max_chars = None
                if max_chars is not None:
                    remaining_max_chars = max_chars - len(main_pack_str)

                if (
                    remaining_max_chars is None
                    or remaining_max_chars >= len(general_header)
                ):
                    gen_pack_str, gen_ids = _build_method_pack(
                        general_methods,
                        conn,
                        general_header,
                        max_chars=remaining_max_chars,
                    )
                    if (
                        gen_pack_str
                        and (max_chars is None or len(main_pack_str) + len(gen_pack_str) <= max_chars)
                    ):
                        main_pack_str += gen_pack_str
                        for gid in gen_ids:
                            if gid not in main_ids:
                                main_ids.append(gid)

        return main_pack_str, main_ids
    finally:
        if close:
            conn.close()


def search_methods_with_scores(
    query: str,
    conn=None,
    max_chars: int | None = None,
    agent_run_id: int | None = None,
) -> tuple[str, list[int], list[dict]]:
    """Search methods and return (formatted_method_pack, method_ids, score_details)."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        methods, _, method_ranks = _search_methods_hybrid(query, conn)
        score_rows: list[dict] = []

        if not methods:
            main_pack_str = _limit_text(
                f"# Method Pack Search Results\n\nNo methods found matching query: '{query}'",
                max_chars,
            )
            main_ids = []
        elif not method_ranks:
            # Empty query fallback: preserve default success-rate ordering while still exposing scores.
            _score_methods(methods, conn, {}, agent_run_id=agent_run_id)
            main_pack_str, main_ids = _build_method_pack(
                methods,
                conn,
                "# Method Pack: Autoresearch Optimization Catalog\n",
                max_chars=max_chars,
            )
            score_rows.extend(_score_rows_for_ids(methods, main_ids))
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
            score_rows.extend(_score_rows_for_ids(ordered_methods, main_ids))

        if methods and query and query.strip():
            cur = conn.cursor()
            cur.execute(
                """
                SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category
                FROM methods m
                WHERE m.category IN ('optimizer', 'regularization')
                """
            )
            general_methods = [dict(row) for row in cur.fetchall()]

            search_result_ids = {m["method_id"] for m in methods}
            general_methods = [m for m in general_methods if m["method_id"] not in search_result_ids]

            if general_methods:
                general_header = "\n# General Optimization & Hyperparameter Tuning Methods\n"
                remaining_max_chars = None
                if max_chars is not None:
                    remaining_max_chars = max_chars - len(main_pack_str)

                if (
                    remaining_max_chars is None
                    or remaining_max_chars >= len(general_header)
                ):
                    _score_methods(general_methods, conn, {}, agent_run_id=agent_run_id)
                    gen_pack_str, gen_ids = _build_method_pack(
                        general_methods,
                        conn,
                        general_header,
                        max_chars=remaining_max_chars,
                    )
                    if (
                        gen_pack_str
                        and (max_chars is None or len(main_pack_str) + len(gen_pack_str) <= max_chars)
                    ):
                        main_pack_str += gen_pack_str
                        for gid in gen_ids:
                            if gid not in main_ids:
                                main_ids.append(gid)
                                score_rows.extend(_score_rows_for_ids(general_methods, [gid]))

        return main_pack_str, main_ids, score_rows
    finally:
        if close:
            conn.close()


def search_methods(query: str, conn=None, max_chars: int | None = None) -> str:
    """Search for methods matching the query keywords and return a formatted method pack."""
    pack, _ = search_methods_and_ids(query, conn, max_chars=max_chars)
    return pack


def _normalize_source_types(source_types) -> list[str] | None:
    """Return a deduplicated source-type filter or None for unfiltered retrieval."""
    if source_types is None:
        return None
    if isinstance(source_types, str):
        values = [source_types]
    elif isinstance(source_types, (list, tuple)):
        values = list(source_types)
    else:
        raise ValueError("source_types must be a string or list of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("source_types must contain only strings")
        label = value.strip()
        if not label:
            raise ValueError("source_types must contain non-empty strings")
        if label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    if not normalized:
        raise ValueError("source_types must contain at least one source type")
    return normalized


def _source_type_filter_sql(source_types: list[str] | None, column: str) -> tuple[str, tuple[str, ...]]:
    if not source_types:
        return "", ()
    placeholders = ",".join("?" for _ in source_types)
    return f" AND {column} IN ({placeholders})", tuple(source_types)

def _filter_method_ids_by_source_types(
    conn,
    method_ids: list[int],
    source_types: list[str] | None,
) -> list[int]:
    """Return method IDs that have at least one source chunk with an allowed source type."""
    if not source_types or not method_ids:
        return list(method_ids)

    method_placeholders = ",".join("?" for _ in method_ids)
    source_placeholders = ",".join("?" for _ in source_types)
    rows = conn.execute(
        f"""
        SELECT DISTINCT ms.method_id
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        JOIN paper_extraction_status pes ON pes.paper_id = c.paper_id
        WHERE ms.method_id IN ({method_placeholders})
          AND pes.source_type IN ({source_placeholders})
        """,
        (*method_ids, *source_types),
    ).fetchall()
    allowed = {row["method_id"] for row in rows}
    return [method_id for method_id in method_ids if method_id in allowed]


def _method_rows_for_ids(conn, method_ids: list[int]) -> list[dict]:
    if not method_ids:
        return []
    placeholders = ",".join("?" for _ in method_ids)
    rows = conn.execute(
        f"""
        SELECT method_id, name, description, implementation_hint, risks, category
        FROM methods
        WHERE method_id IN ({placeholders})
        """,
        tuple(method_ids),
    ).fetchall()
    by_id = {row["method_id"]: dict(row) for row in rows}
    return [by_id[method_id] for method_id in method_ids if method_id in by_id]


def _filtered_method_context(
    query: str,
    conn,
    method_ids: list[int],
    source_types: list[str],
) -> tuple[str, list[int]]:
    filtered_ids = _filter_method_ids_by_source_types(conn, method_ids, source_types)
    method_rows = _method_rows_for_ids(conn, filtered_ids)
    if not method_rows:
        labels = ", ".join(source_types)
        return (
            "# Method Pack Search Results\n\n"
            f"No methods found matching query: '{query}' and source_types: {labels}",
            [],
        )
    order = {method_id: index for index, method_id in enumerate(filtered_ids)}
    return _build_method_pack(
        method_rows,
        conn,
        f"# Method Pack Search Results for: '{query}'\n",
        sort_key=lambda method: order[method["method_id"]],
        max_chars=None,
        source_types=source_types,
    )




def _chunk_fts_brief_sources(
    conn,
    query: str,
    max_papers: int,
    source_types: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Return fallback paper and chunk IDs from chunk FTS rank order."""
    fts_query = _fts_phrase_query(query)
    if not fts_query:
        return [], []

    source_join_sql = ""
    if source_types:
        source_join_sql = (
            "JOIN paper_extraction_status pes ON pes.paper_id = chunks.paper_id"
        )
    source_filter_sql, source_filter_params = _source_type_filter_sql(
        source_types, "pes.source_type"
    )
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT chunks_fts.rowid AS chunk_id, chunks.paper_id, chunks_fts.rank
        FROM chunks_fts
        JOIN chunks ON chunks.chunk_id = chunks_fts.rowid
        {source_join_sql}
        WHERE chunks_fts MATCH ?
        {source_filter_sql}
        ORDER BY chunks_fts.rank, chunks_fts.rowid
        """,
        (fts_query, *source_filter_params),
    )

    paper_ids: list[int] = []
    chunk_ids: list[int] = []
    selected_papers: set[int] = set()
    for row in cur.fetchall():
        paper_id = row["paper_id"]
        if paper_id not in selected_papers:
            if len(paper_ids) >= max_papers:
                continue
            selected_papers.add(paper_id)
            paper_ids.append(paper_id)
        if paper_id in selected_papers:
            chunk_ids.append(row["chunk_id"])
    return paper_ids, chunk_ids


def get_research_brief(
    query: str,
    conn=None,
    max_chars: int | None = None,
    max_papers: int = 3,
    include_scores: bool = False,
    agent_run_id: int | None = None,
    source_types=None,
    source_type=None,
) -> dict:
    """Return a deterministic compressed research brief for one query.

    The brief intentionally reuses ResearchFS' bounded method retrieval and paper
    memory cards. It never calls an LLM and never adds raw chunk text beyond the
    evidence snippets already bounded by ``get_paper_memory``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be non-negative")

    if max_papers <= 0:
        raise ValueError("max_papers must be positive")

    if source_types is None and source_type is not None:
        source_types = source_type
    normalized_source_types = _normalize_source_types(source_types)

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        if include_scores:
            method_context, retrieved_method_ids, scores = search_methods_with_scores(
                query,
                conn=conn,
                max_chars=None,
                agent_run_id=agent_run_id,
            )
        else:
            method_context, retrieved_method_ids = search_methods_and_ids(
                query,
                conn=conn,
                max_chars=None,
                agent_run_id=agent_run_id,
            )
            scores = None

        if normalized_source_types:
            method_context, retrieved_method_ids = _filtered_method_context(
                query,
                conn,
                retrieved_method_ids,
                normalized_source_types,
            )
            if scores is not None:
                allowed_method_ids = set(retrieved_method_ids)
                scores = [
                    score
                    for score in scores
                    if score.get("method_id") in allowed_method_ids
                ]

        cur = conn.cursor()
        paper_ids: list[int] = []
        traversal_chunk_ids: list[int] = []
        selected_papers: set[int] = set()
        source_join_sql = ""
        source_filter_sql, source_filter_params = _source_type_filter_sql(
            normalized_source_types, "pes.source_type"
        )
        if normalized_source_types:
            source_join_sql = (
                "JOIN paper_extraction_status pes ON pes.paper_id = c.paper_id"
            )
        for method_id in retrieved_method_ids:
            cur.execute(
                f"""
                SELECT c.paper_id, ms.chunk_id
                FROM method_sources ms
                JOIN chunks c ON c.chunk_id = ms.chunk_id
                {source_join_sql}
                WHERE ms.method_id = ?
                {source_filter_sql}
                ORDER BY c.paper_id, ms.chunk_id
                """,
                (method_id, *source_filter_params),
            )
            for row in cur.fetchall():
                paper_id = row["paper_id"]
                if paper_id not in selected_papers:
                    if len(paper_ids) >= max_papers:
                        continue
                    selected_papers.add(paper_id)
                    paper_ids.append(paper_id)
                if paper_id in selected_papers:
                    traversal_chunk_ids.append(row["chunk_id"])
        if not paper_ids:
            paper_ids, traversal_chunk_ids = _chunk_fts_brief_sources(
                conn, query, max_papers, normalized_source_types
            )


        header_lines = [
            "# Research Brief",
            f"Query: {query}",
            f"Max papers: {max_papers}",
            f"Selected papers: {len(paper_ids)}",
            "Source types: " + (", ".join(normalized_source_types) if normalized_source_types else "unfiltered"),
            "Source: deterministic compressed method retrieval plus bounded paper memory cards; no LLM call.",
        ]
        lesson_budget = None
        method_budget = None
        if max_chars is not None:
            lesson_budget = max(0, max_chars // 5)
            method_budget = max(0, max_chars // 3)
        standalone_lessons = _standalone_lesson_matches(query, conn)
        sections = [
            "\n".join(header_lines),
            _limit_text(_format_standalone_lessons(standalone_lessons), lesson_budget),
            "## Method retrieval\n" + _limit_text(method_context, method_budget),
        ]

        method_ids = list(retrieved_method_ids)
        chunk_ids = list(traversal_chunk_ids)
        lesson_ids: list[int] = [row["lesson_id"] for row in standalone_lessons]
        paper_memory_provenance: list[dict] = []
        for index, paper_id in enumerate(paper_ids, start=1):
            card_budget = None
            if max_chars is not None:
                remaining_cards = max(1, len(paper_ids) - index + 1)
                current_len = len("\n\n".join(sections))
                remaining = max(0, max_chars - current_len)
                heading_allowance = 80
                card_budget = max(0, remaining // remaining_cards - heading_allowance)
            card = get_paper_memory(paper_id, conn=conn, max_chars=card_budget)
            method_ids.extend(card.get("method_ids") or [])
            chunk_ids.extend(card.get("chunk_ids") or [])
            lesson_ids.extend(card.get("lesson_ids") or [])
            paper_memory_provenance.append(_paper_memory_brief_metadata(card))
            sections.append(
                f"## Paper memory card {index}: paper_id={paper_id}\n"
                + str(card.get("context") or "")
            )

        context = _limit_text("\n\n".join(sections), max_chars)
        result = {
            "ok": True,
            "query": query,
            "context": context,
            "paper_ids": paper_ids,
            "method_ids": _unique_ordered(method_ids),
            "chunk_ids": _unique_ordered(chunk_ids),
            "lesson_ids": _unique_ordered(lesson_ids),
            "paper_memory_provenance": paper_memory_provenance,
            "max_chars": max_chars,
            "max_papers": max_papers,
            "source_types": normalized_source_types,
        }
        if include_scores:
            result["scores"] = scores
        return result
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
    """Start a new agent run; mode is an agent-defined non-empty label."""
    mode = _validate_non_empty_text(mode, "mode")
    if model is not None:
        model = _validate_non_empty_text(model, "model")
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
    agent_run_id = _validate_positive_int_id(agent_run_id, "agent_run_id")
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        _require_existing_id(conn, "agent_runs", "agent_run_id", agent_run_id, "agent_run_id")
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
    commit_hash = _validate_non_empty_text(commit_hash, "commit_hash")
    status = _validate_experiment_status(status)
    description = _validate_non_empty_text(description, "description")
    val_bpb = _validate_optional_non_negative_metric(val_bpb, "val_bpb")
    memory_gb = _validate_optional_non_negative_metric(memory_gb, "memory_gb")
    if agent_run_id is not None:
        agent_run_id = _validate_positive_int_id(agent_run_id, "agent_run_id")
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        if agent_run_id is not None:
            _require_existing_id(conn, "agent_runs", "agent_run_id", agent_run_id, "agent_run_id")
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
    experiment_id = _validate_positive_int_id(experiment_id, "experiment_id")
    val_bpb = _validate_optional_non_negative_metric(val_bpb, "val_bpb")
    memory_gb = _validate_optional_non_negative_metric(memory_gb, "memory_gb")
    if status is not None:
        status = _validate_experiment_status(status)
    if description is not None:
        description = _validate_non_empty_text(description, "description")
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        _require_existing_id(conn, "experiments", "experiment_id", experiment_id, "experiment_id")
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


def _unique_method_ids(method_ids: Optional[list[int]]) -> list[int]:
    """Return method IDs once, preserving caller order."""
    if not method_ids:
        return []
    return list(dict.fromkeys(method_ids))


EXPERIMENT_STATUSES = {"pending", "keep", "discard", "crash"}




def _validate_experiment_status(value) -> str:
    """Return a supported experiment status token."""
    status = _validate_non_empty_text(value, "status")
    if status not in EXPERIMENT_STATUSES:
        allowed = ", ".join(sorted(EXPERIMENT_STATUSES))
        raise ValueError(f"status must be one of: {allowed}")
    return status


def _validate_positive_int_id(value, field_name: str) -> int:
    """Return value as an int ID, rejecting bools and non-positive integers."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validate_non_negative_int(value, field_name: str) -> int:
    """Return value as an int count, rejecting bools and negative integers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_optional_non_negative_cost(value) -> float | None:
    """Return cost as a finite float, allowing None and rejecting bools."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("cost must be a non-negative finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cost must be a non-negative finite number") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError("cost must be a non-negative finite number")
    return numeric_value


def _validate_optional_non_negative_metric(value, field_name: str) -> float | None:
    """Return metric as a finite float, allowing None and rejecting bools."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative finite number") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return numeric_value


def _validate_non_empty_text(value, field_name: str) -> str:
    """Return original text after rejecting empty required public string fields."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_existing_id(conn, table: str, id_column: str, value: int, field_name: str) -> None:
    """Raise ValueError when a referenced row is absent."""
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE {id_column} = ? LIMIT 1",
        (value,),
    ).fetchone()
    if row is None:
        raise ValueError(f"{field_name} does not exist: {value}")


def _validate_method_id_list(method_ids: list[int], conn) -> list[int]:
    """Validate public method ID lists, preserving order and removing duplicates."""
    if not isinstance(method_ids, list) or not method_ids:
        raise ValueError("method_ids must be a non-empty list of positive integers")

    valid_ids: list[int] = []
    seen: set[int] = set()
    for method_id in method_ids:
        valid_id = _validate_positive_int_id(method_id, "method_id")
        if valid_id in seen:
            continue
        seen.add(valid_id)
        valid_ids.append(valid_id)

    if not valid_ids:
        raise ValueError("method_ids must contain at least one method ID")

    placeholders = ",".join("?" for _ in valid_ids)
    rows = conn.execute(
        f"SELECT method_id FROM methods WHERE method_id IN ({placeholders})",
        tuple(valid_ids),
    ).fetchall()
    existing_ids = {row["method_id"] for row in rows}
    missing_ids = [method_id for method_id in valid_ids if method_id not in existing_ids]
    if missing_ids:
        raise ValueError(f"method_ids do not exist: {missing_ids}")
    return valid_ids


def _insert_experiment_methods(
    cur,
    experiment_id: int,
    method_ids: Optional[list[int]],
    role: str = "retrieved",
    note: Optional[str] = None,
) -> None:
    role = role or "retrieved"
    for method_id in _unique_method_ids(method_ids):
        cur.execute(
            """
            INSERT OR IGNORE INTO experiment_methods
            (experiment_id, method_id, role, note)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, method_id, role, note),
        )
        if note is not None:
            cur.execute(
                """
                UPDATE experiment_methods
                SET note = ?
                WHERE experiment_id = ? AND method_id = ? AND role = ?
                """,
                (note, experiment_id, method_id, role),
            )


def link_experiment_methods(
    experiment_id: int,
    method_ids: list[int],
    role: str = "retrieved",
    note: Optional[str] = None,
    conn=None,
) -> None:
    """Connect an experiment to the methods it used or evaluated."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        valid_experiment_id = _validate_positive_int_id(experiment_id, "experiment_id")
        valid_role = _validate_non_empty_text(role, "role")
        _require_existing_id(
            conn, "experiments", "experiment_id", valid_experiment_id, "experiment_id"
        )
        valid_method_ids = _validate_method_id_list(method_ids, conn)
        _insert_experiment_methods(
            cur, valid_experiment_id, valid_method_ids, valid_role, note
        )
        conn.commit()
    finally:
        if close:
            conn.close()


def log_lesson(
    label: str,
    content: str,
    outcome: Optional[str] = None,
    method_id: Optional[int] = None,
    experiment_id: Optional[int] = None,
    conn=None,
) -> int:
    """Record a lesson or failure learned from a method or experiment."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        valid_label = _validate_non_empty_text(label, "label")
        valid_content = _validate_non_empty_text(content, "content")
        valid_method_id = None
        valid_experiment_id = None
        if method_id is not None:
            valid_method_id = _validate_positive_int_id(method_id, "method_id")
            _require_existing_id(conn, "methods", "method_id", valid_method_id, "method_id")
        if experiment_id is not None:
            valid_experiment_id = _validate_positive_int_id(
                experiment_id, "experiment_id"
            )
            _require_existing_id(
                conn,
                "experiments",
                "experiment_id",
                valid_experiment_id,
                "experiment_id",
            )
        cur.execute(
            """
            INSERT INTO lessons (method_id, experiment_id, label, content, outcome)
            VALUES (?, ?, ?, ?, ?)
            """,
            (valid_method_id, valid_experiment_id, valid_label, valid_content, outcome),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        if close:
            conn.close()


def _truncate_text(text: Optional[str], limit: int = 220) -> str:
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _limit_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    suffix = "\n... truncated"
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def _format_stats_summary(stats: dict) -> str:
    feedback_suffix = ""
    if stats.get("feedback_count", 0) > 0:
        feedback_suffix = (
            f"; feedback: {stats['feedback_count']} ratings, "
            f"avg {stats['feedback_avg_rating']:+.2f}"
        )
    if stats["total_runs"] == 0:
        return f"history: no linked experiments{feedback_suffix}"
    return (
        f"history: {stats['total_runs']} runs, {stats['success_rate']:.0%} success "
        f"({stats['keeps']} keep, {stats['discards']} discard, "
        f"{stats['crashes']} crash), avg {stats['avg_bpb_delta']:+.6f} BPB"
        f"{feedback_suffix}"
    )


def _get_code_targets(method_id: int, conn, limit: int = 5) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            mct.file_path,
            mct.symbol_name,
            mct.symbol_kind,
            mct.rationale,
            MIN(cs.start_line) AS start_line,
            MAX(cs.end_line) AS end_line,
            MAX(cs.docstring) AS docstring
        FROM method_code_targets mct
        LEFT JOIN code_symbols cs
          ON cs.file_path = mct.file_path
         AND cs.symbol_name = mct.symbol_name
         AND cs.symbol_kind = mct.symbol_kind
        WHERE mct.method_id = ?
        GROUP BY mct.file_path, mct.symbol_name, mct.symbol_kind, mct.rationale
        ORDER BY mct.file_path, start_line, mct.symbol_name
        LIMIT ?
        """,
        (method_id, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def _format_code_target(row: dict) -> str:
    target = f"{row['symbol_kind']} {row['symbol_name']}".strip()
    location = row["file_path"]
    if row["start_line"] is not None:
        location += f":{row['start_line']}"
        if row["end_line"] and row["end_line"] != row["start_line"]:
            location += f"-{row['end_line']}"
    line = f"- `{target}` at {location}"
    if row["rationale"]:
        line += f" — {_truncate_text(row['rationale'], 160)}"
    if row["docstring"]:
        line += f" ({_truncate_text(row['docstring'], 140)})"
    return line


def _get_history_items(method_id: int, conn, max_items: int) -> tuple[list[dict], list[dict]]:
    cur = conn.cursor()
    cur.execute(
        """
        WITH linked AS (
            SELECT em.experiment_id, em.role, em.note
            FROM experiment_methods em
            WHERE em.method_id = ?
            UNION ALL
            SELECT re.experiment_id, 'retrieved' AS role, re.query AS note
            FROM retrieval_events re
            JOIN retrieved_methods rm ON re.event_id = rm.event_id
            WHERE rm.method_id = ? AND re.experiment_id IS NOT NULL
        )
        SELECT
            e.experiment_id,
            e.status,
            e.val_bpb,
            e.memory_gb,
            e.description,
            e.timestamp,
            GROUP_CONCAT(DISTINCT linked.role) AS roles,
            MAX(linked.note) AS note
        FROM linked
        JOIN experiments e ON e.experiment_id = linked.experiment_id
        GROUP BY
            e.experiment_id,
            e.status,
            e.val_bpb,
            e.memory_gb,
            e.description,
            e.timestamp
        ORDER BY e.timestamp DESC, e.experiment_id DESC
        LIMIT ?
        """,
        (method_id, method_id, max_items),
    )
    experiments = [dict(row) for row in cur.fetchall()]

    cur.execute(
        """
        WITH linked_experiments AS (
            SELECT em.experiment_id
            FROM experiment_methods em
            WHERE em.method_id = ?
            UNION
            SELECT re.experiment_id
            FROM retrieval_events re
            JOIN retrieved_methods rm ON re.event_id = rm.event_id
            WHERE rm.method_id = ? AND re.experiment_id IS NOT NULL
        )
        SELECT DISTINCT
            l.lesson_id,
            l.method_id,
            l.experiment_id,
            l.label,
            l.content,
            l.outcome,
            l.created_at
        FROM lessons l
        WHERE l.method_id = ?
           OR (
               l.experiment_id IS NOT NULL
               AND l.experiment_id IN (SELECT experiment_id FROM linked_experiments)
           )
        ORDER BY l.created_at DESC, l.lesson_id DESC
        LIMIT ?
        """,
        (method_id, method_id, method_id, max_items),
    )
    lessons = [dict(row) for row in cur.fetchall()]
    return experiments, lessons


def _format_experiment_summary(row: dict) -> str:
    parts = [f"status={row['status']}"]
    if row["val_bpb"] is not None:
        parts.append(f"val_bpb={row['val_bpb']:.6f}")
    if row["memory_gb"] is not None:
        parts.append(f"memory_gb={row['memory_gb']:.2f}")
    if row["roles"]:
        parts.append(f"roles={row['roles']}")
    if row["note"]:
        parts.append(f"note={_truncate_text(row['note'], 100)}")
    description = _truncate_text(row["description"], 180)
    if description:
        parts.append(description)
    return f"- Experiment #{row['experiment_id']}: " + "; ".join(parts)


def _format_lesson_summary(row: dict) -> str:
    outcome = f" [{row['outcome']}]" if row["outcome"] else ""
    scope = []
    if row["method_id"] is not None:
        scope.append(f"method #{row['method_id']}")
    if row["experiment_id"] is not None:
        scope.append(f"experiment #{row['experiment_id']}")
    suffix = f" ({', '.join(scope)})" if scope else ""
    return (
        f"- {row['label']}{outcome}{suffix}: "
        f"{_truncate_text(row['content'], 220)}"
    )


def _format_paper_memory_status(status: dict | None) -> list[str]:
    if not status:
        return ["Extraction: no stored extraction status."]

    parts = [f"status={status['status']}"]
    if status.get("source_type"):
        parts.append(f"source={status['source_type']}")
    parts.append(f"chunks={status['chunk_count']}")
    parts.append(f"methods={status['method_count']}")
    if status.get("source_char_count") is not None:
        parts.append(f"source_chars={status['source_char_count']}")
    if status.get("compressed_char_count") is not None:
        parts.append(f"compressed_chars={status['compressed_char_count']}")
    if status.get("compression_ratio") is not None:
        parts.append(f"compression_ratio={status['compression_ratio']:.3f}")
    if status.get("confidence_score") is not None:
        parts.append(f"confidence={status['confidence_score']:.2f}")
    if status.get("confidence_reason"):
        parts.append(f"confidence_reason={_truncate_text(status['confidence_reason'], 160)}")
    if status.get("extractor_model"):
        parts.append(f"extractor_model={_truncate_text(status['extractor_model'], 120)}")
    if status.get("extractor_prompt_version"):
        parts.append(
            f"extractor_prompt_version={_truncate_text(status['extractor_prompt_version'], 120)}"
        )
    if status.get("error_text"):
        parts.append(f"error={_truncate_text(status['error_text'], 160)}")
    return ["Extraction: " + "; ".join(parts) + "."]


def _paper_memory_status_record(row) -> dict | None:
    if row is None:
        return None
    return {
        "paper_id": row["paper_id"],
        "status": row["status"],
        "source_type": row["source_type"],
        "chunk_count": row["chunk_count"],
        "method_count": row["method_count"],
        "source_char_count": row["source_char_count"],
        "compressed_char_count": row["compressed_char_count"],
        "compression_ratio": row["compression_ratio"],
        "confidence_score": row["confidence_score"],
        "confidence_reason": row["confidence_reason"],
        "extractor_model": row["extractor_model"],
        "extractor_prompt_version": row["extractor_prompt_version"],
        "error_text": row["error_text"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def _paper_memory_provenance_source(source: dict) -> dict:
    return {
        "method_id": source["method_id"],
        "chunk_id": source["chunk_id"],
        "section": source["section"],
        "evidence_text": (
            _truncate_text(source["evidence_text"], 320)
            if source.get("evidence_text")
            else None
        ),
        "evidence_start": source["evidence_start"],
        "evidence_end": source["evidence_end"],
        "evidence_strategy": source["evidence_strategy"],
    }


def _paper_memory_brief_metadata(card: dict) -> dict:
    metadata = {
        "paper_id": card["paper_id"],
        "provenance_sources": card.get("provenance_sources") or [],
    }
    if card.get("extraction_status") is not None:
        metadata["extraction_status"] = card["extraction_status"]
    return metadata


def _unique_ordered(values: list[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _paper_memory_recent_experiments(conn, method_ids: list[int], limit: int = 5) -> list[dict]:
    if not method_ids or limit <= 0:
        return []
    placeholders = ",".join("?" for _ in method_ids)
    cur = conn.cursor()
    cur.execute(
        f"""
        WITH linked AS (
            SELECT em.experiment_id, em.role, em.note
            FROM experiment_methods em
            WHERE em.method_id IN ({placeholders})
            UNION ALL
            SELECT re.experiment_id, 'retrieved' AS role, re.query AS note
            FROM retrieval_events re
            JOIN retrieved_methods rm ON rm.event_id = re.event_id
            WHERE rm.method_id IN ({placeholders})
              AND re.experiment_id IS NOT NULL
        )
        SELECT
            e.experiment_id,
            e.status,
            e.val_bpb,
            e.memory_gb,
            e.description,
            e.timestamp,
            GROUP_CONCAT(DISTINCT linked.role) AS roles,
            MAX(linked.note) AS note
        FROM linked
        JOIN experiments e ON e.experiment_id = linked.experiment_id
        GROUP BY
            e.experiment_id,
            e.status,
            e.val_bpb,
            e.memory_gb,
            e.description,
            e.timestamp
        ORDER BY e.timestamp DESC, e.experiment_id DESC
        LIMIT ?
        """,
        (*method_ids, *method_ids, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def _paper_memory_recent_lessons(
    conn, method_ids: list[int], experiment_ids: list[int], limit: int = 5
) -> list[dict]:
    if limit <= 0:
        return []
    clauses: list[str] = []
    params: list[int] = []
    if method_ids:
        clauses.append(f"l.method_id IN ({','.join('?' for _ in method_ids)})")
        params.extend(method_ids)
    if experiment_ids:
        clauses.append(f"l.experiment_id IN ({','.join('?' for _ in experiment_ids)})")
        params.extend(experiment_ids)
    if not clauses:
        return []
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            l.lesson_id,
            l.method_id,
            l.experiment_id,
            l.label,
            l.content,
            l.outcome,
            l.created_at
        FROM lessons l
        WHERE {' OR '.join(clauses)}
        ORDER BY l.created_at DESC, l.lesson_id DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def _paper_memory_feedback_notes(conn, method_id: int, limit: int = 2) -> list[dict]:
    if limit <= 0:
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT feedback_id, rating, label, note, source, timestamp
        FROM method_feedback
        WHERE method_id = ?
        ORDER BY timestamp DESC, feedback_id DESC
        LIMIT ?
        """,
        (method_id, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def get_paper_memory(
    paper_id: int, conn=None, max_chars: Optional[int] = None
) -> dict:
    """Return one deterministic, bounded paper memory card and referenced IDs."""
    paper_id = _validate_positive_int_id(paper_id, "paper_id")
    if max_chars is not None:
        max_chars = _validate_non_negative_int(max_chars, "max_chars")

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT paper_id, title, authors, abstract, url, publish_date, created_at
            FROM papers
            WHERE paper_id = ?
            """,
            (paper_id,),
        )
        paper = cur.fetchone()
        if paper is None:
            raise LookupError(f"No paper found for id {paper_id}")

        cur.execute(
            """
            SELECT
                paper_id,
                status,
                source_type,
                chunk_count,
                method_count,
                source_char_count,
                compressed_char_count,
                compression_ratio,
                confidence_score,
                confidence_reason,
                extractor_model,
                extractor_prompt_version,
                error_text,
                created_at,
                updated_at
            FROM paper_extraction_status
            WHERE paper_id = ?
            """,
            (paper_id,),
        )
        extraction_status = _paper_memory_status_record(cur.fetchone())

        cur.execute(
            """
            SELECT DISTINCT
                m.method_id,
                m.name,
                m.description,
                m.implementation_hint,
                m.risks,
                m.category
            FROM method_sources ms
            JOIN chunks c ON c.chunk_id = ms.chunk_id
            JOIN methods m ON m.method_id = ms.method_id
            WHERE c.paper_id = ?
            ORDER BY m.method_id
            """,
            (paper_id,),
        )
        methods = [dict(row) for row in cur.fetchall()]
        method_ids = [row["method_id"] for row in methods]

        cur.execute(
            """
            SELECT
                ms.method_id,
                ms.chunk_id,
                c.section,
                ms.evidence_text,
                ms.evidence_start,
                ms.evidence_end,
                ms.evidence_strategy
            FROM method_sources ms
            JOIN chunks c ON c.chunk_id = ms.chunk_id
            WHERE c.paper_id = ?
            ORDER BY ms.method_id, ms.chunk_id
            """,
            (paper_id,),
        )
        sources = [dict(row) for row in cur.fetchall()]
        provenance_sources = [_paper_memory_provenance_source(source) for source in sources]
        chunk_ids = _unique_ordered([row["chunk_id"] for row in sources])
        sources_by_method: dict[int, list[dict]] = {}
        for source in sources:
            sources_by_method.setdefault(source["method_id"], []).append(source)

        feedback = {
            method_id: get_method_feedback_summary(method_id, conn=conn)
            for method_id in method_ids
        }
        feedback_notes = {
            method_id: _paper_memory_feedback_notes(conn, method_id)
            for method_id in method_ids
        }
        experiments = _paper_memory_recent_experiments(conn, method_ids)
        experiment_ids = [row["experiment_id"] for row in experiments]
        lessons = _paper_memory_recent_lessons(conn, method_ids, experiment_ids)
        lesson_ids = [row["lesson_id"] for row in lessons]

        lines = [
            f"# Paper Memory: {paper['title']}",
            f"Paper ID: {paper['paper_id']}",
        ]
        if paper["authors"]:
            lines.append(f"Authors: {_truncate_text(paper['authors'], 220)}")
        if paper["publish_date"]:
            lines.append(f"Published: {paper['publish_date']}")
        if paper["url"]:
            lines.append(f"URL: {paper['url']}")
        if paper["abstract"]:
            lines.append(f"Abstract: {_truncate_text(paper['abstract'], 500)}")
        lines.extend(_format_paper_memory_status(extraction_status))

        lines.append("")
        lines.append("## Methods and provenance")
        if not methods:
            lines.append("- No linked methods.")
        for method in methods:
            method_id = method["method_id"]
            label = method["category"] or "uncategorized"
            lines.append(f"### {method['name']} (method_id={method_id}, category={label})")
            method_sources = sources_by_method.get(method_id, [])
            if method_sources:
                lines.append("Evidence:")
                for source in method_sources:
                    location = f"chunk_id={source['chunk_id']}"
                    if source["section"]:
                        location += f", section={source['section']}"
                    if source["evidence_start"] is not None and source["evidence_end"] is not None:
                        location += f", chars={source['evidence_start']}-{source['evidence_end']}"
                    if source["evidence_strategy"]:
                        location += f", strategy={source['evidence_strategy']}"
                    if source["evidence_text"]:
                        evidence = _truncate_text(source["evidence_text"], 320)
                        lines.append(f"- {location}: {evidence}")
                    else:
                        lines.append(
                            f"- {location}: evidence_text missing; no compressed excerpt stored."
                        )
            else:
                lines.append("Evidence: none linked.")
            if method["description"]:
                lines.append(f"Description: {_truncate_text(method['description'], 260)}")
            if method["implementation_hint"]:
                lines.append(
                    f"Implementation hint: {_truncate_text(method['implementation_hint'], 260)}"
                )
            if method["risks"]:
                lines.append(f"Risks: {_truncate_text(method['risks'], 220)}")
            summary = feedback[method_id]
            if summary["feedback_count"] > 0:
                lines.append(
                    "Feedback: "
                    f"{summary['feedback_count']} ratings, "
                    f"avg {summary['feedback_avg_rating']:+.2f}"
                )
                for note in feedback_notes.get(method_id, []):
                    note_parts = [f"rating={note['rating']:+.2f}"]
                    if note["label"]:
                        note_parts.append(f"label={note['label']}")
                    if note["source"]:
                        note_parts.append(f"source={note['source']}")
                    if note["note"]:
                        note_parts.append(_truncate_text(note["note"], 180))
                    lines.append("- Feedback note: " + "; ".join(note_parts))

        lines.append("")
        lines.append("## Recent experiments")
        if experiments:
            lines.extend(_format_experiment_summary(row) for row in experiments)
        else:
            lines.append("- None linked.")

        lines.append("")
        lines.append("## Lessons")
        if lessons:
            lines.extend(_format_lesson_summary(row) for row in lessons)
        else:
            lines.append("- None recorded.")

        context = _limit_text("\n".join(lines), max_chars)
        result = {
            "paper_id": paper_id,
            "max_chars": max_chars,
            "context": context,
            "method_ids": method_ids,
            "chunk_ids": chunk_ids,
            "lesson_ids": lesson_ids,
            "experiment_ids": experiment_ids,
            "provenance_sources": provenance_sources,
        }
        if extraction_status is not None:
            result["extraction_status"] = extraction_status
            result["extraction"] = extraction_status
            result["confidence_score"] = extraction_status["confidence_score"]
            result["confidence_reason"] = extraction_status["confidence_reason"]
            result["compression_ratio"] = extraction_status["compression_ratio"]
            result["source_char_count"] = extraction_status["source_char_count"]
            result["compressed_char_count"] = extraction_status["compressed_char_count"]
        return result
    finally:
        if close:
            conn.close()


def get_method_history(method_id: int, conn=None, max_items: int = 5) -> str:
    """Return recent experiments and lessons linked to a method."""
    method_id = _validate_positive_int_id(method_id, "method_id")
    max_items = _validate_non_negative_int(max_items, "max_items")

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT method_id, name, description, implementation_hint, risks, category
            FROM methods
            WHERE method_id = ?
            """,
            (method_id,),
        )
        method = cur.fetchone()
        if method is None:
            return f"# Method History\n\nNo method found for method_id={method_id}."

        stats = _get_empirical_stats(method_id, conn, {})
        targets = _get_code_targets(method_id, conn, limit=max(1, min(5, max_items or 1)))
        experiments, lessons = _get_history_items(method_id, conn, max_items)

        lines = [
            f"# Method History: {method['name']} (method_id={method_id})",
            f"Score history: {_format_stats_summary(stats)}.",
        ]
        if method["description"]:
            lines.append(f"Description: {_truncate_text(method['description'], 240)}")
        if method["implementation_hint"]:
            lines.append(
                f"Implementation hint: {_truncate_text(method['implementation_hint'], 240)}"
            )
        if method["risks"]:
            lines.append(f"Risks: {_truncate_text(method['risks'], 200)}")

        lines.append("Code targets:")
        if targets:
            lines.extend(_format_code_target(target) for target in targets)
        else:
            lines.append("- None linked.")

        lines.append("Recent experiments:")
        if experiments:
            lines.extend(_format_experiment_summary(row) for row in experiments)
        else:
            lines.append("- None linked.")

        lines.append("Lessons:")
        if lessons:
            lines.extend(_format_lesson_summary(row) for row in lessons)
        else:
            lines.append("- None recorded.")

        return "\n".join(lines)
    finally:
        if close:
            conn.close()


def recommend_next_experiments(
    query: str = "",
    conn=None,
    max_items: int = 5,
    agent_run_id: Optional[int] = None,
) -> str:
    """Recommend methods to try next, including graph history and code targets."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        limit = max(0, max_items)
        methods, _, method_ranks = _search_methods_hybrid(query, conn)
        used_fallback_catalog = False
        if limit == 0:
            return f"# Recommended Next Experiments\n\nNo methods requested for query: '{query}'"
        if not methods:
            methods = _fetch_all_methods(conn.cursor())
            method_ranks = {}
            used_fallback_catalog = True
        if not methods:
            return "# Recommended Next Experiments\n\nNo methods are available."
        scored_methods, _ = _score_methods(
            [dict(method) for method in methods],
            conn,
            method_ranks,
            agent_run_id=agent_run_id,
        )

        header = "# Recommended Next Experiments"
        if query and query.strip():
            header += f" for: '{query}'"
        lines = [header]
        if used_fallback_catalog and query and query.strip():
            lines.append("No direct method matches; showing the ranked method catalog instead.")
        history_limit = max(1, min(3, limit))

        for index, method in enumerate(scored_methods[:limit], start=1):
            method_id = method["method_id"]
            lines.append("")
            lines.append(f"## {index}. {method['name']} (method_id={method_id})")
            lines.append(
                f"Score: {method['_score']:.3f}; "
                f"{_format_stats_summary(method['stats'])}."
            )
            if method["description"]:
                lines.append(f"Description: {_truncate_text(method['description'], 240)}")
            if method["implementation_hint"]:
                lines.append(
                    "Implementation hint: "
                    f"{_truncate_text(method['implementation_hint'], 240)}"
                )
            if method["risks"]:
                lines.append(f"Risks: {_truncate_text(method['risks'], 200)}")

            targets = _get_code_targets(method_id, conn, limit=5)
            lines.append("Code targets:")
            if targets:
                lines.extend(_format_code_target(target) for target in targets)
            else:
                lines.append("- None linked.")

            experiments, lessons = _get_history_items(method_id, conn, history_limit)
            lines.append("Prior experiments:")
            if experiments:
                lines.extend(_format_experiment_summary(row) for row in experiments)
            else:
                lines.append("- None linked.")
            lines.append("Lessons:")
            if lessons:
                lines.extend(_format_lesson_summary(row) for row in lessons)
            else:
                lines.append("- None recorded.")

        return "\n".join(lines)
    finally:
        if close:
            conn.close()


def _get_code_symbols_fallback(
    query: str,
    conn,
    max_chars: Optional[int] = None,
) -> str:
    cur = conn.cursor()
    fts_query = _fts_phrase_query(query)
    try:
        if fts_query:
            cur.execute(
                """
                SELECT
                    cs.file_path,
                    cs.symbol_name,
                    cs.symbol_kind,
                    cs.start_line,
                    cs.end_line,
                    cs.docstring
                FROM code_symbols_fts
                JOIN code_symbols cs ON code_symbols_fts.rowid = cs.symbol_id
                WHERE code_symbols_fts MATCH ?
                ORDER BY code_symbols_fts.rank
                LIMIT 20
                """,
                (fts_query,),
            )
        else:
            cur.execute(
                """
                SELECT file_path, symbol_name, symbol_kind, start_line, end_line, docstring
                FROM code_symbols
                ORDER BY file_path, start_line, symbol_name
                LIMIT 20
                """
            )
        rows = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return "No code symbols indexed."

    if not rows:
        return "No code symbols indexed."

    lines = ["# Code Symbols"]
    for row in rows:
        target = f"{row['symbol_kind']} {row['symbol_name']}".strip()
        location = row["file_path"]
        if row["start_line"] is not None:
            location += f":{row['start_line']}"
            if row["end_line"] and row["end_line"] != row["start_line"]:
                location += f"-{row['end_line']}"
        line = f"- `{target}` at {location}"
        if row["docstring"]:
            line += f" — {_truncate_text(row['docstring'], 160)}"
        lines.append(line)
    return _limit_text("\n".join(lines), max_chars)


def _get_code_context(query: str, conn, max_chars: Optional[int] = None) -> str:
    try:
        from researchfs.codegraph import get_code_map, search_code_symbols
    except ImportError:
        return _get_code_symbols_fallback(query, conn, max_chars=max_chars)

    if query and query.strip():
        return search_code_symbols(query, conn=conn, max_chars=max_chars)
    return get_code_map(conn=conn, max_chars=max_chars)


def get_graph_context(
    query: str = "",
    conn=None,
    max_chars: Optional[int] = None,
    agent_run_id: Optional[int] = None,
) -> str:
    """Return method, experiment, lesson, and code context for loopgraph prompts."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        code_budget = None
        if max_chars is not None:
            code_budget = max(0, max_chars - int(max_chars * 0.7))

        recommendations = recommend_next_experiments(
            query,
            conn=conn,
            max_items=5,
            agent_run_id=agent_run_id,
        )
        code_context = _get_code_context(query, conn, max_chars=code_budget)

        sections = ["# ResearchFS Graph Context", recommendations]
        if code_context:
            sections.append(code_context)
        return _limit_text("\n\n".join(sections), max_chars)
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
        valid_agent_run_id = None
        valid_experiment_id = None
        if agent_run_id is not None:
            valid_agent_run_id = _validate_positive_int_id(
                agent_run_id, "agent_run_id"
            )
            _require_existing_id(
                conn,
                "agent_runs",
                "agent_run_id",
                valid_agent_run_id,
                "agent_run_id",
            )
        if experiment_id is not None:
            valid_experiment_id = _validate_positive_int_id(
                experiment_id, "experiment_id"
            )
            _require_existing_id(
                conn,
                "experiments",
                "experiment_id",
                valid_experiment_id,
                "experiment_id",
            )
        if method_ids is None:
            valid_method_ids = []
        elif not isinstance(method_ids, list):
            raise ValueError("method_ids must be a list of positive integers")
        elif method_ids:
            valid_method_ids = _validate_method_id_list(method_ids, conn)
        else:
            valid_method_ids = []
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO retrieval_events
            (agent_run_id, experiment_id, event_type, query, retrieved_content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                valid_agent_run_id,
                valid_experiment_id,
                event_type,
                query,
                retrieved_content,
            ),
        )
        event_id = cur.lastrowid

        unique_method_ids = valid_method_ids
        for method_id in unique_method_ids:
            cur.execute(
                """
                INSERT INTO retrieved_methods (event_id, method_id)
                VALUES (?, ?)
                """,
                (event_id, method_id),
            )
        if valid_experiment_id is not None and unique_method_ids:
            _insert_experiment_methods(
                cur,
                valid_experiment_id,
                unique_method_ids,
                role="retrieved",
                note=None,
            )
        conn.commit()
        return event_id
    finally:
        if close:
            conn.close()


def log_method_feedback(
    method_id: int,
    rating: float,
    event_id: Optional[int] = None,
    label: Optional[str] = None,
    note: Optional[str] = None,
    source: Optional[str] = None,
    conn=None,
) -> int:
    """Record a human or agent rating for a retrieved method."""
    if isinstance(rating, bool):
        raise ValueError("rating must be a number between -1.0 and 1.0")
    try:
        numeric_rating = float(rating)
    except (TypeError, ValueError) as exc:
        raise ValueError("rating must be a number between -1.0 and 1.0") from exc
    if not -1.0 <= numeric_rating <= 1.0:
        raise ValueError("rating must be between -1.0 and 1.0")
    valid_method_id = _validate_positive_int_id(method_id, "method_id")
    valid_event_id = (
        _validate_positive_int_id(event_id, "event_id")
        if event_id is not None
        else None
    )

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM methods WHERE method_id = ?", (valid_method_id,))
        if cur.fetchone() is None:
            raise ValueError(f"method_id {valid_method_id} does not exist")
        if valid_event_id is not None:
            cur.execute("SELECT 1 FROM retrieval_events WHERE event_id = ?", (valid_event_id,))
            if cur.fetchone() is None:
                raise ValueError(f"event_id {valid_event_id} does not exist")
        cur.execute(
            """
            INSERT INTO method_feedback
            (event_id, method_id, rating, label, note, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (valid_event_id, valid_method_id, numeric_rating, label, note, source),
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
    valid_prompt_tokens = _validate_non_negative_int(prompt_tokens, "prompt_tokens")
    valid_completion_tokens = _validate_non_negative_int(
        completion_tokens, "completion_tokens"
    )
    valid_cost = _validate_optional_non_negative_cost(cost)
    valid_experiment_id = (
        _validate_positive_int_id(experiment_id, "experiment_id")
        if experiment_id is not None
        else None
    )

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        if valid_experiment_id is not None:
            _require_existing_id(
                conn,
                "experiments",
                "experiment_id",
                valid_experiment_id,
                "experiment_id",
            )
        cur.execute(
            """
            INSERT INTO token_usage
            (experiment_id, prompt_tokens, completion_tokens, cost)
            VALUES (?, ?, ?, ?)
            """,
            (
                valid_experiment_id,
                valid_prompt_tokens,
                valid_completion_tokens,
                valid_cost,
            ),
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
