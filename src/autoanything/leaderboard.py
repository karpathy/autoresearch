"""Leaderboard rendering — generate leaderboard.md from history."""

import sqlite3


def export_leaderboard(conn: sqlite3.Connection, output_path: str,
                       direction: str = "minimize"):
    """Export the leaderboard to a markdown file.

    Args:
        conn: SQLite connection to the history database.
        output_path: Path to write leaderboard.md.
        direction: "minimize" or "maximize" — controls sort order.
    """
    order = "ASC" if direction == "minimize" else "DESC"

    # Top scores (accepted + baseline)
    top = conn.execute(f"""
        SELECT score, branch, description, evaluated_at
        FROM evaluations WHERE status IN ('baseline', 'accepted')
        ORDER BY score {order}
    """).fetchall()

    # Recent attempts (last 20)
    recent = conn.execute("""
        SELECT score, status, branch, description, evaluated_at
        FROM evaluations ORDER BY id DESC LIMIT 20
    """).fetchall()

    lines = ["# Leaderboard", ""]
    lines.append("| # | Score | Branch | Description | When |")
    lines.append("|---|-------|--------|-------------|------|")
    for i, (score, branch, desc, when) in enumerate(top, 1):
        score_str = f"{score:.6f}" if score is not None else "crash"
        when_short = when[:16] if when else ""
        lines.append(
            f"| {i} | {score_str} | {branch} | {desc or ''} | {when_short} |"
        )

    lines.extend(["", "## Recent Attempts", ""])
    lines.append("| Score | Status | Branch | Description | When |")
    lines.append("|-------|--------|--------|-------------|------|")
    for score, status, branch, desc, when in recent:
        score_str = f"{score:.6f}" if score is not None else "crash"
        when_short = when[11:16] if when and len(when) > 16 else (when or "")
        lines.append(
            f"| {score_str} | {status} | {branch} | {desc or ''} | {when_short} |"
        )

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
