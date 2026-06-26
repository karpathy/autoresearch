"""Metrics and reporting for NanoChat Autoresearch agent runs."""

from __future__ import annotations

from statistics import mean
from typing import Optional

from researchfs.db import get_connection


COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
}


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: Optional[str]) -> float:
    """Estimate API cost from token counts and model name."""
    if model is None:
        return 0.0
    rates = COST_PER_1K_TOKENS.get(model)
    if rates is None:
        return 0.0
    return (
        prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]
    ) / 1000


def _get_cost(prompt_tokens: int, completion_tokens: int, model: Optional[str]) -> float:
    """Deprecated alias for estimate_cost."""
    return estimate_cost(prompt_tokens, completion_tokens, model)


def load_run(agent_run_id: int, conn=None) -> dict:
    """Load a single agent run with its experiments, tokens, and retrievals."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT agent_run_id, mode, model, started_at, finished_at
            FROM agent_runs
            WHERE agent_run_id = ?
            """,
            (agent_run_id,),
        )
        run_row = cur.fetchone()
        if run_row is None:
            raise ValueError(f"Agent run {agent_run_id} not found")

        cur.execute(
            """
            SELECT
                experiment_id,
                commit_hash,
                val_bpb,
                memory_gb,
                status,
                description,
                timestamp
            FROM experiments
            WHERE agent_run_id = ?
            ORDER BY experiment_id
            """,
            (agent_run_id,),
        )
        experiments = [dict(row) for row in cur.fetchall()]

        cur.execute(
            """
            SELECT
                SUM(prompt_tokens) AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(cost) AS logged_cost
            FROM token_usage
            WHERE experiment_id IN (
                SELECT experiment_id FROM experiments WHERE agent_run_id = ?
            )
            """,
            (agent_run_id,),
        )
        token_row = cur.fetchone()
        prompt_tokens = token_row["prompt_tokens"] or 0
        completion_tokens = token_row["completion_tokens"] or 0
        logged_cost = token_row["logged_cost"] or 0.0
        estimated_cost = estimate_cost(prompt_tokens, completion_tokens, run_row["model"])
        cost = logged_cost if logged_cost else estimated_cost

        cur.execute(
            "SELECT COUNT(*) AS count FROM retrieval_events WHERE agent_run_id = ?",
            (agent_run_id,),
        )
        retrieval_count = cur.fetchone()["count"]

        kept = [e for e in experiments if e["status"] == "keep"]
        valid_bpb = [e["val_bpb"] for e in kept if e["val_bpb"] is not None]
        baseline_experiment = next(
            (e for e in experiments if e["commit_hash"] == "baseline" and e["val_bpb"] is not None),
            None,
        )
        baseline_bpb = baseline_experiment["val_bpb"] if baseline_experiment else None
        best_bpb = min(valid_bpb) if valid_bpb else None
        mean_bpb = mean(valid_bpb) if valid_bpb else None
        improvement = (
            baseline_bpb - best_bpb
            if baseline_bpb is not None and best_bpb is not None
            else None
        )
        total_tokens = prompt_tokens + completion_tokens

        return {
            "agent_run_id": agent_run_id,
            "mode": run_row["mode"],
            "model": run_row["model"],
            "started_at": run_row["started_at"],
            "finished_at": run_row["finished_at"],
            "baseline_bpb": baseline_bpb,
            "best_bpb": best_bpb,
            "mean_bpb": mean_bpb,
            "improvement": improvement,
            "total_experiments": len(experiments),
            "kept": len(kept),
            "crashed": sum(1 for e in experiments if e["status"] == "crash"),
            "discarded": sum(1 for e in experiments if e["status"] == "discard"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "retrieval_count": retrieval_count,
            "tokens_per_improvement": (
                total_tokens / improvement if improvement and improvement > 0 else None
            ),
            "experiments_per_improvement": (
                len(experiments) / improvement if improvement and improvement > 0 else None
            ),
            "experiments": experiments,
        }
    finally:
        if close:
            conn.close()


def compare_runs(baseline_run_id: int, researchfs_run_id: int, conn=None) -> dict:
    """Return side-by-side metrics for baseline and ResearchFS agent runs."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        baseline = load_run(baseline_run_id, conn)
        researchfs = load_run(researchfs_run_id, conn)

        if baseline["mode"] != "baseline":
            raise ValueError(
                f"Run {baseline_run_id} is mode '{baseline['mode']}', expected 'baseline'"
            )
        if researchfs["mode"] != "loopgraph":
            raise ValueError(
                f"Run {researchfs_run_id} is mode '{researchfs['mode']}', expected 'loopgraph'"
            )

        token_reduction = None
        if baseline["total_tokens"] and researchfs["total_tokens"]:
            token_reduction = 1 - researchfs["total_tokens"] / baseline["total_tokens"]

        retrieval_reduction = None
        if baseline["retrieval_count"] and researchfs["retrieval_count"]:
            retrieval_reduction = (
                1 - researchfs["retrieval_count"] / baseline["retrieval_count"]
            )

        return {
            "baseline_run_id": baseline_run_id,
            "researchfs_run_id": researchfs_run_id,
            "baseline": baseline,
            "researchfs": researchfs,
            "best_bpb_delta": (
                researchfs["best_bpb"] - baseline["best_bpb"]
                if baseline["best_bpb"] is not None and researchfs["best_bpb"] is not None
                else None
            ),
            "token_reduction_ratio": token_reduction,
            "retrieval_reduction_ratio": retrieval_reduction,
        }
    finally:
        if close:
            conn.close()


def format_comparison(comparison: dict) -> str:
    """Render a comparison dict as a readable Markdown table."""
    b = comparison["baseline"]
    r = comparison["researchfs"]

    def fmt(val, fmt_spec: str = ".6f"):
        if val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if fmt_spec == ".0f":
                return f"{int(val):,}"
            return f"{val:{fmt_spec}}"
        return str(val)

    lines = [
        "# Agent Run Comparison: Baseline vs ResearchFS",
        "",
        f"| Metric | Baseline ({comparison['baseline_run_id']}) | ResearchFS ({comparison['researchfs_run_id']}) |",
        "|---|---|---|",
        f"| Model | {b['model']} | {r['model']} |",
        f"| Baseline BPB | {fmt(b['baseline_bpb'])} | {fmt(r['baseline_bpb'])} |",
        f"| Best BPB | {fmt(b['best_bpb'])} | {fmt(r['best_bpb'])} |",
        f"| Mean kept BPB | {fmt(b['mean_bpb'])} | {fmt(r['mean_bpb'])} |",
        f"| BPB improvement | {fmt(b['improvement'])} | {fmt(r['improvement'])} |",
        f"| Total experiments | {fmt(b['total_experiments'], '.0f')} | {fmt(r['total_experiments'], '.0f')} |",
        f"| Kept / Crashed / Discarded | {b['kept']}/{b['crashed']}/{b['discarded']} | {r['kept']}/{r['crashed']}/{r['discarded']} |",
        f"| Prompt tokens | {fmt(b['prompt_tokens'], '.0f')} | {fmt(r['prompt_tokens'], '.0f')} |",
        f"| Completion tokens | {fmt(b['completion_tokens'], '.0f')} | {fmt(r['completion_tokens'], '.0f')} |",
        f"| Total tokens | {fmt(b['total_tokens'], '.0f')} | {fmt(r['total_tokens'], '.0f')} |",
        f"| Estimated cost | ${fmt(b['cost'], '.4f')} | ${fmt(r['cost'], '.4f')} |",
        f"| Retrieval/context calls | {fmt(b['retrieval_count'], '.0f')} | {fmt(r['retrieval_count'], '.0f')} |",
        f"| Tokens per BPB improvement | {fmt(b['tokens_per_improvement'], '.0f')} | {fmt(r['tokens_per_improvement'], '.0f')} |",
        "",
    ]

    if comparison["best_bpb_delta"] is not None:
        delta = comparison["best_bpb_delta"]
        direction = "better" if delta < 0 else "worse"
        lines.append(
            f"ResearchFS best BPB is **{abs(delta):.6f} {direction}** than baseline."
        )
    if comparison["token_reduction_ratio"] is not None:
        pct = comparison["token_reduction_ratio"] * 100
        lines.append(
            f"ResearchFS used **{pct:.1f}% fewer tokens** than baseline."
        )
    if comparison["retrieval_reduction_ratio"] is not None:
        pct = comparison["retrieval_reduction_ratio"] * 100
        lines.append(
            f"ResearchFS used **{pct:.1f}% fewer retrieval calls** than baseline."
        )

    return "\n".join(lines)


def list_recent_runs(n: int = 10, conn=None) -> list[dict]:
    """List the most recent agent runs."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT agent_run_id, mode, model, started_at, finished_at
            FROM agent_runs
            ORDER BY agent_run_id DESC
            LIMIT ?
            """,
            (n,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        if close:
            conn.close()
