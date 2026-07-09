"""Metrics and reporting for NanoChat Autoresearch agent runs."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Optional

from researchfs.db import get_connection


NANOCHAT_BENCHMARKS = {
    "autoresearch_at_home": {
        "bpb": 0.9372,
        "source": "https://www.recursive.com/articles/first-steps-toward-automated-ai-research",
        "description": "Previous public best from autoresearch@home, reward hacks removed, mean over 10 random seeds, evaluated on B200.",
        "device": "NVIDIA B200",
    },
    "recursive_best": {
        "bpb": 0.9109,
        "source": "https://www.recursive.com/articles/first-steps-toward-automated-ai-research",
        "description": "Recursive automated research system starting from the Autoresearch seed, evaluated on B200.",
        "device": "NVIDIA B200",
    },
    "recursive_vanilla_transformer": {
        "bpb": 0.9344,
        "source": "https://www.recursive.com/articles/first-steps-toward-automated-ai-research",
        "description": "Recursive automated research system starting from a vanilla Transformer with AdamW, evaluated on B200.",
        "device": "NVIDIA B200",
    },
}


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
                COUNT(*) AS usage_rows,
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
        has_token_usage = token_row["usage_rows"] > 0
        if has_token_usage:
            prompt_tokens = token_row["prompt_tokens"] or 0
            completion_tokens = token_row["completion_tokens"] or 0
            logged_cost = token_row["logged_cost"]
            estimated_cost = estimate_cost(prompt_tokens, completion_tokens, run_row["model"])
            cost = logged_cost if logged_cost is not None else estimated_cost
        else:
            prompt_tokens = None
            completion_tokens = None
            cost = None

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
        total_tokens = (
            prompt_tokens + completion_tokens if has_token_usage else None
        )

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
            "has_token_usage": has_token_usage,
            "retrieval_count": retrieval_count,
            "tokens_per_improvement": (
                total_tokens / improvement
                if total_tokens is not None and improvement and improvement > 0
                else None
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
        baseline_tokens = baseline["total_tokens"]
        researchfs_tokens = researchfs["total_tokens"]
        if (
            baseline_tokens is not None
            and researchfs_tokens is not None
            and baseline_tokens > 0
        ):
            token_reduction = 1 - researchfs_tokens / baseline_tokens

        retrieval_reduction = None
        baseline_retrievals = baseline["retrieval_count"]
        if baseline_retrievals > 0:
            retrieval_reduction = (
                1 - researchfs["retrieval_count"] / baseline_retrievals
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

    def fmt_cost(val):
        if val is None:
            return "N/A"
        return f"${fmt(val, '.4f')}"

    lines = [
        "# Agent Run Comparison: Baseline vs ResearchFS",
        "",
        f"| Metric | Baseline ({comparison['baseline_run_id']}) | ResearchFS ({comparison['researchfs_run_id']}) |",
        "|---|---|---|",
        f"| Model | {b['model'] or 'N/A'} | {r['model'] or 'N/A'} |",
        f"| Baseline BPB | {fmt(b['baseline_bpb'])} | {fmt(r['baseline_bpb'])} |",
        f"| Best BPB | {fmt(b['best_bpb'])} | {fmt(r['best_bpb'])} |",
        f"| Mean kept BPB | {fmt(b['mean_bpb'])} | {fmt(r['mean_bpb'])} |",
        f"| BPB improvement | {fmt(b['improvement'])} | {fmt(r['improvement'])} |",
        f"| Total experiments | {fmt(b['total_experiments'], '.0f')} | {fmt(r['total_experiments'], '.0f')} |",
        f"| Kept / Crashed / Discarded | {b['kept']}/{b['crashed']}/{b['discarded']} | {r['kept']}/{r['crashed']}/{r['discarded']} |",
        f"| Prompt tokens | {fmt(b['prompt_tokens'], '.0f')} | {fmt(r['prompt_tokens'], '.0f')} |",
        f"| Completion tokens | {fmt(b['completion_tokens'], '.0f')} | {fmt(r['completion_tokens'], '.0f')} |",
        f"| Total tokens | {fmt(b['total_tokens'], '.0f')} | {fmt(r['total_tokens'], '.0f')} |",
        f"| Cost (logged/estimated) | {fmt_cost(b['cost'])} | {fmt_cost(r['cost'])} |",
        f"| Retrieval/context calls | {fmt(b['retrieval_count'], '.0f')} | {fmt(r['retrieval_count'], '.0f')} |",
        f"| Tokens per BPB improvement | {fmt(b['tokens_per_improvement'], '.0f')} | {fmt(r['tokens_per_improvement'], '.0f')} |",
        "",
    ]

    if comparison["best_bpb_delta"] is not None:
        delta = comparison["best_bpb_delta"]
        if delta < 0:
            lines.append(
                f"ResearchFS best BPB is **{abs(delta):.6f} better** than baseline."
            )
        elif delta > 0:
            lines.append(
                f"ResearchFS best BPB is **{abs(delta):.6f} worse** than baseline."
            )
        else:
            lines.append("ResearchFS best BPB is **equal** to baseline.")

    def append_usage_change(ratio, noun: str):
        if ratio is None:
            return
        if ratio > 0:
            lines.append(
                f"ResearchFS used **{abs(ratio) * 100:.1f}% fewer {noun}** than baseline."
            )
        elif ratio < 0:
            lines.append(
                f"ResearchFS used **{abs(ratio) * 100:.1f}% more {noun}** than baseline."
            )
        else:
            lines.append(f"ResearchFS used **the same number of {noun}** as baseline.")

    append_usage_change(comparison["token_reduction_ratio"], "tokens")
    append_usage_change(comparison["retrieval_reduction_ratio"], "retrieval calls")

    return "\n".join(lines)


def compare_run_to_benchmarks(
    agent_run_id: int, benchmarks: Optional[dict] = None, conn=None
) -> dict:
    """Compare a single run's best BPB against published external benchmarks."""
    if benchmarks is None:
        benchmarks = NANOCHAT_BENCHMARKS

    run = load_run(agent_run_id, conn)
    run_best = run.get("best_bpb")

    per_benchmark = {}
    for name, meta in benchmarks.items():
        benchmark_bpb = meta.get("bpb")
        delta = (
            run_best - benchmark_bpb
            if run_best is not None and benchmark_bpb is not None
            else None
        )
        per_benchmark[name] = {
            "bpb": benchmark_bpb,
            "delta": delta,
            "source": meta.get("source"),
            "description": meta.get("description"),
            "device": meta.get("device"),
        }

    strongest_name = None
    gap_to_strongest = None
    valid_benchmarks = [
        (name, meta["bpb"])
        for name, meta in per_benchmark.items()
        if meta["bpb"] is not None
    ]
    if valid_benchmarks:
        strongest_name, strongest_bpb = min(valid_benchmarks, key=lambda item: item[1])
        gap_to_strongest = (
            run_best - strongest_bpb if run_best is not None else None
        )

    return {
        "agent_run_id": agent_run_id,
        "model": run.get("model"),
        "mode": run.get("mode"),
        "best_bpb": run_best,
        "baseline_bpb": run.get("baseline_bpb"),
        "benchmarks": per_benchmark,
        "gap_to_sota": gap_to_strongest,
        "strongest_benchmark": strongest_name,
    }


def format_benchmark_comparison(
    comparison: dict,
    title: str = "Benchmark Comparison: Local Run vs Published Results",
    heading_level: int = 1,
) -> str:
    """Render a benchmark comparison dict as Markdown."""

    def fmt(val, fmt_spec: str = ".6f"):
        if val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if fmt_spec == ".0f":
                return f"{int(val):,}"
            return f"{val:{fmt_spec}}"
        return str(val)

    heading = "#" * max(1, min(6, heading_level))
    lines = [
        f"{heading} {title}",
        "",
        f"Run ID: {comparison['agent_run_id']}",
        f"Mode: {comparison['mode']}",
        f"Model: {comparison['model'] or 'N/A'}",
        "",
        "| Benchmark | BPB | Local Best BPB | Delta | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for name, meta in comparison["benchmarks"].items():
        delta = meta["delta"]
        delta_str = fmt(delta)
        if delta is not None:
            if delta < 0:
                delta_str = f"{fmt(delta)} (better)"
            elif delta > 0:
                delta_str = f"{fmt(delta)} (worse)"
            else:
                delta_str = f"{fmt(delta)} (same)"
        note_parts = []
        if meta.get("device"):
            note_parts.append(meta["device"])
        if meta.get("description"):
            note_parts.append(meta["description"])
        note = ", ".join(note_parts)
        lines.append(
            f"| {name} | {fmt(meta['bpb'])} | {fmt(comparison['best_bpb'])} | "
            f"{delta_str} | {note or 'N/A'} |"
        )

    lines.append("")
    strongest_name = comparison.get("strongest_benchmark")
    if strongest_name and comparison.get("gap_to_sota") is not None:
        gap = comparison["gap_to_sota"]
        if gap < 0:
            lines.append(
                f"Local best BPB is **{fmt(abs(gap))} better** than the strongest "
                f"published benchmark ({strongest_name})."
            )
        elif gap > 0:
            lines.append(
                f"Local best BPB is **{fmt(abs(gap))} worse** than the strongest "
                f"published benchmark ({strongest_name})."
            )
        else:
            lines.append(
                f"Local best BPB is **equal to** the strongest published benchmark "
                f"({strongest_name})."
            )
    elif comparison.get("best_bpb") is None:
        lines.append(
            "Local run has no measurable best BPB, so benchmark comparison is unavailable."
        )

    return "\n".join(lines)


def aggregate_seeded_runs(run_ids: list[int], conn=None) -> dict:
    """Aggregate best BPB across multiple seeded runs of the same recipe."""
    if not run_ids:
        raise ValueError("run_ids must not be empty")

    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_ids must be distinct for seed aggregation")

    runs = [load_run(run_id, conn) for run_id in run_ids]

    def values_for(field: str) -> set:
        return {run.get(field) for run in runs}

    modes = values_for("mode")
    models = values_for("model")
    if len(modes) > 1 or len(models) > 1:
        def describe(values: set) -> str:
            return ", ".join(
                sorted("N/A" if value is None else str(value) for value in values)
            )

        raise ValueError(
            "All aggregated seeds must use the same mode and model; "
            f"got modes [{describe(modes)}] and models [{describe(models)}]"
        )

    run_bests = [
        (run["agent_run_id"], run["best_bpb"])
        for run in runs
        if run.get("best_bpb") is not None
    ]
    invalid_run_ids = [
        run["agent_run_id"] for run in runs if run.get("best_bpb") is None
    ]
    if len(run_bests) < 2:
        raise ValueError("Need at least two runs with valid best_bpb to aggregate")

    best_bpbs = [best_bpb for _, best_bpb in run_bests]
    mean_bpb = mean(best_bpbs)
    std_bpb = stdev(best_bpbs)
    min_bpb = min(best_bpbs)
    max_bpb = max(best_bpbs)

    valid_benchmarks = [
        (name, meta.get("bpb"))
        for name, meta in NANOCHAT_BENCHMARKS.items()
        if meta.get("bpb") is not None
    ]
    strongest_name, strongest_bpb = (
        min(valid_benchmarks, key=lambda item: item[1])
        if valid_benchmarks
        else (None, None)
    )
    gap_to_strongest = mean_bpb - strongest_bpb if strongest_bpb is not None else None

    return {
        "run_ids": run_ids,
        "valid_run_ids": [run_id for run_id, _ in run_bests],
        "invalid_run_ids": invalid_run_ids,
        "n_seeds": len(run_ids),
        "valid_seeds": len(run_bests),
        "mean_bpb": mean_bpb,
        "std_bpb": std_bpb,
        "min_bpb": min_bpb,
        "max_bpb": max_bpb,
        "best_bpbs": best_bpbs,
        "mode": runs[0].get("mode"),
        "model": runs[0].get("model"),
        "gap_to_sota": gap_to_strongest,
        "strongest_benchmark": strongest_name,
    }


def detect_reward_hacks(run: dict, z_threshold: float = 3.0) -> list[dict]:
    """Flag suspicious run patterns that need extra validation."""
    flags = []
    kept = [e for e in run.get("experiments", []) if e.get("status") == "keep"]
    valid = [e for e in kept if e.get("val_bpb") is not None]

    if len(valid) >= 2:
        values = [e["val_bpb"] for e in valid]
        avg = mean(values)
        spread = stdev(values)
        if spread > 0:
            for experiment in valid:
                z_score = (experiment["val_bpb"] - avg) / spread
                if abs(z_score) > z_threshold:
                    flags.append(
                        {
                            "experiment_id": experiment["experiment_id"],
                            "val_bpb": experiment["val_bpb"],
                            "reason": (
                                f"BPB z-score {z_score:.2f} exceeds threshold "
                                f"{z_threshold} (mean={avg:.4f}, std={spread:.4f})"
                            ),
                        }
                    )

    best_bpb = run.get("best_bpb")
    strongest_bpb = NANOCHAT_BENCHMARKS.get("recursive_best", {}).get("bpb")
    if best_bpb is not None and strongest_bpb is not None:
        if best_bpb < strongest_bpb - 0.02:
            flags.append(
                {
                    "experiment_id": None,
                    "val_bpb": best_bpb,
                    "reason": (
                        f"Single-seed best BPB {best_bpb:.4f} beats Recursive "
                        f"multi-seed mean {strongest_bpb:.4f} by more than 0.02; "
                        "requires multi-seed validation"
                    ),
                }
            )

    for experiment in kept:
        if experiment.get("val_bpb") is None:
            flags.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "val_bpb": None,
                    "reason": "Kept experiment has no val_bpb; cannot verify it is real",
                }
            )

    return flags


def format_reward_hack_report(run: dict, flags: list[dict]) -> str:
    """Render reward-hack flags as Markdown."""
    run_id = run.get("agent_run_id", "unknown")
    run_label = f"run {run_id}"
    if run.get("mode"):
        run_label = f"{run_label} ({run['mode']})"
    lines = [
        f"## Reward-hack screening: {run_label}",
        "",
    ]
    if not flags:
        lines.append("No suspicious patterns detected.")
        return "\n".join(lines)

    lines.append(
        f"Detected **{len(flags)}** suspicious pattern(s) in run {run_id}:"
    )
    lines.append("")
    for index, flag in enumerate(flags, 1):
        experiment = (
            f"(experiment {flag['experiment_id']})"
            if flag["experiment_id"] is not None
            else "(run-level)"
        )
        bpb = (
            f"BPB {flag['val_bpb']:.4f}" if flag["val_bpb"] is not None else "no BPB"
        )
        lines.append(f"{index}. {experiment} {bpb}: {flag['reason']}")
    return "\n".join(lines)


def format_seeded_aggregate(aggregate: dict) -> str:
    """Render a multi-seed aggregate dict as Markdown."""

    def fmt(val, fmt_spec: str = ".6f"):
        if val is None:
            return "N/A"
        if isinstance(val, (int, float)):
            if fmt_spec == ".0f":
                return f"{int(val):,}"
            return f"{val:{fmt_spec}}"
        return str(val)

    lines = [
        "# Multi-Seed Aggregate Results",
        "",
        f"Mode: {aggregate.get('mode') or 'N/A'}",
        f"Model: {aggregate.get('model') or 'N/A'}",
        f"Seeds evaluated: {aggregate['valid_seeds']} / {aggregate['n_seeds']}",
    ]
    if aggregate.get("invalid_run_ids"):
        skipped = ", ".join(str(run_id) for run_id in aggregate["invalid_run_ids"])
        lines.append(f"Skipped run IDs without valid best BPB: {skipped}")
    lines.extend(
        [
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mean BPB | {fmt(aggregate['mean_bpb'])} |",
            f"| StdDev BPB | {fmt(aggregate['std_bpb'])} |",
            f"| Min BPB | {fmt(aggregate['min_bpb'])} |",
            f"| Max BPB | {fmt(aggregate['max_bpb'])} |",
            "",
            "| Run ID | Best BPB |",
            "|---|---:|",
        ]
    )
    for run_id, best_bpb in zip(aggregate["valid_run_ids"], aggregate["best_bpbs"]):
        lines.append(f"| {run_id} | {fmt(best_bpb)} |")

    strongest_name = aggregate.get("strongest_benchmark")
    if strongest_name and aggregate.get("gap_to_sota") is not None:
        lines.append("")
        gap = aggregate["gap_to_sota"]
        if gap < 0:
            lines.append(
                f"Mean BPB across seeds is **{fmt(abs(gap))} better** than the "
                f"strongest published benchmark ({strongest_name})."
            )
        elif gap > 0:
            lines.append(
                f"Mean BPB across seeds is **{fmt(abs(gap))} worse** than the "
                f"strongest published benchmark ({strongest_name})."
            )
        else:
            lines.append(
                f"Mean BPB across seeds is **equal to** the strongest published "
                f"benchmark ({strongest_name})."
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
