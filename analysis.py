"""
CLI analysis tool for autoresearch experiment results.

Reads results.tsv and produces a summary report, optionally as JSON
or with a progress chart. Designed to be called by the autonomous agent
between experiments for structured feedback on what's working.

Usage:
    uv run analysis.py                          # text report to stdout
    uv run analysis.py --json                   # machine-readable JSON
    uv run analysis.py --plot progress.png      # save progress chart
    uv run analysis.py --tsv path/to/results.tsv  # custom TSV path
"""

import argparse
import json
import sys

import pandas as pd
import numpy as np


def load_results(tsv_path):
    """Load and clean results.tsv."""
    df = pd.read_csv(tsv_path, sep="\t")
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()
    return df


def compute_stats(df):
    """Compute summary statistics from experiment results."""
    counts = df["status"].value_counts()
    n_total = len(df)
    n_keep = int(counts.get("KEEP", 0))
    n_discard = int(counts.get("DISCARD", 0))
    n_crash = int(counts.get("CRASH", 0))
    n_decided = n_keep + n_discard

    kept = df[df["status"] == "KEEP"].copy()

    stats = {
        "total_experiments": n_total,
        "kept": n_keep,
        "discarded": n_discard,
        "crashed": n_crash,
        "keep_rate": round(n_keep / n_decided, 4) if n_decided > 0 else None,
    }

    if len(kept) == 0:
        stats["baseline_bpb"] = None
        stats["best_bpb"] = None
        stats["improvement"] = None
        stats["improvement_pct"] = None
        stats["best_experiment"] = None
        stats["top_hits"] = []
        stats["trajectory"] = "no_data"
        return stats

    baseline_bpb = float(kept.iloc[0]["val_bpb"])
    best_bpb = float(kept["val_bpb"].min())
    best_row = kept.loc[kept["val_bpb"].idxmin()]
    improvement = baseline_bpb - best_bpb

    stats["baseline_bpb"] = round(baseline_bpb, 6)
    stats["best_bpb"] = round(best_bpb, 6)
    stats["improvement"] = round(improvement, 6)
    stats["improvement_pct"] = round(improvement / baseline_bpb * 100, 2) if baseline_bpb > 0 else 0
    stats["best_experiment"] = str(best_row["description"]).strip()

    # Top hits: each kept experiment's delta vs previous kept
    kept = kept.reset_index(drop=True)
    top_hits = []
    for i in range(1, len(kept)):
        delta = float(kept.loc[i - 1, "val_bpb"] - kept.loc[i, "val_bpb"])
        top_hits.append({
            "commit": str(kept.loc[i, "commit"]),
            "val_bpb": round(float(kept.loc[i, "val_bpb"]), 6),
            "delta": round(delta, 6),
            "description": str(kept.loc[i, "description"]).strip(),
        })
    top_hits.sort(key=lambda x: x["delta"], reverse=True)
    stats["top_hits"] = top_hits

    # Trajectory: are recent experiments improving or plateauing?
    if len(kept) >= 3:
        recent_deltas = [
            float(kept.loc[i - 1, "val_bpb"] - kept.loc[i, "val_bpb"])
            for i in range(max(1, len(kept) - 3), len(kept))
        ]
        avg_recent = np.mean(recent_deltas)
        if avg_recent > 0.001:
            stats["trajectory"] = "improving"
        elif avg_recent > 0:
            stats["trajectory"] = "plateauing"
        else:
            stats["trajectory"] = "stuck"
    else:
        stats["trajectory"] = "early"

    return stats


def print_text_report(stats):
    """Print a human-readable text report."""
    print("=" * 60)
    print("AUTORESEARCH EXPERIMENT REPORT")
    print("=" * 60)
    print()

    print(f"Total experiments:  {stats['total_experiments']}")
    print(f"  Kept:             {stats['kept']}")
    print(f"  Discarded:        {stats['discarded']}")
    print(f"  Crashed:          {stats['crashed']}")
    if stats["keep_rate"] is not None:
        print(f"  Keep rate:        {stats['keep_rate']:.1%}")
    print()

    if stats["baseline_bpb"] is None:
        print("No kept experiments yet.")
        return

    print(f"Baseline val_bpb:   {stats['baseline_bpb']:.6f}")
    print(f"Best val_bpb:       {stats['best_bpb']:.6f}")
    print(f"Total improvement:  {stats['improvement']:.6f} ({stats['improvement_pct']:.2f}%)")
    print(f"Best experiment:    {stats['best_experiment']}")
    print(f"Trajectory:         {stats['trajectory']}")
    print()

    if stats["top_hits"]:
        print("Top improvements (by delta):")
        print(f"  {'Rank':>4}  {'Delta':>9}  {'BPB':>10}  Description")
        print("  " + "-" * 56)
        for rank, hit in enumerate(stats["top_hits"][:10], 1):
            print(f"  {rank:4d}  {hit['delta']:+.6f}  {hit['val_bpb']:.6f}  {hit['description']}")
    print()


def save_plot(df, output_path):
    """Save the progress chart to a file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))

    valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
    if len(valid) == 0:
        print(f"No valid experiments to plot.", file=sys.stderr)
        return

    kept_mask = valid["status"] == "KEEP"
    if not kept_mask.any():
        print(f"No kept experiments to plot.", file=sys.stderr)
        return

    baseline_bpb = valid.loc[0, "val_bpb"]
    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]

    # Discarded points
    disc = below[below["status"] == "DISCARD"]
    ax.scatter(disc.index, disc["val_bpb"],
               c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    # Kept points
    kept_v = below[below["status"] == "KEEP"]
    ax.scatter(kept_v.index, kept_v["val_bpb"],
               c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)

    # Running minimum
    kept_idx = valid.index[kept_mask]
    kept_bpb = valid.loc[kept_mask, "val_bpb"]
    running_min = kept_bpb.cummin()
    ax.step(kept_idx, running_min, where="post", color="#27ae60",
            linewidth=2, alpha=0.7, zorder=3, label="Running best")

    # Labels
    for idx, bpb in zip(kept_idx, kept_bpb):
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."
        ax.annotate(desc, (idx, bpb),
                    textcoords="offset points", xytext=(6, 6), fontsize=8.0,
                    color="#1a7a3a", alpha=0.9, rotation=30, ha="left", va="bottom")

    n_total = len(df)
    n_kept = len(df[df["status"] == "KEEP"])
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    best_bpb = kept_bpb.min()
    margin = (baseline_bpb - best_bpb) * 0.15
    if margin > 0:
        ax.set_ylim(best_bpb - margin, baseline_bpb + margin)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Analyze autoresearch experiment results")
    parser.add_argument("--tsv", default="results.tsv", help="Path to results.tsv (default: results.tsv)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    parser.add_argument("--plot", metavar="PATH", help="Save progress chart to file (e.g. progress.png)")
    args = parser.parse_args()

    try:
        df = load_results(args.tsv)
    except FileNotFoundError:
        print(f"Error: {args.tsv} not found. Run some experiments first.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: {args.tsv} is empty.", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(df)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_text_report(stats)

    if args.plot:
        save_plot(df, args.plot)


if __name__ == "__main__":
    main()
