#!/usr/bin/env python3
"""
Autoresearch Experiment Analysis CLI Tool

Analyzes results.tsv from autonomous experiments and provides structured feedback
for both humans and AI agents.

Usage:
    uv run analysis.py                    # text report to stdout
    uv run analysis.py --json            # machine-readable JSON for agents
    uv run analysis.py --plot progress.png # save progress chart
    uv run analysis.py --tsv custom.tsv  # custom results file path
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(tsv_path):
    """Load and validate results.tsv file."""
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
        df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
        df["status"] = df["status"].str.strip().str.upper()
        return df
    except FileNotFoundError:
        print(f"Error: Results file not found: {tsv_path}", file=sys.stderr)
        print("Run some experiments first to generate results.tsv", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_trajectory(df):
    """Analyze experiment trajectory to detect if progress is plateauing."""
    kept = df[df["status"] == "KEEP"].copy()
    if len(kept) < 3:
        return "insufficient_data"
    
    # Look at recent improvements
    recent_kept = kept.tail(5)
    if len(recent_kept) < 3:
        return "insufficient_data"
    
    # Calculate improvement rate in recent experiments
    improvements = []
    for i in range(1, len(recent_kept)):
        prev_bpb = recent_kept.iloc[i-1]["val_bpb"]
        curr_bpb = recent_kept.iloc[i]["val_bpb"]
        improvements.append(prev_bpb - curr_bpb)
    
    avg_improvement = np.mean(improvements)
    
    # Determine trajectory - adjusted thresholds
    if avg_improvement > 0.0005:  # Good improvements
        return "improving"
    elif avg_improvement > 0.00005:  # Small but consistent improvements
        return "plateauing"
    else:
        return "stuck"


def generate_text_report(df):
    """Generate human-readable text report."""
    counts = df["status"].value_counts()
    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_crash = counts.get("CRASH", 0)
    n_total = len(df)
    
    report = []
    report.append("=== AUTORESEARCH EXPERIMENT ANALYSIS ===\n")
    
    # Summary stats
    report.append(f"Total experiments: {n_total}")
    report.append(f"Kept: {n_keep} | Discarded: {n_discard} | Crashed: {n_crash}")
    
    n_decided = n_keep + n_discard
    if n_decided > 0:
        report.append(f"Keep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
    
    # Baseline and best
    if len(df) > 0:
        baseline_bpb = df.iloc[0]["val_bpb"]
        kept = df[df["status"] == "KEEP"].copy()
        
        if len(kept) > 0:
            best_bpb = kept["val_bpb"].min()
            best_row = kept.loc[kept["val_bpb"].idxmin()]
            improvement = baseline_bpb - best_bpb
            improvement_pct = improvement / baseline_bpb * 100
            
            report.append(f"\nBaseline BPB: {baseline_bpb:.6f}")
            report.append(f"Best BPB:     {best_bpb:.6f}")
            report.append(f"Improvement:  {improvement:.6f} ({improvement_pct:.2f}%)")
            report.append(f"Best experiment: {best_row['description']}")
        
        # Trajectory analysis
        trajectory = analyze_trajectory(df)
        report.append(f"\nTrajectory: {trajectory}")
        
        # Top improvements
        if len(kept) > 1:
            kept["prev_bpb"] = kept["val_bpb"].shift(1)
            kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]
            hits = kept.iloc[1:].copy().sort_values("delta", ascending=False)
            
            report.append(f"\nTop improvements:")
            report.append(f"{'Rank':>4}  {'Delta':>8}  {'BPB':>10}  Description")
            report.append("-" * 70)
            
            for rank, (_, row) in enumerate(hits.head(5).iterrows(), 1):
                desc = str(row["description"])[:45]
                if len(str(row["description"])) > 45:
                    desc += "..."
                report.append(f"{rank:4d}  {row['delta']:+.6f}  {row['val_bpb']:.6f}  {desc}")
    
    return "\n".join(report)


def generate_json_report(df):
    """Generate machine-readable JSON report for agents."""
    counts = df["status"].value_counts()
    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_crash = counts.get("CRASH", 0)
    n_total = len(df)
    
    # Basic stats
    result = {
        "total_experiments": int(n_total),
        "kept": int(n_keep),
        "discarded": int(n_discard),
        "crashed": int(n_crash),
        "keep_rate": float(n_keep / (n_keep + n_discard)) if (n_keep + n_discard) > 0 else 0.0,
    }
    
    # Performance metrics
    if len(df) > 0:
        baseline_bpb = float(df.iloc[0]["val_bpb"])
        result["baseline_bpb"] = baseline_bpb
        
        kept = df[df["status"] == "KEEP"].copy()
        if len(kept) > 0:
            best_bpb = float(kept["val_bpb"].min())
            best_row = kept.loc[kept["val_bpb"].idxmin()]
            improvement = baseline_bpb - best_bpb
            
            result.update({
                "best_bpb": best_bpb,
                "improvement": improvement,
                "improvement_pct": float(improvement / baseline_bpb * 100),
                "best_experiment": str(best_row["description"]),
            })
    
    # Trajectory analysis
    result["trajectory"] = analyze_trajectory(df)
    
    # Top hits for agent analysis
    kept = df[df["status"] == "KEEP"].copy()
    if len(kept) > 1:
        kept["prev_bpb"] = kept["val_bpb"].shift(1)
        kept["delta"] = kept["prev_bpb"] - kept["val_bpb"]
        hits = kept.iloc[1:].copy().sort_values("delta", ascending=False)
        
        result["top_hits"] = [
            {
                "rank": rank,
                "delta": float(row["delta"]),
                "val_bpb": float(row["val_bpb"]),
                "description": str(row["description"])
            }
            for rank, (_, row) in enumerate(hits.head(5).iterrows(), 1)
        ]
    else:
        result["top_hits"] = []
    
    return result


def create_progress_plot(df, output_path):
    """Create progress plot and save to file."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Filter out crashes for plotting
    valid = df[df["status"] != "CRASH"].copy()
    valid = valid.reset_index(drop=True)
    
    if len(valid) == 0:
        print("No valid experiments to plot", file=sys.stderr)
        return
    
    baseline_bpb = valid.loc[0, "val_bpb"]
    
    # Only plot points at or below baseline (the interesting region)
    below = valid[valid["val_bpb"] <= baseline_bpb + 0.0005]
    
    # Plot discarded as faint background dots
    disc = below[below["status"] == "DISCARD"]
    if len(disc) > 0:
        ax.scatter(disc.index, disc["val_bpb"],
                   c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")
    
    # Plot kept experiments as prominent green dots
    kept_v = below[below["status"] == "KEEP"]
    if len(kept_v) > 0:
        ax.scatter(kept_v.index, kept_v["val_bpb"],
                   c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)
        
        # Running minimum step line
        kept_mask = valid["status"] == "KEEP"
        kept_idx = valid.index[kept_mask]
        kept_bpb = valid.loc[kept_mask, "val_bpb"]
        running_min = kept_bpb.cummin()
        ax.step(kept_idx, running_min, where="post", color="#27ae60",
                linewidth=2, alpha=0.7, zorder=3, label="Running best")
        
        # Label each kept experiment
        for idx, bpb in zip(kept_idx, kept_bpb):
            desc = str(valid.loc[idx, "description"]).strip()
            if len(desc) > 45:
                desc = desc[:42] + "..."
            ax.annotate(desc, (idx, bpb),
                        textcoords="offset points",
                        xytext=(6, 6), fontsize=8.0,
                        color="#1a7a3a", alpha=0.9,
                        rotation=30, ha="left", va="bottom")
    
    n_total = len(df)
    n_kept = len(df[df["status"] == "KEEP"])
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # Y-axis: from just below best to just above baseline
    if len(kept_v) > 0:
        best_bpb = kept_bpb.min()
        margin = (baseline_bpb - best_bpb) * 0.15
        ax.set_ylim(best_bpb - margin, baseline_bpb + margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Progress plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze autoresearch experiment results")
    parser.add_argument("--tsv", default="results.tsv", help="Path to results.tsv file")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--plot", help="Save progress plot to file")
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.tsv)
    
    # Generate output
    if args.json:
        report = generate_json_report(df)
        print(json.dumps(report, indent=2))
    else:
        report = generate_text_report(df)
        print(report)
    
    # Generate plot if requested
    if args.plot:
        create_progress_plot(df, args.plot)


if __name__ == "__main__":
    main()
