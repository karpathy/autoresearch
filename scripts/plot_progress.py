"""Generate progress.png chart from agent results — matches Karpathy's autoresearch style."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_file = sys.argv[1] if len(sys.argv) > 1 else "agent_results_pubmed.tsv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "progress.png"

    # Load results (columns: commit, val_bpb, memory_gb, status, description, sample_text)
    df = pd.read_csv(results_file, sep="\t", header=None,
                     names=["commit", "val_bpb", "memory_gb", "status", "description", "sample_text"])
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
    df = df.dropna(subset=["val_bpb"])

    # Filter out crashes and bogus values
    valid = df[(df["status"].isin(["keep", "discard"])) & (df["val_bpb"] > 0) & (df["val_bpb"] < 20)].copy()
    valid = valid.reset_index(drop=True)

    kept = valid[valid["status"] == "keep"]
    discarded = valid[valid["status"] == "discard"]

    # Running best (cumulative minimum of kept experiments)
    kept_sorted = kept.copy()
    kept_sorted["running_min"] = kept_sorted["val_bpb"].cummin()

    fig, ax = plt.subplots(figsize=(16, 8))

    # Discarded experiments — faint gray dots
    ax.scatter(discarded.index, discarded["val_bpb"], c="#cccccc", s=12, alpha=0.5,
               zorder=2, label="Discarded")

    # Kept experiments — green dots with black edges
    ax.scatter(kept.index, kept["val_bpb"], c="#2ecc71", s=50, zorder=4,
               edgecolors="black", linewidths=0.8, label="Kept")

    # Running minimum step line
    ax.step(kept.index.tolist(), kept_sorted["running_min"].tolist(),
            where="post", color="#27ae60", linewidth=2, zorder=3, label="Running best")

    # Annotate kept experiments
    for idx, row in kept.iterrows():
        # Truncate long descriptions
        desc = row["description"]
        if len(desc) > 50:
            desc = desc[:47] + "..."
        ax.annotate(desc, (idx, row["val_bpb"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, rotation=30, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

    # Focus y-axis on interesting region (baseline down to best, with margins)
    baseline_bpb = kept["val_bpb"].iloc[0] if len(kept) > 0 else valid["val_bpb"].max()
    best_bpb = valid["val_bpb"].min()
    margin = (baseline_bpb - best_bpb) * 0.15
    y_max = baseline_bpb + margin
    # Show discards above baseline but cap at reasonable level
    y_max = min(y_max, baseline_bpb * 1.3)
    y_min = best_bpb - margin
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"autoresearch: {baseline_bpb:.4f} → {best_bpb:.4f} val_bpb "
                 f"({(1 - best_bpb/baseline_bpb)*100:.1f}% improvement over "
                 f"{len(valid)} experiments)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
