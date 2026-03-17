"""Generate progress.png chart from agent results — matches Karpathy's autoresearch style."""

import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_file = sys.argv[1] if len(sys.argv) > 1 else "agent_results_pubmed.tsv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "progress.png"

    # Load results (columns: commit, val_bpb, memory_gb, status, description, sample_text)
    df = pd.read_csv(results_file, sep="\t", header=None,
                     names=["commit", "val_bpb", "memory_gb", "status", "description", "sample_text"])
    df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")

    total_experiments = len(df)

    # Use original row number as experiment number (preserves crashes in numbering)
    df["exp_num"] = range(len(df))

    # Split by status — keep original experiment numbers
    kept = df[(df["status"] == "keep") & (df["val_bpb"] > 0) & (df["val_bpb"] < 20)].copy()
    discarded = df[(df["status"] == "discard") & (df["val_bpb"] > 0) & (df["val_bpb"] < 20)].copy()
    crashed = df[df["status"] == "crash"].copy()

    # Running best (cumulative minimum of kept experiments)
    kept = kept.copy()
    kept["running_min"] = kept["val_bpb"].cummin()

    fig, ax = plt.subplots(figsize=(16, 8))

    # Crashed experiments — small red x's at the top
    if len(crashed) > 0:
        crash_y = [kept["val_bpb"].iloc[0] * 1.03] * len(crashed)  # plot near top
        ax.scatter(crashed["exp_num"], crash_y, c="#ff6b6b", s=15, alpha=0.4,
                   marker="x", zorder=2, label=f"Crashed ({len(crashed)})")

    # Discarded experiments — faint gray dots
    ax.scatter(discarded["exp_num"], discarded["val_bpb"], c="#cccccc", s=12, alpha=0.5,
               zorder=2, label=f"Discarded ({len(discarded)})")

    # Kept experiments — green dots with black edges
    ax.scatter(kept["exp_num"], kept["val_bpb"], c="#2ecc71", s=50, zorder=4,
               edgecolors="black", linewidths=0.8, label=f"Kept ({len(kept)})")

    # Running minimum step line — extend to the end
    step_x = kept["exp_num"].tolist() + [total_experiments - 1]
    step_y = kept["running_min"].tolist() + [kept["running_min"].iloc[-1]]
    ax.step(step_x, step_y, where="post", color="#27ae60", linewidth=2, zorder=3,
            label="Running best")

    # Annotate kept experiments
    for _, row in kept.iterrows():
        desc = str(row["description"])
        if desc == "baseline":
            label = "baseline"
        elif len(desc) > 50:
            label = desc[:47] + "..."
        else:
            label = desc
        ax.annotate(label, (row["exp_num"], row["val_bpb"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, rotation=30, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5))

    # Y-axis: baseline down to best, with margins
    baseline_bpb = kept["val_bpb"].iloc[0]
    best_bpb = kept["val_bpb"].min()
    margin = (baseline_bpb - best_bpb) * 0.15
    y_max = min(baseline_bpb + margin, baseline_bpb * 1.3)
    y_min = best_bpb - margin
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-1, total_experiments + 1)

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.set_title(f"autoresearch: {baseline_bpb:.4f} \u2192 {best_bpb:.4f} val_bpb "
                 f"({(1 - best_bpb/baseline_bpb)*100:.1f}% improvement over "
                 f"{total_experiments} experiments)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
