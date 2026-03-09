"""Generate progress plot from loop_results.tsv showing val_bpb and robustness_gap."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUBSAMPLE = 5  # Show every Nth experiment in history, plus all keeps

rows = []
with open("loop_results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

experiments = list(range(len(rows)))
val_bpbs = [float(r["val_bpb"]) for r in rows]
statuses = [r["status"].strip() for r in rows]
robustness_gaps = [float(r["robustness_gap"]) for r in rows]
descriptions = [r["description"].strip() for r in rows]

# Compute running best val_bpb
running_best = []
best = val_bpbs[0]
for v, s in zip(val_bpbs, statuses):
    if s == "keep" and v < best:
        best = v
    running_best.append(best)

# Split into kept/discarded
kept_x = [i for i, s in enumerate(statuses) if s == "keep"]
kept_y = [val_bpbs[i] for i in kept_x]
disc_x = [i for i, s in enumerate(statuses) if s != "keep"]
disc_y = [val_bpbs[i] for i in disc_x]

# Subsampled discards (every SUBSAMPLE-th)
sub_disc_x = [x for x in disc_x if x % SUBSAMPLE == 0]
sub_disc_y = [val_bpbs[x] for x in sub_disc_x]

# Robustness gap for kept experiments (non-zero only)
gap_kept_x = [i for i in kept_x if robustness_gaps[i] > 0]
gap_kept_y = [robustness_gaps[i] for i in gap_kept_x]
# Discarded that had gap tested
gap_disc_x = [i for i, s in enumerate(statuses) if s != "keep" and robustness_gaps[i] > 0]
gap_disc_y = [robustness_gaps[i] for i in gap_disc_x]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle(f"Closed Loop Progress: {len(rows)} total, {len(kept_x)} kept, best={running_best[-1]:.3f}", fontsize=13, fontweight="bold")

# Top: val_bpb (full history, subsampled discards)
ax1.scatter(sub_disc_x, sub_disc_y, color="lightgray", s=25, zorder=2, label=f"Discarded (1/{SUBSAMPLE})", alpha=0.6)
ax1.scatter(kept_x, kept_y, color="#2ecc71", s=60, zorder=3, label="Kept")
ax1.plot(experiments, running_best, color="#2ecc71", linewidth=2, zorder=1, label="Running best")

# Label kept points
for x in kept_x:
    desc = descriptions[x].split(" — ")[0] if " — " in descriptions[x] else descriptions[x][:25]
    y = val_bpbs[x]
    if len(kept_x) <= 15 or kept_x.index(x) % 2 == 0:
        ax1.annotate(desc, (x, y), textcoords="offset points", xytext=(5, 8),
                     fontsize=6, color="#2ecc71", rotation=25, alpha=0.9)

ax1.set_ylabel("Validation BPB (lower is better)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, len(rows))

# Bottom: robustness_gap (full history)
if gap_disc_x:
    ax2.scatter(gap_disc_x, gap_disc_y, color="lightcoral", s=30, zorder=2, marker="x", label="Discarded (gap tested)", alpha=0.6)
if gap_kept_x:
    ax2.scatter(gap_kept_x, gap_kept_y, color="#e74c3c", s=60, zorder=3, label="Kept (gap tested)")
    ax2.plot(gap_kept_x, gap_kept_y, color="#e74c3c", linewidth=1.5, zorder=1, alpha=0.7)
    for x in gap_kept_x:
        desc = descriptions[x].split(" — ")[0] if " — " in descriptions[x] else ""
        if len(gap_kept_x) <= 15 or gap_kept_x.index(x) % 3 == 0:
            ax2.annotate(desc, (x, robustness_gaps[x]), textcoords="offset points", xytext=(5, 5),
                         fontsize=6, color="#e74c3c", alpha=0.8)

ax2.set_xlabel("Experiment #", fontsize=11)
ax2.set_ylabel("Robustness Gap (adversarial)", fontsize=11)
if gap_kept_x or gap_disc_x:
    ax2.legend(loc="upper left", fontsize=9)
else:
    ax2.text(0.5, 0.5, "No gap-tested experiments yet", transform=ax2.transAxes,
             ha="center", va="center", fontsize=10, color="gray", alpha=0.7)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, len(rows))

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print(f"Saved progress.png ({len(rows)} experiments, subsampled every {SUBSAMPLE})")
