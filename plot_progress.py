"""Generate progress plot from loop_results.tsv showing val_bpb and robustness_gap."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
kept_labels = [descriptions[i].split(" — ")[0] if " — " in descriptions[i] else descriptions[i][:30] for i in kept_x]
disc_x = [i for i, s in enumerate(statuses) if s != "keep"]
disc_y = [val_bpbs[i] for i in disc_x]

# Robustness gap for kept experiments (non-zero only)
gap_kept_x = [i for i in kept_x if robustness_gaps[i] > 0]
gap_kept_y = [robustness_gaps[i] for i in gap_kept_x]
# Discarded that had gap tested
gap_disc_x = [i for i, s in enumerate(statuses) if s != "keep" and robustness_gaps[i] > 0]
gap_disc_y = [robustness_gaps[i] for i in gap_disc_x]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
fig.suptitle(f"Closed Loop Progress: {len(rows)} Experiments, {len(kept_x)} Kept", fontsize=14, fontweight="bold")

# Top: val_bpb
ax1.scatter(disc_x, disc_y, color="lightgray", s=30, zorder=2, label="Discarded")
ax1.scatter(kept_x, kept_y, color="#2ecc71", s=60, zorder=3, label="Kept")
ax1.plot(experiments, running_best, color="#2ecc71", linewidth=1.5, zorder=1, label="Running best")

# Label kept experiments (every other to reduce clutter)
for i, (x, y, label) in enumerate(zip(kept_x, kept_y, kept_labels)):
    if i % 2 == 0 or x == kept_x[-1]:
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=(5, 8),
                     fontsize=6, color="gray", rotation=25)

ax1.set_ylabel("Validation BPB (lower is better)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

# Bottom: robustness_gap
ax2.scatter(gap_disc_x, gap_disc_y, color="lightcoral", s=40, zorder=2, marker="x", label="Discarded (gap tested)")
ax2.scatter(gap_kept_x, gap_kept_y, color="#e74c3c", s=60, zorder=3, label="Kept (gap tested)")

# Running gap for kept
if gap_kept_x:
    gap_running = []
    for x, y in zip(gap_kept_x, gap_kept_y):
        gap_running.append(y)
    ax2.plot(gap_kept_x, gap_running, color="#e74c3c", linewidth=1.5, zorder=1, alpha=0.7)

ax2.set_xlabel("Experiment #", fontsize=11)
ax2.set_ylabel("Robustness Gap (adversarial)", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

# Highlight DOJO-caught discards (improved val_bpb but gap spiked)
for i in range(len(rows)):
    if statuses[i] != "keep" and robustness_gaps[i] > 0:
        ax2.annotate(descriptions[i].split(" — ")[0] if " — " in descriptions[i] else "",
                     (i, robustness_gaps[i]), textcoords="offset points", xytext=(5, 5),
                     fontsize=6, color="red", alpha=0.7)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
