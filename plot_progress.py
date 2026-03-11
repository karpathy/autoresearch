#!/usr/bin/env python3
"""Generate progress.png from results.tsv"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rows = []
with open('results.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        v = float(r['val_loss'])
        s = r['status'].strip()
        if v <= 0 or v >= 10:
            continue
        rows.append({
            'val_loss': v,
            'status': s,
            'desc': r['description'].strip(),
        })

n = len(rows)
xs = list(range(len(rows)))
ys = [r['val_loss'] for r in rows]

# Running best (keeps only) — step line connecting kept improvements
best_x, best_y = [], []
best = float('inf')
for i, r in enumerate(rows):
    if r['status'] == 'keep' and r['val_loss'] < best:
        best = r['val_loss']
    if r['status'] == 'keep':
        best_x.append(i)
        best_y.append(best)

# Light gray background style
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

# Grid — very light, y-axis only
ax.grid(axis='y', color='#e0e0e0', linewidth=0.5)
ax.grid(axis='x', visible=False)

# Running best step line
if best_x:
    ax.step(best_x, best_y, where='post', color='#2ecc71', linewidth=1.8,
            alpha=0.7, label='Running best', zorder=3)

# Scatter: green keeps (larger), gray discards (smaller, lighter)
for x, y, r in zip(xs, ys, rows):
    if r['status'] == 'keep':
        ax.scatter(x, y, color='#2ecc71', s=70, zorder=5,
                   edgecolors='white', linewidths=0.5)
    else:
        ax.scatter(x, y, color='#cccccc', s=30, zorder=4,
                   edgecolors='#bbbbbb', linewidths=0.3, alpha=0.7)

# Annotate only running-best keeps (new records)
prev_best = float('inf')
for x, y, r in zip(xs, ys, rows):
    if r['status'] == 'keep' and r['val_loss'] < prev_best:
        prev_best = r['val_loss']
        short = r['desc'].split('(')[0].strip()
        if len(short) > 45:
            short = short[:42] + '...'
        ax.annotate(short, (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='#888888',
                    ha='left', va='bottom', rotation=45)

n_kept = sum(1 for r in rows if r['status'] == 'keep')
ax.set_xlabel('Experiment #', fontsize=12, color='#555555')
ax.set_ylabel('Validation Loss (lower is better)', fontsize=12, color='#555555')
ax.set_title(f'ANE Autoresearch Progress: {n} Experiments, {n_kept} Kept Improvements',
             fontsize=14, fontweight='bold', color='#333333', pad=15)

# Clean spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#cccccc')

ax.tick_params(colors='#888888', labelsize=10)

legend = [
    mpatches.Patch(color='#cccccc', label='Discarded'),
    mpatches.Patch(color='#2ecc71', label='Kept'),
    plt.Line2D([0], [0], color='#2ecc71', linewidth=1.8, alpha=0.7, label='Running best'),
]
ax.legend(handles=legend, fontsize=10, loc='upper right',
          framealpha=0.9, edgecolor='#dddddd')

# Tight y-axis: show range of actual progress with some padding
valid_ys = [y for y in ys if y < 8]
if valid_ys:
    ymin, ymax = min(valid_ys), max(valid_ys)
    ax.set_ylim(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.35)

plt.tight_layout()
plt.savefig('progress.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved progress.png ({n} experiments, best={min(ys):.4f})')
