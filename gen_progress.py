#!/usr/bin/env python3
"""Generate progress.png from results.tsv in the style of the original autoresearch chart."""
import csv
from pathlib import Path

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

ROOT = Path(__file__).parent
TSV = ROOT / 'results.tsv'
OUT = ROOT / 'progress.png'

# Parse results
rows = []
with open(TSV) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for i, r in enumerate(reader):
        vl = float(r['val_loss'])
        rows.append({
            'run': i + 1,
            'val_loss': vl,
            'status': r['status'],
            'desc': r['description'],
            'ane': float(r['ane_util_pct']),
        })

# Filter out crashes / diverged for plotting
valid = [r for r in rows if r['val_loss'] > 0 and r['val_loss'] < 10]
keeps = [r for r in valid if r['status'] == 'keep']
discards = [r for r in valid if r['status'] != 'keep']

# Running best
best_so_far = float('inf')
running_best_x = []
running_best_y = []
for r in rows:
    if r['status'] == 'keep' and 0 < r['val_loss'] < 10:
        if r['val_loss'] < best_so_far:
            best_so_far = r['val_loss']
            running_best_x.append(r['run'])
            running_best_y.append(best_so_far)

# Extend running best as staircase
staircase_x = []
staircase_y = []
for i, (x, y) in enumerate(zip(running_best_x, running_best_y)):
    if i > 0:
        staircase_x.append(x)
        staircase_y.append(running_best_y[i-1])
    staircase_x.append(x)
    staircase_y.append(y)
# Extend to last experiment
if rows:
    staircase_x.append(valid[-1]['run'])
    staircase_y.append(running_best_y[-1])

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Discards (grey, small)
ax.scatter([r['run'] for r in discards], [r['val_loss'] for r in discards],
           c='#cccccc', s=30, zorder=2, label='Discarded', alpha=0.7)

# Kept (green, larger)
ax.scatter([r['run'] for r in keeps], [r['val_loss'] for r in keeps],
           c='#2ecc71', s=60, zorder=3, label='Kept', edgecolors='#27ae60', linewidths=0.5)

# Running best staircase
ax.plot(staircase_x, staircase_y, c='#2ecc71', linewidth=2, zorder=2, label='Running best', alpha=0.8)

# ncdrone reference line
ax.axhline(y=5.81, color='#e67e22', linewidth=1, linestyle='--', alpha=0.7, zorder=1)
ax.text(rows[-1]['run'] + 0.5, 5.82, 'ncdrone ref: 5.81', fontsize=8, color='#e67e22', va='bottom')

# Label kept experiments (angled text)
labeled = set()
for r in keeps:
    # Shorten description for label
    desc = r['desc']
    # Remove common prefixes
    for prefix in ['baseline: ', 'baseline ', 'continue ', 'phase1: ', 'phase2: ', 'phase3: ']:
        if desc.startswith(prefix):
            desc = desc[len(prefix):]
    # Truncate
    if len(desc) > 45:
        desc = desc[:42] + '...'

    # Avoid overlapping labels — skip if too close to a previously labeled point
    skip = False
    for lx, ly in labeled:
        if abs(r['run'] - lx) < 2 and abs(r['val_loss'] - ly) < 0.03:
            skip = True
            break
    if skip:
        continue
    labeled.add((r['run'], r['val_loss']))

    ax.annotate(desc, (r['run'], r['val_loss']),
                textcoords='offset points', xytext=(8, -5),
                fontsize=6.5, color='#555555', rotation=25,
                ha='left', va='top')

# Formatting
total_kept = len(keeps)
total_exp = len(rows)
ax.set_title(f'ANE Autoresearch Progress: {total_exp} Experiments, {total_kept} Kept Improvements',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Experiment #', fontsize=11)
ax.set_ylabel('Validation Loss (lower is better)', fontsize=11)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Y-axis: show more decimal places
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')
