#!/usr/bin/env python3
"""Generate Phase 2 visualization plots."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load manual trial data
manual_data = []
with open('results.tsv', 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    for i, line in enumerate(lines, 1):
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                manual_data.append({
                    'trial': i,
                    'method': 'manual',
                    'val_bpb': float(parts[1]),
                    'status': parts[3],
                    'description': parts[4] if len(parts) > 4 else ''
                })

# Load optimization data
def load_optuna_results(method_name, pattern):
    results = []
    for exp_dir in sorted(Path('experiments').glob(pattern)):
        results_file = exp_dir / 'results.jsonl'
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    data = json.loads(line)
                    results.append({
                        'trial': data['trial_number'],
                        'method': method_name,
                        'val_bpb': data['results']['val_bpb'],
                        'status': 'keep' if data['results']['status'] == 'success' else 'crash',
                        'params': data['params']
                    })
    return results

bayesian_unfocused = load_optuna_results('bayesian_unfocused', 'bayesian_run_2026*')
genetic_unfocused = load_optuna_results('genetic_unfocused', 'genetic_run_2026*')
bayesian_focused = load_optuna_results('bayesian_focused', 'bayesian_focused_*')
genetic_focused = load_optuna_results('genetic_focused', 'genetic_focused_*')

# Combine all data
all_data = manual_data + bayesian_unfocused + genetic_unfocused + bayesian_focused + genetic_focused
df = pd.DataFrame(all_data)

# Create main progress plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

# Plot 1: Overall progress
colors = {
    'manual': '#2ecc71',
    'bayesian_unfocused': '#e74c3c',
    'genetic_unfocused': '#f39c12',
    'bayesian_focused': '#3498db',
    'genetic_focused': '#9b59b6'
}

markers = {
    'manual': 'o',
    'bayesian_unfocused': 's',
    'genetic_unfocused': '^',
    'bayesian_focused': 'D',
    'genetic_focused': 'v'
}

trial_counter = 0
running_best = float('inf')
best_points = []

for method in ['manual', 'bayesian_unfocused', 'genetic_unfocused', 'bayesian_focused', 'genetic_focused']:
    method_data = df[df['method'] == method].copy()
    if len(method_data) == 0:
        continue

    method_data = method_data.sort_values('trial')

    for _, row in method_data.iterrows():
        trial_counter += 1
        val_bpb = row['val_bpb']

        # Handle crashes/failures
        if val_bpb > 10:  # Very high values indicate failure
            ax1.scatter(trial_counter, 1.52, marker='x', color='gray', alpha=0.3, s=50)
            continue

        # Plot point
        color = colors[method]
        marker = markers[method]
        alpha = 0.8 if row['status'] == 'keep' else 0.3

        ax1.scatter(trial_counter, val_bpb, marker=marker, color=color,
                   alpha=alpha, s=100, edgecolors='white', linewidth=1)

        # Track running best
        if val_bpb < running_best:
            running_best = val_bpb
            best_points.append((trial_counter, val_bpb))

# Plot running best line
if best_points:
    best_x, best_y = zip(*best_points)
    ax1.plot(best_x, best_y, 'g-', linewidth=2, alpha=0.7, label='Running Best')
    ax1.scatter(best_x, best_y, s=200, color='gold', marker='*',
               edgecolors='green', linewidth=2, zorder=5, label='New Best')

# Add baseline line
baseline = 1.451763
ax1.axhline(baseline, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')

# Add Phase 1 target
phase1_best = 1.371
ax1.axhline(phase1_best, color='purple', linestyle='--', alpha=0.5, label=f'Phase 1 Best ({phase1_best:.3f})')

ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation BPB (lower is better)', fontsize=12, fontweight='bold')
ax1.set_title('Phase 2: Hybrid Manual + Automated Hyperparameter Optimization',
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1.36, 1.53)

# Plot 2: Method comparison
method_stats = []
for method in colors.keys():
    method_data = df[(df['method'] == method) & (df['val_bpb'] < 10)]
    if len(method_data) > 0:
        method_stats.append({
            'method': method.replace('_', ' ').title(),
            'best': method_data['val_bpb'].min(),
            'median': method_data['val_bpb'].median(),
            'count': len(method_data)
        })

if method_stats:
    stats_df = pd.DataFrame(method_stats)
    stats_df = stats_df.sort_values('best')

    x = np.arange(len(stats_df))
    width = 0.35

    ax2.bar(x - width/2, stats_df['best'], width, label='Best', alpha=0.8, color='#2ecc71')
    ax2.bar(x + width/2, stats_df['median'], width, label='Median', alpha=0.8, color='#3498db')

    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation BPB', fontsize=12, fontweight='bold')
    ax2.set_title('Method Performance Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df['method'], rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for i, row in stats_df.iterrows():
        ax2.text(x[i], ax2.get_ylim()[1] * 0.98, f"n={row['count']}",
                ha='center', va='top', fontsize=9)

plt.tight_layout()
plt.savefig('phase2_progress.png', dpi=150, bbox_inches='tight')
print("Saved phase2_progress.png")

# Create parameter exploration plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get focused optimization data
focused_data = df[df['method'].str.contains('focused', na=False)]

if len(focused_data) > 0:
    # Extract parameters
    params_list = []
    for _, row in focused_data.iterrows():
        if 'params' in row and row['params']:
            params = row['params'].copy()
            params['val_bpb'] = row['val_bpb']
            params['method'] = row['method']
            params_list.append(params)

    if params_list:
        params_df = pd.DataFrame(params_list)

        # Plot parameter relationships
        param_pairs = [
            ('matrix_lr', 'val_bpb', 'Matrix LR vs Performance'),
            ('embedding_lr', 'val_bpb', 'Embedding LR vs Performance'),
            ('weight_decay', 'val_bpb', 'Weight Decay vs Performance'),
            ('warmdown_ratio', 'val_bpb', 'Warmdown Ratio vs Performance'),
        ]

        for idx, (param_x, param_y, title) in enumerate(param_pairs):
            ax = axes[idx // 2, idx % 2]

            if param_x in params_df.columns:
                for method in ['bayesian_focused', 'genetic_focused']:
                    method_data = params_df[params_df['method'] == method]
                    if len(method_data) > 0:
                        ax.scatter(method_data[param_x], method_data[param_y],
                                  label=method.replace('_', ' ').title(),
                                  alpha=0.6, s=100, color=colors[method])

                ax.set_xlabel(param_x.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Validation BPB', fontweight='bold')
                ax.set_title(title, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase2_parameter_exploration.png', dpi=150, bbox_inches='tight')
print("Saved phase2_parameter_exploration.png")

# Print summary statistics
print("\n" + "="*80)
print("PHASE 2 SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal trials: {len(df)}")
print(f"  Manual: {len(df[df['method'] == 'manual'])}")
print(f"  Bayesian (unfocused): {len(df[df['method'] == 'bayesian_unfocused'])}")
print(f"  Genetic (unfocused): {len(df[df['method'] == 'genetic_unfocused'])}")
print(f"  Bayesian (focused): {len(df[df['method'] == 'bayesian_focused'])}")
print(f"  Genetic (focused): {len(df[df['method'] == 'genetic_focused'])}")

print(f"\nBest results by method:")
for method in colors.keys():
    method_data = df[(df['method'] == method) & (df['val_bpb'] < 10)]
    if len(method_data) > 0:
        best = method_data['val_bpb'].min()
        print(f"  {method:25s}: {best:.6f}")

overall_best = df[df['val_bpb'] < 10]['val_bpb'].min()
print(f"\n{'='*80}")
print(f"OVERALL BEST: {overall_best:.6f}")
print(f"Baseline:     {baseline:.6f}")
print(f"Improvement:  {baseline - overall_best:.6f} ({(baseline - overall_best)/baseline*100:.2f}%)")
print(f"Gap to Phase 1 best: {overall_best - phase1_best:.6f}")
print(f"{'='*80}\n")
