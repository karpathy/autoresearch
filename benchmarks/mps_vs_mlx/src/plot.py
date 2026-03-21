#!/usr/bin/env python3
"""
Generate plots from benchmark results.

Usage:
    uv run benchmarks/mps_vs_mlx/src/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.mps_vs_mlx.src.config import RESULTS_DIR, PLOTS_DIR

plt.style.use('seaborn-v0_8-whitegrid')


def load_results() -> dict:
    """Load benchmark results."""
    results_path = RESULTS_DIR / 'benchmark.json'
    if not results_path.exists():
        raise FileNotFoundError(f"Run benchmark first: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def plot_throughput(results: dict, output_dir: Path):
    """Throughput comparison bar chart."""
    mps = results['pytorch_mps']
    mlx = results['mlx']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        ['PyTorch MPS\n(GPU Muon)', 'MLX\n(CPU Muon)'],
        [mps['tokens_per_sec'], mlx['tokens_per_sec']],
        color=['#2ecc71', '#e74c3c'],
        edgecolor='black',
        linewidth=1.5
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:,.0f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    
    speedup = results['comparison']['throughput_speedup']
    ax.set_title(f'Training Throughput\nPyTorch MPS is {speedup:.2f}x faster',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Tokens/sec', fontsize=12)
    ax.set_ylim(0, max(mps['tokens_per_sec'], mlx['tokens_per_sec']) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=150)
    plt.close()


def plot_breakdown(results: dict, output_dir: Path):
    """Step time breakdown stacked bar."""
    mps = results['pytorch_mps']
    mlx = results['mlx']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['PyTorch MPS\n(GPU Muon)', 'MLX\n(CPU Muon)']
    forward = [mps['forward_ms']['mean'], mlx['forward_ms']['mean']]
    backward = [mps['backward_ms']['mean'], mlx['backward_ms']['mean']]
    optimizer = [mps['optimizer_ms']['mean'], mlx['optimizer_ms']['mean']]
    
    x = np.arange(len(categories))
    width = 0.5
    
    ax.bar(x, forward, width, label='Forward', color='#3498db')
    ax.bar(x, backward, width, bottom=forward, label='Backward', color='#9b59b6')
    ax.bar(x, optimizer, width, bottom=np.array(forward) + np.array(backward),
           label='Optimizer', color='#e67e22')
    
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Step Time Breakdown', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    
    for i, (f, b, o) in enumerate(zip(forward, backward, optimizer)):
        total = f + b + o
        ax.annotate(f'{total:.0f} ms', xy=(i, total + 5),
                    ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'breakdown.png', dpi=150)
    plt.close()


def plot_optimizer(results: dict, output_dir: Path):
    """Optimizer comparison (key insight)."""
    mps = results['pytorch_mps']
    mlx = results['mlx']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(
        ['GPU Muon\n(PyTorch MPS)', 'CPU Muon\n(MLX/numpy)'],
        [mps['optimizer_ms']['mean'], mlx['optimizer_ms']['mean']],
        color=['#2ecc71', '#e74c3c'],
        edgecolor='black',
        linewidth=1.5
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.1f} ms',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    
    opt_speedup = results['comparison']['optimizer_speedup']
    ax.set_title(f'Muon Optimizer: GPU vs CPU\nGPU is {opt_speedup:.1f}x faster',
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimizer.png', dpi=150)
    plt.close()


def plot_summary(results: dict, output_dir: Path):
    """Summary table as image."""
    mps = results['pytorch_mps']
    mlx = results['mlx']
    comp = results['comparison']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    headers = ['Metric', 'PyTorch MPS', 'MLX', 'Winner']
    data = [
        ['Throughput (tok/s)', f"{mps['tokens_per_sec']:,.0f}",
         f"{mlx['tokens_per_sec']:,.0f}", f"MPS ({comp['throughput_speedup']:.2f}x)"],
        ['Step time (ms)', f"{mps['total_ms']['mean']:.1f}",
         f"{mlx['total_ms']['mean']:.1f}", 'MPS'],
        ['Optimizer (ms)', f"{mps['optimizer_ms']['mean']:.1f}",
         f"{mlx['optimizer_ms']['mean']:.1f}", f"MPS ({comp['optimizer_speedup']:.1f}x)"],
        ['Forward+Backward (ms)',
         f"{mps['forward_ms']['mean'] + mps['backward_ms']['mean']:.1f}",
         f"{mlx['forward_ms']['mean'] + mlx['backward_ms']['mean']:.1f}", 'Similar'],
    ]
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(data) + 1):
        table[(i, 3)].set_facecolor('#d5f5e3')
        table[(i, 3)].set_text_props(fontweight='bold')
    
    ax.set_title('Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    results = load_results()
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_throughput(results, PLOTS_DIR)
    plot_breakdown(results, PLOTS_DIR)
    plot_optimizer(results, PLOTS_DIR)
    plot_summary(results, PLOTS_DIR)
    
    print(f"Plots saved to: {PLOTS_DIR}")
    for f in PLOTS_DIR.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
