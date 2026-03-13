#!/usr/bin/env python3
"""
Benchmark runner: PyTorch MPS vs MLX

Usage:
    uv run benchmarks/mps_vs_mlx/src/run.py
    uv run benchmarks/mps_vs_mlx/src/run.py --steps 100
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.mps_vs_mlx.src.config import BenchConfig, RESULTS_DIR
from benchmarks.mps_vs_mlx.src.runners.pytorch_mps import benchmark_pytorch_mps
from benchmarks.mps_vs_mlx.src.runners.mlx_cpu_muon import benchmark_mlx


def main():
    parser = argparse.ArgumentParser(description='MPS vs MLX Benchmark')
    parser.add_argument('--steps', type=int, default=50, help='Benchmark steps')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup steps')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    
    config = BenchConfig(
        warmup_steps=args.warmup,
        benchmark_steps=args.steps,
        batch_size=args.batch_size,
    )
    
    print("=" * 70)
    print("Benchmark: PyTorch MPS vs MLX")
    print("=" * 70)
    print(f"Config: {asdict(config)}")
    print()
    
    # Run benchmarks
    print("[1/2] PyTorch MPS (GPU Muon)...")
    mps_results = benchmark_pytorch_mps(config)
    print(f"  Throughput: {mps_results['tokens_per_sec']:.0f} tok/s")
    print(f"  Step time: {mps_results['total_ms']['mean']:.1f} ms")
    print()
    
    print("[2/2] MLX (CPU Muon)...")
    mlx_results = benchmark_mlx(config)
    print(f"  Throughput: {mlx_results['tokens_per_sec']:.0f} tok/s")
    print(f"  Step time: {mlx_results['total_ms']['mean']:.1f} ms")
    print()
    
    # Comparison
    speedup = mps_results['tokens_per_sec'] / mlx_results['tokens_per_sec']
    opt_speedup = mlx_results['optimizer_ms']['mean'] / mps_results['optimizer_ms']['mean']
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Throughput: MPS {mps_results['tokens_per_sec']:.0f} vs MLX {mlx_results['tokens_per_sec']:.0f} tok/s")
    print(f"  → PyTorch MPS is {speedup:.2f}x faster")
    print(f"Optimizer: MPS {mps_results['optimizer_ms']['mean']:.1f} vs MLX {mlx_results['optimizer_ms']['mean']:.1f} ms")
    print(f"  → GPU Muon is {opt_speedup:.1f}x faster")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': asdict(config),
        'pytorch_mps': mps_results,
        'mlx': mlx_results,
        'comparison': {
            'throughput_speedup': speedup,
            'optimizer_speedup': opt_speedup,
        }
    }
    
    output_path = RESULTS_DIR / 'benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {output_path}")


if __name__ == '__main__':
    main()
