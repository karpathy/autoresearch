"""Benchmark configuration."""

from dataclasses import dataclass
from pathlib import Path

# Paths
BENCHMARK_ROOT = Path(__file__).parent.parent
RESULTS_DIR = BENCHMARK_ROOT / 'results'
PLOTS_DIR = BENCHMARK_ROOT / 'plots'


@dataclass
class BenchConfig:
    """Benchmark configuration."""
    warmup_steps: int = 10
    benchmark_steps: int = 50
    batch_size: int = 4
    seq_len: int = 512
    vocab_size: int = 8192
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4


def generate_batch(batch_size: int, seq_len: int, vocab_size: int):
    """Generate random batch as numpy arrays."""
    import numpy as np
    x = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    y = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    return x, y


def compute_stats(times: list) -> dict:
    """Compute statistics from timing list."""
    import numpy as np
    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
    }
