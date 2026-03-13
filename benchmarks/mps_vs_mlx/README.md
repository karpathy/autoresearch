# Benchmark: PyTorch MPS vs MLX

Comparison of Apple Silicon training approaches:
- **PyTorch MPS** — unified `train.py` with GPU Muon optimizer
- **MLX** — PR #202 `train_mlx.py` with CPU Muon optimizer

## Results

| Metric | PyTorch MPS | MLX | Speedup |
|--------|-------------|-----|---------|
| Throughput | 11,597 tok/s | 6,622 tok/s | **1.75x** |
| Optimizer | 52 ms | 215 ms | **4.1x** |
| Forward | 48 ms | 38 ms | 0.79x |
| Backward | 76 ms | 56 ms | 0.74x |

## Conclusion

PyTorch MPS is 1.75x faster overall due to GPU Muon optimizer (4.1x faster than CPU Muon).
MLX forward/backward is slightly faster, but CPU Muon bottleneck kills performance.

## Structure

```
benchmarks/mps_vs_mlx/
├── src/                    # Source code
│   ├── run.py              # Standalone benchmark runner
│   ├── plot.py             # Standalone plot generator
│   ├── config.py           # Configuration
│   └── runners/            # Backend implementations
│       ├── pytorch_mps.py
│       └── mlx_cpu_muon.py
├── results/                # JSON data
│   └── benchmark.json
├── plots/                  # PNG visualizations
│   ├── throughput.png
│   ├── breakdown.png
│   └── optimizer.png
└── reference/              # PR #202 original code
    └── train_mlx_pr202.py
```

## Usage

### Via stdio_bus (recommended)

```bash
# Start worker
stdio_bus --config stdio_bus/configs/benchmark.json --tcp 0.0.0.0:9000

# Run benchmark
(echo '{"jsonrpc":"2.0","id":1,"method":"benchmark.run","params":{"steps":50},"sessionId":"s1"}'; sleep 120) | nc localhost 9000

# Generate plots
(echo '{"jsonrpc":"2.0","id":2,"method":"benchmark.plot","params":{},"sessionId":"s1"}'; sleep 5) | nc localhost 9000

# Get analysis
(echo '{"jsonrpc":"2.0","id":3,"method":"benchmark.analyze","params":{},"sessionId":"s1"}'; sleep 2) | nc localhost 9000
```

### Standalone

```bash
# Install MLX
uv sync --extra benchmark

# Run benchmark
uv run benchmarks/mps_vs_mlx/src/run.py

# Generate plots
uv run benchmarks/mps_vs_mlx/src/plot.py
```

## JSON-RPC Methods

| Method | Params | Description |
|--------|--------|-------------|
| `benchmark.status` | — | Current state |
| `benchmark.run` | steps, warmup, batch_size | Run benchmark |
| `benchmark.history` | limit | Get history |
| `benchmark.analyze` | — | AI analysis prompt |
| `benchmark.plot` | — | Save plots to `plots/` |
