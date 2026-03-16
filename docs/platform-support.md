# Platform Support

Autoresearch automatically detects your hardware at startup via `platform_config.py` and configures device, attention backend, compilation, and recommended hyperparameters.

## Supported Platforms

### Linux + NVIDIA CUDA

The primary target. Full feature support.

| Feature | Status |
|---|---|
| Device | `cuda` |
| Flash Attention 3 | Yes (Hopper sm_90+) or FA3 community fallback |
| `torch.compile` | Yes (Triton backend) |
| Mixed precision | bfloat16 (sm_80+) or float16 |
| Sliding window attention | Yes (via FA3) |

**Recommended defaults scale by VRAM:**

| VRAM | Depth | Batch Size | Seq Len |
|---|---|---|---|
| 70+ GB (H100/A100 80GB) | 8 | 524,288 | 1024 |
| 20-70 GB (A5000, RTX 4090) | 6 | 262,144 | 512 |
| 8-20 GB (RTX 3070/4060) | 4 | 131,072 | 256 |
| < 8 GB | 3 | 65,536 | 128 |

### macOS Apple Silicon (M1/M2/M3/M4)

Runs via the MPS (Metal Performance Shaders) backend.

| Feature | Status |
|---|---|
| Device | `mps` |
| Flash Attention 3 | No — uses PyTorch SDPA fallback |
| `torch.compile` | No (Triton not available on MPS) |
| Mixed precision | float16 (bf16 limited on MPS) |
| Sliding window attention | No — full context via SDPA |

**Recommended defaults scale by unified memory:**

| Unified Memory | Depth | Batch Size | Seq Len |
|---|---|---|---|
| 64+ GB | 6 | 131,072 | 512 |
| 32-64 GB | 5 | 65,536 | 512 |
| 16-32 GB | 4 | 32,768 | 256 |
| < 16 GB | 3 | 16,384 | 128 |

### macOS Intel / Linux CPU

CPU fallback for development and CI. Not recommended for real training.

| Feature | Status |
|---|---|
| Device | `cpu` |
| Flash Attention 3 | No — uses PyTorch SDPA fallback |
| `torch.compile` | No |
| Mixed precision | Disabled (float32) |
| Sliding window attention | No — full context via SDPA |

Defaults: depth=2, batch=4096, seq_len=64.

## Environment Variable Overrides

All platform defaults can be overridden:

```bash
# Override model depth
AUTORESEARCH_DEPTH=4 uv run train.py

# Override device batch size
AUTORESEARCH_DEVICE_BATCH=8 uv run train.py

# Select a different model architecture
AUTORESEARCH_MODEL=gpt2 uv run train.py

# Combine overrides
AUTORESEARCH_DEPTH=3 AUTORESEARCH_DEVICE_BATCH=4 AUTORESEARCH_MODEL=gpt2 uv run train.py
```

## SDPA Fallback

On non-CUDA platforms, attention uses `torch.nn.functional.scaled_dot_product_attention` (SDPA) instead of Flash Attention 3. Key differences:

- **No sliding window**: SDPA doesn't support the `window_size` parameter, so all layers use full-context attention. The `WINDOW_PATTERN` setting is ignored.
- **GQA support**: The SDPA path handles grouped-query attention by expanding KV heads via `repeat_interleave`.
- **Performance**: Slower than FA3 but correct. Acceptable for development and small experiments.

## Dependency Handling

Platform-specific dependencies are managed in `pyproject.toml`:

- **`kernels`** package: only installed on Linux (`sys_platform == 'linux'`), as it provides CUDA-only FA3 bindings.
- **`torch`**: installed from the CUDA 12.8 index on Linux, from PyPI on macOS.
- **numpy, pyarrow, etc.**: platform-independent, installed everywhere.

```bash
# Install (platform auto-detected)
uv sync

# Install with dev/test dependencies
uv sync --extra dev
```

## Verifying Your Setup

```bash
# Check detected platform
uv run python -c "from platform_config import PLATFORM; PLATFORM and __import__('platform_config').print_platform_info()"

# Run unit tests (no GPU required)
uv sync --extra dev
uv run pytest tests/ -m unit -v
```

## torch.compile Behavior

On CUDA, three things are compiled:
1. `adamw_step_fused` — fused AdamW parameter update
2. `muon_step_fused` — fused Muon optimizer step with polar express orthogonalization
3. The full model via `torch.compile(model, dynamic=False)`

On MPS and CPU, all three are skipped. The functions run as plain Python — slower but correct.

## RMSNorm Compatibility

The nanochat model uses `F.rms_norm` for normalization, which was added in PyTorch 2.4. On older torch versions (2.2-2.3, common on Intel Mac), a manual fallback is used:

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
```

This is mathematically equivalent and used transparently.
