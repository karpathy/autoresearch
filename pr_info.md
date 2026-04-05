# Port autoresearch to Intel XPU

## Summary

- Port training from NVIDIA CUDA to Intel XPU (tested on Arc Pro B60, 24GB)
- Replace Flash Attention 3 with `F.scaled_dot_product_attention` with explicit causal mask
- Enable `torch.compile` on XPU via PyTorch 2.9 inductor backend
- Add XPU device detection and memory tracking in `train.py` and `prepare.py`
- Update dependencies: remove `kernels`, switch to PyTorch XPU wheels, pin Python 3.12

## Changes

- `train.py`: XPU platform detection, SDPA fallback with sliding window, GPU FLOPS map for Intel GPUs, torch.compile support
- `prepare.py`: device-agnostic helper `_get_device()` for dataloader and evaluation
- `pyproject.toml`: XPU torch source, remove CUDA-only deps, add `index-strategy`
- `.python-version`: 3.10 -> 3.12

## Results (Arc Pro B60, DEPTH=6, 26.3M params)

```bash
intel@arda-multiarc-002:~/ziteng/autoresearch$ uv run train.py
Detected GPU: Intel(R) Graphics [0xe210] -> peak BF16 FLOPS: 9.8e+13
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 6, 'n_head': 3, 'n_kv_head': 3, 'n_embd': 384, 'window_pattern': 'L'}
Parameter counts:
  wte                     : 3,145,728
  value_embeds            : 9,437,184
  lm_head                 : 3,145,728
  transformer_matrices    : 10,617,120
  scalars                 : 12
  total                   : 26,345,772
Estimated FLOPs per token: 1.392002e+08
Scaling AdamW LRs by 1/sqrt(384/768) = 1.414214
Time budget: 300s
Gradient accumulation steps: 4
step 00309 (99.8%) | loss: 3.452771 | lrm: 0.00 | dt: 1014ms | tok/sec: 129,238 | mfu: 18.3% | epoch: 1 | remaining: 0s      
---
val_bpb:          1.227996
training_seconds: 300.4
total_seconds:    354.7
peak_vram_mb:     8076.0
mfu_percent:      18.50
total_tokens_M:   40.6
num_steps:        310
num_params_M:     26.3
depth:            6
```

