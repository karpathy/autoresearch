# V100 Notes for autoresearch

## Goal
Get `karpathy/autoresearch` to run successfully on a single NVIDIA Tesla V100S-PCIE-32GB GPU.

## Hardware
- GPU: Tesla V100S-PCIE-32GB
- Compute capability: 7.0
- Driver version: 535.288.01
- CUDA version shown by `nvidia-smi`: 12.2

## Initial problem
The original repository did not run successfully on V100.

Main issues observed:
1. FlashAttention failed because it only supports Ampere GPUs or newer.
2. `torch.compile` / Triton triggered compilation errors involving `Python.h`.
3. The original bf16-oriented path was not suitable for V100.
4. The original default training configuration was too large/heavy for this GPU.

## Main changes made
### In `train.py`
- Removed the dependency on FlashAttention for V100 bring-up.
- Replaced the attention path with a V100-safe fallback.
- Disabled `torch.compile` to avoid Triton / `Python.h` issues.
- Removed bf16-dependent training behavior.
- Reduced the model/training configuration to a smaller V100-safe setting.

### In `prepare.py`
- Reduced sequence length and evaluation workload to make runs practical on V100.

## Working configuration
### `prepare.py`
- `MAX_SEQ_LEN = 512`
- `EVAL_TOKENS = 4 * 524288`

### `train.py`
- `DEPTH = 4`
- `WINDOW_PATTERN = "L"`
- `TOTAL_BATCH_SIZE = 2**14`
- `DEVICE_BATCH_SIZE = 4`

## Successful run result
A full training + evaluation run completed successfully on V100.

Final run summary:
- `val_bpb: 1.394976`
- `training_seconds: 300.2`
- `total_seconds: 312.0`
- `peak_vram_mb: 602.2`
- `mfu_percent: 0.21`
- `total_tokens_M: 17.0`
- `num_steps: 1038`
- `num_params_M: 11.5`
- `depth: 4`

## Status
Current status: successful proof-of-concept V100 run completed.

This means the repo can be made to work on V100 with a reduced and V100-compatible configuration.

