# ML Training Experiment (RTX 4060)

## Overview

Reference implementation of the autoresearch framework. Trains a small GPT language model and autonomously optimizes architecture, hyperparameters, and training loop for lowest val_bpb within a 5-minute time budget on RTX 4060 (8GB VRAM).

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted for consumer GPU constraints.

## Key Differences from Upstream (H100)

| Parameter | Upstream (H100) | This Workflow (RTX 4060) |
|-----------|-----------------|--------------------------|
| MAX_SEQ_LEN | 2048 | 512 |
| EVAL_TOKENS | 40 * 524288 | 5 * 524288 |
| DEPTH | 8 | 4 |
| DEVICE_BATCH_SIZE | 128 | 16 |
| TOTAL_BATCH_SIZE | 2^19 (~524K) | 2^16 (~65K) |
| WINDOW_PATTERN | SSSL | L (full attention only) |
| Attention | Flash Attention 3 | SDPA fallback |
| Peak FLOPS | 989.5 TFLOPS | 121 TFLOPS |

## Files

| File | Role | Who Edits |
|------|------|-----------|
| workflow.yaml | Targets, metric, run command | Human (once) |
| program.md | Research strategy and constraints | Human (iteratively) |
| train.py | Model, optimizer, training loop | Agent (each experiment) |
| prepare.py | Data prep, tokenizer, evaluation | Nobody (read-only) |
| pyproject.toml | Dependencies | Nobody (read-only) |

## Constraints

- **8GB VRAM**: OOM is a constant risk. Estimate memory before scaling up.
- **No Flash Attention 3**: Using PyTorch SDPA fallback. Do not import FA3.
- **No torch.compile**: No Triton on Windows. Compilation is disabled.
- **MAX_SEQ_LEN = 512**: Fixed in prepare.py. Do not assume 2048.
- **5-minute time budget**: Fixed in prepare.py (TIME_BUDGET = 300).

## Running

```bash
uv sync                          # install dependencies
uv run prepare.py --num-shards 2 # download data (one-time)
uv run train.py                  # run one experiment (~5 min)
```

## Data

Training data is cached at `~/.cache/autoresearch/`. Run `uv run prepare.py` if data is missing.
