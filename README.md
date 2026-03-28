# autoresearch_v100

V100-compatible fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Full credit to [@karpathy](https://github.com/karpathy) for the original idea and codebase. This fork keeps the same basic structure and goal of the upstream repository, but modifies the training setup so it can run successfully on a single NVIDIA Tesla V100 GPU.

The goal of this fork is simple: make `autoresearch` run end-to-end on V100 hardware with a reduced and compatible configuration.

## Quick start

Requirements:
- NVIDIA Tesla V100 GPU
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

```bash
# install dependencies
uv sync

# one-time data + tokenizer prep
uv run prepare.py

# run one training experiment
uv run train.py

## What matters
prepare.py — data prep, tokenizer, dataloader, and evaluation
train.py — model, optimizer, and training loop
program.md — autonomous experiment protocol from upstream
V100_NOTES.md — notes on the V100 bring-up and successful run
logs/ — saved outputs from successful V100 runs

## Why this fork was needed

The upstream repository was developed primarily for newer NVIDIA GPUs. On Tesla V100 hardware, the original setup ran into multiple issues, including:

-FlashAttention requiring Ampere or newer GPUs
-torch.compile / Triton triggering Python.h compilation issues
-bf16-oriented behavior not being a good fit for V100
-default training settings being too large for practical V100 bring-up

This fork focuses on a smaller and V100-compatible working configuration.

## Working V100 configuration
prepare.py
- MAX_SEQ_LEN = 512
- reduced evaluation workload
train.py
- reduced model depth
- simplified attention path for V100 compatibility
- disabled torch.compile
- reduced total batch size
- reduced device batch size

## Successful V100 run

A full training + evaluation run completed successfully on:

GPU: Tesla V100S-PCIE-32GB
Compute capability: 7.0

Example successful run summary:
```bash
val_bpb: 1.394976
training_seconds: 300.2
total_seconds: 312.0
peak_vram_mb: 602.2
mfu_percent: 0.21
total_tokens_M: 17.0
num_steps: 1038
num_params_M: 11.5
depth: 4
```
## Differences from upstream
V100-safe training path
no FlashAttention dependency for bring-up
no torch.compile dependency for bring-up
reduced sequence length and model size
reduced training/eval workload for successful V100 execution
## Acknowledgments
https://github.com/karpathy/autoresearch — original autoresearch
upstream contributors and notable fork authors for making platform ports easier to think about
## License
MIT. See the upstream repository for the original project license details
