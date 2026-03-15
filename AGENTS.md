# autoresearch (Brett's RTX 4060 Fork)

## Overview

Autonomous ML training research agent. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted for RTX 4060 (8GB VRAM).

## Key Differences from Upstream

| Parameter | Upstream (H100) | This Fork (RTX 4060) |
|-----------|-----------------|---------------------|
| MAX_SEQ_LEN | 2048 | 512 |
| EVAL_TOKENS | 40 * 524288 | 5 * 524288 |
| DEPTH | 8 | 4 |
| DEVICE_BATCH_SIZE | 128 | 16 |
| TOTAL_BATCH_SIZE | 2^19 (~524K) | 2^16 (~65K) |
| WINDOW_PATTERN | SSSL | L (full attention only) |
| Attention | Flash Attention 3 | SDPA fallback (Ada Lovelace SM 8.9) |
| Peak FLOPS | 989.5 TFLOPS | 121 TFLOPS |

## Skills

This repo uses the `autonomous-iteration` skill from `~/.copilot/skills/autonomous-iteration/SKILL.md`.

The project-specific instructions are in `program.md`.

## File Structure

```
prepare.py      -- constants, data prep + runtime utilities (modified for RTX 4060)
train.py        -- model, optimizer, training loop (agent modifies this)
program.md      -- agent instructions (human modifies this)
AGENTS.md       -- this file
results.tsv     -- experiment log (untracked)
musings.md      -- experiment reflections (untracked)
```

## Running

```bash
uv sync                          # install dependencies
uv run prepare.py --num-shards 2 # download data (start small)
uv run train.py                  # run baseline (~5 min)
```

## Upstream Sync

```bash
git remote add upstream https://github.com/karpathy/autoresearch.git
git fetch upstream
git merge upstream/master  # merge upstream improvements
```
