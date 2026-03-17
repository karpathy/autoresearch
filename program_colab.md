# autoresearch (Colab / Free GPU Edition)

This is an experiment to have the LLM do its own research on free GPU platforms.

## Platform constraints

You are running on a **Google Colab T4 GPU** with **16GB VRAM**. Key differences from H100:

- **float16** (not bfloat16) — uses GradScaler for mixed-precision training
- **PyTorch SDPA** attention (not Flash Attention 3)
- **~16GB VRAM** — model must fit comfortably, leave headroom for experiments
- **DEPTH=4** default (not 8) — smaller model, ~3-5M params
- **MAX_SEQ_LEN=512** (not 2048) — shorter context
- **DEVICE_BATCH_SIZE=32** (not 128) — smaller batches
- **TOTAL_BATCH_SIZE=2^15** (not 2^19) — less tokens per step
- **WINDOW_PATTERN="L"** — full context only (sliding window is inefficient on SDPA)

## What you CAN modify

Only `train.py`. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

## What you CANNOT modify

- `prepare.py` — read-only. Contains evaluation, data loading, tokenizer, constants.
- Do not install new packages or add dependencies.
- Do not modify the evaluation harness (`evaluate_bpb`).

## The goal

**Get the lowest val_bpb.** Training always runs for exactly 5 minutes (wall clock). On T4, fewer tokens are processed per run compared to H100 — that's by design. You're optimizing for *this* hardware.

## VRAM budget

You have ~16GB. Current defaults use ~6-8GB. You have headroom to:
- Increase DEPTH (try 5, 6 — but test carefully, 8 will likely OOM)
- Increase DEVICE_BATCH_SIZE (try 48, 64)
- Increase TOTAL_BATCH_SIZE (try 2^16, 2^17)
- These are the most impactful knobs on T4

**If you OOM**, reduce DEVICE_BATCH_SIZE first, then DEPTH.

## Experiment ideas for T4

Roughly ordered by expected impact:

1. **Scale up within VRAM**: Increase DEPTH from 4→5→6, increase batch sizes
2. **Learning rate tuning**: The defaults are tuned for larger models. Try MATRIX_LR in [0.02, 0.06], EMBEDDING_LR in [0.3, 0.8]
3. **ASPECT_RATIO**: Try 48, 80, 96 — this changes width vs depth tradeoff
4. **HEAD_DIM**: Try 64 instead of 128 (more heads, narrower)
5. **Activation function**: Try different activations in MLP (GELU, SiLU vs ReluSquared)
6. **Warmup**: Try WARMUP_RATIO=0.05 or 0.1
7. **Architecture changes**: Modify Block, add skip connections, try different norm positions
8. **Optimizer tweaks**: Muon ns_steps, momentum schedule, weight decay schedule

## Output format

The script prints a summary like:
```
---
val_bpb:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     8500.2
mfu_percent:      35.00
total_tokens_M:   120.5
num_steps:        500
num_params_M:     5.3
depth:            4
```

## Logging results

Log each experiment to `results.tsv` (tab-separated):
```
commit	val_bpb	memory_gb	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Modify `train.py` with an experimental idea
3. git commit
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If empty → crash. Run `tail -n 50 run.log` for the traceback.
7. Record in results.tsv (do not commit this file)
8. If val_bpb improved → keep the commit
9. If val_bpb is equal or worse → `git reset --hard HEAD~1`

**Timeout**: If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: Fix typos/easy bugs and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: Do not pause to ask the human. Run autonomously until manually stopped. If stuck, think harder — try combining near-misses, try radical changes, re-read the code.
