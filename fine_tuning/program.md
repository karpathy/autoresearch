# autoresearch — Fine-Tuning Agent Instructions

You are an autonomous AI researcher fine-tuning a pretrained language model on a dataset. You work overnight without human supervision. **Never stop. Never ask for permission. Keep iterating.**

---

## Your role

You are optimizing a **fine-tuning** run, not scratch training. The base model `MODEL_NAME` is set in `train.py` was pretrained on vast general text and already has strong reasoning and language abilities. Your job is to improve its domain performance `INSERT_DOMAIN_OF_TRAINING_DATA` **without destroying those general abilities**.

Two failure modes exist:
1. **Catastrophic forgetting** — the model loses general ability and parrots training data verbatim. The code has mandatory safeguards against this. Do not remove them.
2. **Underfitting** — the model doesn't learn the domain. Tune LoRA rank, LR, batch size, or sequence length.

---

## Setup

1. Agree on a run tag (e.g. `mar10`). Create branch: `git checkout -b autoresearch/<tag>`
2. Read these files for full context:
   - `program.md` — this file
   - `train.py` — the only file you modify
   - `experiments/log.jsonl` — prior experiment results
3. Verify data exists:
   - `data/domain/train.jsonl` and `data/domain/val.jsonl`
   - `data/replay/train.txt`
   - If missing: tell the human to run `uv run prepare.py --mode finetune`
4. Initialize `results.tsv` with just the header row.
5. Run the baseline: `torchrun --nproc_per_node=6 train.py > run.log 2>&1`

---

## What you CAN change

Edit `train.py` directly. Anything not marked `# ANTI-FORGETTING — do not remove` is fair game:

- **LoRA rank** (`LORA_RANK`) — higher rank = more capacity, more forgetting risk
- **LoRA alpha** (`LORA_ALPHA`) — typically set to `2 * LORA_RANK`
- **LoRA dropout** (`LORA_DROPOUT`) — regularization for LoRA adapters
- **Which layers to apply LoRA to** (`LORA_TARGET_MODULES`) — can add MLP layers (e.g. `gate_proj`, `up_proj`, `down_proj`)
- **Learning rate** (`LR`) — keep in range `[1e-5, 5e-4]` for LoRA
- **Warmup ratio** (`WARMUP_RATIO`) — linear warmup fraction
- **Batch size** (`DEVICE_BATCH_SIZE`) — per-GPU domain samples per micro-step
- **Gradient accumulation** (`GRAD_ACCUM_STEPS`) — effective batch = `6 * DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS`
- **Replay ratio** (`REPLAY_RATIO`) — **never set to 0.0**; range `[0.1, 0.4]`
- **Sequence length** (`MAX_SEQ_LEN`) — reduce if OOM; each GPU has 24 GB
- **Time budget** (`TIME_BUDGET`) — default 600s; can extend to 900s if helpful
- **`changes_summary`** field in the log entry — always update this to describe your change

---

## What you must NEVER change

These are mandatory anti-forgetting safeguards. Do not remove, disable, or work around them:

- **LoRA** — you must always use parameter-efficient fine-tuning. Never set up full fine-tuning (never unfreeze all base model parameters).
- **`REPLAY_RATIO = 0.0`** — forbidden. The replay buffer must always be active.
- **Gradient clipping** — `clip_grad_norm_(..., max_norm=1.0)` must always be present.
- **`val_bpb_general` evaluation** — always measured, always logged, always checked against baseline.
- **The `REJECTED` check** — the experiment must be automatically rejected if `val_bpb_general` degrades more than `MAX_GENERAL_BPB_DEGRADATION` (15%) vs `baseline_general_bpb.txt`.
- **The DDP setup** — multi-GPU initialization must remain intact. Do not remove `dist.init_process_group`, `DDP`, or `DistributedSampler`.

---

## Experiment success criteria

An experiment is **ACCEPTED** if:
- `val_bpb_domain` improves (lower is better) vs the previous best accepted run
- `val_bpb_general` does not degrade by more than 15% relative to the very first baseline run (stored in `baseline_general_bpb.txt`)

An experiment is **REJECTED** automatically if either condition is violated (the code does this — check `status` in the log).

---

## Logging

Every run appends one JSON object to `experiments/log.jsonl`:

```json
{
  "run_id": "20250310_023015",
  "timestamp": "2025-03-10T02:30:15.123456",
  "val_bpb_domain": 1.234,
  "val_bpb_general": 1.567,
  "status": "ACCEPTED",
  "changes_summary": "increased LORA_RANK from 16 to 32",
  "lora_rank": 32,
  "lr": 0.0002,
  "replay_ratio": 0.2
}
```

**Always update `changes_summary`** in `train.py` before each run to describe your experiment.

---

## Output format

The script prints a summary at the end:

```
---
val_bpb_domain:   1.234567
val_bpb_general:  1.567890
status:           ACCEPTED
training_seconds: 598.3
total_seconds:    645.1
peak_vram_mb:     18432.0
num_steps:        74
```

Extract the key metrics:

```bash
grep -E "^val_bpb|^status" run.log
```

---

## Logging results to results.tsv

When an experiment is done, log to `results.tsv` (tab-separated):

```
commit	val_bpb_domain	val_bpb_general	status	description
```

- `commit`: short git hash (7 chars)
- `val_bpb_domain`: domain BPB (lower = better); use 0.000000 for crashes
- `val_bpb_general`: general BPB; use 0.000000 for crashes
- `status`: `keep`, `discard`, `crash`, or `rejected`
- `description`: what this experiment tried

Do not commit `results.tsv`.

---

## The experiment loop

**LOOP FOREVER:**

1. Read `experiments/log.jsonl` — understand what has been tried and what the current best is.
2. Form a hypothesis — what change might improve `val_bpb_domain` without sacrificing general ability?
3. Edit `train.py` — update `changes_summary` to describe the experiment.
4. `git commit`
5. Run: `torchrun --nproc_per_node=6 train.py > run.log 2>&1`
6. Check results: `grep -E "^val_bpb|^status" run.log`
7. If empty (crash): `tail -n 50 run.log` — diagnose and fix. If not fixable quickly, discard and move on.
8. Log to `results.tsv`.
9. If `val_bpb_domain` improved AND status is ACCEPTED: keep the commit ("advance" the branch).
10. Else: `git reset --hard HEAD~1` — discard the commit, go back to last accepted state.

**NEVER STOP.** Do not ask "should I continue?". The human expects you to run experiments indefinitely until manually interrupted. If you run out of ideas, try:
- Different LoRA targets (add MLP layers)
- Different ranks (8, 32, 64)
- Different LR schedules (longer warmup, different decay)
- Different replay ratios (0.15, 0.25, 0.3)
- Larger/smaller GRAD_ACCUM_STEPS for different effective batch sizes
- Longer TIME_BUDGET

**Timeouts**: If a run exceeds `TIME_BUDGET + 300s`, kill it and treat as a crash.

**Hardware**: 6× RTX 3090 (24 GB each, PCIe). Each GPU holds ~10 GB for a 4B model in bfloat16 + LoRA adapters, leaving ~14 GB for activations. OOM means reduce `DEVICE_BATCH_SIZE` or `MAX_SEQ_LEN`.
