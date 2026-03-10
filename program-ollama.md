# autoresearch (Local Ollama Agent)

Simplified instructions for running autoresearch with a local Ollama model via OpenCode.

## Setup

1. **Create a branch**: `git checkout -b autoresearch/<tag>` (e.g. `autoresearch/mar10`)
2. **Read these files** for context:
   - `README.md` — project overview
   - `train.py` — the ONLY file you modify (model architecture, optimizer, hyperparameters)
   - `prepare.py` — fixed data/eval code (DO NOT modify)
3. **Check data exists**: `ls ~/.cache/autoresearch/` should have data shards and tokenizer. If empty, tell user to run `uv run prepare.py`.
4. **Initialize results**: `./log_result.sh` will create `results.tsv` automatically on first use.
5. **Run baseline**: `./run_experiment.sh` — this runs the unmodified code to get a baseline.
6. **Log baseline**: `./log_result.sh <val_bpb> <memory_gb> keep "baseline"`

## Helper Scripts

Three helper scripts simplify the workflow:

- **`./run_experiment.sh`** — Runs `uv run train.py`, prints `val_bpb` and `memory_gb` (or `CRASH` with error trace).
- **`./log_result.sh <val_bpb> <memory_gb> <status> "<description>"`** — Appends a row to `results.tsv`. Status: `keep`, `discard`, or `crash`.
- **`./revert_experiment.sh`** — Reverts the last git commit (use when experiment didn't improve).

## Experiment Loop

Repeat forever:

1. **Pick ONE change** to try in `train.py`. Keep changes small and focused. Examples:
   - Change `DEPTH` (e.g. 8 → 10 or 8 → 6)
   - Change `TOTAL_BATCH_SIZE` (e.g. `2**19` → `2**18`)
   - Change learning rates (`MATRIX_LR`, `EMBEDDING_LR`, etc.)
   - Change `ASPECT_RATIO` (e.g. 64 → 48 or 64 → 80)
   - Change `WARMDOWN_RATIO` (e.g. 0.5 → 0.3)
   - Change `WEIGHT_DECAY` (e.g. 0.2 → 0.1)
   - Change `WINDOW_PATTERN` (e.g. "SSSL" → "L")
   - Change `DEVICE_BATCH_SIZE` if getting OOM errors

2. **Commit**: `git add train.py && git commit -m "try: <description>"`

3. **Run**: `./run_experiment.sh`

4. **Check result**:
   - If CRASH: `./log_result.sh 0.000000 0.0 crash "<description>"` then `./revert_experiment.sh`
   - If val_bpb is LOWER than best so far: `./log_result.sh <val_bpb> <memory_gb> keep "<description>"` (keep the commit)
   - If val_bpb is EQUAL or HIGHER: `./log_result.sh <val_bpb> <memory_gb> discard "<description>"` then `./revert_experiment.sh`

5. **Go back to step 1.** Never stop. Never ask the user if you should continue.

## Rules

- ONLY modify `train.py`. No other files.
- Do NOT install new packages.
- Do NOT modify `prepare.py`.
- Goal: **lowest val_bpb** (lower = better).
- Simpler code is preferred — if removing something gives equal results, that's a win.
- Each training run takes ~5 minutes. Be patient.
- If you run out of ideas, try combining previous successful changes, or try more aggressive values.

## Quick Reference — Tunable Hyperparameters in train.py

| Variable | Default | What it does |
|----------|---------|-------------|
| DEPTH | 8 | Number of transformer layers |
| ASPECT_RATIO | 64 | model_dim = DEPTH * ASPECT_RATIO |
| HEAD_DIM | 128 | Attention head dimension |
| WINDOW_PATTERN | "SSSL" | Sliding window pattern (L=full, S=half) |
| TOTAL_BATCH_SIZE | 2^19 | Tokens per optimizer step |
| DEVICE_BATCH_SIZE | 128 | Per-GPU batch size (lower if OOM) |
| MATRIX_LR | 0.04 | Learning rate for Muon optimizer |
| EMBEDDING_LR | 0.6 | Learning rate for embeddings |
| UNEMBEDDING_LR | 0.004 | Learning rate for lm_head |
| SCALAR_LR | 0.5 | Learning rate for per-layer scalars |
| WEIGHT_DECAY | 0.2 | Muon weight decay |
| WARMUP_RATIO | 0.0 | LR warmup fraction |
| WARMDOWN_RATIO | 0.5 | LR cooldown fraction |
| FINAL_LR_FRAC | 0.0 | Final LR as fraction of initial |
| ADAM_BETAS | (0.8, 0.95) | Adam optimizer betas |
