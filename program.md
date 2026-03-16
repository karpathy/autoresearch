# autoresearch — agent instructions

## 1. Orientation (do this first, every run)

Before making any changes, read and understand the codebase:

1. Read `ground.json` — the user-owned, read-only configuration:
   - `mode`: `"test"` or `"train"` — determines which time budget applies.
   - `training.time_budget_test` / `training.time_budget_train` — the wall-clock seconds the training loop is allowed to run. **Respect this strictly.**
   - `training.max_seq_len` — sequence length, fixed.
   - `processor` — dtype, compile, flash_attention, peak_flops overrides (all `"auto"` by default).
2. Read `model.json` — your hyperparameter file (you own this):
   - `architecture`: depth, aspect_ratio, head_dim, window_pattern.
   - `optimization`: batch sizes, learning rates, weight decay, adam betas, warmup/warmdown ratios, final_lr_frac.
   - `evaluation`: batch_size, tokens (for the fast eval after training).
3. Read `prepare.py` — understand but **never edit**:
   - Exports: `MAX_SEQ_LEN`, `TIME_BUDGET`, `PLATFORM`, `Tokenizer`, `make_dataloader`, `evaluate_bpb`, `get_token_bytes`.
   - `PLATFORM` dict: device, dtype, use_grad_scaler, attention, compile, peak_flops (auto-detected from GPU hardware specs).
4. Read `train.py` — the model and training loop (you own this):
   - Loads all hyperparameters from `model.json` at startup.
   - Imports platform config from `prepare.py`.
   - Prints a `---` separator followed by key=value summary lines at the end of training.
5. Note the key metric: **`val_bpb`** (bits per byte) — lower is better. This is printed by `train.py` after the training loop completes.

## 2. Decision metrics

Use these to guide your experiment choices:

| Metric | Source | Meaning |
|---|---|---|
| `val_bpb` | train.py stdout | Primary objective — minimize this |
| `peak_vram_mb` | train.py stdout | Must not OOM — watch this when increasing batch/model size |
| `mfu_percent` | train.py stdout | Hardware utilization — indicates if compute is bottlenecked |
| `training_seconds` | train.py stdout | Must stay within `TIME_BUDGET` |
| `total_tokens_M` | train.py stdout | Throughput — more tokens = more learning within budget |
| `num_params_M` | train.py stdout | Model capacity — larger is not always better under time constraint |

## 3. File ownership

| File | Owner | Editable | Purpose |
|---|---|---|---|
| `ground.json` | User | **NO** | Platform config, data paths, time budgets |
| `prepare.py` | User | **NO** | Data prep, tokenizer, dataloader, eval, platform detection |
| `model.json` | Agent | **YES** | Architecture + optimization hyperparameters |
| `train.py` | Agent | **YES** | Model definition, optimizer, training loop |
| `results.tsv` | Agent | **YES** | Experiment log — append only |
| `program.md` | User | **NO** | This document |

## 4. Execution sequence

### First run (setup)

1. Run `uv run prepare.py` to ensure data and tokenizer are cached.
2. Initialize `results.tsv` with this exact header (tab-separated):

   ```
   run_id val_bpb peak_vram_mb mfu_percent training_seconds total_tokens_M num_params_M status description
   ```

3. Run `uv run train.py`, capturing stdout to `sessions/<run_id>.log`.
   - `run_id` = short git commit hash or a timestamp tag — unique per run.
4. Parse the `---` block from the log to extract metrics.
5. Append one row to `results.tsv` with the extracted values and `status=baseline`.

### Subsequent runs (experiment loop)

1. Form one hypothesis from the current code and most recent run metrics.
2. Edit `model.json` and/or `train.py`.
3. Commit with a message describing the hypothesis.
4. Run `uv run train.py`, capturing stdout to `sessions/<run_id>.log` (use the new commit hash as `run_id`).
5. Parse the `---` block. Extract `val_bpb`, `peak_vram_mb`, `mfu_percent`, `training_seconds`, `total_tokens_M`, `num_params_M`.
6. Append one row to `results.tsv`:
   - `status=keep` if val_bpb improved.
   - `status=discard` if val_bpb did not improve.
   - `status=crash` if the run failed.
7. If `discard` or `crash`: revert with `git reset --hard HEAD~1`.
8. Continue to next hypothesis.

## 5. Logging rules

- Every run MUST have its own log file: `sessions/<run_id>.log`.
- Every run MUST have exactly one row appended to `results.tsv`.
- The `run_id` in `results.tsv` must match the log filename (without `.log`).
- Never overwrite or delete previous log files or results rows.

## 6. Constraints

1. **Time budget**: `train.py` self-enforces via `TIME_BUDGET` from `ground.json`. Do not circumvent this.
2. **No new packages**: use only what is already installed in the environment.
3. **Do not edit** `ground.json`, `prepare.py`, or `program.md`.
4. **Prefer simpler changes** when two options yield similar `val_bpb`.
5. **VRAM**: if a run OOMs, reduce `device_batch_size` in `model.json` or model size before retrying.

## 7. Autonomy

Continue iterating experiments until manually stopped. Do not pause for permission between runs.
