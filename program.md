# autoresearch — RF-DETR VDA combined detection

This is an autonomous-research loop for the **RF-DETR VDA combined-detection** model.
You (the agent) iterate on the *real* training code in the `videa-ml` repo, run
fixed-wall-clock experiments, and keep changes that raise `val_mAP50`.

> This fork does **not** train the upstream nanochat GPT. The `train.py` / `prepare.py`
> files in this repo are the upstream toy and are **ignored**. The training code you edit
> lives in `~/Repositories/videa-ml` (see paths below).

## Topology (read this first)

- **Instructions + results live here**: `~/Repositories/autoresearch/` holds this
  `program.md` and `results.tsv`.
- **Code you edit + git history live in `videa-ml`**: all edits, commits, branch, and
  keep/discard `git reset` operations happen in `~/Repositories/videa-ml`. The branch,
  not this repo, is what advances.
- All paths below are **absolute** so commands work regardless of your current directory.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun26`). Per-GPU
   variant if running several at once: `jun26-gpu7`. The branch `autoresearch/<tag>` must
   not already exist in `videa-ml`.
2. **Create the branch in videa-ml** (off the current RF-DETR branch `suraj/rf-detr-vda2`):
   ```bash
   cd ~/Repositories/videa-ml && git checkout -b autoresearch/<tag>
   ```
3. **Read the in-scope files** for full context (the repo is large; read only these):
   - `~/Repositories/videa-ml/libs/VDA2/RF_DETR_VDA_GUIDE.md` — **canonical** gotchas,
     parity methodology, next-run recommendations. Read this fully.
   - `~/Repositories/videa-ml/libs/VDA2/VDA2/training_configs/rfdetr_vda1_combined_fda60.yaml`
     — the config you tune.
   - `~/Repositories/videa-ml/libs/vh_nnet/vh_nnet/networks/detection/rf_detr/{model,boxes,collate,optimization}.py`
     — the model/optimizer code you may tune.
4. **Verify data + weights exist** (do not modify these):
   - Parquets: `/data/projects/vda/2026_retraining/data/vda1_detection/processed/annotations/fda60/{combined_train,combined_val}.parquet`
   - Val subsample: `…/fda60/combined_val_sub.parquet` (the fast eval set — see "Fixed
     eval harness" below). Regenerate with `make_val_subsample.py` if it's missing or
     the training `sources` changed.
   - COCO warm-start: `/data/projects/vda/2026_retraining/pretrained/rf-detr-base.pth`
   If missing, stop and tell the user.
5. **Cap `steps_per_epoch` so val runs inside the budget**: a full combined epoch (~59k
   images) takes far longer than the 30-minute budget, so with the default `null` you'd
   get **zero** `val_mAP50` points. Set `train_params_config.steps_per_epoch` to a small
   value (start ~150–300) and a large `n_epochs` (e.g. 50) so several `val_mAP50`
   evaluations land inside the window. `steps_per_epoch` also rescales the cosine schedule
   (`total_steps = n_epochs * steps_per_epoch`), so keep it fixed across experiments for
   comparability — treat it as part of the fixed budget, not a tuning knob.
6. **Initialize `results.tsv`**: create `~/Repositories/autoresearch/results.tsv` with just
   the header row (it is gitignored — leave it untracked). The baseline is recorded after
   the first run.
7. **Confirm and go.**

### Fixed eval harness (the comparable-metric contract — do NOT change)

Validation must stay identical across experiments or `val_mAP50` is not comparable. Two
deliberate choices make the val pass fast *and* fixed; treat both like upstream's
read-only `prepare.py`:

1. **Simple val mAP@0.5 only.** The val callback is the base
   `vh_nnet.networks.mask2former.callbacks.map_metrics.MAPCallback` with
   `iou_thresholds: [0.5]`, `compute_mask_metrics: false`. We removed
   `GroupedClassReportCallback` (it piled per-class/per-group FROC + F1 + HTML scoring on
   top of the bs=1 inference, taking >10 min/epoch) and the per-epoch
   `EpochHTMLReportCallback`. The base callback logs `val_mAP50` (+ a cheap `val_f1`);
   `EarlyStopping` still monitors `val_mAP50`.
2. **Subsampled val set.** `MAPCallback` runs inference one image at a time
   (`batch_size=1`) over the *entire* val set, which dominates the budget. So
   `val_dataset_config` points at `combined_val_sub.parquet` — a **deterministic
   (seed=42), stratified 1000-image subsample** of the 3209-image val set, built by
   `libs/VDA2/scripts/make_val_subsample.py`: proportional by `source`, rarest-class-first
   so all 57 val classes are covered, then random fill. This is a *progress signal*, not
   the parity gate — regenerate it (same script/seed) if `sources` change, and never edit
   it ad hoc mid-run.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU for a **fixed wall-clock budget of 30 minutes**
(plus torch.compile/startup overhead). The metric is **`val_mAP50`** (COCO IoU-0.5
progress signal logged by the base `MAPCallback`) — **higher is better**.

> Note: `val_mAP50` is a progress signal, NOT the official VDA1.0 parity gate (that is the
> `ml_base` centroid-IoU best-F1/FROC/AP harness in the guide). It is the right *fast*
> signal for autonomous keep/discard; periodic true-parity checks are the user's call.

**What you CAN edit** (in the config yaml, or the `rf_detr/*.py` model code):
- Optimizer + LR recipe: `lr`, `lr_encoder`, `lr_vit_layer_decay`, `lr_component_decay`,
  `weight_decay`, `warmup_epochs`, `lr_scheduler`, `ema_decay`, `use_ema`.
- Augmentation: the entire `bw_pa_aug` block (and `train` preprocessor kwargs).
- Compute: `batch_size`, `grad_accum_steps`, `use_amp`, `amp_dtype`, `compile_model`.
- Architecture: `resolution` (**MUST be a multiple of 56** — 560, 616, …, 1008, 1064),
  `projector_scale` (e.g. `[P4]`, `[P3, P4]`), and decoder/backbone knobs in `rf_detr/*.py`.

**What you CANNOT change** (these define the comparable metric and data integrity — they
are this loop's equivalent of upstream's read-only `prepare.py`; see "Fixed eval harness"):
- `val_dataset_config` (the subsampled val set, `combined_val_sub.parquet`) and the
  `MAPCallback` config (`iou_thresholds: [0.5]`, `num_classes`, score/detection thresholds,
  `compute_mask_metrics`) — this is the `val_mAP50` ground truth.
- The data parquet paths, `sources`, `num_classes`, the class map, and `mask_partial_labels`.
- The fixed RF-DETR invariants from the guide — violating these silently corrupts training:
  `positional_encoding_size=37` (fixed by the COCO checkpoint); the head must be reinit to
  `N+1` after warm-start; loss is computed in `train()` mode for BOTH train and val
  (group-DETR); padding-mask convention is `True = padding`; always train on RAW images
  (the preprocessor does 12-bit→8-bit windowing).

**Simplicity criterion**: all else equal, simpler is better. A tiny `val_mAP50` gain that
adds ugly complexity isn't worth it; a gain (or wash) from *removing* code is a clear win.

**The first run**: your very first run establishes the baseline — run the config as-is
(after only the `steps_per_epoch` cap from setup step 5), no other changes.

## Running an experiment

Launch **detached** with `setsid` so the run survives even if your shell/monitor process
is reaped — a long-lived foreground/background task that dies takes the training process
group down with it:

```bash
cd ~/Repositories/videa-ml
setsid bash -c 'timeout 1800 uv run --package VDA2 python libs/VDA2/scripts/train_vda2_model.py \
  --fpath-config libs/VDA2/VDA2/training_configs/rfdetr_vda1_combined_fda60.yaml \
  --gpu 5 > ~/Repositories/autoresearch/run.log 2>&1' >/dev/null 2>&1 < /dev/null &
disown
```
- `timeout 1800` enforces the 30-minute budget (the trainer has no built-in time stop).
  torch.compile startup is *extra* on top of this; for very short experiments you may set
  `compile_model: false` to spend the window on training instead of compilation.
- `--gpu N`: pick an idle GPU (omit to auto-select the lowest-memory GPU). Use distinct
  GPUs for parallel `-gpuN` branches. One run per 20 GB card (~17 GB at 560 res, batch 16)
  — confirm the target GPU is free first (`nvidia-smi`).
- Redirect to `run.log` — do NOT `tee` or let training output flood your context. Poll
  progress with short reads of `run.log` / `history.csv`, not a long-lived watcher.
- To stop a run, `kill <pid>` the specific GPU PID
  (`nvidia-smi --query-compute-apps=pid --format=csv,noheader -i <gpu>`); free the GPU
  before relaunching so two runs don't OOM the card.

## Reading the result

Find the newest run dir for **this config's `dirname_prefix`** and read the best
`val_mAP50` from its `history.csv`. Match the prefix (not a bare `*`) so a concurrent run
sharing the model-runs dir (e.g. a parallel `strongaug` experiment) isn't picked up:
```bash
RUN_DIR=$(ls -dt /data/projects/vda/2026_retraining/model_runs/vda1_combined_fda60/rfdetr_vda1_combined_fda60_r560_*/ \
  2>/dev/null | head -1)
python - "$RUN_DIR/history.csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(f"best_val_mAP50={df['val_mAP50'].max():.6f} epochs={len(df)}")
PY
```
If `history.csv` is missing or has no `val_mAP50` row, the run crashed (or no epoch
finished in the window) → `tail -n 50 ~/Repositories/autoresearch/run.log` to diagnose.
Peak VRAM: `grep -iE "peak.*mem|reserved" ~/Repositories/autoresearch/run.log` or `nvidia-smi`.

## Logging results

When an experiment is done, append a row to `~/Repositories/autoresearch/results.tsv`
(tab-separated, NOT comma — commas break descriptions). Header + 6 columns:

```
commit	val_mAP50	mem_gb	epochs	status	description
```

1. videa-ml git commit hash (short, 7 chars)
2. best `val_mAP50` (e.g. `0.474000`) — use `0.000000` for crashes
3. peak GPU memory in GB, `.1f` — use `0.0` if unknown/crash
4. epochs (val points) completed in the window — use `0` for crash
5. status: `keep`, `discard`, or `crash`
6. short description of what the experiment tried

Example:
```
commit	val_mAP50	mem_gb	epochs	status	description
a1b2c3d	0.421000	15.2	6	keep	baseline (steps_per_epoch=200)
b2c3d4e	0.447000	15.4	6	keep	lr_encoder 1.5e-4 -> 2.5e-4
c3d4e5f	0.410000	15.2	6	discard	projector_scale [P4] only
d4e5f6g	0.000000	0.0	0	crash	resolution 600 (not a multiple of 56)
```

## The experiment loop

The experiment runs on a dedicated branch in `videa-ml` (e.g. `autoresearch/jun26` or
`autoresearch/jun26-gpu7`).

LOOP FOREVER:

1. Check git state in `~/Repositories/videa-ml`: current branch/commit.
2. Tune the config yaml (or `rf_detr/*.py`) with one experimental idea.
3. `git commit` in videa-ml.
4. Run the experiment (the `timeout … train_vda2_model.py` command above).
5. Read the best `val_mAP50` (the snippet above).
6. If there's no `val_mAP50`, the run crashed → read the tail of `run.log` and attempt a
   fix. Give up after a few attempts.
7. Record the row in `results.tsv` (leave `results.tsv` untracked — do NOT commit it).
8. **Disk hygiene**: on `discard` or `crash`, delete that experiment's run dir under
   `/data/projects/vda/2026_retraining/model_runs/vda1_combined_fda60/` — RF-DETR
   checkpoints are large and ~100 overnight runs will fill `/data`. Keep only `keep` runs.
9. If `val_mAP50` improved (higher), **advance** the branch (keep the commit). If it's
   equal or worse, `git reset --hard` the videa-ml branch back to where you started.

You are a fully autonomous researcher: try an idea, keep if it helps, discard if it
doesn't, advance the branch, iterate. Rewind only very sparingly, if ever.

**Timeout**: each experiment should take ~30 min + startup. If a run exceeds ~45 min of
wall clock (e.g. the `timeout` didn't fire), kill it and treat it as a failed run.

**Crashes**: if a run crashes (OOM, bug, etc.), use judgment. Something dumb (typo, bad
value, OOM you can shrink batch for) → fix and re-run. Fundamentally broken idea → log
`crash` and move on.

**NEVER STOP**: once the loop has begun (after setup), do NOT pause to ask the human
whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human may be asleep and expects you to work *indefinitely* until manually stopped. If
you run out of ideas, think harder: re-read `RF_DETR_VDA_GUIDE.md` §9 (next-run
recommendations), re-read the `rf_detr/*.py` code for new angles, combine previous
near-misses, or try more radical architecture/optimizer changes. The loop runs until the
human interrupts you, period.
