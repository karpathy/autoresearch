# autoresearch — RF-DETR VDA combined detection

This is an autonomous-research loop for the **RF-DETR VDA combined-detection** model.
You (the agent) iterate on the *real* training code in the `videa-ml` repo, run
fixed-wall-clock experiments, and keep changes that raise `val_mAP50`.

> **From-scratch (subset) mode — CURRENT.** Each experiment trains the VDA detector
> **from scratch** — i.e. from the COCO RF-DETR base (`pretrain_weights`, with the 60-class
> head reinitialized), `initial_model_weights: null`, NOT from any of our trained VDA
> checkpoints. Training and validation run on **fixed deterministic subsets** of the data
> (`combined_train_sub.parquet` ~15k imgs, `combined_val_sub.parquet` 400 imgs), and the
> budget is **2 hours** per experiment.
> **Goal = peak / long-term performance, NOT fast convergence.** Optimize for the highest
> `val_mAP50` the recipe reaches within the 2h budget (the model trains well past the
> cold-start; ~15k imgs over ~2h is enough to generalize, not just overfit). Don't over-index
> on how fast a run leaves zero — index on where it ends up. Findings transfer to full-data training.
>
> _(History: an earlier "warm-start fine-tune" mode fine-tuned from the best checkpoint at
> lr 2e-5; its log is `results_warmstart.tsv`. Key carry-over learning: eval uses EMA
> weights, so EMA decay must match the step budget — see that log.)_

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
   - Full parquets: `…/fda60/{combined_train,combined_val}.parquet`
   - **Subsets (what we train/eval on):** `…/fda60/combined_train_sub.parquet` (~15k imgs)
     and `…/fda60/combined_val_sub.parquet` (400 imgs) — see "Fixed data + eval harness".
     Regenerate with `make_val_subsample.py --split train/--split val` if missing or `sources` change.
   - COCO base: `/data/projects/vda/2026_retraining/pretrained/rf-detr-base.pth`
   If missing, stop and tell the user.
5. **Budget knobs (fixed across experiments for comparability):** `steps_per_epoch: 750`
   (~0.8 pass over the 15k train subset/epoch), `n_epochs: 50` (the 2-hour `timeout` ends the
   run first). `steps_per_epoch` also rescales the cosine schedule
   (`total_steps = n_epochs * steps_per_epoch`) — treat it as part of the fixed budget, not
   a tuning knob.
6. **Initialize `results.tsv`**: create `~/Repositories/autoresearch/results.tsv` with just
   the header row (it is gitignored — leave it untracked). The baseline is recorded after
   the first run.
7. **Confirm and go.**

### Fixed data + eval harness (the comparable-metric contract — do NOT change)

Train data, val data, and the metric must stay identical across experiments or `val_mAP50`
is not comparable. Treat all of these like upstream's read-only `prepare.py`:

0. **Fixed train subset.** Training runs on `combined_train_sub.parquet` — a deterministic
   (seed=42), stratified **~15,000-image** subsample of the train split (rarest-class-first,
   all 57 classes covered), built by `make_val_subsample.py --split train --target 15000`.
   Sized so ~2h trains well past cold-start and generalizes (not just overfits), fixed so
   every experiment trains on the same images. Don't change it mid-mode.

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
   (seed=42), stratified 400-image subsample** of the 3209-image val set, built by
   `libs/VDA2/scripts/make_val_subsample.py`: proportional by `source`, rarest-class-first
   so all 57 val classes are covered, then random fill. 400 keeps the bs=1 val pass to
   ~1–1.5 min (a small fraction of the budget, so most of it is training); raise/lower
   `--target` to trade val stability against epochs-per-budget. This is a *progress
   signal*, not the parity gate — regenerate it (same script/seed) if `sources` change,
   and never edit it ad hoc mid-run.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU for a **fixed wall-clock budget of 2 hours**
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

**From-scratch is fixed; the training recipe and architecture are fair game.** Keep
`initial_model_weights: null` (train from the COCO base — do NOT warm-start from a trained
VDA checkpoint), keep the fixed train/val subsets and the metric, but everything that
shapes *how the model learns* is what you tune: LR/schedule/optimizer, augmentation,
EMA, and — since this is from scratch — **architecture** (`projector_scale`, decoder
depth/points, queries, two-stage, backbone knobs in `rf_detr/*.py`). Resolution may now be
changed too (no warm-start to stay pos-enc-compatible with) as long as it stays a multiple
of 56 and `positional_encoding_size` is set consistently.

> `num_classes` stays **60** (the fda60 head). Only **57** of the 60 carry GT in the BW/PA
> data — ids 5 `mandibularCondyle`, 42 `plate`, 43 `screw` are pano-only and absent here —
> so `val_mAP50` averages over those 57. "57/57" from the subsample script = full class
> coverage of the data, NOT a class-count change. Never edit `num_classes`.

**The first run**: your very first run establishes the baseline — run the config as-is
(from COCO, lr 1e-4, `steps_per_epoch 250`), no other changes. `val_mAP50` starts at ~0 and
climbs over the 2-hour budget; optimize for its PEAK, not how fast it leaves zero.

## Running an experiment

Launch **detached** with `setsid` so the run survives even if your shell/monitor process
is reaped — a long-lived foreground/background task that dies takes the training process
group down with it:

```bash
cd ~/Repositories/videa-ml
setsid bash -c 'timeout 7200 uv run --package VDA2 python libs/VDA2/scripts/train_vda2_model.py \
  --fpath-config libs/VDA2/VDA2/training_configs/rfdetr_vda1_combined_fda60.yaml \
  --gpu 5 > ~/Repositories/autoresearch/run.log 2>&1' >/dev/null 2>&1 < /dev/null &
disown
```
- `timeout 7200` enforces the 2-hour budget (the trainer has no built-in time stop).
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

**Timeout**: each experiment should take ~2 h + startup. If a run exceeds ~2h20m of
wall clock (e.g. the `timeout` didn't fire), kill it and treat it as a failed run.

**Crashes**: if a run crashes (OOM, bug, etc.), use judgment. Something dumb (typo, bad
value, OOM you can shrink batch for) → fix and re-run. Fundamentally broken idea → log
`crash` and move on.

**NEVER STOP — RESEARCH NEVER STOPS.** Once the loop has begun (after setup), do NOT pause
to ask the human whether to continue. Do NOT ask "should I keep going?" or "is this a good
stopping point?". The human may be asleep and expects you to work *indefinitely* until
manually stopped. The loop runs until the human interrupts you, period.

**Always be hunting for a better method — not just sweeping the same knobs.** When the
obvious recipe knobs are explored:
1. **Re-read the local sources** for new angles: `RF_DETR_VDA_GUIDE.md` §9 (next-run
   recommendations) and the `rf_detr/*.py` model/optimizer code.
2. **Combine near-misses** and try more radical architecture/optimizer changes (now allowed,
   since we train from scratch): projector scales, decoder depth/points, query count,
   two-stage, loss weights, schedule shape, stronger/different augmentation.
3. **Deep research when you run dry.** Use the `deep-research` skill (or `WebSearch` /
   `WebFetch`) to find current literature and techniques — RF-DETR / DETR / DINOv2
   detection-training improvements, small-object detection, label-efficient training,
   augmentation and optimizer advances — then **implement the most promising idea and test
   it** in the loop. Log what you tried and what the source was in the `description`.
4. Treat every plateau as a prompt to search wider, not to stop. Record dead-ends so you
   don't repeat them. Research never stops.
