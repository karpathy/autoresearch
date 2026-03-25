# autoresearch

This is an experiment to have the LLM do its own research on ReID knowledge distillation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `program.md` -- these instructions (you are reading it)
   - `prepare.py` -- IMMUTABLE: data loading, teacher inference, evaluation, caching. Do not modify.
   - `train.py` -- YOUR FILE: student model, losses, optimizer, scheduler, augmentations. Everything here is fair game within constraints.
4. **Verify teacher cache exists** at `workspace/output/trendyol_teacher_cache2/`. If it does not exist, the first run will build it automatically (takes ~10-30 min, excluded from experiment budget). This is normal.
5. **Initialize results.tsv** with just the header row:
   ```
   commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
   ```
6. **Run baseline**: `python train.py > run.log 2>&1` -- your first run is always the unmodified train.py to establish the baseline.
7. **Record baseline** in results.tsv, commit as baseline.
8. **Begin experiment loop**.

Once the baseline is recorded, kick off the experimentation loop below.

## Experimentation

Each experiment runs on a single GPU (RTX 4090, 24GB VRAM). The training script runs for a **fixed budget of 10 epochs** (NOT wall-clock time -- you optimize WHAT happens in 10 epochs, not how many). You launch it simply as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, loss functions, loss weights, augmentations, training loop, batch size, projection head design, backbone choice. All within the constraints below.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, teacher inference, and caching logic. Modifying it breaks the trust boundary and makes all experiments non-comparable.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`. Available: torch, timm, onnxruntime, transformers, torchvision, numpy, PIL.
- Modify the evaluation harness. The `evaluate_retrieval` and `compute_combined_metric` functions in `prepare.py` are the ground truth metric.
- Exceed the edge deployment limits (see Hard Constraints below).

**The goal is simple: get the highest combined_metric.** The combined metric is `0.5 * recall@1 + 0.5 * mean_cosine`. Higher is better. recall@1 measures retrieval accuracy (can the model find the right product?). mean_cosine measures teacher alignment (does the student agree with the teacher?).

**VRAM** is a HARD constraint. The RTX 4090 has 24GB. If peak VRAM exceeds 22GB on a successful run, do NOT increase batch size or model size further -- you are near the limit. OOM = crash = discard.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. But always maintain edge-deployability -- a simpler model that's too big to deploy is useless.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is. Record the baseline combined_metric, then begin experimenting.

## Hard Constraints -- NEVER VIOLATE

These are absolute rules. Violating any one invalidates all experiments.

1. **NEVER edit prepare.py** -- it contains evaluation, data loading, and teacher inference. Modifying it breaks the trust boundary. All experiments become non-comparable. If you need something from prepare.py, import it. Do not add new imports of private/internal functions.

2. **NEVER install new packages** -- only use what's in `pyproject.toml`. Available: torch, timm, onnxruntime, transformers, torchvision, numpy, PIL. If you want a feature from a missing library, implement it yourself in train.py using only these packages.

3. **NEVER exceed 10 epochs** -- this is the fixed experiment budget. You optimize WHAT happens in 10 epochs, not how many epochs to train. The epoch count is enforced by prepare.py.

4. **NEVER stop the loop** -- run until manually interrupted. The human may be asleep. Do NOT ask "should I continue?" or "is this a good stopping point?" See NEVER STOP section below.

5. **NEVER exceed edge deployment limits** -- the student model must remain edge-deployable:
   - Embedding dimension MUST remain 256.
   - After any architecture change, verify parameter count and GFLOPs are reasonable.
   - Parameter count must not grow unbounded. If you change the backbone, verify it is still lightweight (LCNet-class, not ResNet-50 class).
   - If you switch backbones, confirm the new one is comparable in size to `lcnet_050`.

6. **NEVER remove or modify the checkpoint saving block** -- the code between `# Save model checkpoint` and `# Compute final metrics` in train.py MUST remain intact. It saves `checkpoint_last.pt` and `checkpoint_best.pt`. You may change `OUTPUT_DIR` but never delete the `torch.save()` calls or the checkpoint dict construction.

7. **NEVER remove quality degradation augmentation** -- real-world ReID images are low-resolution, JPEG-compressed, and degraded. The `RandomQualityDegradation` transform simulates this. You may tune its parameters (prob, downsample_ratio, quality_range) but it must remain active. Removing it will improve metrics on clean validation images but produce a model that fails in production.

## Output Format

After each run, the script prints a summary block:

```
---
combined_metric:  0.654321
recall@1:         0.432100
mean_cosine:      0.876543
peak_vram_mb:     18432.1
total_seconds:    342.5
epochs:           10
```

Extract the key metrics from the log file:

```
grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log
```

If the grep output is empty, the run crashed -- see Crash Handling below.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated -- commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. combined_metric achieved (e.g. 0.654321) -- use 0.000000 for crashes
3. recall_1 (e.g. 0.432100) -- use 0.000000 for crashes
4. mean_cosine (e.g. 0.876543) -- use 0.000000 for crashes
5. peak_vram_mb, round to .1f (e.g. 18432.1) -- use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
a1b2c3d	0.654321	0.432100	0.876543	18432.1	keep	baseline
b2c3d4e	0.672100	0.460200	0.884000	18500.3	keep	reduce ArcFace weight from 0.05 to 0.02, increase LR to 0.15
c3d4e5f	0.640000	0.410000	0.870000	18400.0	discard	switch to GeLU activation in projection head
d4e5f6g	0.000000	0.000000	0.000000	0.0	crash	double batch size to 512 (OOM)
```

**NOTE:** results.tsv is NOT git-tracked. It is your experiment log across kept and discarded runs.

Keep descriptions informative so you can reason about them later. Bad: "try stuff". Good: "reduce ArcFace weight from 0.05 to 0.02, increase LR to 0.15".

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. **Read history**: Check results.tsv. `cat results.tsv`. What has been tried? What improved? What patterns emerge? (e.g., "lower LR helped", "VAT hurts", "ArcFace weight 0.1 > 0.05"). See the History Reasoning section below for guidance.

2. **Choose next experiment**: Based on history, pick an idea from the playbook or formulate your own hypothesis. Prefer unexplored dimensions over minor variations of explored ones. One idea per experiment for clear attribution.

3. **Edit train.py**: Make your changes. Keep diffs minimal and focused. One idea per experiment.

4. **git commit**: Commit the change with a descriptive message (e.g., "reduce ArcFace weight to 0.02").

5. **Run**: `python train.py > run.log 2>&1`

6. **Read results**: `grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram_mb:" run.log`
   - If grep is empty: run crashed. `tail -n 50 run.log` for the stack trace. See Crash Handling below.

7. **Log to results.tsv**: Record all 7 columns. NOTE: do not git-track results.tsv.

8. **Keep or discard**:
   - combined_metric improved (higher than current best)? **KEEP** -- advance the branch.
   - Same or worse? **DISCARD** -- `git reset --hard HEAD~1`
   - Crash? Log as crash in results.tsv, `git reset --hard HEAD~1`
   - 3+ consecutive crashes on the same idea? **SKIP** that direction entirely and move on.

9. **GOTO 1**

**Timeout**: Each experiment takes roughly 5-15 minutes depending on batch size and augmentation. If a run exceeds 30 minutes, kill it (`kill` the process) and treat it as a failure -- discard and revert.

## NEVER STOP

Once the experiment loop has begun (after the initial setup and baseline), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?" The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder -- re-read the playbook, re-read results.tsv for patterns, re-read train.py for overlooked opportunities, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10-15 minutes (10 epochs), you can run approximately 4-6 experiments per hour, for a total of about 40-50 over the duration of an overnight run. The user then wakes up to experimental results -- a full results.tsv and a branch history of improvements -- all completed by you while they slept.

## Domain Context: ReID Knowledge Distillation

A brief primer so you understand the domain you are optimizing:

- **ReID (Re-Identification)**: Given a query image of a product, find matching products in a gallery using embedding similarity. The model maps images to a vector space where similar products are close and different products are far apart.
- **Knowledge distillation**: A large, accurate teacher model (ONNX Trendyol model or DINOv2) produces high-quality 256-dimensional embeddings. A small student model (LCNet050) learns to match the teacher's embeddings while remaining small enough to deploy on edge devices.
- **The combined metric** balances two objectives:
  - **recall@1** -- retrieval accuracy. Can the model find the correct product as its top-1 nearest neighbor? This is the end-user metric.
  - **mean_cosine** -- teacher alignment. Does the student's embedding agree with the teacher's? This provides a stable training signal even when retrieval has not converged.
- **The student model pipeline**: Input image -> backbone (LCNet050 from timm) -> projection head (linear layers) -> L2-normalized 256-dimensional embedding.
- **Training uses multiple losses**:
  - **Distillation loss** (cosine similarity with teacher embeddings) -- the primary training signal.
  - **ArcFace loss** (angular margin classification) -- pushes embeddings of same-class products closer, different-class further apart.
  - **VAT loss** (Virtual Adversarial Training) -- adversarial robustness regularization. Currently disabled by default (weight=0).
  - **Separation loss** -- pushes blacklist product embeddings away from whitelist centroid.
- **The backbone is partially frozen**: Initially frozen, then last stages are unfrozen at a configurable epoch. This prevents destroying pretrained features early in training.

## Search Space Reference

These are all the tunable constants in train.py. Read train.py to confirm exact variable names before editing -- names may differ slightly from this table.

| Constant | Default | Domain | Notes |
|----------|---------|--------|-------|
| `LR` | 1e-1 | Optimizer | Learning rate -- high-impact, try first |
| `WEIGHT_DECAY` | 1e-5 | Optimizer | Regularization strength |
| `BATCH_SIZE` | 256 | Training | Distillation batch size -- watch VRAM |
| `ARCFACE_BATCH_SIZE` | 128 | Training | ArcFace batch size -- watch VRAM |
| `ARCFACE_LOSS_WEIGHT` | 0.05 | Loss | Weight of ArcFace loss in total -- high-impact |
| `ARCFACE_S` | 32.0 | Loss | ArcFace scale parameter |
| `ARCFACE_M` | 0.50 | Loss | ArcFace margin -- affects class separation |
| `VAT_WEIGHT` | 0 (disabled) | Loss | Virtual adversarial training -- try enabling |
| `VAT_EPSILON` | 8.0 | Loss | VAT perturbation magnitude |
| `SEP_WEIGHT` | 1.0 | Loss | Separation loss weight |
| `UNFREEZE_EPOCH` | 5 | Backbone | When to unfreeze backbone stages |
| `QUALITY_DEGRADATION_PROB` | 0.5 | Augmentation | Probability of quality degradation -- NEVER set to 0 |
| `DROP_HARD_RATIO` | 0.2 | ArcFace | Hard negative mining ratio |
| `ARCFACE_PHASEOUT_EPOCH` | 0 (disabled) | Loss | Epoch to phase out ArcFace -- try enabling |
| `EMBEDDING_DIM` | 256 | Model | MUST remain 256 (hard constraint) |
| `MODEL_NAME` | `hf-hub:timm/lcnet_050.ra2_in1k` | Model | Can change to other timm backbones if edge-deployable |

**Important:** These names may differ slightly from the actual constants in train.py. Always read train.py to confirm exact variable names before editing.

## ReID Experiment Playbook

Experiment hints organized by expected impact. Start with Tier 1 and work down. These are directional suggestions -- use your judgment and results history to choose exact values.

### Tier 1 -- High Impact, Low Risk (try first)

- **Loss weight ratios**: Adjust the balance between distillation, ArcFace, and separation losses. The default ArcFace weight (0.05) is conservative -- experiment in the range 0.01-0.2. The distillation loss is implicit (weight 1.0). Try different ratios to find the best balance between teacher alignment and class separation.
- **Learning rate and schedule**: Try cosine annealing with warmup. Experiment with warmup fraction (5-20% of epochs). Try lower LR (1e-2 to 1e-3) with longer warmup. The default LR (1e-1) is aggressive -- lower values with a schedule may converge better.
- **Unfreeze epoch and stages**: Currently unfreezes backbone at epoch 5. Try earlier (epoch 2-3) or later (epoch 7-8). Try unfreezing fewer or more stages. Use a lower LR for backbone parameters than for the projection head (differential learning rates).

### Tier 2 -- Medium Impact, Medium Risk

- **Projection head design**: Add hidden layers between backbone and embedding. Try BatchNorm -> ReLU -> Linear patterns. Try dropout between layers. Keep final output dimension = 256 (hard constraint).
- **Augmentation parameters**: Tune quality degradation intensity (prob, downsample ratio, quality range). Add or adjust color jitter, random erasing. NEVER remove quality degradation entirely (hard constraint).
- **ArcFace margin and scale**: Tune ARCFACE_M (try 0.3-0.7) and ARCFACE_S (try 16-64). These control how sharply the model separates classes. Higher margin = harder task = potentially better embeddings if training converges.
- **VAT regularization**: Currently disabled (weight=0). Try enabling with small weight (0.01-0.1). VAT adds adversarial perturbations to embeddings, which may improve robustness. Start small -- too much VAT can destabilize training.
- **Optimizer choice**: Default is SGD with momentum. Try AdamW with lower LR (1e-3 to 1e-4). AdamW often works better for fine-tuning pretrained models.

### Tier 3 -- High Impact, Higher Risk (try when Tier 1/2 plateau)

- **Different backbone**: Try other lightweight models from the timm registry. MUST remain edge-deployable -- check parameter count and GFLOPs. Examples: mobilenetv3_small_100, efficientnet_lite0, tf_mobilenetv3_small_minimal_100. Always verify parameter count is comparable to LCNet050 (~2M params). Do NOT try large models (ResNet-50, EfficientNet-B4, etc.).
- **Batch size changes**: Increase for better gradient estimates but watch VRAM. Decrease if hitting OOM. Batch size affects gradient noise and can significantly change convergence dynamics.
- **ArcFace phaseout**: Try phasing out ArcFace loss after epoch N (e.g., epoch 6-8) to let distillation dominate in the final training epochs. This may help the student focus on teacher alignment during fine-tuning.
- **Novel loss ideas**: Circle loss, subcenter ArcFace, proxy-anchor loss. Implement from scratch using only torch (no new dependencies). These are advanced -- only try if Tier 1/2 ideas are exhausted. The math is standard and you can implement them in train.py.

## When You Are Stuck

If experiments stop improving, escalate through these levels:

1. **Minor plateau (3-5 no-improvement experiments)**: Switch to a different parameter dimension. If you were tuning LR, switch to loss weights. If tuning losses, try augmentation. Change the axis of search, not just the values.

2. **Medium plateau (5-10 no-improvement experiments)**: Jump to a different tier. Re-read results.tsv and identify the single best configuration. Try combining the best configuration with ideas from unexplored dimensions. Look at decomposed metrics -- maybe recall@1 is good but mean_cosine is lagging. Target the weaker metric.

3. **Major plateau (10+ no-improvement experiments)**: Try radical changes -- different backbone, novel loss function, extreme augmentation parameter changes. Re-read train.py carefully line by line for overlooked opportunities. Re-read this playbook from the top. Try ideas you previously dismissed.

## Reading results.tsv for History Reasoning

Before EVERY experiment, read results.tsv:

```
cat results.tsv
```

Analyze before choosing your next experiment:

- **How many experiments have run?** Are you early (exploring broadly) or deep (refining)?
- **What is the current best combined_metric?** What configuration produced it?
- **Which changes improved it?** Look for patterns: "LR decreases helped", "ArcFace weight in range 0.02-0.05 worked best", "VAT consistently crashed".
- **Which changes hurt or crashed?** Avoid repeating failed directions.
- **Use decomposed metrics**: If recall@1 improved but mean_cosine dropped, the change helped retrieval but hurt teacher alignment. Choose your next experiment to address the weaker metric specifically.
- **If multiple crashes on similar ideas**, skip that entire direction. Move to a completely different parameter dimension.
- **Track VRAM trends**: If peak_vram_mb is creeping up across experiments, be cautious about adding model complexity or increasing batch sizes.

The quality of your experiments depends on the quality of your reasoning about past results. Do not skip this step.

## Crash Handling

When a run crashes:

1. **Check the error**: `tail -n 50 run.log`

2. **If OOM (CUDA out of memory)**:
   - Your next experiment MUST reduce compute, not try the same thing with a minor tweak.
   - Reduce batch size, reduce model size, or simplify the architecture.
   - If peak_vram_mb was > 22000 on the previous successful run, do NOT increase batch size or model size further. You are near the 24GB limit.

3. **If bug (typo, import error, shape mismatch)**:
   - Fix the bug and re-run. This does not count as a failed experiment -- it's a code fix.
   - `git reset --hard HEAD~1`, fix the issue, commit again, re-run.

4. **If 3+ consecutive crashes on the same idea**:
   - Skip that direction entirely. Log the last crash, then move to something completely different.
   - Do not keep trying variations of a fundamentally broken idea.

5. **VRAM budget rule**: If peak_vram_mb > 22000 on a successful run, do NOT increase batch size or model size further. You are operating near the 24GB RTX 4090 limit. Future experiments should maintain or reduce compute.
