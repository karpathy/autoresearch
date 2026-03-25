# DINOv3 Fine-tuning -- Agent Instructions

This is the autonomous experimentation guide for DINOv3 ViT-H+ LoRA fine-tuning on the product dataset. You are fine-tuning DINOv3 ViT-H+ (840M parameters, 1280-dimensional embeddings) using LoRA adapters and supervised InfoNCE contrastive loss. Your goal: maximize recall@1 on the validation set.

## Setup

To set up a new DINOv3 fine-tuning experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `dino-mar25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The sub-project is self-contained. Read these files for full context:
   - `dino_finetune/program_dino.md` -- these instructions (you are reading it)
   - `dino_finetune/prepare_dino.py` -- IMMUTABLE: model loading, data pipeline, evaluation, adapter save/load. Do not modify.
   - `dino_finetune/train_dino.py` -- YOUR FILE: LoRA config, loss function, optimizer, scheduler, augmentation. Everything here is fair game within constraints.
4. **Initialize results.tsv** in the `dino_finetune/` directory with just the header row:
   ```
   commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
   ```
5. **Run baseline**: `cd dino_finetune && python train_dino.py > run.log 2>&1` -- your first run is always the unmodified train_dino.py to establish the baseline.
6. **Record baseline** in results.tsv, commit as baseline.
7. **Begin experiment loop**.

## Hard Constraints -- NEVER VIOLATE

These are absolute rules. Violating any one invalidates all experiments.

1. **NEVER edit prepare_dino.py** -- it contains model loading, data pipeline, evaluation, and adapter save/load. Modifying it breaks the trust boundary and makes all experiments non-comparable. If you need something from prepare_dino.py, import it.

2. **NEVER add new dependencies** -- only use what is already available: `peft`, `transformers`, `torch`, `torchvision`, `numpy`, `PIL`, `loguru`. If you want a feature from a missing library, implement it yourself in train_dino.py using only these packages.

3. **NEVER exceed EPOCHS=10 per experiment** -- this is the fixed experiment budget enforced by prepare_dino.py. You optimize WHAT happens in 10 epochs, not how many epochs to train.

4. **NEVER stop the loop** -- run until manually interrupted. The human may be asleep. Do NOT ask "should I continue?" or "is this a good stopping point?" See NEVER STOP section below.

5. **NEVER disable gradient checkpointing** -- the DINOv3 ViT-H+ model is 840M parameters. Without gradient checkpointing you WILL hit OOM on a 24GB GPU. Keep `USE_GRADIENT_CHECKPOINTING = True`.

6. **NEVER set BATCH_SIZE > 16** -- physical batch size is limited by VRAM on RTX 4090 (24GB). To increase the effective batch size, use `GRADIENT_ACCUMULATION_STEPS` instead. For example, BATCH_SIZE=8 with GRADIENT_ACCUMULATION_STEPS=32 gives an effective batch of 256.

## Search Space -- Experiment Variables

These are all the tunable constants at the top of `train_dino.py`. Read the file to confirm exact variable names before editing.

### LoRA Configuration

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `LORA_R` | 16 | [4, 64] | LoRA rank -- number of low-rank dimensions. Higher = more capacity but more VRAM and risk of overfitting. Start exploring 8, 16, 32. |
| `LORA_ALPHA` | 32 | [8, 128] | LoRA scaling factor. Typically set to 2x LORA_R. The effective scaling is alpha/r, so alpha=32 with r=16 gives scaling=2.0. |
| `LORA_DROPOUT` | 0.05 | [0.0, 0.2] | Dropout on LoRA layers. Regularization against overfitting. Higher values reduce overfitting but may hurt convergence. |
| `LORA_TARGET_MODULES` | `["q_proj", "v_proj"]` | See below | Which attention projections get LoRA adapters. Can expand to include `"k_proj"`, `"out_proj"`, or MLP modules like `"mlp.fc1"`, `"mlp.fc2"`. More modules = more capacity but more VRAM. |

**LORA_TARGET_MODULES expansion guide:**
- Default `["q_proj", "v_proj"]` -- standard and safe, ~0.2% trainable params
- Add `"k_proj"` -- marginal capacity increase, low risk
- Add `"out_proj"` -- attention output projection, moderate increase
- Add `"mlp.fc1"`, `"mlp.fc2"` -- significant capacity increase, watch VRAM closely
- Adding all of the above with high LORA_R (e.g., 64) WILL cause OOM

### Training Hyperparameters

| Constant | Default | Safe Range | What It Controls |
|----------|---------|------------|------------------|
| `BATCH_SIZE` | 8 | [4, 16] | Physical batch size per GPU step. VRAM-limited. Default 8 is safe for 24GB. Going to 16 is possible but tight -- monitor peak_vram_mb. |
| `GRADIENT_ACCUMULATION_STEPS` | 16 | [4, 64] | Number of forward passes before optimizer step. Effective batch = BATCH_SIZE x GRADIENT_ACCUMULATION_STEPS. Default gives effective batch 128. |
| `LR` | 2e-4 | [1e-5, 1e-3] | Learning rate for AdamW optimizer. This is the most impactful hyperparameter after temperature. Lower values (1e-5 to 5e-5) are conservative; higher values (5e-4 to 1e-3) are aggressive and risk divergence. |
| `WEIGHT_DECAY` | 0.01 | [0.0, 0.1] | AdamW weight decay. Regularization that prevents large weights. 0.01 is standard; try 0.0 to see if it helps or 0.05-0.1 for stronger regularization. |
| `WARMUP_RATIO` | 0.1 | [0.0, 0.3] | Fraction of total training steps used for linear LR warmup. 0.1 means first 10% of steps ramp up LR linearly. Higher warmup is safer for large learning rates. |
| `TEMPERATURE` | 0.07 | [0.03, 0.2] | **CRITICAL** -- InfoNCE temperature scaling. Controls the sharpness of the similarity distribution. Lower = sharper (harder positives/negatives), higher = softer. This is the single most impactful parameter for contrastive learning. |

### Other Constants

| Constant | Default | Notes |
|----------|---------|-------|
| `SEED` | 42 | Random seed. Change if you want to test variance across seeds. |
| `USE_GRADIENT_CHECKPOINTING` | True | **NEVER set to False** -- OOM risk. |
| `EVAL_EVERY_N_EPOCHS` | 1 | How often to evaluate. 1 = every epoch. Set to 2 if you want faster epochs (skips eval). |
| `NUM_WORKERS` | 4 | DataLoader workers. Increase if data loading is the bottleneck. |

## Experiment Strategy (Prioritized)

Work through these priorities in order. Exhaust each priority level before moving to the next.

### Priority 1: Temperature Tuning (MOST IMPACTFUL)

Temperature is the single most impactful hyperparameter in contrastive learning. The default 0.07 is a common starting point but may not be optimal for this dataset.

**Suggested experiments:**
- T=0.05 (sharper, harder negatives)
- T=0.03 (very sharp -- watch for NaN loss)
- T=0.10 (softer, more gradual learning)
- T=0.15 (softer still)

**What to expect:** Small temperature changes produce large metric swings. If loss goes to NaN, temperature is too low.

### Priority 2: Learning Rate Sweep

After finding optimal temperature, sweep learning rate.

**Suggested experiments:**
- LR=5e-5 (conservative)
- LR=1e-4 (moderate)
- LR=5e-4 (aggressive)
- LR=1e-3 (very aggressive -- likely needs higher warmup)

**Combine with warmup:** If using high LR (5e-4+), increase WARMUP_RATIO to 0.2 or 0.3.

### Priority 3: LoRA Rank Exploration

Test whether more or less LoRA capacity helps.

**Suggested experiments:**
- LORA_R=4, LORA_ALPHA=8 (minimal capacity)
- LORA_R=8, LORA_ALPHA=16 (reduced)
- LORA_R=32, LORA_ALPHA=64 (increased)
- LORA_R=64, LORA_ALPHA=128 (high -- watch VRAM)

**Rule of thumb:** Keep LORA_ALPHA = 2 * LORA_R for consistent effective scaling.

### Priority 4: Effective Batch Size

Contrastive learning benefits from larger batches (more negatives per anchor).

**Suggested experiments:**
- Effective batch 64: BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=8
- Effective batch 256: BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=32
- Effective batch 512: BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=64

**Note:** Larger effective batch means fewer optimizer steps per epoch. Compensate with higher LR if needed.

### Priority 5: Target Module Expansion

Add more LoRA adapters to increase model capacity.

**Suggested experiments:**
- Add k_proj: `["q_proj", "v_proj", "k_proj"]`
- Add output projection: `["q_proj", "v_proj", "k_proj", "out_proj"]`
- Add MLP: `["q_proj", "v_proj", "mlp.fc1", "mlp.fc2"]` (significant VRAM increase)

**Warning:** Combining target module expansion with high LORA_R (32+) can cause OOM. Test incrementally.

### Priority 6: Augmentation Changes

Modify data augmentation in train_dino.py if you add custom transforms.

**Ideas:**
- Modify crop scale (default from processor -- try tighter crops for product images)
- Add color jitter (brightness, contrast, saturation)
- Add random horizontal flip probability adjustments
- Add random erasing for regularization

## Workflow -- The Experiment Loop

LOOP FOREVER:

1. **Read history**: `cat dino_finetune/results.tsv`. What has been tried? What improved? What patterns emerge?

2. **Choose next experiment**: Based on history, pick the next experiment from the priority list above. Prefer unexplored dimensions over minor variations. One idea per experiment for clear attribution.

3. **Edit train_dino.py**: Make your changes. Keep diffs minimal and focused. One idea per experiment.

4. **git commit**: Commit the change with a descriptive message (e.g., "dino: reduce temperature to 0.05").

5. **Run**: `cd dino_finetune && python train_dino.py > run.log 2>&1`

6. **Read results**:
   ```bash
   grep "RESULT:\|METRIC:" dino_finetune/run.log
   ```
   If grep is empty, the run crashed -- `tail -n 50 dino_finetune/run.log` for the stack trace.

7. **Log to results.tsv**: Record all 7 columns. Do NOT git-track results.tsv.

8. **Keep or discard**:
   - combined_metric improved (higher than current best)? **KEEP** -- advance the branch.
   - Same or worse? **DISCARD** -- `git reset --hard HEAD~1`
   - Crash? Log as crash in results.tsv, `git reset --hard HEAD~1`
   - 3+ consecutive crashes on the same idea? **SKIP** that direction entirely.

9. **GOTO 1**

## NEVER STOP

Once the experiment loop has begun (after the initial setup and baseline), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" The human might be asleep and expects you to continue working indefinitely until manually stopped. You are autonomous.

If you run out of ideas:
1. Re-read results.tsv for patterns
2. Re-read train_dino.py line by line for overlooked opportunities
3. Re-read this program_dino.md from the top
4. Try combining the best settings from different experiments
5. Try more radical changes (extreme temperature, very different LR, full module expansion)

## Common Pitfalls

- **Temperature too low (< 0.03)**: Causes NaN loss because exp(sim/T) overflows. If you see NaN, increase temperature immediately.
- **Temperature too high (> 0.5)**: Makes the loss insensitive -- all similarities look the same. Loss will be nearly constant and model won't learn meaningful distinctions.
- **BATCH_SIZE > 16**: Causes OOM on 24GB GPU with DINOv3 ViT-H+ even with gradient checkpointing. Use GRADIENT_ACCUMULATION_STEPS for larger effective batch instead.
- **High LORA_R + many target modules**: LORA_R=64 with all attention + MLP modules causes OOM. Test incrementally.
- **Forgetting to check peak_vram_mb**: If peak_vram_mb > 22000 on a successful run, you are near the 24GB limit. Do NOT increase model complexity or batch size further.
- **Not reading results.tsv**: Every experiment must be informed by history. Repeating failed directions wastes compute time.

## Output Format

After each run, the training script logs a RESULT line and prints a METRIC line:

```
RESULT: recall@1=0.4321 mean_cosine=0.8765 combined=0.6543 peak_vram_mb=18432
METRIC: 0.654300
```

Extract metrics from the log:
```bash
grep "RESULT:\|METRIC:" dino_finetune/run.log
```

If grep is empty, the run crashed -- see Crash Handling below.

## Logging Results

Log every experiment to `dino_finetune/results.tsv` (tab-separated). The TSV has 7 columns:

```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. combined_metric (e.g. 0.654321) -- use 0.000000 for crashes
3. recall_1 (e.g. 0.432100) -- use 0.000000 for crashes
4. mean_cosine (e.g. 0.876543) -- use 0.000000 for crashes
5. peak_vram_mb, round to .1f -- use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

Example:
```
commit	combined_metric	recall_1	mean_cosine	peak_vram_mb	status	description
a1b2c3d	0.654321	0.432100	0.876543	18432.1	keep	baseline
b2c3d4e	0.672100	0.460200	0.884000	18500.3	keep	reduce temperature to 0.05
c3d4e5f	0.000000	0.000000	0.000000	0.0	crash	LORA_R=64 with all MLP modules (OOM)
```

results.tsv is NOT git-tracked. It is your experiment log across kept and discarded runs.

## Crash Handling

When a run crashes:

1. **Check the error**: `tail -n 50 dino_finetune/run.log`

2. **If OOM (CUDA out of memory)**:
   - Log as crash in results.tsv
   - `git reset --hard HEAD~1`
   - Next experiment MUST reduce compute (lower LORA_R, fewer target modules, smaller batch size)
   - If peak_vram_mb was > 22000 on the previous successful run, do NOT increase model complexity

3. **If bug (typo, import error, shape mismatch)**:
   - Fix the bug and re-run. This is a code fix, not a failed experiment.
   - `git reset --hard HEAD~1`, fix the issue, commit again, re-run.

4. **If NaN loss**:
   - Almost certainly temperature too low. Increase temperature.
   - `git reset --hard HEAD~1`

5. **If 3+ consecutive crashes on same idea**:
   - Skip that direction entirely. Log the last crash, move to something completely different.

## Domain Context: DINOv3 Contrastive Fine-tuning

- **DINOv3 ViT-H+**: A 840M parameter vision transformer pre-trained with DINOv3 self-supervised method on LVD-1689M dataset. Produces 1280-dimensional CLS token embeddings.
- **LoRA (Low-Rank Adaptation)**: Injects low-rank trainable matrices into frozen transformer layers. Only ~0.2% of parameters are trained, keeping VRAM manageable for the 840M model.
- **InfoNCE Loss**: Supervised contrastive loss. Same-product images are positives, different products are negatives. Temperature controls how sharply the model discriminates between similar and dissimilar pairs.
- **The combined metric**: `0.5 * recall@1 + 0.5 * mean_cosine`. recall@1 measures retrieval accuracy (can the fine-tuned model find the right product?). mean_cosine measures embedding quality (high cosine similarity between same-class embeddings).
- **Why fine-tune DINOv3?**: The fine-tuned model becomes a teacher for the lightweight LCNet student in the main autoresearch pipeline. Better DINOv3 embeddings = better distillation signal = better student model.

## Reading results.tsv for History Reasoning

Before EVERY experiment, read results.tsv:

```bash
cat dino_finetune/results.tsv
```

Analyze before choosing your next experiment:

- **How many experiments have run?** Early = explore broadly. Deep = refine best configuration.
- **What is the current best combined_metric?** What configuration produced it?
- **Which changes improved?** Look for patterns: "lower temperature helped", "higher LR crashed", "more LoRA modules helped".
- **Which changes hurt or crashed?** Avoid repeating failed directions.
- **Use decomposed metrics**: If recall@1 improved but mean_cosine dropped, the change helped retrieval but hurt embedding quality. Target the weaker metric next.
- **Track VRAM trends**: If peak_vram_mb is creeping toward 22000, be cautious about adding model complexity.
