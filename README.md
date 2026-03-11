# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## ANE Backend (Apple Neural Engine)

This fork adds an **ANE training backend** that runs transformer training directly on the Apple Neural Engine via reverse-engineered private APIs. No GPU required — trains on the 15.8 TFLOPS ANE available in every Apple Silicon Mac.

### How it works

- Uses TinyStories dataset with Llama2 32K BPE tokenizer (ANE's native data format)
- **Dynamic weight pipeline**: 10 ANE kernels are compiled once at startup (~470ms). Weights are passed via IOSurface spatial dimensions using `slice_by_size`, not baked into kernels — no recompilation during training
- After each Adam update, weights are transposed and re-staged to per-layer IOSurfaces (~50ms)
- **Scaled initialization**: Wo and W2 weights initialized with `1/sqrt(2*NLAYERS)` residual scaling
- Metric is `val_loss` (cross-entropy), not `val_bpb` — experiments are compared within this framework
- Agent edits only `ane/experiment_config.h` (architecture + optimizer hyperparameters)

### Current best results

**val_loss = 3.55** (~56 autonomous experiment cycles, ~67M param model, 5-min budget)

Starting from 6.109 baseline, key improvements discovered through autonomous experimentation:

| Phase | Change | val_loss | Steps/5min |
|---|---|---|---|
| Static kernels | Baseline (NL=12, SEQ=256) | 6.109 | ~400 |
| | NL=6, SEQ=512 + ACCUM=1 | 5.978 | ~120 |
| | Optimizer tuning (betas, WD, anneal cycles) | 5.414 | ~60 |
| + ncdrone optimizer | Loss scaling, softcap, diff LR, cosine sched | 5.023 | ~120 |
| | Extended training (15 min) | 4.836 | ~120 |
| **Dynamic pipeline** | **One-time compile, no recompilation** | **3.89** | **~1340** |
| | Continued training + ACCUM tuning | **3.68** | **~1340** |
| | EMBED_LR_SCALE=2.0 (reduce embed LR near plateau) | **3.55** | **~1340** |

The dynamic weight pipeline was the single biggest improvement: eliminating per-batch recompilation turned ~60% of wall time from compilation into training, yielding 11x more steps per 5-minute budget.

### Hyperparameters

The agent edits `ane/experiment_config.h`. All hyperparameters and their current best values:

**Architecture** (changing these resets checkpoint):

| Parameter | Value | Notes |
|---|---|---|
| `DIM` | 768 | Model dimension |
| `HIDDEN` | 2048 | FFN hidden dimension |
| `HEADS` | 12 | Attention heads (DIM must be divisible by HEADS) |
| `SEQ` | 512 | Sequence length. 512 is optimal; 1024 hits ANE SRAM wall |
| `NLAYERS` | 6 | Transformer layers. 6 is the sweet spot — fewer layers = faster steps = more training in the 5-min budget |

**Optimizer** (safe to change between runs):

| Parameter | Value | Notes |
|---|---|---|
| `LEARNING_RATE` | 3e-4f | Base learning rate (scaled by differential LR multipliers below) |
| `ADAM_BETA1` | 0.9f | First moment decay |
| `ADAM_BETA2` | 0.95f | Second moment decay (ncdrone uses 0.95 vs default 0.999) |
| `ADAM_EPS` | 1e-8f | Adam epsilon |
| `ACCUM_STEPS` | 4 | Gradient accumulation steps per Adam update + weight re-staging (~50ms) |
| `GRAD_CLIP_MAX` | 1.0f | Global L2 gradient norm clip threshold |
| `WEIGHT_DECAY` | 0.2f | Decoupled weight decay (AdamW). Applied only to weight matrices, not embeddings or RMSNorm |
| `LR_WARMUP_STEPS` | 100 | Linear warmup steps before cosine decay |
| `LR_MIN_FRAC` | 0.1f | Cosine schedule decays LR to this fraction of max |
| `LOSS_SCALE` | 256.0f | Loss scaling factor — prevents FP16 gradient underflow. Undone during gradient averaging |
| `SOFTCAP` | 15.0f | Logit softcapping: `cap * tanh(logits/cap)`, clamps logits to [-cap, cap] to prevent explosion |
| `EMBED_LR_SCALE` | 5.0f | Embedding LR = base LR × this (embeddings learn faster) |
| `MATRIX_LR_SCALE` | 0.05f | Weight matrix LR = base LR × this (matrices learn slower) |

**Optimizer features** (from maderix/ANE + ncdrone research):

- **AdamW** — decoupled weight decay applied before Adam step, only on weight matrices
- **Gradient clipping** — global L2 norm across all parameters using vDSP
- **Cosine LR schedule** — linear warmup for `LR_WARMUP_STEPS`, then cosine decay to `LR_MIN_FRAC` of max LR
- **Loss scaling (256×)** — scales gradients up before FP16 backward pass, undone during gradient averaging. Prevents underflow that causes the 5.5 plateau (maderix)
- **Logit softcapping** — `cap * tanh(logits/cap)` before softmax with chain-rule correction in backward. Prevents logit explosion during training
- **Differential learning rates** — embeddings at 5× base LR, weight matrices at 0.05× base LR, norm params at 1× base LR (ncdrone)
- **Residual scaling** — residual connections scaled by `1/sqrt(2*NLAYERS)` to stabilize deep residual streams

### Differences from the CUDA backend

The ANE backend is a separate training stack, not a port of `train.py`. Key differences:

| | CUDA (`train.py`) | ANE (`train_ane.m`) |
|---|---|---|
| **Optimizer** | Muon + AdamW (per-parameter-group LRs, weight decay, momentum scheduling) | AdamW (differential LR, cosine schedule, loss scaling, logit softcap, residual scaling) |
| **Data** | climbmix-400b, custom 8K BPE | TinyStories, Llama2 32K BPE |
| **Metric** | `val_bpb` (bits per byte) | `val_loss` (cross-entropy) |
| **Language** | Python / PyTorch | Objective-C / raw MIL kernels |
| **Attention** | Flash Attention, sliding window patterns | Standard attention via ANE conv ops |

Results are not comparable across backends — each is its own self-contained research loop.

### Setup

```bash
# 1. Download TinyStories data (~1 GB)
cd ane && bash download_data.sh

# 2. Compile the training binary
make -C ane train_ane

# 3. Run a single experiment (5 minutes)
python harness_ane.py

# 4. Or run with custom wall time
ANE_WALL_TIME=60 python harness_ane.py
```

### Running the autonomous agent

Point Claude at `program_ane.md`:

```
Hi, have a look at program_ane.md and let's kick off a new ANE experiment!
```

The agent will modify `ane/experiment_config.h`, run experiments via `harness_ane.py`, and iterate autonomously.

### ANE-specific files

```
ane/                         # ANE training backend
├── experiment_config.h      # Agent's ONLY editing target
├── stories_config.h         # Model config, structs (includes experiment_config.h)
├── stories_io.h             # IOSurface I/O, dynamic weight staging, request helpers
├── stories_mil_dynamic.h    # Dynamic MIL kernel generators (10 kernels, slice_by_size weights)
├── stories_cpu_ops.h        # CPU ops (RMSNorm, SiLU bwd, cross-entropy, Adam, vocab compaction)
├── train_ane.m              # Training binary (dynamic pipeline, one-time compile, wall-time budget)
├── download_data.sh         # TinyStories data download
└── Makefile                 # Build train_ane binary
harness_ane.py               # ANE orchestrator
program_ane.md               # ANE agent instructions
```

The original CUDA files (`prepare.py`, `train.py`, `program.md`) remain untouched and work as before if you have a GPU.

## Knowledge sources

This project builds on and references the following repositories:

- **[maderix/ANE](https://github.com/maderix/ANE)** — First project to train transformers directly on the Apple Neural Engine using Objective-C and raw MIL kernel compilation. The ANE training backend in this repo is based on this work.
- **[miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)** — MacOS fork of autoresearch adapted for Apple Silicon. Early reference for running autonomous research on Mac hardware.
- **[ncdrone/autoresearch-ANE](https://github.com/ncdrone/autoresearch-ANE)** — ANE-native autoresearch fork. Key finding: "more steps > bigger model" — NL=6 SEQ=512 gets ~3000 steps/5min vs ~400 at NL=12 SEQ=256, achieving val_loss=5.81. This insight drove the architecture change that broke through the initial plateau.


## License

MIT
