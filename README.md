# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is still intentionally small, but the training runtime is now split into a thin entrypoint plus a compact package:

- **`prepare.py`** — fixed constants, one-time data prep, tokenizer training, dataloader, and evaluation. This remains the evaluation harness.
- **`train.py`** — tiny entrypoint that configures the environment and forwards into the trainer package.
- **`autoresearch_trainer/`** — model, optimizer, compile integration, runtime loop, mmap train-token cache, and validated experiment profiles.
- **`program.md`** — baseline instructions for an autonomous coding agent.
- **`benchmarks/`** — curated benchmark history plus the distilled findings from recent tuning work.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
#    On supported Windows machines, uv downloads the pinned Flash Attention wheel automatically.
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py

# Optional: use the package module entrypoint
uv run python -m autoresearch_trainer
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Git safety

This repo installs the pinned Windows Flash Attention wheel directly from its upstream download URL, so the main setup no longer depends on a tracked local wheel. If you choose to keep local wheels under `vendor/` as a personal cache, those files must stay untracked.

We install a versioned `pre-push` hook with `git config core.hooksPath .githooks` so pushes fail fast if they include files larger than `90 MB` or common ML artifact formats such as `*.whl`, `*.pt`, `*.ckpt`, `*.safetensors`, or `*.onnx`.

If you need a large binary, keep it as a local cache or publish it through GitHub Releases or object storage instead of committing it into Git history.

## Performance profiles

Two validated profiles are built in:

- `baseline` / `default` / `mfu50` — the deeper default, using MLP-only checkpointing to stay near `50% MFU` on the current Windows RTX 3060 setup
- `throughput` — the preserved higher-`tok/s` profile for day-to-day throughput comparisons

Examples:

```bash
uv run train.py --benchmark-steps 20
uv run train.py --experiment-profile baseline --benchmark-steps 20
uv run train.py --experiment-profile throughput --benchmark-steps 20
uv run train.py --experiment-profile mfu50 --benchmark-steps 20
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py                 — fixed data prep + evaluation harness
train.py                   — thin entrypoint
autoresearch_trainer/      — modular training runtime
    token_cache.py         — mmap train-token cache + random token-window loader
    utils/platform.py      — isolated OS and environment hacks
benchmarks/                — benchmark archive + preserved findings
program.md                 — agent instructions
verify_flash_attn.py       — local Flash Attention smoke test
pyproject.toml             — dependencies
vendor/                    — optional untracked local wheel cache
```

## Design choices

- **Small modular core.** The training code is split by responsibility so it stays easy to read and easier to extend than a monolithic script. The main loop is fully object-oriented via a `Trainer` class and `TrainingState` data class, providing isolated state tracking and clean execution contexts.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Validated profiles.** High-throughput and high-MFU settings are kept as named profiles instead of being scattered through ad-hoc notes.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **Agent-Ready Analytics.** The trainer automatically emits structured `metrics.jsonl` files per run, split into `warmup_excluded` and `end_to_end` groups so external swarms or plotting utilities can compare steady-state and full-run behavior without manual log scraping.

## Windows notes

- Flash Attention and `torch.compile` support on Windows are documented in `docs/windows_flash_attention.md`.
- The pinned Windows Flash Attention wheel is downloaded by `uv sync` from its direct URL and validated with a SHA256 hash.
- Recent benchmark analysis and preserved findings live in `benchmarks/SUMMARY_2026-03-15.md` and `benchmarks/KEY_FINDINGS.md`.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. The effective tokens per optimizer step are `DEVICE_BATCH_SIZE * MAX_SEQ_LEN * grad_accum_steps`. On smaller machines, lower `DEVICE_BATCH_SIZE` first and only raise `--grad-accum-steps` if you explicitly want more accumulation.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
