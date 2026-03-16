# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and one of:
- **Linux** with an NVIDIA GPU (tested on H100) — full performance
- **macOS Apple Silicon** (M1/M2/M3/M4) — MPS backend, reduced scale
- **macOS Intel / Linux CPU** — CPU fallback, for development and testing

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (platform-detected automatically)
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py

# Optional: select a different model architecture
AUTORESEARCH_MODEL=gpt2 uv run train.py

# Optional: override platform defaults
AUTORESEARCH_DEPTH=4 AUTORESEARCH_DEVICE_BATCH=8 uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

See [docs/platform-support.md](docs/platform-support.md) for detailed platform information and [docs/model-selection.md](docs/model-selection.md) for available model architectures.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py          — constants, data prep + runtime utilities (do not modify)
train.py            — training loop + hyperparameters (agent modifies this)
program.md          — agent instructions
platform_config.py  — auto-detects CUDA/MPS/CPU, sets recommended defaults
models/             — model architectures (nanochat, gpt2)
  __init__.py       — registry + create_model() factory
  base.py           — TrainableModel protocol
  nanochat.py       — default GPT (RoPE, GQA, value embeddings, Muon optimizer)
  gpt2.py           — standard GPT-2 variant (LayerNorm, GELU, AdamW)
tests/              — unit tests (run with: uv sync --extra dev && uv run pytest tests/ -m unit)
docs/               — platform support, model selection, adding new models
pyproject.toml      — dependencies (platform-conditional)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

Autoresearch now runs on multiple platforms. Platform detection is automatic via `platform_config.py`:

| Platform | Device | Flash Attention | torch.compile | Recommended for |
|---|---|---|---|---|
| Linux + NVIDIA GPU | CUDA | Yes (Hopper+) | Yes | Production training |
| macOS Apple Silicon | MPS | No (SDPA fallback) | No | Development, small experiments |
| macOS Intel / CPU | CPU | No (SDPA fallback) | No | Testing, CI |

The platform auto-sets reasonable defaults for depth, batch size, and sequence length based on available hardware. Override via environment variables:

```bash
AUTORESEARCH_DEPTH=4 AUTORESEARCH_DEVICE_BATCH=8 uv run train.py
```

See [docs/platform-support.md](docs/platform-support.md) for full details.

### Tips for smaller compute

For Macbooks and small GPUs, also consider:

1. Use a dataset with less entropy, e.g. [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
2. Decrease `vocab_size` (4096, 2048, 1024, or even byte-level 256).
3. Lower `MAX_SEQ_LEN` in `prepare.py` (down to 256 etc.) and increase `DEVICE_BATCH_SIZE` to compensate.
4. Decrease `EVAL_TOKENS` in `prepare.py` for faster validation.
5. Lower `DEPTH` (the platform auto-detects a reasonable default).
6. Use `WINDOW_PATTERN = "L"` (full context) instead of `"SSSL"` (sliding window may be inefficient without FA3).
7. Lower `TOTAL_BATCH_SIZE` (keep powers of 2, e.g. `2**14` ~16K).

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
