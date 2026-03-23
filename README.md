# autoresearch on Modal

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

This fork is set up to run autoresearch loops through a lightweight local Modal control shim. The goal is still the same: let an agent edit `train.py`, run a fixed-budget training experiment, compare the result, and iterate. The difference is that the local machine is only the orchestrator. GPU training happens remotely on Modal, so the local agent can drive long-running experiments without needing a local NVIDIA box.

The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching most of the Python files manually. Instead, you are programming the `program.md` file that tells the agent how to run the loop and evaluate experiments. A bit more context on the original project is in this [tweet](https://x.com/karpathy/status/2029701092347630069).

`progress.png` is a generated artifact and is intentionally not tracked in git. Render it from the current `results.tsv` after the first recorded run.

## How it works

The repo is deliberately kept small and only really has a few files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`modal_control.py`** — local control surface for `start`, `status`, `logs`, `result`, and `stop`.
- **`modal_train.py`** — Modal-backed remote runner implementation. This is infrastructure, not the main experiment surface.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+ locally, a Modal account, and a GPU-backed Modal runtime. The current default training profile is H100-oriented.

```bash
# 1. Install/authenticate Modal locally
python3 -m pip install modal
modal setup

# 2. Put Modal credentials in `.env` 

# 3. Start your ai agent with ability to run without getting blocked. by permissions
claude --dangerously-skip-permissions

# 4. Tell your AI to start working:
> "Hi have a look at program.md and let's kick off a new experiment! let's do the setup first."
```

## Modal-backed local control

This fork includes a small local control shim so the agent has a stable execution interface:

```bash
# Start a cold run in the background
python3 modal_control.py start

# Warm runs can skip data prep and reuse the Modal volume cache
python3 modal_control.py start --skip-prepare

# Inspect a run
python3 modal_control.py status <run_id>
python3 modal_control.py logs <run_id> --tail 80
python3 modal_control.py result <run_id>

# Stop a run
python3 modal_control.py stop <run_id>
```

The controller stores local run state under `.modal-control/`. Remote training
data persists in a Modal Volume mounted at `/root/.cache/autoresearch`. If a
local `.env` file exists, `modal_control.py` loads it automatically before
spawning Modal commands. You do not need `uv sync` just to orchestrate remote
Modal runs.

The agent should use `modal_control.py`, not `modal_train.py`, as the main execution surface.


## Project structure

```
prepare.py         — constants, data prep + runtime utilities (do not modify)
train.py           — model, optimizer, training loop (agent modifies this)
program.md         — agent instructions
modal_control.py   — local control surface for Modal-backed runs
modal_train.py     — remote Modal runner
parse_run.py       — parses run logs into deterministic JSON
pyproject.toml     — optional local research dependencies if you want to run directly
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **Local control, remote execution.** The local machine just starts, polls, and evaluates runs. Modal owns the GPU runtime and persistent cache volume.

## Platform support

This fork no longer assumes the local machine has a GPU. The default path is: local agent loop plus remote Modal GPU execution.

The current baseline is still tuned for a large NVIDIA GPU and was validated against H100-shaped hardware. It is not yet a cheap-GPU profile. In particular, the current defaults are not a good fit for T4 out of the box; if you want to target cheaper GPUs, make a smaller profile first and treat that as a separate experiment track.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
