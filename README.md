# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

Give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

This fork adds `scripts/agent.py` — a standalone autonomous agent with a Rich TUI dashboard, GPU thermal management, crash recovery, and auto-push to GitHub.

## Quick start

**Requirements:** NVIDIA GPU, WSL2 (Ubuntu), Python 3.10+

```bash
# Clone and setup (one command does everything)
git clone https://github.com/bmdhodl/autoresearch.git
cd autoresearch
bash scripts/setup.sh --api-key sk-ant-api03-YOUR-KEY-HERE

# Run the agent
uv run scripts/agent.py
```

`setup.sh` installs system deps, uv, Python packages, downloads training data, and configures git credentials.

## How it works

1. Agent asks Claude Sonnet to propose a code change to `train.py`
2. Applies the change, commits it, runs training for 5 minutes
3. If val_bpb improves → keep and push. If not → revert.
4. Repeat. All results logged to `agent_results.tsv`.

## Project structure

```
train.py          — model + optimizer + training loop (agent modifies this)
prepare.py        — data prep + evaluation utilities (read-only)
program.md        — original agent instructions
scripts/
  agent.py        — autonomous research agent with dashboard
  setup.sh        — one-command install script
  dashboard.py    — standalone dashboard wrapper
pyproject.toml    — dependencies
```

## Agent features

- **Rich TUI dashboard** — live training metrics, GPU stats, experiment history
- **GPU thermal management** — pauses between experiments, aborts if GPU overheats
- **Crash recovery** — logs state to disk, resumes cleanly after crashes
- **Anti-repetition** — sends full experiment history to LLM, blocks known-bad ideas
- **Auto-push** — pushes kept improvements to GitHub automatically
- **10-minute hard timeout** — kills hung training runs (e.g. from PC sleep)
- **Auto-detects GPU** — adapts VRAM limits and prompt to your hardware

## Agent usage

```bash
uv run scripts/agent.py                  # full dashboard mode
uv run scripts/agent.py --resume         # continue from prior experiments
uv run scripts/agent.py --max-runs 50    # cap total experiments
uv run scripts/agent.py --dataset pubmed # train on PubMed medical abstracts
uv run scripts/agent.py --no-dashboard   # text-only mode
```

## Custom datasets

Train on PubMed medical abstracts (27.7M abstracts, ~14.6GB download):

```bash
# Download data to a specific drive (default: ~/.cache/autoresearch/)
export AUTORESEARCH_DATA_DIR=/mnt/g/autoresearch-data
uv run prepare.py --dataset pubmed

# Or pass it via setup.sh
bash scripts/setup.sh --api-key sk-ant-... --data-dir /mnt/g/autoresearch-data
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`.
- **Fixed 5-minute time budget.** Makes experiments comparable regardless of architecture changes. Expect ~10 experiments/hour.
- **Self-contained.** One GPU, one file, one metric (val_bpb — lower is better).

## Platform notes

Requires WSL2 on Windows for flash-attn3 / Triton / torch.compile support. The `train.py` on master has a Windows native fallback (slower, no flash attention).

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## License

MIT
