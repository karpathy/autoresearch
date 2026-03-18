# autoresearch

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) by [DeepBlueDynamics](https://github.com/DeepBlueDynamics)

Give an AI agent a real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up to a log of experiments and a better model.

## What's different in this fork

- **Agent harness** (`agent.py`) — structured tool-calling agent that works with Claude, GPT, or Gemini. The agent gets tools to tweak hyperparameters, edit architecture, run experiments, and keep/discard results. No more copy-pasting into a chat window.
- **SDR entropy seeding** — replaces the fixed `torch.manual_seed(42)` with true hardware randomness from an RTL-SDR radio receiver via [sdr-random](https://github.com/DeepBlueDynamics/sdr-random).
- **Optimized defaults** — hyperparameters tuned from 215 experiments across Karpathy's sessions ([Discussion #32](https://github.com/karpathy/autoresearch/discussions/32), [#43](https://github.com/karpathy/autoresearch/discussions/43)): depth 9, halved batch size, SSSSL window pattern, RoPE base 200K, weight decay on embeddings/VE/lm_head, 0.68x init scale, longer warmdown. Baseline 0.9979 -> 0.9697 val_bpb on H100.

## Quick start

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data + train tokenizer (one-time, ~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

## Running the agent

The agent harness gives any LLM provider structured tools to run experiments autonomously.

```bash
# Install agent dependencies (pick your provider)
pip install anthropic openai google-genai

# Run with Claude
python agent.py --provider anthropic --model claude-sonnet-4-20250514

# Run with GPT
python agent.py --provider openai --model gpt-4o

# Run with Gemini
python agent.py --provider gemini --model gemini-2.0-flash

# Run on a named branch, limit experiments
python agent.py --provider anthropic --model claude-sonnet-4-20250514 --tag mar18 --max-experiments 20
```

Set your API key as an environment variable: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`.

### Agent tools

The agent gets 8 tools it can call during the experiment loop:

| Tool | What it does |
|------|-------------|
| `get_config` | Read current hyperparameters from train.py |
| `set_hyperparams` | Modify hyperparameters (batch size, LR, depth, etc.) |
| `edit_code` | Replace entire sections of train.py (model, optimizer, training loop) |
| `run_experiment` | Execute 5-min training run, return val_bpb + metrics |
| `get_history` | Read results.tsv — full experiment log |
| `keep` | Git commit + log improvement to results.tsv |
| `discard` | Revert changes + log failure to results.tsv |
| `read_code` | Inspect specific lines of train.py |

The agent loops autonomously: check config, propose a change, run it, evaluate, keep or discard, repeat. Context auto-compresses so it can run indefinitely.

### Manual mode

You can also run experiments the original way — point Claude Code, Codex, or any coding agent at `program.md`:

```
Hi have a look at program.md and let's kick off a new experiment!
```

## SDR entropy seeding

This fork seeds PyTorch's RNG with true hardware randomness from an RTL-SDR radio receiver. The entropy comes from ADC quantization noise — physically random, not pseudorandom.

Requires [sdr-random](https://github.com/DeepBlueDynamics/sdr-random) running on a machine with an RTL-SDR dongle:

```bash
# On the SDR host
sdr-rand local --port 9090

# train.py auto-fetches entropy at startup, falls back to os.urandom if unavailable
uv run train.py
```

## Project structure

```
train.py        — model, optimizer, training loop (agent modifies this)
prepare.py      — constants, data prep, evaluation (do not modify)
agent.py        — autonomous experiment agent (Claude / GPT / Gemini)
program.md      — manual-mode agent instructions
pyproject.toml  — dependencies
results.tsv     — experiment log (auto-generated)
```

## Optimized defaults

This fork ships with hyperparameters validated across 215 experiments on H100:

| Setting | Upstream | This fork | Impact |
|---------|----------|-----------|--------|
| Depth | 8 | 9 | -0.004 val_bpb |
| Aspect ratio | 64 | 57 | depth-over-width |
| Batch size | 524K | 262K | -0.012 (more steps in 5 min) |
| Window pattern | SSSL | SSSSL | -0.004 cumulative |
| Short window | seq_len/2 | seq_len/8 | narrower local attention |
| RoPE base | 10K | 200K | -0.001 |
| Embedding LR | 0.6 | 0.9 | -0.005 |
| Warmdown ratio | 0.5 | 0.75 | -0.001 to -0.027 |
| Final LR frac | 0.0 | 0.05 | -0.006 |
| Init scale | 1.0x | 0.68x | -0.016 cumulative |
| x0_lambda init | 0.1 | 0.05 | -0.001 |
| Embedding WD | 0.0 | 0.001 | regularization |
| VE WD | 0.0 | 0.003 | -0.003 cumulative |
| LM head WD | 0.0 | 0.01 | -0.009 |
| Softcap | float32 before tanh | bf16 tanh, then float32 | saves ~4GB VRAM |

## License

MIT
