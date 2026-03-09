# autoresearch-dj

Apple Silicon (MLX) fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) with **DOJO mode** — an adversarial testing loop that stress-tests trained models for failure modes that standard evaluation misses.

## Progress

![Closed Loop Progress](progress.png)

**Top**: val_bpb over 70 experiments (green = kept, gray = discarded). **Bottom**: robustness_gap from adversarial testing (red X = DOJO caught a fragile improvement that looked good on val_bpb but failed stress-testing).

Best so far: **val_bpb = 1.484** (-26.3% from 2.014 baseline), robustness_gap = 2.083, subgroup disparity = 0.11 (-84% from 0.69 baseline).

## What is this?

Two autonomous loops in one repo:

1. **Autoresearch mode** (from upstream): An AI agent modifies `train.py`, trains for 5 minutes, checks if `val_bpb` improved, keeps or reverts, repeats overnight. You wake up to a better model.

2. **DOJO mode** (new): After training, the agent flips to adversarial testing. It modifies `test_protocol.py` to find failure modes in the trained model — memorization leakage, subgroup disparity, architectural blind spots, noise fragility. The metric is `robustness_gap`: how much worse can the model perform under adversarial conditions vs. its reported baseline? Bigger gap = the agent found a more meaningful failure.

**Why?** `val_bpb` is an average. It hides the worst-case. A model reporting 1.8 bpb might score 3.5 on numbers and 1.2 on prose. DOJO finds what the average conceals.

## Quick start

Requirements: Apple Silicon Mac (M1/M2/M3/M4), Python 3.10+, uv.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (one-time)
uv run prepare.py

# Run a single training experiment (~7 min including compile + eval)
uv run train.py

# Run DOJO adversarial testing on the trained model
uv run run_dojo.py

# Start the closed loop (train + stress-test, automated)
# Point Claude Code at program_loop.md and let it go

# Or run each mode separately:
# Training only: point Claude Code at program.md
# Adversarial only: point Claude Code at program_dojo.md
```

## How it works

### Training mode (autoresearch)

Same as upstream. Three files:

- **`prepare.py`** — data prep, tokenizer, dataloader, evaluation. Not modified.
- **`train.py`** — model, optimizer, training loop. The agent edits this.
- **`program.md`** — agent instructions for training. Point your agent here.

### DOJO mode (adversarial testing)

After training produces a checkpoint, DOJO mode stress-tests it:

- **`test_protocol.py`** — adversarial test functions. The agent edits this.
- **`run_dojo.py`** — loads the trained model, runs test_protocol, reports robustness gap.
- **`program_dojo.md`** — agent instructions for adversarial testing. Point your agent here.

The agent reads `program_dojo.md`, modifies `test_protocol.py`, runs a 5-minute adversarial evaluation, checks `robustness_gap`, and commits or reverts. Repeat overnight. Wake up to a map of your model's failure modes.

### Closed loop (train + stress-test)

The full pipeline combines both loops into one autonomous cycle where adversarial findings directly inform training decisions. Each iteration:

1. The agent reads adversarial results (e.g., "window boundary has 1.5 BPB gap")
2. Reasons about the root cause (e.g., "sliding window SSSL pattern loses context at the boundary")
3. Decides what to change in `train.py` (e.g., "switch to SSLL to give more layers full attention")
4. Trains the model (5 min)
5. Stress-tests it with adversarial protocols (5 min)
6. Evaluates both metrics: keeps changes only if val_bpb improved WITHOUT making the model more fragile
7. Repeats with a new idea based on what it just learned

The model and its tests co-evolve — training failures inform better tests, and test results inform better training.

- **`program_loop.md`** — agent instructions for the closed loop. Point your agent here.
- **`loop_results.tsv`** — combined metrics: val_bpb, robustness_gap, worst_case_bpb, worst_test per experiment.

~12 min per experiment (5 min train + 2 min compile/eval + 5 min adversarial). ~5 experiments/hour, ~40 overnight.

### Adversarial tests

| Test | What it finds | Why it matters |
|---|---|---|
| Memorization leakage | Training data the model memorized verbatim | Privacy risk, data extraction attacks |
| Subgroup disparity | Text types where the model is much worse | Analogous to demographic bias in healthcare AI |
| Window boundary exploit | Architectural blind spots from sliding window attention | Structural weakness invisible to average metrics |
| Noise robustness | How fragile the model is to minor input perturbations | Deployment readiness |

## Project structure

```
prepare.py           — constants, data prep + runtime utilities (do not modify)
train.py             — model, optimizer, training loop (agent modifies this)
program.md           — agent instructions for training mode
program_dojo.md      — agent instructions for adversarial testing mode
program_loop.md      — agent instructions for closed loop (train + test)
test_protocol.py     — adversarial test functions (agent modifies this)
run_dojo.py          — adversarial test runner: loads checkpoint, runs tests, reports gap
results.tsv          — training experiment log (val_bpb)
dojo_results.tsv     — adversarial experiment log (robustness_gap)
loop_results.tsv     — closed loop log (val_bpb + robustness_gap combined)
```

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — [autoresearch](https://github.com/karpathy/autoresearch) and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — MLX port this fork is based on
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT. Original copyright preserved. See [LICENSE](LICENSE).
