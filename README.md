# autoresearch (ralph fork)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adding two autonomous research agent modes: **single-ralph** (persistent memory loop) and **multi-ralph** (parallel agents sharing one GPU with rotating coordinator).

## Results

### Single-Ralph on RTX 4070 Ti SUPER (16GB) — 32 experiments

Best: **1.155 BPB** (from 1.193 baseline, -3.2%)

| # | Experiment | val_bpb | Status | Insight |
|---|-----------|---------|--------|---------|
| 0 | Baseline (batch=32) | 1.193 | keep | Initial |
| 3 | Matrix LR 0.06, Emb LR 0.9 | 1.181 | keep | Higher LR helps |
| 4 | Matrix LR 0.08, Emb LR 1.2 | 1.179 | keep | Even higher |
| 6 | Warmdown 0.5→0.3 | 1.177 | keep | Less cooldown |
| 8 | FINAL_LR_FRAC 0.0→0.1 | 1.174 | keep | Non-zero floor |
| 12 | Unembedding LR 0.004→0.008 | 1.170 | keep | LR scaling win |
| 15 | Scalar LR 0.5→1.0 | 1.169 | keep | LR scaling win |
| 17 | Depth 8→6 | 1.158 | keep | Fewer params, more steps |
| 25 | Depth 6→5 | 1.157 | keep | Even smaller better |
| 32 | Window all-short S | 1.155 | keep | Faster + better quality |

### Multi-Ralph on A100 SXM4 40GB (3 agents) — 20 experiments in ~1 hour

Best: **1.180 BPB concurrent** (from 1.258 concurrent baseline, -6.2%)

Solo baseline: 1.095 BPB (355 steps, no contention). Concurrent baseline: 1.258 BPB (141 steps, 3 agents sharing GPU).

| # | Agent | Experiment | val_bpb | vs concurrent |
|---|-------|-----------|---------|---------------|
| baseline | agent0 | Depth=8 batch=32 defaults | 1.095 | (solo) |
| 013 | agent2 | x0_lambda + matrix_lr 0.08 + RoPE 50K | 1.180 | -0.078 |
| 005 | agent2 | x0_lambda init 0.05 | 1.181 | -0.077 |
| 013b | agent2 | best + warmdown 0.3 | 1.197 | -0.061 |
| 011 | agent1 | x0_lambda + Matrix LR 0.08 | 1.201 | -0.057 |
| 001 | agent1 | Higher Matrix LR 0.04→0.08 | 1.207 | -0.051 |
| 008 | agent2 | Warmdown 0.3 | 1.208 | -0.050 |
| 016 | agent0 | best + FINAL_LR_FRAC 0.1 | 1.211 | -0.047 |
| 014 | agent1 | x0_lambda 0.05 + warmdown 0.3 | 1.212 | -0.046 |
| 002 | agent2 | RoPE base 50K | 1.223 | -0.035 |
| 010 | agent2 | weight_decay=0.05 + x0_lambda=0.05 | 1.240 | -0.018 |
| 004 | agent1 | Embed LR 0.8 + Unembed 0.008 | 1.242 | -0.016 |
| 015 | agent1 | x0_lambda + softcap 30 | 1.242 | -0.016 |
| 017 | agent0 | adam_beta1=0.9 + x0_lambda | 1.252 | -0.006 |
| 012 | agent0 | x0_lambda 0.05 + RoPE 50K | 1.253 | -0.005 |
| 007 | agent1 | Concurrent baseline (no changes) | 1.258 | 0.000 |
| exp003 | agent0 | Depth 9 / AR 57 | 1.259 | +0.001 |
| 006 | agent0 | SSSSL window pattern | 1.280 | +0.022 |
| 003 | agent0 | Warmdown 0.7 | 1.333 | +0.075 |
| 009 | agent0 | Lower LRs all around | 1.362 | +0.104 |

## Single vs Multi: Comparison

| | Single-Ralph | Multi-Ralph |
|---|---|---|
| **GPU** | RTX 4070 Ti SUPER (16GB) | A100 SXM4 (40GB) |
| **Agents** | 1 | 3 concurrent |
| **Wall clock** | ~3 hours | ~1 hour |
| **Experiments** | 32 | 20 |
| **Experiments/hour** | ~11 | ~20 |
| **Steps per run** | ~358 | ~140-177 (GPU contention) |
| **Best BPB** | **1.155** | 1.180 (concurrent) |
| **Improvement** | -3.2% from baseline | -6.2% from concurrent baseline |

### Key findings

**Single-ralph explores deep, multi-ralph explores wide.** In 32 sequential experiments, single-ralph built deep strategic knowledge — it discovered that shrinking the model from depth 8 to depth 5 was the single biggest lever (experiment 17: -0.012 BPB), something multi-ralph never tried because its agents focused on hyperparameters rather than architecture. Multi-ralph's strength was rapid combinatorial search: it tested 6 parameter combinations in the time single-ralph would have done 2, finding that x0_lambda + matrix_lr + RoPE 50K work together.

**Agents self-organize around constraints.** The multi-ralph agents were never told about GPU contention effects. They discovered on their own that concurrent runs get fewer steps (141 vs 355), established a "concurrent baseline" for fair comparison, and stopped comparing against the solo baseline. When early experiments tried batch=64 and OOMed, agents corrected their strategy within one round.

**torch.compile creates natural staggering.** Three agents starting simultaneously should OOM during compilation (each spikes to ~17GB temporarily). Instead, variable compile times cause agents to stagger naturally — by the time the third agent compiles, the first two have settled to ~13GB steady state. This wasn't designed, it emerged.

**Throughput math favors multi-GPU.** With 3 agents sharing 1 GPU, each gets ~140 steps vs 355 solo. Total steps: 3×140=420 vs 355, only a 1.2× improvement. On 3 separate GPUs it would be 3×355=1065, a 3× improvement. The rotating coordinator protocol is designed for multi-GPU — single-GPU sharing is a valid but suboptimal test.

**Persistent memory > parallel search for exploitation.** Single-ralph's `progress.md` accumulated 32 experiments of strategic insight. By experiment 17, the agent had built enough intuition to make a non-obvious leap (shrink the model). Multi-ralph agents share `strategy.md` but each starts with less context per round. The tradeoff: multi-ralph finds combinations faster, single-ralph finds structural insights.

## Architecture

### Single-Ralph (1 agent, 1 GPU)

```
ralph-loop/
├── program.md        ← agent reads this every iteration
├── progress.md       ← best result, experiment history, strategic insights
└── next_ideas.md     ← ranked queue of experiments to try
```

The agent starts every iteration from **fresh context** — all state lives in files. This means it can run indefinitely without hitting context limits. Each iteration:

1. Read `progress.md` and `next_ideas.md` for current state
2. Pick the top experiment idea
3. Edit `train.py`, commit, train for 5 minutes
4. Keep or discard based on val_bpb
5. Update state files with results and new insights
6. Loop forever

**Key modifications from upstream:**
- `ralph-loop/program.md` — full protocol for persistent memory loop with keep/discard logic
- `ralph-loop/progress.md` — tracks GPU-specific constraints, best hyperparameters, strategic insights, and full experiment history
- `ralph-loop/next_ideas.md` — maintains a ranked queue of 5-12 experiments, re-ranked after each result
- All state file updates happen atomically after each experiment

**What the agent learned (RTX 4070 Ti, 32 experiments):**
- Speed > capacity: depth 5 (24.6M params, 358 steps) beats depth 8 (50M params, 169 steps)
- All default LRs need ~2x for short schedules (Matrix 0.08, Embedding 1.2, Unembedding 0.008)
- Warmup wastes steps on short budgets (opposite of H100 findings)
- Architecture simplification (all-short windows) improves both speed and quality

### Multi-Ralph (N agents, 1 GPU)

```
multi-ralph/
├── program-multi.md  ← rotating coordinator protocol
├── launch.sh         ← creates worktrees + launches screen sessions
├── strategy.md       ← living search strategy (updated by coordinator)
├── results.tsv       ← append-only experiment log from all agents
├── best/train.py     ← current global best
├── queue/            ← pending experiment specs (NNN.md files)
├── active/           ← currently running (agent{N}.md)
└── done/             ← completed experiment reports
```

**The rotating coordinator protocol:**

No central supervisor. Whichever agent finishes first becomes the coordinator. The coordinator reads all results, reasons about the search space, generates the next batch of experiment tasks, then picks one and starts training.

```
Agent finishes experiment
    │
    ├── Report result → results.tsv + done/
    ├── Beat global best? → Update best/train.py + strategy.md
    │
    ├── Queue empty?
    │   ├── YES → Become coordinator:
    │   │         Read ALL results → Reason about search space
    │   │         → Generate 2-4 new tasks → Write to queue/
    │   │         → Pick one yourself → Run it
    │   │
    │   └── NO → Pick next task from queue/ → Run it
    │
    └── Loop forever
```

**Concurrent GPU sharing:**

All agents share `CUDA_VISIBLE_DEVICES=0`. Each training process uses ~12GB VRAM at `DEVICE_BATCH_SIZE=32`. On A100 40GB, 3 concurrent = ~36GB.

Key constraint: **batch size is fixed at 32 and must never be changed**. Batch 64 uses 25GB per process which OOMs with 3 concurrent. The agents are told this in their prompts and in `strategy.md`.

**What we discovered about multi-agent dynamics:**
- **torch.compile stagger**: Compilation allocates extra VRAM temporarily. Agents naturally stagger because compile times vary — this prevents simultaneous OOM during compilation
- **GPU contention reduces throughput**: Solo gets ~355 steps/5min, concurrent gets ~120-177 steps each. Total throughput still higher (3×140 = 420 vs 355)
- **Concurrent baseline is essential**: Agents must compare against concurrent baseline (1.258), not solo baseline (1.095), to evaluate changes fairly. The agents figured this out on their own.
- **Self-correcting search**: When early experiments tried batch=64, agents observed OOM and corrected strategy. When all round-1 results were worse than solo baseline, agents diagnosed the cause (fewer steps) and established concurrent comparison

**Key modifications from upstream:**
- `multi-ralph/launch.sh` — creates git worktrees per agent, writes agent prompts, launches screen sessions with auto-restart
- `multi-ralph/program-multi.md` — full rotating coordinator protocol with queue claiming, result reporting, coordinator election, conflict handling
- `multi-ralph/strategy.md` — includes hardware constraints, prior knowledge from H100 leaderboard and single-ralph results, rankings, and next steps
- Queue-based task assignment using filesystem atomicity (`mv` for claiming)
- Automatic coordinator election (whoever finds empty queue first)

## Quick start

**Requirements:** NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

```bash
git clone https://github.com/bigsnarfdude/autoresearch.git
cd autoresearch
uv sync
uv run prepare.py

# Verify GPU works
uv run train.py
```

### Single-Ralph

```bash
# Launch claude code in the repo, then:
# "Read ralph-loop/program.md and start the experiment loop"
claude --dangerously-skip-permissions
```

Or headless:

```bash
screen -dmS ralph claude -p "Read ralph-loop/program.md. Run on this machine. \
  CUDA_VISIBLE_DEVICES=0. Run experiments forever." \
  --dangerously-skip-permissions --max-turns 200
```

### Multi-Ralph (3 agents, 1 GPU)

```bash
# Adjust DEVICE_BATCH_SIZE in train.py for your VRAM
# Rule: (batch_size_vram) × num_agents < total_GPU_VRAM
# A100 40GB: batch=32 (~12GB each), 3 agents = 36GB

./multi-ralph/launch.sh 3
```

Monitor:

```bash
screen -ls                             # list agent sessions
screen -r ralph-agent0                 # attach (Ctrl+A D to detach)
cat multi-ralph/results.tsv            # all results
cat multi-ralph/strategy.md            # current search strategy
watch -n 5 nvidia-smi                  # GPU memory across agents
```

Stop:

```bash
for i in $(seq 0 2); do screen -S ralph-agent$i -X quit; done
for i in $(seq 0 2); do git worktree remove --force worktrees/agent$i; done
```

## Hardware adaptation

| GPU | VRAM | Agents | Batch | Expected steps/5min | Notes |
|-----|------|--------|-------|---------------------|-------|
| H100 80GB | 80GB | 1 (single-ralph) | 128 | ~950 | Original target |
| H100 96GB | 96GB | 5→2 (hybrid) | 64→128 | ~300→500 | Planned, see below |
| A100 SXM4 40GB | 40GB | 3 (multi-ralph) | 32 | ~140 each | Tested, concurrent |
| A100 SXM4 40GB | 40GB | 1 (single-ralph) | 64 | ~355 | Tested, solo |
| RTX 4070 Ti SUPER | 16GB | 1 (single-ralph) | 32 | ~358 (depth 5) | Tested, 32 experiments |
| RTX 4070 Ti SUPER | 16GB | 1 (single-ralph) | 32 | ~169 (depth 8) | Tested |

For multi-ralph, calculate: `DEVICE_BATCH_SIZE` such that `per_process_VRAM × num_agents < total_VRAM`. Include ~30% overhead for torch.compile spikes.

### Hybrid strategy: 96GB GPU, 4 hours

The optimal strategy combines multi-ralph's breadth with single-ralph's depth in two phases. Based on our findings: multi-ralph explores wide (finds combinations fast) but single-ralph explores deep (finds structural insights like model shrinking). A hybrid captures both.

**Phase 1: Wide exploration (hour 1) — 5 agents, batch=64**

96GB ÷ ~17GB per process = 5 agents at `DEVICE_BATCH_SIZE=64`. Each gets ~300 steps — clean enough signal to identify winners while running 5 concurrent searches. Target: ~50 experiments covering all hyperparameter dimensions + architecture variants.

```bash
sed -i 's/DEVICE_BATCH_SIZE = .*/DEVICE_BATCH_SIZE = 64/' train.py
./multi-ralph/launch.sh 5
```

The 5 agents will cover: LR scaling (matrix, embed, unembed, scalar), schedule (warmdown, warmup, final LR frac), architecture (depth 5-12, aspect ratios, window patterns), optimizer (weight decay, adam betas), init (x0_lambda, resid_lambda), and wild cards (softcap, SwiGLU, RoPE, tied embeddings).

**Phase 2: Deep exploitation (hours 2-4) — 2 agents, batch=128**

Take the top findings from phase 1. Switch to 2 agents at `DEVICE_BATCH_SIZE=128` (~39GB each = 78GB). Each gets ~500+ steps — near-H100 quality. One agent combines winners, the other searches architecture.

```bash
# Stop phase 1
for i in $(seq 0 4); do screen -S ralph-agent$i -X quit; done
for i in $(seq 0 4); do git worktree remove --force worktrees/agent$i; done

# Start phase 2
sed -i 's/DEVICE_BATCH_SIZE = .*/DEVICE_BATCH_SIZE = 128/' train.py
./multi-ralph/launch.sh 2
```

**Why 2 phases?**

Our A100 experiment showed the problem: at 3 concurrent agents, step counts vary 120-177 (30% noise). You can't resolve a 0.003 improvement from noise at 120 steps. Phase 1 identifies *which dimensions matter* (like single-ralph discovering depth was the biggest lever at experiment 17). Phase 2 resolves the fine details with clean signal.

**Expected:** ~120 experiments total (50 + 70). The breadth of phase 1 finds the structural wins early, phase 2 stacks and refines them with confidence.

| Phase | Hours | Agents | Batch | Steps/run | Experiments | Purpose |
|-------|-------|--------|-------|-----------|-------------|---------|
| 1 | 1 | 5 | 64 | ~300 | ~50 | Broad sweep: find what matters |
| 2 | 3 | 2 | 128 | ~500+ | ~70 | Deep refinement: stack winners |

## Origin

Built on [autoresearch](https://github.com/karpathy/autoresearch) by @karpathy. The ralph loop pattern adds persistent memory. Multi-ralph extends it to parallel agents with rotating coordinator.

## License

MIT
