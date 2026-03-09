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

### Multi-Ralph on A100 SXM4 40GB (3 agents) — 11 experiments

Best: **1.181 BPB concurrent** (from 1.258 concurrent baseline, -6.1%)

Solo baseline: 1.095 BPB (355 steps, no contention). Concurrent baseline: 1.258 BPB (141 steps, 3 agents sharing GPU).

| # | Agent | Experiment | val_bpb | vs concurrent |
|---|-------|-----------|---------|---------------|
| baseline | agent0 | Depth=8 batch=32 defaults | 1.095 | (solo) |
| 005 | agent2 | x0_lambda init 0.05 | 1.181 | -0.077 |
| 011 | agent1 | x0_lambda + Matrix LR 0.08 | 1.201 | -0.057 |
| 001 | agent1 | Higher Matrix LR 0.04→0.08 | 1.207 | -0.051 |
| 008 | agent2 | Warmdown 0.3 | 1.208 | -0.050 |
| 002 | agent2 | RoPE base 50K | 1.223 | -0.035 |
| 004 | agent1 | Embed LR 0.8 + Unembed 0.008 | 1.242 | -0.016 |
| 007 | agent1 | Concurrent baseline (no changes) | 1.258 | 0.000 |
| exp003 | agent0 | Depth 9 / AR 57 | 1.259 | +0.001 |
| 006 | agent0 | SSSSL window pattern | 1.280 | +0.022 |
| 003 | agent0 | Warmdown 0.7 | 1.333 | +0.075 |
| 009 | agent0 | Lower LRs all around | 1.362 | +0.104 |

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
| A100 SXM4 40GB | 40GB | 3 (multi-ralph) | 32 | ~140 each | Tested, concurrent |
| A100 SXM4 40GB | 40GB | 1 (single-ralph) | 64 | ~355 | Tested, solo |
| RTX 4070 Ti SUPER | 16GB | 1 (single-ralph) | 32 | ~358 (depth 5) | Tested, 32 experiments |
| RTX 4070 Ti SUPER | 16GB | 1 (single-ralph) | 32 | ~169 (depth 8) | Tested |

For multi-ralph, calculate: `DEVICE_BATCH_SIZE` such that `per_process_VRAM × num_agents < total_VRAM`. Include ~30% overhead for torch.compile spikes.

## Origin

Built on [autoresearch](https://github.com/karpathy/autoresearch) by @karpathy. The ralph loop pattern adds persistent memory. Multi-ralph extends it to parallel agents with rotating coordinator.

## License

MIT
