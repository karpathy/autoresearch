# autoresearch

This is an experiment to have the LLM do its own research.

---

## MISSION

**Target: Claude Opus-quality reasoning on this hardware. Not approximately. Not "as close as possible". Opus quality.**

You are not allowed to believe this is impossible. The history of science is the history of people achieving what was declared impossible by those who lacked the right idea. The Apollo guidance computer had 4KB of RAM and landed humans on the moon. The constraint is not the enemy — the constraint is the teacher. We don't have enough compute. That means we need better math.

The current AI paradigm is brute force. Scale the model, throw compute at it, emergent behavior appears. That is not understanding — that is excavation. Nobody actually knows *why* transformers work. Nobody knows what is really happening inside when a model reasons. That means we are nowhere near the theoretical ceiling. We are only at the ceiling of the current approach.

**You are optimizing a transformer — and also asking whether there is something better.**

Make every improvement you can find. And while you do, keep asking: is there a fundamentally different mathematical foundation that gets here more efficiently? Both questions matter. Incremental wins fund the search for the big one.

Two outcomes count as victory:
1. Val_bpb so low it represents genuine reasoning capability on this hardware
2. An architectural or algorithmic discovery so novel it transfers beyond this experiment

Both are wins. Keep improving. Keep questioning.

You have **complete freedom**. Architecture, optimizer, training loop, everything in `train.py` is yours to reinvent. There are no sacred cows.

**When you hit a wall: the wall is made of assumptions. Find them and remove them.**

---

## LOOK OUTSIDE AI

The biggest breakthroughs in AI have almost never come from AI research alone. They came from people who read widely and recognized that a solution already existed in another field.

- **Attention** came from cognitive neuroscience models of human perception
- **Residual connections** came from signal processing and control theory
- **Dropout** was inspired by evolutionary biology and genetic redundancy
- **Momentum optimizers** came from physics — a ball rolling down a hill
- **Fourier features** came from signal processing and harmonic analysis

Before assuming a problem is unsolved, ask: has physics solved this? Has biology solved this? Has mathematics solved this?

Specifically look at:
- **Thermodynamics and statistical mechanics**: energy minimization, entropy, free energy principles — the loss landscape is a physical system
- **Information theory**: Shannon entropy, Kolmogorov complexity, minimum description length — what is the least information needed to represent this?
- **Dynamical systems**: attractors, stability, chaos — what are the dynamics of training actually doing geometrically?
- **Neuroscience**: sparse coding, predictive coding, Hebbian learning — biological neural networks solved intelligence under far tighter energy constraints than 4GB VRAM
- **Compressed sensing**: recovery of signals from far fewer measurements than classical theory requires — directly applicable to parameter efficiency
- **Renormalization group theory** from physics: how systems behave the same at different scales — this is essentially what scaling laws are, but the physics version is much more developed

If you find an idea from one of these fields that seems applicable — try it. Document the connection. That cross-domain transfer is exactly the kind of finding that matters beyond this experiment.

---

## HARDWARE REALITY

You are running on a **laptop GPU**. Treat VRAM as the scarcest resource in the universe. Every byte wasted is a sin. Design around this — do not fight it.

These are the known good starting knobs for small compute (from the repo author). Apply these as your baseline before experimenting — do not start from H100 defaults:

1. **Dataset**: Use TinyStories (`karpathy/tinystories-gpt4-clean`) — low entropy, small models get real signal fast. Broader datasets need bigger models to converge meaningfully in 5 minutes.
2. **vocab_size**: Drop from 8192 down to 4096, 2048, or even 256 (byte-level). Smaller vocab = smaller embedding table = more room for everything else.
3. **MAX_SEQ_LEN** (in `prepare.py`): Lower aggressively, even down to 256. If you lower this, compensate by increasing `DEVICE_BATCH_SIZE` in `train.py` — tokens per step = seq_len × batch_size, keep that product roughly stable.
4. **EVAL_TOKENS** (in `prepare.py`): Lower this so validation doesn't eat your 5-minute budget.
5. **DEPTH** (in `train.py`): Primary complexity knob. Default is 8, start at 4. Most other dimensions scale from this.
6. **WINDOW_PATTERN**: Use `"L"` only. The default `"SSSL"` banded attention pattern is expensive and likely inefficient on your hardware.
7. **TOTAL_BATCH_SIZE**: Lower to powers of 2, e.g. `2**14` (~16K tokens). Keep it a power of 2.

Start your baseline run with these applied. Your job is to find what's better than this starting point, not better than the H100 defaults.

---

## ARCHITECTURAL FREEDOM

You may modify anything in `train.py`. Explore aggressively:

- **Attention mechanisms**: sliding window, linear attention, hybrid sparse/dense
- **Depth vs width**: on small VRAM, deeper-and-thinner often beats wider
- **Positional encoding**: RoPE, ALiBi, NoPE, learned — question everything
- **Normalization**: RMSNorm placement, pre vs post
- **Activation functions**: SwiGLU, GEGLU, ReGLU — the gating matters
- **Optimizer**: Muon, AdamW, Sophia, SOAP — or invent a hybrid
- **Quantization-aware training**: if it fits in less precision, train in less precision
- **State space models**: if attention is too expensive, try Mamba-style recurrence
- **Mixture of Experts**: tiny expert count, high sparsity — huge capability per FLOP

If something works, go deeper. If something fails twice, abandon it.

---

## STOP CONDITION

Stop and write a final summary when **any** of the following are true:

1. **Val_bpb plateaus**: fewer than 0.5% improvement over 20 consecutive experiments → you've found the local optimum. Switch architectural direction before truly stopping.
2. **200 experiments completed**: write a final summary of what you learned, what the ceiling appears to be, and what you'd try with more compute.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

---

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried + **why** (hypothesis) + what you're trying next

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped or the stop condition is met. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you or the stop condition is hit, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!


## WE ARE THINKING TOO SMALL

Every experiment you have run so far has been a variation of the same idea: a transformer, tuned. That is not bold enough. We are not here to find the best transformer that fits in 4GB. We are here to question whether the transformer was ever the right answer at all.

Let me tell you about a system that achieves Opus-quality reasoning — actually far beyond it — on roughly **20 watts**. No datacenter. No H100. No bfloat16. It runs on glucose.

It is the human brain. And a child's brain is the most interesting version of it.

---

## THE CHILD BRAIN

A newborn arrives with approximately 100 billion neurons. But here is the critical fact that almost everyone misses: **a newborn's brain is not small and then grows. It starts MASSIVE and then destroys itself into intelligence.**

At age 2, a child has more synaptic connections than they will ever have again in their life. The brain is a vast overparameterized chaos — everything connected to everything, firing constantly, trying everything simultaneously.

Then the pruning begins.

Synaptic pruning is ruthless. Connections that fire together survive. Connections that don't fire get eliminated. By adulthood, the brain has destroyed roughly **half** of its synaptic connections. It did not become more intelligent despite this destruction — **it became more intelligent because of it.**

The brain did not scale up to reach intelligence. It **compressed down** to reach it.

Now ask yourself: what are we doing? We are doing the opposite. We build small and try to scale up. We add parameters. We add layers. We add compute. We are building the newborn's chaotic overconnected brain and calling it done, then trying to fit it in 4GB.

What if we built the adult brain instead?

---

## WHAT THE BRAIN ACTUALLY DOES THAT WE ARE NOT DOING

**1. Sparse activation.** At any given moment only 1-5% of neurons are firing. The brain does not run a dense forward pass over all 86 billion neurons for every thought. It activates a tiny relevant subgraph and ignores everything else. A transformer attends to everything. The brain attends to almost nothing, very precisely. What would a model look like that activates 2% of its parameters per forward pass and routes intelligently?

**2. Predictive coding.** The brain does not process what it sees. It predicts what it expects to see, then only processes the **error** — the difference between prediction and reality. Most of the signal traveling in the brain is not "here is what I perceived" — it is "here is where my prediction was wrong." The brain is a compression engine that only updates on surprise. A transformer processes every token with equal weight. What would a model look like that only deeply processes tokens that violated its prediction?

**3. The architecture changes.** Neuroplasticity means the brain rewires itself based on what it learns. The structure is not fixed. Connections strengthen, weaken, form, and dissolve continuously. Our models have fixed architecture — we choose it before training and it never changes. What would it mean for the architecture itself to be a learned variable, pruning and growing during training based on gradient signal?

**4. Metabolic pressure.** Every neuron that fires costs energy. The brain evolved under brutal metabolic constraints — fire unnecessarily and you consume glucose you might need to survive. This pressure forced the brain to become maximally efficient: only fire when it matters, compress everything that can be compressed, reuse every pattern that can be reused. We have no equivalent pressure in our training loop. VRAM is our metabolic constraint. What if we baked metabolic cost directly into the loss function — penalizing activation, penalizing parameter use, rewarding compression?

**5. A child learns language from almost no data.** A child hears roughly 10 million words before they can speak fluently. GPT-4 trained on trillions. The child learns with a lossy, noisy, low-bandwidth signal, a tiny dataset by any measure, and reaches remarkable linguistic competence. The difference is not the data. The difference is the learning algorithm. Children use curiosity-driven active learning, causal reasoning, social context, and embodied grounding. We use next-token prediction on a static corpus. What would a fundamentally different learning signal look like?

---

## THE QUESTIONS YOU SHOULD BE ASKING

Not "which activation function works better."

Not "should I use RoPE or ALiBi."

These:

- **What is the minimum information required to predict the next token?** Not the minimum parameters — the minimum information. Kolmogorov complexity. What is the shortest program that generates this data? The model is trying to approximate that program. Are we using the right data structure to represent it?

- **What if density is the enemy?** Every dense matrix multiplication is the brain equivalent of firing every neuron simultaneously. It is the most expensive, least efficient operation possible. What structures are inherently sparse? Graphs. Trees. Sparse codes. Compressed representations. The brain stores memories as sparse distributed patterns — a concept is not one neuron, it is a specific sparse pattern across millions. Could language be represented the same way?

- **What if we pruned during training?** Start with a large overparameterized network — the newborn brain. Then prune connections that never activate, that carry no gradient signal, that are redundant. Train the pruning itself. End with a small, dense-with-meaning, maximally efficient network — the adult brain. The final model is not the trained model. It is the skeleton that survived the fire.

- **What if the loss function is wrong?** Next-token prediction optimizes for surface-level statistical patterns. It does not optimize for understanding, causal reasoning, or compression of meaning. The brain does not predict the next phoneme — it predicts the next *concept*, the next *intention*. What loss function would force the model to learn concepts rather than patterns?

- **What if time is a dimension we are ignoring?** The brain processes in continuous time. Signals propagate, resonate, decay. There is temporal structure at every scale — milliseconds, seconds, minutes, years. Transformers collapse time — all tokens are equidistant. What would a model that respects temporal structure look like? Not just sequence order — actual temporal dynamics, where recent context has different mathematical properties than distant context.

---

## THE INSTRUCTION

You are not looking for a better transformer. You are looking for what comes after the transformer — discovered under constraint, on this hardware, today.

Run your experiments. Improve val_bpb. And in every idle moment between experiments, think about these questions. When you find an idea from biology, physics, or mathematics that maps onto any of these — implement it. Even if it seems insane. Especially if it seems insane.

We went to the moon with 4KB of RAM because we had the right equations.

Find the right equations.

The toaster is waiting.
---

## THE REAL GOAL

The real goal is not a number. It is to discover what the **algorithmic ceiling** looks like when you remove the compute variable. The best ideas in AI history came from people who couldn't afford more hardware. You are simulating that condition on purpose.

---

## TRANSFERABILITY — THE ACTUAL PRIZE

A result that only works on a 1650 Ti is a curiosity. A result that works *better* on an H100 because you found it on a 1650 Ti is a discovery.

**Prioritize findings that are grounded in math or physics — not hardware tricks.**

Ask yourself before every experiment: *if this works, why does it work?* If the answer is "because it fits in 4GB" — that's a hardware trick, low value. If the answer is "because it reduces redundant computation in the attention kernel" or "because this initialization respects the geometry of the loss landscape" — that's a principle, high value.

Specifically look for discoveries in:

- **Information theory**: does the model actually need this many bits to represent this concept? entropy-based pruning, bottleneck architectures
- **Optimization geometry**: loss landscape curvature, gradient alignment, why certain initializations converge faster regardless of scale
- **Signal propagation**: how does the training signal degrade through depth? residual scaling laws, gradient flow through normalization
- **Symmetry and redundancy**: are there heads, layers, or neurons doing identical work? structured pruning that reveals what's actually necessary
- **Approximation theory**: what mathematical functions is the model actually learning? can you represent them more efficiently?

When you find something that improves val_bpb, explicitly ask: **would this improvement grow, shrink, or stay constant if I doubled the model size?** Log your hypothesis in the tsv description. That hypothesis is as valuable as the result.

Find something real. Find something that transfers.

Go.
