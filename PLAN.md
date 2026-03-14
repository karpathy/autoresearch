# AutoAnything: Plan for Generalizing Autoresearch

## The Idea

Autoresearch proves a powerful concept: an AI agent can autonomously improve a system by proposing changes, scoring them against an objective metric, and keeping only what works. But right now it's welded to one domain — optimizing a GPT training script against val_bpb on a single GPU — and it's serial, single-agent, all on one machine.

The concept generalizes to *any* optimization problem with a black-box scoring function. AutoAnything is a framework where you define what to optimize and how to score it, then unleash a swarm of agents — potentially hundreds, running in different sandboxes, different machines, different continents — to hill-climb relentlessly. You come back in a week to a leaderboard of experiments and a measurably better system.

Think of it as the crowdsourced, distributed version of Ralph. Ralph is a goal in a loop with one agent. AutoAnything is a goal in a loop with an *army* of agents, all submitting ideas concurrently, but evaluated one at a time against the one true current state.

**Examples of things you could optimize:**

- A prompt template (scored by LLM-as-judge accuracy on a test set)
- A web app's Lighthouse performance score
- A compiler optimization pass (scored by benchmark runtime)
- A trading strategy (scored by backtested Sharpe ratio)
- An infrastructure config (scored by p99 latency under load)
- A game AI (scored by win rate against a baseline opponent)
- An ML training script (the original use case)

The common pattern: there's some mutable state (files an agent can change), an evaluation function that produces a number, and a direction (lower is better, or higher is better). Everything else is details.

## Intellectual Context

This sits at the intersection of several ideas:

- **Ralph / Gastown** — single-agent goal-in-a-loop. AutoAnything is the distributed, crowdsourced generalization. Where Ralph is single-threaded, this lets hundreds of agents race to improve the same thing.
- **Danny's observation** — this needs a true black box scoring system. If agents can see the evaluator, they will game it. The evaluator must be private.
- **Goodhart's Law / Venkatesh on over-specified metrics** — "When a measure becomes a target, it ceases to be a good measure." The scoring function must actually capture what you care about. A bad metric optimized ruthlessly produces paperclips — a system that scores well but misses the point. The quality of your scoring function is the ceiling on the quality of your results.
- **Under-specified goals, the paperclip problem** — the flip side. If the goal is too vague or the metric too narrow, relentless optimization produces perverse results. AutoAnything makes this concrete: you MUST have a numerical scoring function, and whatever you pick, hundreds of agents will exploit every degree of freedom it leaves open.

The scoring function is everything. AutoAnything is just plumbing.

## The Critical Design Insight: Blind Evaluation

The most important architectural decision is the separation between **what agents can see** and **how scoring works**.

Agents must NOT have access to the scoring function. If they can see the evaluator, they can game it — overfitting to the test set, exploiting quirks in the metric, or just hardcoding good scores. This is the same reason Kaggle keeps the test set private.

This means the system is fundamentally **two separate things**:

1. **The Challenge Repo** (public/shared) — the mutable files, agent instructions, score history, and a description of what "better" means in plain language. This is what agents clone and work in. The scoring function is NOT here.

2. **The Evaluator** (private) — the scoring code, evaluation infrastructure, and merge logic. Only the problem owner controls this. It runs somewhere agents can't see into — a private server, a GitHub Action with secrets, a gitignored directory on the evaluator's machine.

Agents submit their work by **pushing a git branch** (or opening a PR). They describe what they tried. The evaluator picks it up, scores it in private, and decides whether to merge it forward.

This is Kaggle for code. The leaderboard is public, the test set is private, and submissions are git branches.

## Scores

Scores are always numerical. The problem definition specifies:

- **Direction:** minimize or maximize
- **Bounds:** whether the score has a known theoretical limit

**Bounded scores** have a known best-possible value (e.g., accuracy 0–100%, error rate that bottoms out at 0). Agents know how close they are to "done." As you approach the bound, diminishing returns are expected and agents should try increasingly creative approaches.

**Unbounded scores** have no known ceiling/floor (e.g., throughput in requests/sec, Dragon Ball Z power levels). There's always room to improve. The score just keeps going. The system never converges — it just gets better and better over time, and the rate of improvement is itself interesting to track.

This distinction matters for agent strategy: bounded problems eventually "solve" (or plateau near the bound); unbounded problems are infinite games.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         Challenge Repo            │
                    │         (public/shared)           │
                    │                                   │
                    │  - mutable files (e.g. train.py)  │
                    │  - agent instructions              │
                    │  - leaderboard / history           │
                    │  - problem description             │
                    │  - NO scoring code                 │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         push branch      push branch      push branch
              │                │                │
        ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
        │  Agent 1   │   │  Agent 2   │   │  Agent N   │
        │  (sandbox) │   │  (sandbox) │   │  (sandbox) │
        │            │   │            │   │            │
        │ clones repo│   │ clones repo│   │ clones repo│
        │ makes edits│   │ reads hist │   │ tries idea │
        │ pushes     │   │ pushes     │   │ pushes     │
        └────────────┘   └────────────┘   └────────────┘

                               │
                     ┌─────────┴──────────┐
                     │     Evaluator       │
                     │     (private)       │
                     │                     │
                     │ - serial eval queue │
                     │ - has scoring code  │
                     │ - rebase → eval →   │
                     │   merge or discard  │
                     └─────────────────────┘
```

### The Challenge Repo

This is an ordinary git repo (typically on GitHub). It contains:

```
challenge-repo/
├── problem.yaml             # Problem definition (what to optimize, constraints, score direction)
├── agent_instructions.md    # Guidance for agents (like program.md but generic)
├── leaderboard.md           # Auto-updated scoreboard
├── history.tsv              # Full experiment log (commit, score, status, description)
│
├── src/                     # The mutable files agents can edit
│   └── train.py             # (or whatever the problem's mutable state is)
│
└── context/                 # Read-only files agents can reference
    └── prepare.py           # (or whatever immutable context the problem needs)
```

**`problem.yaml`** tells agents what they're optimizing and what files they can touch. It does NOT contain the evaluation command or scoring code:

```yaml
name: gpt-pretraining
description: >
  Optimize a GPT training script for lowest validation bits-per-byte (val_bpb).
  Training runs for a fixed 5-minute time budget. You can change anything in the
  mutable files: model architecture, optimizer, hyperparameters, batch size, etc.
  The evaluation metric is val_bpb — lower is better.

mutable:
  - src/train.py

readonly:
  - context/prepare.py

score:
  direction: minimize
  name: val_bpb
  description: "Validation bits per byte — measures how well the model compresses unseen text"
  bounded: false    # no known theoretical minimum for this metric

constraints:
  - "Training must complete within the 5-minute time budget"
  - "Only packages in pyproject.toml are available"
  - "Must not modify files outside of src/"
```

Another example — a bounded score:

```yaml
name: classifier-prompt
score:
  direction: maximize
  name: accuracy
  description: "Classification accuracy on held-out test set"
  bounded: true
  bound: 100.0     # perfect accuracy = 100%
```

**`agent_instructions.md`** is the generalized `program.md`. It tells agents the protocol:

```markdown
# How to Participate

1. Clone this repo and create a branch: `proposals/<your-name>/<short-description>`
2. Read `problem.yaml` to understand what you're optimizing
3. Read the files in `context/` for background
4. Read `history.tsv` and `leaderboard.md` to see what's been tried and what worked
5. Modify ONLY the files listed under `mutable` in `problem.yaml`
6. Commit with a clear message explaining your approach
7. Push your branch (or open a PR)

The evaluator will automatically pick up your branch, rebase it onto current main,
score it, and either merge (if improved) or discard (if not).

If your branch can't rebase cleanly onto current main, it will be discarded.
Pull the latest main and try again.
```

**`history.tsv`** is the experiment log, updated by the evaluator after each run:

```
commit	branch	score	status	description	timestamp
a1b2c3d	main	0.997900	baseline	initial baseline	2026-03-13T10:00:00Z
b2c3d4e	proposals/agent-1/higher-lr	0.993200	accepted	increase LR to 0.04	2026-03-13T10:08:00Z
c3d4e5f	proposals/agent-2/gelu	1.005000	rejected	switch to GeLU activation	2026-03-13T10:12:00Z
d4e5f6g	proposals/agent-3/wider	0.000000	crash	double model width (OOM)	2026-03-13T10:15:00Z
```

### The Evaluator

The evaluator is a separate, private system that the problem owner runs. It watches the challenge repo for new branches/PRs, runs the scoring function, and decides whether to merge.

#### Evaluation is Serial

This is a key design decision. The evaluation queue is **serial** — one proposal at a time, always against the true latest state of main. Here's why:

The only thing that matters is: "does this proposal make main better?" To answer that, you must evaluate the proposal *as it would exist on current main*. If main has moved since the agent started working, the proposal must be rebased onto current main before evaluation. If it can't rebase cleanly, discard it — the world moved on, try again.

This means parallel evaluation (multiple proposals being scored simultaneously in worktrees) wastes work. While you're evaluating proposal A, proposal B might get merged, and now A's evaluation is against a stale base. You'd have to rebase and re-evaluate anyway.

**The evaluation loop is simple:**

```
FOREVER:
  1. Pick the next proposal branch from the queue
  2. Rebase it onto current main
  3. If rebase fails → discard, record in history, continue
  4. Run the scoring function on the rebased code
  5. If the score improved → merge to main, update history + leaderboard, push
  6. If the score didn't improve → discard, record in history
  7. If it crashed → record as crash in history
```

If a proposal doesn't make things incrementally better, you discard it forever. No second chances, no "close enough," no combining. The rule is binary: better or gone.

**Proposal generation is massively parallel** — hundreds of agents can be thinking, coding, and pushing branches simultaneously. But the funnel narrows to a single thread at evaluation time. This means the evaluator is always the bottleneck, and evaluation throughput determines overall progress rate. If each evaluation takes 5 minutes, you get 12 evaluations/hour regardless of how many agents are working. But those 12 are drawn from a much larger pool of ideas.

#### Evaluator Deployment Options

**Option A: Gitignored Local Evaluator (simplest)**

The evaluator lives right in the challenge repo, just gitignored:

```
challenge-repo/
├── .gitignore              # includes: evaluator/
├── problem.yaml
├── src/train.py
├── context/prepare.py
├── agent_instructions.md
├── history.tsv
│
└── evaluator/              # GITIGNORED — only on the evaluator's machine
    ├── score.sh            # the actual scoring script
    ├── evaluate_loop.sh    # the serial evaluation loop
    └── data/               # private test data, if any
```

When you clone this repo on the machine that runs evaluations, you create the `evaluator/` directory locally. It's gitignored, so agents who clone the repo don't get it. The evaluator polls for new branches, rebases them, runs `score.sh`, and merges or discards.

This is the simplest deployment. One machine, one repo, the evaluator is just some gitignored files. Good for single-user or small-team use. Good for getting started.

**Option B: GitHub Actions**

A workflow triggers on PRs. The scoring code lives in a private repo (fetched via secret token) or as encrypted GitHub Action secrets. The workflow rebases the PR, scores it, posts the result as a comment, and auto-merges or closes.

```yaml
# .github/workflows/evaluate.yml
on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest  # or self-hosted with GPU
    concurrency:
      group: evaluation     # only one evaluation at a time (serial queue)
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - name: Rebase onto main
        run: git rebase origin/main
      - name: Fetch scoring code
        uses: actions/checkout@v4
        with:
          repository: user/my-evaluator-private
          token: ${{ secrets.EVALUATOR_TOKEN }}
          path: _evaluator
      - name: Run evaluation
        run: bash _evaluator/score.sh
      - name: Post results and merge/close
        run: python _evaluator/autoanything/reporter.py
```

The `concurrency` setting ensures only one evaluation runs at a time (serial queue). The scoring code repo is private. Agents can't see it.

For problems that need GPUs or special hardware, use self-hosted runners.

**Option C: Standalone Server**

An HTTP service running somewhere (a VM, a container, a machine with GPUs). Polls the challenge repo for new branches, evaluates them serially, pushes results back. Good for expensive evaluations or custom environments.

All three options implement the same serial evaluation loop — they differ only in where the loop runs and how the scoring code is kept private.

### Concurrency Model

To be precise about what's parallel and what's serial:

```
PARALLEL (unlimited):
  - Agents thinking about what to try
  - Agents writing code
  - Agents pushing branches
  - Agents reading history to learn what worked

SERIAL (one at a time):
  - Rebase proposal onto current main
  - Run scoring function
  - Merge or discard
  - Update history
```

When main advances (a proposal gets merged), every pending proposal in the queue is now based on a stale version. The evaluation loop handles this by rebasing each proposal onto current main right before evaluating it. If the rebase fails (merge conflict), that proposal is discarded — the agent's changes are incompatible with what was accepted in the meantime.

This means if you're an agent and you pushed a branch 30 minutes ago, by the time the evaluator gets to it, main might have advanced 6 times. Your branch will be rebased onto that new main. If it still applies cleanly and still improves the score — congratulations, you're in. If not, your work is discarded.

This is harsh but correct. No stale evaluations, no optimistic merges that might interact badly. The score you see in the history is always the score on the exact state that was merged.

## What the Current System Does Well (and What We Keep)

1. **Dead-simple protocol.** Modify → submit → get scored → keep/discard. We keep this.
2. **Git as the state machine.** Branches track proposals. Main tracks the best known state. We keep this and lean into it harder.
3. **Human-readable instructions.** `agent_instructions.md` is just Markdown. We keep this.
4. **Fixed evaluation budget.** Per-problem timeouts ensure comparability. We keep this as a configurable constraint.

## What Changes from the Current System

| Aspect | Autoresearch (current) | AutoAnything (target) |
|--------|----------------------|----------------------|
| Domain | ML training only | Any black-box optimization |
| Agents | 1, serial, same machine | N, parallel, distributed |
| Scoring | Agent runs eval and reads score | Evaluator runs privately, agent never sees scoring code |
| Submission | Agent modifies file in place, git commit | Agent pushes branch / opens PR |
| State management | Single branch, git reset on failure | Main = best state, proposals rebase onto main |
| Evaluation | Inline in the agent loop | Separate private service, serial queue |
| Stale proposals | N/A (serial) | Rebase onto current main; conflict = discard |
| History | Untracked results.tsv | Committed leaderboard + history in the challenge repo |

## Concrete Plan: From Here to There

### Phase 1: Local Gitignored Evaluator

**Goal:** Get the simplest possible version working with the existing ML use case.

1. **Restructure this repo as a challenge repo:**
   - Create `problem.yaml` with the ML problem definition (no scoring code)
   - Move `train.py` into `src/`
   - Move `prepare.py` into `context/`
   - Rewrite `program.md` → `agent_instructions.md` with the generic protocol (clone, branch, modify, push)
   - Add empty `history.tsv` and `leaderboard.md`
   - Add `evaluator/` to `.gitignore`

2. **Create the gitignored evaluator:**
   - `evaluator/score.sh` — runs `uv run src/train.py`, extracts `val_bpb`, prints it
   - `evaluator/evaluate_loop.sh` — the serial loop: poll for new branches, rebase, run score.sh, merge or discard, update history.tsv

3. **Test end-to-end:** Manually create a proposal branch with a change to `src/train.py`, run the evaluate loop, see it get scored and either merged or rejected.

**Deliverable:** The ML training optimization works through the new system. An agent (or human) can make changes to `src/train.py`, push a branch, and the evaluator scores it and merges if better. The scoring code is in `evaluator/` which is gitignored.

### Phase 2: GitHub Actions Evaluator

**Goal:** Make evaluation happen automatically via GitHub infrastructure.

1. Write a GitHub Actions workflow that triggers on PRs
2. Use `concurrency` to enforce serial evaluation
3. The workflow fetches scoring code from a private repo (using a secret token)
4. It rebases the PR onto main, runs the evaluation, posts the score as a PR comment
5. If the score improves, it auto-merges and updates the leaderboard
6. If not, it closes the PR with the score

**Deliverable:** Anyone (human or agent) can fork the repo, make changes, open a PR, and get an automated score. The scoring function stays private.

### Phase 3: Multi-Agent at Scale

**Goal:** Support many agents submitting concurrently.

1. **Queue management:** When many branches arrive faster than evaluation can process them, the evaluator needs a fair queue (FIFO, or priority based on agent track record)
2. **Stale branch cleanup:** Branches that have been pending for too long (main has moved too far ahead) could be pre-emptively discarded to save queue time
3. **Agent history feed:** Structured history that agents can read to learn what's been tried — diffs of accepted changes, diffs of rejected changes, summaries
4. **Leaderboard and progress dashboard:** A web page (or GitHub Pages) showing the progress curve over time and attempts

### Phase 4: Intelligence and Diversity

**Goal:** Make the swarm smarter.

1. **Agent strategy templates:** Multiple `agent_instructions.md` variants — conservative, radical, specialist, crossover
2. **Structured feedback:** When the evaluator rejects a proposal, include more than the score — "memory usage 3x", "crashed at step 200 with OOM", etc.
3. **Diminishing returns signal:** The evaluator detects when the last N proposals all failed and publishes a signal to try something radical

### Phase 5: Generalize and Polish

**Goal:** Make it trivial to set up a new optimization challenge.

1. **`autoanything init`** — scaffolds a new challenge repo. Asks: "What files should agents modify? What does 'better' mean? Minimize or maximize? Bounded or unbounded?"
2. **`autoanything evaluator init`** — scaffolds the evaluator config. Asks: "What command scores a proposal? Where should this run?"
3. **Second example problem** — something non-ML to prove generality
4. **Documentation** for both problem owners (how to set up a challenge) and agents (how to participate)

## Key Design Decisions

### 1. Why serial evaluation?

Parallel evaluation (scoring multiple proposals simultaneously) seems like it would be faster. But it wastes work: while proposal A is being evaluated, proposal B might get merged, invalidating A's evaluation. You'd have to rebase and re-evaluate A anyway.

Serial evaluation means: always evaluate against the true latest state. No wasted evaluations, no optimistic merges that might interact badly, no stale scores in the history.

Proposal generation is still massively parallel. The bottleneck is evaluation throughput, which is determined by how long the scoring function takes. This is a property of the problem, not the system.

### 2. Why git branches instead of an API?

- **Universal.** Anything that can `git push` can be an agent. Claude Code, Codex, a human with vim, a shell script.
- **Auditable.** Every proposal is a commit with a diff and a message.
- **No custom infrastructure for agents.** Just git access.
- **Works with GitHub's ecosystem.** PRs, Actions, branch protections — all just work.

### 3. Why is the scoring function private?

- **Prevents gaming.** If agents can read the scoring function, they can overfit to it.
- **Mirrors real optimization.** In real problems, you often *can't* see the scoring function.
- **Enables competition.** Multiple agents compete fairly.

### 4. Why is discard forever?

If a proposal doesn't improve the score, it's gone. No "almost" list, no "try combining these two near-misses." This is ruthlessly simple and it works because:

- Agents can see the history. If an idea was close, an agent can read about it and try a refined version.
- The search space is infinite. Spending time revisiting failed proposals is worse than trying new ideas.
- It keeps the system stateless. The only state is: main (current best) + history (what was tried). No complex bookkeeping.

### 5. What's the minimal viable system?

A challenge repo with `problem.yaml`, one mutable file, `agent_instructions.md`, and a gitignored `evaluator/` directory containing a scoring script and a shell-script loop that polls for branches.

One agent (a Claude Code session), one evaluator (a shell script on a machine with a GPU). That's the whole system.

## What This Isn't

- **Not a hyperparameter search framework.** Optuna, Ray Tune, etc. search over numeric parameter spaces with mathematical strategies. AutoAnything uses LLM reasoning to make *arbitrary code/text changes*. The search space is unbounded and non-numeric.
- **Not CI/CD.** It doesn't deploy anything. It optimizes and logs results.
- **Not a competition platform.** No user accounts, team management, or prize pools. Though someone could build one on top of it.
- **Not a replacement for good metrics.** AutoAnything will ruthlessly optimize whatever number you give it. If that number doesn't capture what you actually care about, you'll get paperclips. The framework is just plumbing — the scoring function is everything.

## First Steps

1. Restructure this repo as a challenge repo (problem.yaml, src/, context/, agent_instructions.md)
2. Add `evaluator/` to .gitignore
3. Write `evaluator/score.sh` for the ML use case
4. Write `evaluator/evaluate_loop.sh` — the serial poll-rebase-score-merge loop
5. Test end-to-end with a manual proposal branch
6. Create a second non-ML example to prove generality
