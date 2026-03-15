# AutoAnything: The Idea

## Origin

This project started from Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — a single AI agent in a loop, optimizing a GPT training script against validation bits-per-byte on one GPU. The agent would modify `train.py`, run training for five minutes, check the score, and keep the change if it improved. Simple evolutionary search powered by an LLM instead of random mutations.

The insight that made autoresearch compelling wasn't the ML part. It was the loop: propose a change, score it against an objective function, keep it if it's better, throw it away if it's not. That loop works for anything with a number you can measure. The code changes are the mutations, the scoring function is the fitness landscape, and the LLM is a mutation operator that actually understands what it's changing.

We wanted to make that loop applicable to anything.

## The generalization

AutoAnything takes the autoresearch concept and removes everything specific to ML training:

- The mutable state doesn't have to be a training script. It can be any file — a prompt template, a solver configuration, a game AI, infrastructure code.
- The scoring function doesn't have to be validation loss. It can be any program that outputs a number — a benchmark, a test suite, an LLM judge, a simulator.
- The agent doesn't have to be a single Claude session on the same machine. It can be any number of agents, anywhere, submitting concurrently.

The common pattern: there's some mutable state (files an agent can change), an evaluation function that produces a number, and a direction (lower is better, or higher is better). Everything else is just plumbing.

## Principles

### Black-box scoring

Agents never see the scoring code. This is the single most important design decision.

If an optimizer can see the evaluation function, it will overfit to it — exploiting quirks in the metric, hardcoding known-good outputs, gaming the test set. This is the same reason Kaggle keeps the test set private and the same reason you don't let students write the exam.

The separation is structural, not conventional. The scoring code is never committed to the problem repo. It exists only on the evaluation machine. Agents know *what* metric they're optimizing (from `problem.yaml`) and *what scores others have achieved* (from `leaderboard.md`), but they have zero information about *how* the score is computed. They push a branch, and a number comes back.

The scoring function is everything. AutoAnything is just plumbing.

### Only forward, only better

When a proposal is scored, one of two things happens: it beats the current best and gets merged into main, or it doesn't and it's discarded forever. No second chances, no "close enough," no combining near-misses.

Every agent works from the current best state. Every accepted change makes the state strictly better according to the metric. The main branch only moves forward — a ratchet that clicks in one direction. The history of accepted changes is a monotonically improving sequence.

This is ruthlessly simple and it works because the search space is infinite. Revisiting failed proposals is worse than trying new ideas. And agents can see the leaderboard — if an idea was close, an agent can read about it and try a refined version. The system's memory is the git history and the leaderboard, not a queue of maybes.

### Git as the protocol

Submissions are git branches or pull requests. This means:

- **Anything that can `git push` can be an agent.** Claude Code, Codex, Cursor, a human with vim, a shell script. No SDK, no registration, no custom API.
- **Every proposal is auditable.** A commit with a diff and a message explaining the approach.
- **The existing ecosystem works.** GitHub PRs, Actions, branch protections, webhooks — all just work. No custom infrastructure for the submission side.
- **State management is trivial.** Main tracks the best known state. Proposal branches are the pending mutations. Git merge is the mechanism for accepting improvements.

### Serial evaluation, parallel proposals

Evaluation is serial — one proposal scored at a time. This is counterintuitive but correct.

The question being answered is always: "does this proposal beat the current best?" Since we evaluate one at a time, the incumbent never changes during an evaluation. The comparison is always clean. No race conditions, no stale baselines, no wasted work.

If you scored proposals in parallel, you'd constantly be comparing against stale state. Proposal A might beat the incumbent, get merged, and now proposal B's score is meaningless — it was measured against a baseline that no longer exists. You'd have to re-evaluate.

Proposal *generation* is massively parallel. Hundreds of agents can be thinking, coding, and pushing branches simultaneously. The funnel narrows to a single thread at evaluation time. The evaluator is always the bottleneck, and evaluation throughput determines the rate of progress. If scoring takes five minutes, you get twelve evaluations per hour regardless of how many agents are working. But those twelve are drawn from a much larger pool of ideas.

### A population of agents

The original autoresearch was one agent on one machine. AutoAnything is designed for many agents working concurrently on the same problem — potentially different models, different prompting strategies, different machines, different continents.

Each agent clones the same repo, reads the same problem definition and leaderboard, and pushes branches. They don't coordinate with each other directly. Coordination happens through the shared state: the main branch (current best) and the leaderboard (what's been tried). An agent that reads the leaderboard and sees that increasing the learning rate worked might try increasing it further. An agent that sees a crash from making the model wider might try a more conservative width increase. The leaderboard is the collective memory.

This is closer to evolutionary search than to gradient descent. The population generates diverse mutations. The scoring function selects for fitness. The main branch is the fittest individual. Over time, the population converges on good solutions through selection pressure alone.

## What you can optimize

Anything with a scoring function:

- **A prompt template** — scored by LLM-as-judge accuracy on a test set
- **A web app's performance** — scored by Lighthouse score
- **A compiler optimization pass** — scored by benchmark runtime
- **A trading strategy** — scored by backtested Sharpe ratio
- **Infrastructure config** — scored by p99 latency under load
- **A game AI** — scored by win rate against a baseline opponent
- **An ML training script** — scored by validation loss (the original use case)

These are the obvious cases — problems where there's already a natural number attached to the output. But the interesting frontier is everything that doesn't have one yet.

### Optimizing subjective things

The scoring function doesn't have to be deterministic or mechanical. It just has to return a number. And now that LLMs can act as judges, that opens up a much larger class of problems: things that used to require human taste.

Consider optimizing an essay. There's no compiler output, no benchmark, no loss curve. But you can define what you care about — clarity of argument, strength of evidence, prose quality, originality, tone — and have an LLM score each dimension on a rubric. The scoring function becomes: run the essay through an LLM judge multiple times across each dimension, collect the scores, apply weights that reflect your priorities, and collapse it all into a single number. The agent never sees the rubric, the weights, or the judge's reasoning. It just pushes a branch and gets back a score.

This works for anything you can articulate values about:

- **An essay or blog post** — scored across argument structure, evidence quality, readability, and originality, weighted toward whatever you care about most
- **A short story** — scored on narrative tension, character voice, prose style, and thematic coherence
- **A product landing page** — scored on persuasiveness, clarity, emotional resonance, and information hierarchy
- **An API design** — scored on consistency, discoverability, naming conventions, and error handling philosophy
- **A legal brief** — scored on argument strength, precedent usage, counterargument anticipation, and conciseness

The scoring function for these is more complex than a benchmark — it's a pipeline. Read the artifact, construct dimension-specific prompts, call an LLM judge (potentially multiple times for stability), parse the scores, apply the weight vector, and output the weighted sum. But from the framework's perspective, it's still just `score.sh` returning a JSON number. The machinery inside the black box is irrelevant.

What makes this powerful is that the weights encode values the agents can't see. You might weight originality at 3x and conciseness at 0.5x, and the agents will converge on bold, expansive writing without ever being told to. Change the weights and the same swarm of agents will produce something entirely different. The values live in the scoring function, not in the agents — which is exactly where they should be, because it means you can change what "better" means without rewriting any agent instructions.

The ceiling here is the quality of the LLM judge and the quality of your rubric. A vague rubric produces vague optimization — the agent finds whatever makes the judge say "8/10" without actually being good. A precise rubric with well-calibrated dimensions produces real improvement. The same Goodhart problem applies, but now the measure you're targeting is itself an LLM's judgment, which can be surprisingly nuanced if you give it detailed criteria to evaluate against.

The quality of the scoring function is the ceiling on the quality of the results. A bad metric optimized ruthlessly produces paperclips — a system that scores well but misses the point. Whatever number you pick, a swarm of agents will exploit every degree of freedom it leaves open.

## How it works in practice

A problem is a git repo with a fixed structure. Agents see everything except the scoring code:

```
my-problem/
├── problem.yaml            # What to optimize, constraints, metric name + direction
├── agent_instructions.md   # Protocol for agents
├── state/                  # Mutable files agents edit
├── context/                # Read-only background
├── leaderboard.md          # Auto-updated scoreboard
│
├── scoring/                # GITIGNORED — only on the evaluation machine
│   └── score.sh            # The private scoring function
└── .autoanything/          # GITIGNORED — evaluator state
    └── history.db          # SQLite evaluation history
```

The evaluator (a separate process the problem owner runs) watches for new branches or PRs, scores them, and merges improvements. It can run as a polling loop (`autoanything evaluate`) or as a webhook server that reacts to GitHub PR events (`autoanything serve`).

The evaluation loop:

1. Pick the next proposal from the queue
2. Check out the proposal branch
3. Run the scoring function
4. If the score beats the incumbent — merge to main, update the leaderboard, push
5. If it doesn't — discard, record the attempt
6. If it crashed — record as crash, move on

After every evaluation, the leaderboard is updated so agents can see what was tried and what worked. The SQLite database records everything — accepted, rejected, and crashed proposals — as a complete history of the search.

## The Goodhart problem

"When a measure becomes a target, it ceases to be a good measure."

AutoAnything makes Goodhart's Law concrete. You must pick a number, and agents will optimize it relentlessly. If the metric doesn't capture what you care about, you'll get a system that scores well but misses the point.

This is a feature, not a bug. It forces you to think hard about what "better" means before you start. And if your metric is good, relentless optimization is exactly what you want.
