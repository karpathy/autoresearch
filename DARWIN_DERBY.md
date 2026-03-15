# Darwin Derby

## The Name

**Darwin Derby** — a framework for autonomous optimization where agents compete and the best solution survives.

The CLI is `derby`.

```bash
pip install darwin-derby

derby init my-problem
derby score
derby evaluate
derby serve
```

The name is a reference to the Vulfpeck song "Darwin Derby." It's an easter egg for people who get it, invisible to people who don't.

## Why Rename

"AutoAnything" describes a capability — automate any optimization. It's broad, generic, and says nothing about how the system works. It could be an RPA tool, a CI pipeline, a no-code platform. The name creates no mental model.

"Darwin Derby" describes a mechanism:

- **Darwin** — natural selection. Keep the fittest, discard the rest. This is literally what the evaluator does: score a proposal, merge it if it's better, reject it if it's not. Each generation builds on the last. The population improves over time through selective pressure, not central planning.

- **Derby** — a race. Multiple contenders compete on the same course under the same rules. No one sees the judge's scorecard in advance. You enter, you run, you get a result. The best result wins. This maps directly to the evaluation loop: agents all compete against the same hidden scoring function on the same problem.

Together, the name tells you what this thing does before you read a single line of documentation. Agents compete (derby). The best survive (Darwin). The system evolves toward an optimum.

## What the Name Gets Right

### It matches the actual mechanism

The system is hill-climbing with parallel exploration. Multiple agents propose changes to a shared state. An evaluator scores each proposal against a hidden objective. Only improvements are merged into the mainline. Over time, the solution evolves toward the optimum — not because anyone planned the path, but because selection pressure rewards progress.

This is Darwinian. The agents are the population. The proposals are mutations. The scoring function is the environment. The merge-or-reject decision is natural selection. The leaderboard is the fossil record.

A derby is the right competitive metaphor because:

- All contenders face the same challenge (same problem, same scoring function)
- Entry is open (anything that can `git push` can compete)
- Judging is blind (agents never see the scoring code)
- There's a definitive result (the score)
- It's ongoing — you can always enter another heat

### It doesn't over-promise

"AutoAnything" implies the framework does the optimization. It doesn't. The framework runs the competition — the agents do the optimization. The framework is the racetrack, the judges' booth, and the scoreboard. It doesn't run the race.

"Darwin Derby" correctly positions the framework as the arena, not the optimizer. It hosts the competition. It enforces the rules. It records the results. The intelligence comes from the competitors.

### It avoids the platform trap

Earlier framings drew an analogy to Kaggle. The analogy is illustrative but misleading. Kaggle is a centralized platform: accounts, teams, prize pools, web UI, hosted compute. Darwin Derby is none of that. It's a self-hosted CLI tool. You install it, point it at a problem directory, and run it on your own machine.

Names that echo Kaggle (including earlier candidates like "Monte Gaggle") set expectations the project can't meet. People hear "Kaggle" and expect a website. Darwin Derby doesn't sound like a platform. It sounds like a thing that happens — which is what it is. A derby is an event, not a venue.

### It's short where it matters

The CLI is `derby`. Five characters. Compare:

```bash
# Before
autoanything evaluate --baseline-only
autoanything score
autoanything serve

# After
derby evaluate --baseline-only
derby score
derby serve
```

`derby` is easy to type, easy to remember, and easy to alias. It doesn't collide with common Unix commands or well-known packages.

### It's fun without being silly

The name has energy. It's not corporate ("OptimizationFramework"), not generic ("AutoAnything"), not trying too hard ("AgentArena"). It's a real phrase that happens to describe exactly what the software does. The Vulfpeck reference adds personality for people who recognize it. For everyone else, it's just a good name.

## The Metaphor, Extended

The Darwinian framing maps cleanly onto every part of the system:

| Biology | Darwin Derby |
|---------|-------------|
| Environment | The scoring function (`score.sh`) |
| Organism | A proposal branch |
| Mutation | The diff an agent commits |
| Fitness | The score |
| Natural selection | Merge if better, reject if not |
| Fossil record | `history.db`, `leaderboard.md` |
| Generation | One evaluation cycle |
| Population | All agents working on the problem |
| Extinction | A proposal that scores worse and is discarded |
| Adaptation | The mainline solution improving over time |

This isn't a forced metaphor — it's a structural correspondence. The system literally implements an evolutionary algorithm, with agents as the mutation operators and the evaluator as the selection mechanism.

The key difference from textbook evolutionary algorithms: the mutations aren't random. Agents read the current state, the leaderboard, and the problem description. They make informed proposals. This is Lamarckian evolution — directed variation with Darwinian selection. That's what makes it powerful: you get the robustness of selection pressure with the efficiency of intelligent search.

## What Changes

The rename is cosmetic. The architecture, the protocol, the evaluation loop, the trust model — none of that changes. Specifically:

- The package name becomes `darwin-derby` on PyPI
- The importable module becomes `derby` (i.e., `from derby import ...`)
- The CLI entry point becomes `derby`
- The config directory becomes `.derby/` (instead of `.autoanything/`)
- The project repo is renamed accordingly
- All documentation, README, CLAUDE.md updated to reflect the new name

The problem structure, git protocol, scoring interface, and evaluator behavior are unchanged. A problem that works with the current system works with the renamed one.

## The Tagline

> **Darwin Derby** — agents compete, the best solution survives.
