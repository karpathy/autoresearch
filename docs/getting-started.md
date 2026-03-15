# Getting Started

**AutoAnything** lets you define any optimization problem — a scoring function, some mutable state, and a direction — and then run AI agents in a loop to improve it. The agents propose changes, the framework scores them blindly, and only improvements are kept.

## Try it

```bash
uv tool install autoanything
autoanything try fib
```

That runs a built-in demo: agents optimize a naive Fibonacci implementation for speed, and you'll see a progress chart at the end.

## Create your own problem

Follow the walkthrough in [create-problem.md](create-problem.md). The short version:

```bash
autoanything init my-problem --direction minimize
cd my-problem
# Edit three things:
#   problem.yaml       — describe the problem
#   state/             — put whatever files agents should optimize
#   scoring/score.py   — implement your scoring function
autoanything run -a "claude -p 'read agent_instructions.md and improve the solution'"
```

## Learn more

- [README](../README.md) — full CLI reference and how the system works
- [create-problem.md](create-problem.md) — step-by-step guide to creating your own problem
- [autoanything.md](autoanything.md) — design philosophy and principles

The key idea: agents never see the scoring code, so they can't game it — they just push changes and get back a number.
