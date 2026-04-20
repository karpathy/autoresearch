# autoresearch

Canonical reference harness for agent-assisted development. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), evolved into a framework for optimizing how AI agents work in any codebase. This repo is both the framework and its own reference implementation.

## What is a harness?

A harness is the code around a fixed model that decides what to store, retrieve, and present. It encompasses agent instructions, lifecycle hooks, skills, guardrails, quality gates, and structural conventions -- everything that shapes how an agent behaves in a workspace. Improving the harness improves every task the agent runs, without changing the model itself.

## Quick start

```bash
git clone https://github.com/BrettReifs/autoresearch && cd autoresearch
uv sync

# Generate a harness.yaml via guided conversation
python scaffold.py onboard

# Generate workspace files from a harness config
python scaffold.py generate harness.yaml

# Create a new research workflow
python scaffold.py workflow my-experiment
```

## Repo structure

```
autoresearch/
├── AGENTS.md                 # Root manifest/router (start here)
├── harness.yaml              # Machine-readable harness config
├── agents/                   # Agent behavioral contracts
├── skills/                   # Reusable agent capabilities
├── hooks/                    # Deterministic lifecycle scripts
├── tests/                    # Benchmark suite for harness quality
├── workflows/                # Concrete experiment configurations
│   ├── _template/
│   ├── examples/ml-training/
│   └── exec-summarizer/
└── scaffold.py               # CLI for onboarding and scaffolding
```

Each sub-directory has its own `AGENTS.md` with local context. The root `AGENTS.md` is the discovery index.

## The autoresearch loop

An AI agent modifies a target, measures a metric, keeps improvements, discards regressions, and repeats. The protocol:

1. Edit target file(s)
2. Commit with descriptive message
3. Run experiment, capture output
4. Extract metric
5. Check quality gates (if configured)
6. Keep (improved + gates passed) or discard (revert)
7. Log to `results.tsv` and `musings.md`
8. Repeat

The agent cycles through three modes -- the Artisan's Triad -- to explore the design space without getting stuck:

| Mode | Action | Example |
|------|--------|---------|
| Additive | Introduce new elements | New features, logging, optimizations |
| Reductive | Remove or simplify | Delete dead code, reduce parameters |
| Reformative | Reshape without adding/removing | Refactor structure, change ratios |

## Metrics (Pareto ratchet)

Harness quality is measured across five metrics organized into tiers. Optimizing a lower tier never regresses a higher one.

| Metric | Direction | Tier | Role |
|--------|-----------|------|------|
| Task success rate | Higher | T1 | Gate -- must not regress |
| Quality-gate pass rate | Higher | T1 | Gate -- must not regress |
| Rework rate | Lower | T2 | Constraint -- subject to T1 floors |
| Token consumption | Lower | T3 | Optimization -- subject to T1+T2 |
| Time per turn | Lower | T3 | Optimization -- subject to T1+T2 |

## Reference workflows

| Workflow | Domain | Target | Metric |
|----------|--------|--------|--------|
| `examples/ml-training/` | ML training | `train.py` | `val_bpb` (lower) |
| `exec-summarizer/` | Prompt engineering | `prompt.txt` | `quality_score` (higher) |

## License

MIT
