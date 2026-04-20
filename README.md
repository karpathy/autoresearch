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
├── .github/skills/           # Skill definitions (SKILL.md per skill)
├── hooks/                    # Deterministic lifecycle scripts
├── tests/                    # Benchmark suite (8 synthetic tasks)
├── workflows/                # Concrete experiment configurations
│   ├── _template/
│   ├── examples/ml-training/
│   └── exec-summarizer/
└── scaffold.py               # CLI for onboarding and scaffolding
```

Each sub-directory has its own `AGENTS.md` with local context. The root `AGENTS.md` is the discovery index. Skills live in `.github/skills/` (one `SKILL.md` per skill), including `autonomous-iteration` for the experiment loop and `map-harness` for structure hygiene.

## The autoresearch loop

An AI agent modifies a target, measures a metric, keeps improvements, discards regressions, and repeats: edit, commit, run, extract metric, check gates, keep or discard, log to `results.tsv`, repeat. See `.github/skills/autonomous-iteration/SKILL.md` for the full protocol.

The agent cycles through three modes -- the Artisan's Triad -- to avoid local optima:

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

## Benchmarks

The benchmark suite (`tests/benchmark.py`) runs 8 synthetic tasks against a fixed model baseline (Sonnet 4.6, highest reasoning effort). Tasks cover constraint adherence, context retrieval, data flow, index navigation, instruction following, memory retrieval, quality gates, and tool orchestration. Results are evaluated with a Pareto ratchet -- optimizing lower-tier metrics never regresses higher-tier ones.

Run benchmarks: `uv run python tests/benchmark.py --all`

## Observability

Benchmark runs emit telemetry to `tests/results/telemetry.jsonl` (token counts, latencies, error taxonomy). Errors are classified as: constraint violation, tool failure, timeout, quality gate fail, or unknown.

## Conventions

- AGENTS.md and SKILL.md files stay under 1000 characters -- context window budget
- Pipe-compressed index format for instruction files
- Constraints include rationale: `No X -- reason`
- TSV for structured logs, gitignored
- No emoji, no em dashes (use `--`)

## Reference workflows

| Workflow | Domain | Target | Metric |
|----------|--------|--------|--------|
| `examples/ml-training/` | ML training | `train.py` | `val_bpb` (lower) |
| `exec-summarizer/` | Prompt engineering | `prompt.txt` | `quality_score` (higher) |
| `harness-optimize/` | Meta-optimization | Harness config files | `harness_score` (higher) |
| `report-design/` | Report UX | HTML templates + CSS | `quality_score` (higher) |

## Report generation

After completing an autoresearch run, generate a shareable HTML report:

```bash
python scaffold.py report exec-summarizer        # generate report
python scaffold.py report exec-summarizer --open  # generate and open in browser
```

Reports follow a narrative arc -- Situation > Challenge > Experiments > Findings > Impact -- with interactive Chart.js charts, experiment timelines, and full provenance. Output goes to `<workflow>/outputs/report/index.html`.

## License

MIT
