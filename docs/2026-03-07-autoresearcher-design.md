# OpenCastor AutoResearcher Design

**Date**: 2026-03-07
**Status**: Approved

## Overview

Adapt [karpathy/autoresearch](https://github.com/karpathy/autoresearch) to run autonomous overnight improvement cycles on the OpenCastor codebase. Instead of ML training experiments, the agent runs software improvement experiments: writing tests, improving docs, and generating RCAN presets. Each experiment is measured against a metric, kept if it improves things, reverted if not.

## Analogy to autoresearch

| autoresearch | OpenCastor variant |
|---|---|
| Edits `train.py` | Edits files in `castor/`, `docs/`, `config/presets/` |
| Runs 5-min GPU training | Runs `pytest` + `ruff` (~2–5 min per experiment) |
| Metric: `val_bpb` (lower is better) | Track-specific metric (see below) |
| H100 GPU required | Runs on Raspberry Pi 5 |

## Research Tracks

The agent rotates through three tracks, one per nightly run:

### Track A — Tests
- **What**: Read source files, identify untested code paths, write new pytest tests
- **Files modified**: `tests/` (new test files or appending to existing)
- **Metric**: `pytest --co -q | wc -l` — test count must increase; all must pass
- **Keep if**: count increases AND zero regressions
- **Discard if**: any existing test breaks or test is trivially wrong

### Track B — Docs
- **What**: Scan for missing docstrings, stale docs, undocumented CLI flags; rewrite them
- **Files modified**: `castor/` source files (docstrings), `docs/` markdown
- **Metric**: `ruff check castor/ | wc -l` — warning count must not increase; plus docstring coverage
- **Keep if**: ruff clean, docstring count increases
- **Discard if**: ruff introduces new violations

### Track C — Presets
- **What**: Generate new RCAN config presets for hardware combinations not yet in `config/presets/`
- **Files modified**: `config/presets/<name>.rcan.yaml`
- **Metric**: `castor validate --config <preset>` passes
- **Keep if**: validation passes, no duplicate
- **Discard if**: validation fails after 2 fix attempts

## LLM Architecture (Hybrid)

```
gemma3:1b (Ollama, on-device, free)
    └─ proposes code change / new file content

Claude Haiku 4.5 (cloud, ~$0.001/experiment)
    └─ reviews: "Does this improve quality without breaking conventions?"
    └─ Pass → apply change, run experiment
    └─ Fail → discard, log reason, try next idea
```

On-device model handles high-volume drafting (free). Cloud model gates quality (cheap).

## Cost Estimate

| Item | Rate | Per Night |
|---|---|---|
| gemma3:1b drafts | free (Ollama) | $0.00 |
| Claude Haiku review (~2K in + 500 out per experiment) | $0.25/MTok in, $1.25/MTok out | ~$0.07 |
| 12 experiments/hr × 6 hrs = ~72 experiments | — | **~$0.07–0.15** |

## Nightly Cron Schedule

```
00:00 — start agent on branch autoresearch/YYYY-MM-DD
        track rotates: Mon/Thu=A (tests), Tue/Fri=B (docs), Wed/Sat=C (presets), Sun=all
06:00 — agent stops, commits results.tsv
        opens GitHub PR with experiment summary
```

## Repository Structure

Fork `karpathy/autoresearch` → `craigm26/opencastor-autoresearch`

Key files adapted:
- `program.md` — OpenCastor-specific agent instructions (replaces ML training instructions)
- `results.tsv` — experiment log (same format as autoresearch)
- `run_agent.py` — orchestrator: Ollama draft → Haiku review → apply → test → keep/revert
- `.env` — `ANTHROPIC_API_KEY`, `OPENCASTOR_REPO_PATH`
- `cron.sh` — wrapper: git branch, run agent, open PR

## program.md Summary

The `program.md` for OpenCastor will:
1. Set context: what OpenCastor is, its test suite, code conventions
2. Define the active track for this run
3. List files in scope
4. Define the metric command to run
5. Define keep/revert criteria
6. Set the loop: propose → review → apply → measure → keep/revert → log → repeat
7. Set NEVER STOP rule (same as autoresearch)

## Output Artifacts

Each nightly run produces:
- `results.tsv` — tab-separated: `commit | metric_before | metric_after | delta | status | description`
- `run.log` — full experiment output
- GitHub PR — auto-opened at 6am with summary table

## Safety Constraints

- Agent only modifies `tests/`, `docs/`, `config/presets/` and docstrings in `castor/`
- Agent NEVER modifies `castor/api.py`, `castor/safety.py`, or any auth/security files
- All changes go on a dated branch; nothing merges to main automatically
- `pytest` must pass 100% before any commit is kept
