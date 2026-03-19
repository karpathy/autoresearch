# CLAUDE.md — opencastor-autoresearch Development Guide

> **Agent context file.** Read this before making any changes.

## What Is opencastor-autoresearch?

AI-assisted research pipeline that continuously improves the OpenCastor codebase and discovers optimal agent harness configurations.

**Repo**: craigm26/opencastor-autoresearch | **Python**: 3.10+

## Ecosystem Versions (as of 2026-03-19)

- **RCAN spec**: v1.6.1 (primary compliance reference) — rcan.dev
- **OpenCastor runtime**: 2026.3.17.13 (yyyy.month.day.iteration)
- **rcan-py**: 0.6.0 | **rcan-ts**: 0.6.0

## Two Research Tracks

### Codebase Autoresearch (run_agent.py)
Nightly LLM-assisted research against the OpenCastor codebase.
- Draft model: local Ollama (gemma3 or similar)
- Reviewer: Gemini 2.0 Flash (Google ADC)
- Tracks: A=tests, B=docs, C=RCAN presets, D=skill evals, E=harness tests, F=trajectory mining
- Results: results.tsv, ~/.config/opencastor/trajectories.db

### Harness Research (harness_research/)
Discovers optimal agent harness YAML configurations.
- Generates N=5 candidates via Gemini
- Evaluates against 30 scenarios across home/industrial/general environments
- Reports winners to opencastor-ops/harness-research/
- Approval flow: Craig adds label `approve-harness` → PR auto-merges in OpenCastor on CI pass

## Auth Pattern

```python
import google.auth
import google.auth.transport.requests
from google import genai

_creds, _project = google.auth.default()
_creds.refresh(google.auth.transport.requests.Request())
client = genai.Client(vertexai=True, project=_project, location="us-central1")
```

## Run

```bash
# Codebase research
python run_agent.py

# Harness research
python harness_research/run.py --dry-run   # preview
python harness_research/run.py             # full run

# Cron (managed by cron.sh)
bash cron.sh
```

## OpenCastor Version Format

`yyyy.month.day.iterationnumber` — e.g. `2026.3.17.13`
RCAN spec v1.6.1 is the primary compliance reference.
