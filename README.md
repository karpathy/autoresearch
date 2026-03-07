# opencastor-autoresearch

Autonomous overnight improvement agent for [OpenCastor](https://github.com/craigm26/OpenCastor).

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted for
software improvement (tests, docs, presets) instead of ML training.

## How it works

- **Draft model**: gemma3:1b via Ollama (on-device, free)
- **Review model**: Gemini 2.0 Flash via Google ADC (no API key needed)
- **Loop**: propose → review → apply → test → keep/revert → log
- **Schedule**: 12am–6am nightly via cron
- **Cost**: free (ADC quota)

## Tracks (rotates by day)

| Day | Track | What it does |
|-----|-------|-------------|
| Mon/Thu | A | Write new pytest tests |
| Tue/Fri | B | Add missing docstrings |
| Wed/Sat | C | Generate RCAN presets |
| Sun | A | Tests (default) |

## Setup

1. `cp .env.example .env` — no API key needed, uses Google ADC
2. Ensure ADC is configured: `gcloud auth application-default login`
3. `pip install google-genai google-auth ollama gitpython`
4. Ensure Ollama is running: `ollama pull gemma3:1b`
5. Add cron job: `crontab -e` and add:
   ```
   0 0 * * * /home/craigm26/opencastor-autoresearch/cron.sh >> /home/craigm26/autoresearch.log 2>&1
   ```

## Manual run

```bash
source .env
export OPENCASTOR_REPO_PATH=/home/craigm26/OpenCastor
export TODAY_TRACK=A
python3 run_agent.py
```

## Results

Each night produces a `results.tsv` with columns:
`commit | metric_before | metric_after | delta | status | description`

A GitHub PR is automatically opened in OpenCastor at 6am with the experiment summary.

## Baseline metrics (2026-03-07)

- Tests: 4323
- Missing docstrings: 1197
- RCAN presets: 18
