# Executive Summarizer

## Overview

Prompt optimization workflow. Iterates a system prompt for summarizing news articles for executive professional audiences, scored against a 4-dimension quality rubric using GPT-5.4-mini via the Copilot SDK.

## Workflow Structure

```
exec-summarizer/
├── AGENTS.md          -- this file
├── program.md         -- research program and constraints
├── workflow.yaml      -- manifest (targets, metric, run command)
├── prompt.txt         -- THE TARGET (agent edits this system prompt)
├── evaluate.py        -- evaluation harness (read-only)
├── articles.json      -- test articles (read-only)
├── results/
│   ├── results.tsv    -- experiment log (untracked)
│   └── musings.md     -- reflections (untracked)
└── outputs/
```

## Key Files

| File | Role | Who Edits |
|------|------|-----------|
| workflow.yaml | Declares targets, metric, run command | Human (once) |
| program.md | Research strategy and constraints | Human (iteratively) |
| prompt.txt | System prompt being optimized | Agent (each experiment) |
| evaluate.py | Scoring harness (Copilot SDK) | Nobody (read-only) |
| articles.json | Test dataset | Nobody (read-only) |

## Scoring Rubric

Each summary is scored 1-10 across 4 dimensions:
1. **Conciseness**: 2-3 sentences, no filler
2. **Executive Relevance**: Impact, implications, "so what?" for decision-makers
3. **Source Provenance**: Data and quotes attributed with URI links
4. **ECQ Alignment**: Framing connects to Executive Core Qualifications

## Constraints

- Only modify prompt.txt
- Do not modify evaluate.py or articles.json
- Each evaluation takes ~30-60 seconds (API calls)
- System prompt should stay under 500 words

## Running

```bash
python evaluate.py                    # run evaluation
python evaluate.py > run.log 2>&1     # run with log capture
grep '^quality_score:' run.log        # extract metric
```
