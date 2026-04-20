---
name: report-generator
description: Generate professional HTML research reports from experiment data.
---

## When to use
After autoresearch completes (synthesis stage), or on demand via `scaffold.py report`.

## Steps
1. Read workflow.yaml + results.tsv + musings.md + git log
2. Assemble ReportData
3. Render Jinja2 templates with Chart.js
4. Write to outputs/report/

## Constraints
- Read-only on experiment data -- never modify results
- Design tokens in design-tokens.json -- do not inline styles

## Triggers
report, synthesize, summarize results

## Output
Slide-like HTML: Situation > Challenge > Experiments > Findings > Impact
