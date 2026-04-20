# design-optimize
Use ui-ux-pro-max design intelligence to optimize report templates.

## Files
- `workflow.yaml` | targets, metric, run command
- `program.md` | research strategy | human-written
- `evaluate.py` | quality evaluator | fixed

## Design Intelligence
`python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "<q>" --domain <d>`
Domains: style, chart, ux, typography, color. Use `--design-system` for full system.

## Run
1. Read `program.md` for strategy
2. Query ui-ux-pro-max for guidance
3. Follow `autonomous-iteration` skill
4. Apply changes to CSS/tokens, measure, keep/discard

## Constraints
- Never modify fixed files -- evaluation integrity
- Evidence-based changes only -- query ui-ux-pro-max first
- No emoji as icons, cursor:pointer on interactive elements
