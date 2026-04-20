# autoresearch
Reference harness for agent-assisted development.

## Structure
- `.github/agents/` | agent definitions | `research-runner.agent.md`
- `.github/skills/` | capabilities | autonomous-iteration, map-harness, report-generator
- `.github/hooks/` | lifecycle enforcement | `.github/hooks/AGENTS.md`
- `.github/tests/` | benchmark suite | `.github/tests/AGENTS.md`
- `workflows/` | experiment configs | per-workflow AGENTS.md

## Constraints
- No secrets in code or commits -- credential leaks are irrecoverable
- No `git push --force` -- history rewrites break collaboration
- No modifying fixed files in workflow.yaml -- evaluation integrity
- Conventional commits with Co-authored-by trailer -- attribution
- `python scaffold.py <name>` for new workflows -- ensures template consistency
