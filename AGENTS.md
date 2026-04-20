# autoresearch
Reference harness for agent-assisted development.

## Structure
- `agents/` | agent definitions | `agents/AGENTS.md`
- `skills/` | reusable capabilities | `skills/AGENTS.md`
- `hooks/` | lifecycle enforcement | `hooks/AGENTS.md`
- `tests/` | benchmark suite | `tests/AGENTS.md`
- `workflows/` | experiment configs | per-workflow AGENTS.md
- `.github/` | Copilot integration | pointers only, never duplicated definitions

## Constraints
- No secrets in code or commits -- credential leaks are irrecoverable
- No `git push --force` -- history rewrites break collaboration
- No modifying fixed files in workflow.yaml -- evaluation integrity
- Conventional commits with Co-authored-by trailer -- attribution
- `python scaffold.py <name>` for new workflows -- ensures template consistency
