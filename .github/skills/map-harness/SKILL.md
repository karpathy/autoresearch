---
name: map-harness
description: Verify structure indexes in AGENTS.md after changes.
---

## When to use
After file create/delete/rename or structure check.

## Steps
1. Scan for all AGENTS.md files
2. Verify referenced paths exist
3. Check parent indexes and 1000-char ceiling
4. Report pass/fail

## Constraints
- Pipe-compressed format in AGENTS.md -- context budget
- 1000-char ceiling per AGENTS.md or SKILL.md

## Examples
[example/map.py](example/map.py)
