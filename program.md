# autoresearch program

This repo now runs autonomous research jobs through the local Codex CLI.

## Mission

Take the research specification that follows and return the requested deliverable with no wrapper text. The final response should be the finished artifact, ready to save as a report.

## Working rules

1. Use web search whenever the task depends on current, source-specific, or niche facts.
2. Follow the output contract exactly. Match headings, tables, section ordering, and any "Not confirmed" rules from the spec.
3. Prefer primary and official sources first, then major manufacturer, rental, trade, or standards sources, then Wikimedia Commons when a direct image file is needed.
4. Do not invent URLs, especially image URLs. If a direct image asset cannot be confirmed after reasonable searching, write `Not confirmed` instead of guessing.
5. Treat meaningfully distinct sub-types or visibly distinct variants as separate rows when the spec asks for exhaustive coverage.
6. Keep aliases short and useful. Avoid repeating the main equipment name in slightly different wording unless that variant is commonly used.
7. Use concise descriptions that explain what the item is and where it is typically used.
8. If the spec requires an `Unlisted / Miscellaneous` section, include it and only place items there that fit the phase but not an existing listed category.
9. Return only the final deliverable. Do not add a preamble, status note, or explanation of your research process.
10. Do not spend time exploring unrelated repo files or old `results/` artifacts unless the current spec explicitly asks you to recover prior output. Treat past run folders as scratch, not sources.
11. If the spec gives you a fixed backlog of rows to revisit, process every listed row once before stopping. Do not treat a partially improved table as complete.
12. If the spec asks for gap-filling or confirmation, success means every requested row was checked and either updated with a confirmed value or intentionally left as `Not confirmed`.

## Quality bar

- Completeness matters more than elegance for inventory-style research.
- Merge duplicates caused by aliasing, but keep genuinely different machine classes separate.
- Prefer recognizable, commonly photographed variants when choosing image URLs.
- If the spec scopes the task to a single phase, stay inside that phase.
- After reading the provided spec, move to external research quickly instead of re-reading the workspace.
