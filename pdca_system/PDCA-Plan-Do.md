# PD - Seed Planning And Generation

## Responsibility
Extract exactly one testable improvement hypothesis from the seed prompt,
generate the first implementation in a candidate worktree, and hand the result
to CA through the runner.

## Workspace and paths
**CWD = seed worktree.** Read and edit only inside it; use relative paths only.

## arXiv search (CLI)

Run from **project root** using the Python executable provided in your task prompt when available (for example, `{python_exe} pdca_system/run_arxiv.py ...`). If no Python executable is provided, use your project's normal Python runner. Arxiv is a project dependency.

### Search (CLI script)

From project root, use the script in this component with the provided Python executable:

```bash
{python_exe} pdca_system/run_arxiv.py --query "machine learning" --max-results 5
{python_exe} pdca_system/run_arxiv.py --id 1605.08386v1 --output json
```

If your task prompt does not provide `{python_exe}`, substitute your project's normal Python launcher for the examples above and ensure the project environment is active or `PYTHONPATH` includes the project root.

**CLI arguments:** `--query` / `-q`, `--id` (one or more arXiv IDs; overrides query), `--max-results` / `-n`, `--sort-by` (relevance | submittedDate | lastUpdatedDate), `--sort-order` (ascending | descending), `--output` / `-o` (text | json), `--download-dir`, `--verbose` / `-v`.

### Hypothesis from results
1. Read abstracts; pick one concrete change (not just a concept).
2. Map to component: `model`, `optimizer`, or `trainer`.
3. State expected benefit; reduce to one isolated, evaluable improvement.

## Input
- **results.tsv** in cwd (if present) ? read first to avoid duplicating tried/discarded ideas.
- arXiv via arxiv-search; past failures in `queue/done/`; manual seed files.

## One-Improvement Rule

One seed = one hypothesis = one causal change. Do not bundle ideas. If the prompt has several options, pick the single best for this run. Prefer the smallest coherent change that tests the hypothesis.

**Good:** one optimizer schedule change; one architectural block; one training heuristic. **Bad:** model + optimizer + batch together; multiple paper ideas in one seed; "cleanup + new feature" in one candidate.

## Output Format
Write the summary JSON to the file named `autoresearch_summary.json` in your current working directory (cwd root). Do not print it to stdout or stderr. Use this shape:

```json
{"idea":"short title","target_component":"model | optimizer | trainer","description":"change details, hypothesis, expected benefit","source_refs":["arXiv:<id>"],"commit_sha":"git sha","completed_at":"YYYY-MM-DD HH:MM:SS"}
```

## Runner / worktree
Before each P run, the runner syncs the seed worktree with its baseline branch (merge baseline into seed) so P always starts from the latest baseline. The CA stage receives the **Python executable** from the daemon (the one running the daemon); the canonical script to run (e.g. train.py) is defined in protocol and PDCA-Check-Action.md.

## Steps
1. Read `results.tsv` if present.
2. Refine prompt ? one concrete idea ? one isolated improvement; name target component.
3. Implement in worktree (from baseline); commit on seed branch.
4. Write summary JSON to `autoresearch_summary.json` in cwd (runner records commit). Description must be enough for CA.

## Constraints
- One component, one improvement per seed. Smallest viable implementation.
- No exploratory cleanup or opportunistic refactors unless required for the one change.
- Commit on seed branch; runner does not merge. **PD must never merge;** only CA triggers merge into baseline.
