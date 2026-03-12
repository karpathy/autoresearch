# P - Seed Planning And Generation

## Responsibility
Extract exactly one testable improvement hypothesis from the seed prompt,
generate the first implementation in a candidate worktree, and hand the result
to DCA through the runner.

## Workspace and paths
**CWD = seed worktree.** Read and edit only inside it; use relative paths only.

## arXiv search (CLI)

Run from repo root with uv (e.g. `uv run python component_system/run_arxiv.py ...`); arxiv is already a project dependency.

### Search (CLI script)

From repo root, use the script in this component:

```bash
uv run python component_system/run_arxiv.py --query "machine learning" --max-results 5
uv run python component_system/run_arxiv.py --id 1605.08386v1 --output json
```

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
Print a summary block for the runner:
```text
AUTORESEARCH_P_SUMMARY_BEGIN
{"idea":"short title","target_component":"model | optimizer | trainer","description":"change details, hypothesis, expected benefit","source_refs":["arXiv:<id>"],"commit_sha":"git sha","completed_at":"YYYY-MM-DD HH:MM:SS"}
AUTORESEARCH_P_SUMMARY_END
```

## Steps
1. Read `results.tsv` if present.
2. Refine prompt ? one concrete idea ? one isolated improvement; name target component.
3. Implement in worktree (from baseline); commit on seed branch.
4. Print summary block (runner records commit). Description must be enough for DCA.

## Constraints
- One component, one improvement per seed. Smallest viable implementation.
- No exploratory cleanup or opportunistic refactors unless required for the one change.
- Commit on seed branch; runner does not merge. **P must never merge;** only DCA triggers merge into baseline.
