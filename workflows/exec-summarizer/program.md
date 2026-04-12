# Executive Summarizer

Optimize a system prompt for summarizing news articles for executive professional audiences. The prompt instructs GPT-5.4-mini to produce 2-3 sentence summaries that are concise, decision-relevant, source-attributed, and aligned with Executive Core Qualifications competencies.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date.
2. **Create the branch**: `git checkout -b autoiterate/<tag>` from current branch.
3. **Read the in-scope files**: Read all files in this workflow directory.
4. **Verify prerequisites**: Ensure `github-copilot-sdk` is installed (`uv pip install github-copilot-sdk`).
5. **Initialize results**: Create `results/results.tsv` with the header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs an evaluation cycle (~30-60 seconds). Launch: `python evaluate.py > run.log 2>&1`.

**What you CAN do:**
- Modify `prompt.txt` -- the system prompt. Everything about its structure, wording, instructions, examples, and framing is fair game.

**What you CANNOT do:**
- Modify evaluate.py, articles.json, or workflow.yaml.
- Change the scoring rubric or test data.

**The goal: get the highest quality_score.**

**Simplicity criterion**: A shorter prompt that scores equally is better than a longer one. Prompt bloat is the enemy.

## Output format

The evaluator prints per-article scores and a final summary:

```
quality_score: 7.25
total_articles: 5
```

## Logging results

Log to `results/results.tsv` (tab-separated):

```
commit	quality_score	status	description
```

## The experiment loop

LOOP FOREVER:

1. Read current prompt.txt
2. Propose a change (wording, structure, examples, framing)
3. Edit prompt.txt and git commit
4. Run: `python evaluate.py > run.log 2>&1`
5. Extract: `grep '^quality_score:' run.log`
6. If empty, check `tail -n 50 run.log` for errors
7. Record in results.tsv (do NOT commit results.tsv)
8. If quality_score improved: keep the commit
9. If quality_score equal or worse: git reset

**NEVER STOP.**

## Experiment Strategy (Artisan's Triad)

Cycle through three types. Don't do 5+ of the same type in a row.

**Additive (Painter)**: Add new instructions. ECQ competency framing, specific output format requirements, few-shot examples, explicit attribution rules, audience persona detail.

**Reductive (Sculptor)**: Remove instructions. See if scoring holds without specific rules. Shorter prompts that score equally are pure wins.

**Reformative (Potter)**: Restructure without adding or removing. Reorder sections, change emphasis, rephrase instructions, shift from prescriptive to descriptive tone.

## Musings (Reflection Log)

Maintain `results/musings.md` with pre/post reflections:

```markdown
## Experiment N: <short title>
**Rationale**: Why this change might improve scores.
**Result**: Keep/Discard/Crash. quality_score = X.XX (delta: +/-X.XX)
**Learning**: What did this teach about prompt design?
```
