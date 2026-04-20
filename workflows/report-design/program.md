# Report Design Optimization

Optimize the HTML research report template for maximum visual quality, narrative coherence, and professional shareability. The report generator transforms raw autoresearch data (results.tsv, musings.md, git log) into a slide-like strategy deck. This workflow iterates on the template's visual design, information architecture, and data presentation.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date.
2. **Create the branch**: `git checkout -b autoiterate/<tag>` from current branch.
3. **Read the in-scope files**: Read all files in this workflow directory and the report-generator skill directory.
4. **Generate a sample report**: Run `python scaffold.py report exec-summarizer` (or any workflow with results data) to produce a baseline report.
5. **Initialize results**: Create `results/results.tsv` with the header row.
6. **Confirm and go**.

## Experimentation

Each experiment modifies the report templates, regenerates the report, and measures quality. Launch: `python evaluate.py > run.log 2>&1`.

**What you CAN do:**
- Modify templates in `../../.github/skills/report-generator/templates/` -- all section templates, partials, and CSS
- Modify `../../.github/skills/report-generator/design-tokens.json` -- colors, typography, spacing, chart config

**What you CANNOT do:**
- Modify evaluate.py, report_data.py, or generate_report.py
- Change the narrative arc structure (Situation > Challenge > Experiments > Findings > Impact)
- Add external dependencies (no npm, no Tailwind build step)

**The goal: get the highest quality_score.**

**Simplicity criterion**: Cleaner CSS that scores equally is better. Unnecessary visual complexity is the enemy.

## Evaluation Dimensions

The evaluator scores reports on 8 dimensions (each 1-10):

1. **Narrative coherence** -- Does the arc flow logically? Is the "so what" clear?
2. **Information density** -- Right amount of data per section? No filler, no overload?
3. **Chart comprehension** -- Can a reader extract the key insight in <5 seconds?
4. **Visual hierarchy** -- Clear heading levels, scannable structure, professional feel?
5. **Accessibility/contrast** -- WCAG AA compliance, readable on projectors?
6. **Source attribution** -- Are claims backed by data? Can a reader trace to source?
7. **Engagement** -- Does the scroll/transition design encourage continued reading?
8. **Responsiveness** -- Usable at common viewport widths?

Composite: structural dims (1-3) weighted 2x, presentational (4-6) at 1x, polish (7-8) at 0.5x.

## Experiment Order

Run experiments in this order for maximum leverage:

1. **Narrative structure** -- story skeleton first
2. **Data density** -- how much per section
3. **Chart types** -- right visualization for each data point
4. **Typography/spacing** -- slide-like professional feel
5. **Color/contrast** -- palette, accessibility
6. **Citation styling** -- provenance visible but not disruptive
7. **Scroll behavior** -- transitions, progress indicators
8. **Responsive breakpoints** -- graceful degradation

## Logging results

Log to `results/results.tsv` (tab-separated):

```
commit	quality_score	status	description
```

## The experiment loop

LOOP FOREVER:

1. Read current template files
2. Propose a visual/structural change
3. Edit template files and git commit
4. Run: `python evaluate.py > run.log 2>&1`
5. Extract: `grep '^quality_score:' run.log`
6. If empty, check `tail -n 50 run.log` for errors
7. Record in results.tsv (do NOT commit results.tsv)
8. If quality_score improved: keep the commit
9. If quality_score equal or worse: git reset

**NEVER STOP.**

## Experiment Strategy (Artisan's Triad)

Cycle through three types. Don't do 5+ of the same type in a row.

**Additive (Painter)**: Add visual elements -- progress indicators, animated transitions, callout boxes, hover effects, gradient accents, data labels on charts.

**Reductive (Sculptor)**: Remove visual noise -- simplify CSS rules, reduce color palette, eliminate decorative elements that don't aid comprehension, strip unused styles.

**Reformative (Potter)**: Restructure layout -- change grid proportions, swap chart types, reorder information within sections, adjust spacing ratios, change typography scale relationships.

## Musings (Reflection Log)

Maintain `results/musings.md` with pre/post reflections:

```markdown
## Experiment N: <short title>
**Rationale**: Why this visual change might improve quality scores.
**Result**: Keep/Discard/Crash. quality_score = X.XX (delta: +/-X.XX)
**Learning**: What did this teach about report design?
```
