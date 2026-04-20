# Design Optimization Research Program

## Objective
Use ui-ux-pro-max design intelligence to iteratively improve report template visual quality, measured by an 8-dimension composite quality_score (structural 2x, presentational 1x, polish 0.5x).

## Design Intelligence Tools

### Generate baseline design system
```bash
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "" --design-system > design-system.json
```

Produces a JSON design system with colors, typography, spacing, shadows, and responsive breakpoints.

### Query by domain
```bash
# Style patterns (layout, composition, whitespace)
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "data visualization layout" --domain style

# Chart best practices (color palettes, axis formatting, legends)
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "line chart readability" --domain chart

# UX patterns (navigation, affordances, progressive disclosure)
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "scroll-based navigation" --domain ux

# Typography (scale, hierarchy, readability)
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "modular typography scale" --domain typography

# Color theory (contrast, harmony, semantic meaning)
python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "accessible color contrast" --domain color
```

## Experiment Strategy

Each experiment follows this pattern:

1. **Query ui-ux-pro-max** for a design recommendation related to a specific quality dimension
2. **Apply the design change** to CSS (styles.css) or design tokens (design-tokens.json) or structure (base.html)
3. **Measure quality_score** using evaluate.py (scores 8 dimensions, composite 0-10)
4. **Keep or discard** based on whether quality_score improved

### Target quality dimensions
- **narrative_coherence** (2x): all 5 sections present, logical flow
- **information_density** (2x): content-to-markup ratio
- **chart_comprehension** (2x): chart count, clarity
- **visual_hierarchy** (1x): heading levels, spacing rhythm
- **accessibility_contrast** (1x): alt text, ARIA, color contrast
- **source_attribution** (1x): commit refs, citations visible
- **engagement** (0.5x): scroll-snap, transitions, progress indicators
- **responsiveness** (0.5x): media queries, viewport units

## The Artisan's Triad (Adapted for Design)

Cycle through three modes to avoid local optima:

### Additive: Add design patterns
- Hover effects on interactive elements
- Smooth transitions (opacity, transform)
- Visual indicators (active nav dots, scroll progress)
- Chart annotations, legends, tooltips
- Drop shadows for depth
- Border accents for hierarchy

### Reductive: Remove visual noise
- Simplify color palette (fewer hues)
- Reduce padding/margins (tighten layout)
- Fewer shadows (minimize elevation layers)
- Remove decorative elements
- Consolidate typography scale (fewer sizes)
- Reduce animation count

### Reformative: Reshape layout
- Change grid system (columns, gaps, alignment)
- New typography scale (different ratio, e.g., 1.25 vs 1.618)
- Swap color palette (warm to cool, high to low contrast)
- Different chart type (bar to line, stacked to grouped)
- Rearrange section order
- Shift from horizontal to vertical rhythm

## Pre-Delivery Checklist (from ui-ux-pro-max/PROMPT.md)

Before marking an experiment as "keep", validate:

- [ ] No emoji used as icons (use UTF-8 symbols: ▲ ▼ ● ◆ ✓ ✗ ← → ↑ ↓)
- [ ] `cursor: pointer` on all clickable elements
- [ ] Contrast ratio meets WCAG AA (4.5:1 for text, 3:1 for UI)
- [ ] Hover states defined for interactive elements
- [ ] Focus states visible for keyboard navigation
- [ ] Chart colors distinguishable for colorblind users
- [ ] Responsive breakpoints cover mobile (320px+), tablet (768px+), desktop (1024px+)
- [ ] Typography scale uses consistent ratio (1.25, 1.333, or 1.618)
- [ ] Spacing follows 4px or 8px rhythm

## Example Experiment Sequence

### Experiment 1: Baseline
- **Action**: Generate design system, apply to design-tokens.json
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "" --design-system`
- **Rationale**: Establish baseline design system
- **Expected**: quality_score establishes floor

### Experiment 2: Typography scale
- **Action**: Query typography best practices, apply modular scale
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "modular typography scale for data reports" --domain typography`
- **Rationale**: Improve visual_hierarchy dimension
- **Expected**: +0.3 to +0.8 points if scale creates better hierarchy

### Experiment 3: Chart color palette
- **Action**: Query chart color best practices, update chart tokens
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "accessible chart color palettes" --domain chart`
- **Rationale**: Improve chart_comprehension and accessibility_contrast
- **Expected**: +0.2 to +0.5 points if palette improves legibility

### Experiment 4: Scroll navigation UX
- **Action**: Query scroll-based navigation, add scroll-snap and nav dots
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "scroll-snap navigation patterns" --domain ux`
- **Rationale**: Improve engagement dimension
- **Expected**: +0.1 to +0.3 points (engagement is 0.5x weight)

### Experiment 5: Reduce visual noise (Reductive)
- **Action**: Simplify color palette, reduce shadows
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "minimalist data visualization" --domain style`
- **Rationale**: Test if simplicity improves information_density
- **Expected**: May improve or regress -- test hypothesis

### Experiment 6: Contrast optimization
- **Action**: Query color contrast, adjust text/background colors
- **Query**: `python ../../.github/prompts/ui-ux-pro-max/scripts/search.py "WCAG AA color contrast" --domain color`
- **Rationale**: Improve accessibility_contrast dimension
- **Expected**: +0.2 to +0.4 points if contrast violations fixed

## Constraints

- **Never modify fixed files**: report_data.py, generate_report.py, evaluate.py are read-only
- **Evidence-based changes**: Always query ui-ux-pro-max before making a design change
- **Measure before discarding**: Run evaluate.py after every change, even if you think it won't help
- **Follow pre-delivery checklist**: Validate all checkpoints before marking "keep"
- **No cargo cult design**: Don't add patterns just because they're trendy -- justify with query results

## Success Criteria

- Achieve quality_score >= 7.0 (out of 10)
- All 8 dimensions score >= 6.0
- Pass all pre-delivery checklist items
- Maintain valid HTML/CSS (no syntax errors)
