# Design System Integration - Report Generator

The autoresearch report generator now supports optional design system overrides via MASTER.md files.

## Feature Overview

When generating reports, you can provide a MASTER.md design system file to override the default design tokens. This allows you to apply custom branding, color schemes, and typography to your reports.

## Usage

### Command Line

```bash
# Using generate_report.py directly
python .github/skills/report-generator/generate_report.py workflows/my-workflow --design-system design-system/

# Using scaffold.py
python scaffold.py report my-workflow --design-system test-design-system/

# Can also point directly to MASTER.md
python scaffold.py report my-workflow --design-system path/to/MASTER.md
```

### Programmatic

```python
from pathlib import Path
from generate_report import main

main(
    workflow_dir=Path("workflows/exec-summarizer"),
    open_browser=True,
    design_system=Path("design-system/")
)
```

## MASTER.md Format

The MASTER.md file should follow this structure:

```markdown
# Design System: Your Project Name

## Colors
- Primary: #1E40AF
- Secondary: #3B82F6
- CTA: #F59E0B
- Background: #F8FAFC
- Text: #1E3A8A

## Typography
- Heading: Fira Code
- Body: Fira Sans
- Google Fonts: https://fonts.google.com/share?selection.family=...

## Spacing
(Optional section - not currently used by report generator)

## Notes
(Optional section - not parsed)
```

### Supported Fields

**Colors:**
- `Primary` - Maps to `--color-brand-primary`, `--color-brand-primary-light`, `--color-border-focus`
- `Secondary` - Maps to `--color-brand-secondary`
- `CTA` - Maps to `--color-brand-accent`
- `Background` - Maps to `--color-bg-primary`
- `Text` - Maps to `--color-text-primary`

**Typography:**
- `Heading` - Maps to `--font-heading`
- `Body` - Maps to `--font-body`

All fields are optional. If a field is missing, the default design token value is used.

## Generating MASTER.md

You can generate a MASTER.md using the ui-ux-pro-max tool:

```bash
# This will create design-system/MASTER.md
python -m ui_ux_pro_max --design-system --persist
```

## Implementation Details

### Parser (generate_report.py)

The `load_design_overrides()` function:
1. Accepts a Path to either a MASTER.md file or directory containing one
2. Uses regex to extract color hex codes and font names from markdown sections
3. Returns a dictionary of overrides (empty dict if path is None)
4. Prints warnings if file not found, but doesn't fail (graceful degradation)

### Template (base.html)

The base template now includes a conditional style block:
```html
{% if design_overrides %}
<style>
    :root {
        {% if design_overrides.primary %}--color-brand-primary: {{ design_overrides.primary }};{% endif %}
        ...
    }
</style>
{% endif %}
```

This approach:
- Only injects CSS when overrides are present
- Uses CSS custom property cascade (overrides win over defaults)
- Preserves all default tokens that aren't overridden
- Works without JavaScript

## Testing

Test the feature with the included test design system:

```bash
# Create test MASTER.md (already included in repo)
cat test-design-system/MASTER.md

# Test parsing (should print 7 overrides)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.github/skills/report-generator')
import generate_report
overrides = generate_report.load_design_overrides('test-design-system')
print(f'Loaded {len(overrides)} overrides')
"
```

## Backwards Compatibility

This feature is **100% optional**:
- If `--design-system` is not provided, reports generate exactly as before
- If MASTER.md is not found at the specified path, a warning is printed but generation continues
- If MASTER.md exists but has no matching sections, an empty dict is returned (no overrides)
- The `main()` function signature maintains backwards compatibility with `design_system` as an optional parameter

## Future Enhancements

Potential additions:
- Support for spacing overrides from MASTER.md
- Support for border radius values
- Support for shadow definitions
- Validation of hex color format
- Support for RGB/HSL color formats
- Auto-generation of light/dark variants from primary color
