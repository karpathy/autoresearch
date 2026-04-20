#!/usr/bin/env python3
"""Generate professional HTML research reports from autoresearch experiment data.

Usage:
    python generate_report.py <workflow_dir>
    python generate_report.py workflows/exec-summarizer
    python generate_report.py workflows/exec-summarizer --design-system design-system/
    
Output goes to <workflow_dir>/outputs/report/
"""

import argparse
import re
import shutil
import sys
import webbrowser
from pathlib import Path

# Handle imports - report_data.py is in the same directory
try:
    from . import report_data
except ImportError:
    # If running as script, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    import report_data

try:
    import jinja2
except ImportError:
    print("Error: jinja2 is not installed.", file=sys.stderr)
    print("Install with: pip install jinja2", file=sys.stderr)
    print("Or with uv: uv pip install jinja2", file=sys.stderr)
    sys.exit(1)


def load_design_overrides(path: Path | str | None) -> dict[str, str]:
    """Load design system overrides from a MASTER.md file.
    
    Args:
        path: Path to MASTER.md file or directory containing it. If None, returns empty dict.
    
    Returns:
        Dictionary with design override keys (primary, secondary, cta, background, text, 
        heading_font, body_font) mapped to their values.
    """
    if path is None:
        return {}
    
    path = Path(path)
    
    # If path is a directory, look for MASTER.md inside it
    if path.is_dir():
        master_path = path / "MASTER.md"
        if not master_path.exists():
            print(f"Warning: MASTER.md not found in {path}", file=sys.stderr)
            return {}
    else:
        master_path = path
        if not master_path.exists():
            print(f"Warning: Design system file not found: {master_path}", file=sys.stderr)
            return {}
    
    try:
        content = master_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not read {master_path}: {e}", file=sys.stderr)
        return {}
    
    overrides = {}
    
    # Extract colors from ## Colors section
    # Expected format:
    # ## Colors
    # - Primary: #1E40AF
    # - Secondary: #3B82F6
    # - CTA: #F59E0B
    # - Background: #F8FAFC
    # - Text: #1E3A8A
    colors_section = re.search(r'##\s+Colors\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if colors_section:
        colors_text = colors_section.group(1)
        
        # Match color entries like "- Primary: #1E40AF" or "- primary: #1E40AF"
        primary_match = re.search(r'-\s+Primary:\s+(#[0-9A-Fa-f]{6})', colors_text, re.IGNORECASE)
        if primary_match:
            overrides['primary'] = primary_match.group(1)
        
        secondary_match = re.search(r'-\s+Secondary:\s+(#[0-9A-Fa-f]{6})', colors_text, re.IGNORECASE)
        if secondary_match:
            overrides['secondary'] = secondary_match.group(1)
        
        cta_match = re.search(r'-\s+CTA:\s+(#[0-9A-Fa-f]{6})', colors_text, re.IGNORECASE)
        if cta_match:
            overrides['cta'] = cta_match.group(1)
        
        bg_match = re.search(r'-\s+Background:\s+(#[0-9A-Fa-f]{6})', colors_text, re.IGNORECASE)
        if bg_match:
            overrides['background'] = bg_match.group(1)
        
        text_match = re.search(r'-\s+Text:\s+(#[0-9A-Fa-f]{6})', colors_text, re.IGNORECASE)
        if text_match:
            overrides['text'] = text_match.group(1)
    
    # Extract typography from ## Typography section
    # Expected format:
    # ## Typography
    # - Heading: Fira Code
    # - Body: Fira Sans
    typo_section = re.search(r'##\s+Typography\s*\n(.*?)(?:\n##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if typo_section:
        typo_text = typo_section.group(1)
        
        # Match font entries like "- Heading: Fira Code"
        heading_font_match = re.search(r'-\s+Heading:\s+([^\n]+)', typo_text, re.IGNORECASE)
        if heading_font_match:
            overrides['heading_font'] = heading_font_match.group(1).strip()
        
        body_font_match = re.search(r'-\s+Body:\s+([^\n]+)', typo_text, re.IGNORECASE)
        if body_font_match:
            overrides['body_font'] = body_font_match.group(1).strip()
    
    if overrides:
        print(f"Loaded design overrides from {master_path}")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
    
    return overrides


def main(workflow_dir: Path | str | None = None, open_browser: bool = False, design_system: Path | str | None = None) -> None:
    """Generate HTML report from workflow experiment data.
    
    Args:
        workflow_dir: Path to workflow directory. If None, uses command line args.
        open_browser: Whether to open the report in browser after generation.
        design_system: Optional path to MASTER.md or directory containing it. Overrides design tokens.
    """
    # Parse arguments if workflow_dir not provided
    if workflow_dir is None:
        parser = argparse.ArgumentParser(
            description="Generate professional HTML research reports from autoresearch experiment data."
        )
        parser.add_argument(
            "workflow_dir",
            type=Path,
            help="Path to workflow directory (e.g., workflows/exec-summarizer)"
        )
        parser.add_argument(
            "--open",
            action="store_true",
            help="Open report in default browser after generation"
        )
        parser.add_argument(
            "--design-system",
            type=Path,
            help="Path to MASTER.md or directory containing it (overrides design tokens)"
        )
        args = parser.parse_args()
        workflow_dir = args.workflow_dir
        open_browser = args.open
        design_system = args.design_system
    
    # Convert to Path if string
    workflow_dir = Path(workflow_dir)
    
    # Validate workflow directory
    if not workflow_dir.exists():
        print(f"Error: Workflow directory not found: {workflow_dir}", file=sys.stderr)
        sys.exit(1)
    
    workflow_yaml = workflow_dir / "workflow.yaml"
    if not workflow_yaml.exists():
        print(f"Error: workflow.yaml not found in {workflow_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Collect report data
    print(f"Collecting data from {workflow_dir}...")
    try:
        report_obj = report_data.collect(workflow_dir)
    except Exception as e:
        print(f"Error collecting report data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load design system overrides if provided
    design_overrides = load_design_overrides(design_system)
    
    # Set up Jinja2 environment
    templates_dir = Path(__file__).parent / "templates"
    if not templates_dir.exists():
        print(f"Error: Templates directory not found: {templates_dir}", file=sys.stderr)
        sys.exit(1)
    
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        autoescape=True
    )
    
    # Convert dataclass to dict for tojson filter compatibility
    # Templates access fields as report.field_name (works with both dicts and objects)
    report_dict = report_data.report_data_to_dict(report_obj)
    
    # Load and render template
    print("Rendering report template...")
    try:
        template = env.get_template("base.html")
        html_content = template.render(report=report_dict, design_overrides=design_overrides)
    except jinja2.TemplateError as e:
        print(f"Error rendering template: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = workflow_dir / "outputs" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy static assets
    static_src = templates_dir / "static"
    static_dst = output_dir / "static"
    
    if static_src.exists():
        print("Copying static assets...")
        # Remove existing static dir if present
        if static_dst.exists():
            shutil.rmtree(static_dst)
        shutil.copytree(static_src, static_dst)
    
    # Write HTML file
    output_file = output_dir / "index.html"
    output_file.write_text(html_content, encoding="utf-8")
    
    print(f"\nReport generated: {output_file}")
    
    # Open in browser if requested
    if open_browser:
        print("Opening report in browser...")
        webbrowser.open(output_file.resolve().as_uri())


if __name__ == "__main__":
    main()
