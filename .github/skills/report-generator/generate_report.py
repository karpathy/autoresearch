#!/usr/bin/env python3
"""Generate professional HTML research reports from autoresearch experiment data.

Usage:
    python generate_report.py <workflow_dir>
    python generate_report.py workflows/exec-summarizer
    
Output goes to <workflow_dir>/outputs/report/
"""

import argparse
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


def main(workflow_dir: Path | str | None = None, open_browser: bool = False) -> None:
    """Generate HTML report from workflow experiment data.
    
    Args:
        workflow_dir: Path to workflow directory. If None, uses command line args.
        open_browser: Whether to open the report in browser after generation.
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
        args = parser.parse_args()
        workflow_dir = args.workflow_dir
        open_browser = args.open
    
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
        html_content = template.render(report=report_dict)
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
