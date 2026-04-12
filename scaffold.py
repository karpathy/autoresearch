#!/usr/bin/env python3
"""Scaffold a new research workflow from the template."""

import argparse
import shutil
import sys
from pathlib import Path


def replace_placeholders(file_path: Path, workflow_name: str) -> None:
    """Replace {workflow_name} placeholders in a file."""
    content = file_path.read_text(encoding="utf-8")
    updated = content.replace("{workflow_name}", workflow_name)
    file_path.write_text(updated, encoding="utf-8")


def scaffold_workflow(workflow_name: str) -> None:
    """Create a new workflow from the template."""
    repo_root = Path(__file__).parent
    template_dir = repo_root / "workflows" / "_template"
    target_dir = repo_root / "workflows" / workflow_name

    # Validate template exists
    if not template_dir.exists():
        print(f"Error: Template directory not found at {template_dir}", file=sys.stderr)
        sys.exit(1)

    # Refuse to overwrite existing workflow
    if target_dir.exists():
        print(f"Error: Workflow already exists at {target_dir}", file=sys.stderr)
        sys.exit(1)

    # Copy template to new workflow directory
    shutil.copytree(template_dir, target_dir)

    # Replace placeholders in markdown files
    for md_file in target_dir.glob("*.md"):
        replace_placeholders(md_file, workflow_name)

    # Print success message
    print(f"Created workflow: workflows/{workflow_name}/")
    print()
    print("Next steps:")
    print(f"  1. Edit workflows/{workflow_name}/workflow.yaml (define targets, metric, run command)")
    print(f"  2. Write workflows/{workflow_name}/program.md (research strategy and constraints)")
    print("  3. Point your AI agent at the workflow and start experimenting")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold a new research workflow from the template."
    )
    parser.add_argument(
        "workflow_name",
        help="Name of the workflow to create (e.g., my-experiment, prompt-tuning)"
    )
    args = parser.parse_args()

    scaffold_workflow(args.workflow_name)


if __name__ == "__main__":
    main()
