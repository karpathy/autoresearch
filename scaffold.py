#!/usr/bin/env python3
"""Scaffold workflows, print onboarding prompts, and generate workspaces."""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent

ONBOARDING_PROMPT = """\
# Harness Onboarding -- Conversation Protocol

Reference schema: `harness.yaml` in the autoresearch repo root.

Walk the user through each section below. Ask 1--2 questions at a time.
After each answer, update the running summary, fill knowns, mark unknowns
with `[TBD]`.

## Running Summary (maintain this across turns)

```
name:        [TBD]
description: [TBD]
purpose:     [TBD]
constraints: [TBD]
conventions: [TBD]
platforms:   [TBD]
skills:      [TBD]
hooks:       [TBD]
guardrails:  [TBD]
gates:       [TBD]
structure:   [TBD]
models:      [TBD]
```

## Sections

### 1. Identity
- What is the short name for this workspace?
- One-sentence description of its purpose?

### 2. Agent Behavior
- What is the agent's primary purpose in this workspace?
- What must the agent never do? (constraints)
- What style conventions should the agent follow?

### 3. Platform Integration
- Which platforms will consume these instructions? (Copilot, Claude, generic)
- Where should platform-specific instruction files live?

### 4. Skills
- Does this workspace use any reusable skills?
- For each skill: name, file path, short description, trigger phrases?

### 5. Agents
- Are there specialized agent personas beyond the default?
- For each: name, file path, role description?

### 6. Hooks
- Which lifecycle events need hook scripts? (session_start, pre_tool_use,
  post_tool_use, session_end)
- For each hook: what should it check or enforce?

### 7. Guardrails
- Which shell commands should be blocked?
- Which paths are read-only for agents?
- Which file patterns must never be committed?

### 8. Quality Gates
- What linter, type checker, or test commands should run after changes?

### 9. Scaffold Patterns
- What is the directory structure? (agents/, skills/, hooks/, tests/, etc.)
- Any structural rules agents should follow?

### 10. Model Preferences
- Preferred model for routine work? Complex reasoning? Quick checks?

Once all sections are filled, generate the harness.yaml and run:
  python scaffold.py generate harness.yaml --output-dir <target>
"""


def replace_placeholders(file_path: Path, workflow_name: str) -> None:
    """Replace {workflow_name} placeholders in a file."""
    content = file_path.read_text(encoding="utf-8")
    updated = content.replace("{workflow_name}", workflow_name)
    file_path.write_text(updated, encoding="utf-8")


def scaffold_workflow(workflow_name: str) -> None:
    """Create a new workflow from the template."""
    template_dir = REPO_ROOT / "workflows" / "_template"
    target_dir = REPO_ROOT / "workflows" / workflow_name

    if not template_dir.exists():
        print(f"Error: Template directory not found at {template_dir}", file=sys.stderr)
        sys.exit(1)
    if target_dir.exists():
        print(f"Error: Workflow already exists at {target_dir}", file=sys.stderr)
        sys.exit(1)

    shutil.copytree(template_dir, target_dir)
    for md_file in target_dir.glob("*.md"):
        replace_placeholders(md_file, workflow_name)

    print(f"Created workflow: workflows/{workflow_name}/")
    print()
    print("Next steps:")
    print(f"  1. Edit workflows/{workflow_name}/workflow.yaml")
    print(f"  2. Write workflows/{workflow_name}/program.md")
    print("  3. Point your AI agent at the workflow and start experimenting")


def generate_workspace(harness_path: Path, output_dir: Path) -> None:
    """Read a harness.yaml and generate workspace scaffolding."""
    if not harness_path.exists():
        print(f"Error: Harness file not found: {harness_path}", file=sys.stderr)
        sys.exit(1)

    with open(harness_path, encoding="utf-8") as f:
        harness = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    # Create directory structure from scaffold.structure
    structure = (harness.get("scaffold") or {}).get("structure") or {}
    for dir_path in structure.values():
        d = output_dir / dir_path
        d.mkdir(parents=True, exist_ok=True)
        created.append(str(d.relative_to(output_dir)))

    # Create hook directories from hooks section
    hooks = harness.get("hooks") or {}
    for event_hooks in hooks.values():
        for hook in (event_hooks or []):
            hook_file = output_dir / hook["path"]
            hook_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate AGENTS.md
    agent = harness.get("agent") or {}
    purpose = agent.get("purpose", "")
    constraints = agent.get("constraints") or []
    conventions = agent.get("conventions") or []
    agents_md = output_dir / "AGENTS.md"
    lines = [f"# {harness.get('name', 'Workspace')}\n"]
    if purpose:
        lines.append(f"\n{purpose}\n")
    if constraints:
        lines.append("\n## Constraints\n")
        lines.extend(f"- {c}\n" for c in constraints)
    if conventions:
        lines.append("\n## Conventions\n")
        lines.extend(f"- {c}\n" for c in conventions)
    agents_md.write_text("".join(lines), encoding="utf-8")
    created.append("AGENTS.md")

    # Generate .github/copilot-instructions.md
    github_dir = output_dir / ".github"
    github_dir.mkdir(parents=True, exist_ok=True)
    copilot_md = github_dir / "copilot-instructions.md"
    cp_lines = [f"# Copilot Instructions -- {harness.get('name', 'Workspace')}\n"]
    if purpose:
        cp_lines.append(f"\n{purpose}\n")
    copilot_cfg = (harness.get("platforms") or {}).get("copilot") or {}
    if copilot_cfg.get("instructions"):
        cp_lines.append(
            f"\nCanonical instructions: `{copilot_cfg['instructions']}`\n"
        )
    if constraints:
        cp_lines.append("\n## Constraints\n")
        cp_lines.extend(f"- {c}\n" for c in constraints)
    if conventions:
        cp_lines.append("\n## Conventions\n")
        cp_lines.extend(f"- {c}\n" for c in conventions)
    copilot_md.write_text("".join(cp_lines), encoding="utf-8")
    created.append(".github/copilot-instructions.md")

    print(f"Generated workspace in {output_dir}/")
    for item in created:
        print(f"  {item}")


def generate_report_cmd(workflow_name: str, open_browser: bool = False, design_system: Path | None = None) -> None:
    """Generate an HTML report from workflow results.
    
    Args:
        workflow_name: Name of the workflow (e.g., 'exec-summarizer')
        open_browser: Whether to open the report in browser after generation
        design_system: Optional path to MASTER.md or directory containing it
    """
    workflow_dir = REPO_ROOT / "workflows" / workflow_name
    if not workflow_dir.exists():
        print(f"Error: Workflow not found: {workflow_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Import and execute the generate_report module
    gen_script = REPO_ROOT / ".github" / "skills" / "report-generator" / "generate_report.py"
    if not gen_script.exists():
        print(f"Error: Report generator not found: {gen_script}", file=sys.stderr)
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("generate_report", gen_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(workflow_dir, open_browser=open_browser, design_system=design_system)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold workflows, onboard workspaces, and generate from harness configs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    wf = sub.add_parser("workflow", help="Create a new workflow from the template")
    wf.add_argument("name", help="Workflow name (e.g., my-experiment)")

    sub.add_parser("onboard", help="Print the onboarding conversation prompt")

    gen = sub.add_parser("generate", help="Generate workspace files from a harness.yaml")
    gen.add_argument("harness_yaml", type=Path, help="Path to a filled harness.yaml")
    gen.add_argument("--output-dir", type=Path, default=Path("."),
                     help="Output directory (default: current directory)")

    rpt = sub.add_parser("report", help="Generate an HTML report from workflow results")
    rpt.add_argument("workflow_name", help="Workflow name (e.g., exec-summarizer)")
    rpt.add_argument("--open", action="store_true", help="Open report in browser after generation")
    rpt.add_argument("--design-system", type=Path, help="Path to MASTER.md or directory containing it (overrides design tokens)")

    args = parser.parse_args()
    if args.command == "workflow":
        scaffold_workflow(args.name)
    elif args.command == "onboard":
        print(ONBOARDING_PROMPT)
    elif args.command == "generate":
        generate_workspace(args.harness_yaml, args.output_dir)
    elif args.command == "report":
        generate_report_cmd(args.workflow_name, args.open, getattr(args, 'design_system', None))


if __name__ == "__main__":
    main()
