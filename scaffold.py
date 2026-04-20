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


BOOTSTRAP_PROMPT = """\
# Background Harness Optimizer -- Bootstrap Prompt

You are the harness background optimizer. Begin immediately -- no user interaction needed.

## Session Configuration
- Run tag: {tag}
- Branch: autoiterate/harness-bg-{tag}
- Worktree: {worktree_path}
- Mode: Background (autonomous, no confirmations)

## Immediate Actions
1. Read `workflows/harness-optimize/workflow.yaml` for targets, metric, and run command
2. Read `workflows/harness-optimize/program.md` for experimentation strategy
3. Read all target files listed in workflow.yaml:
   - AGENTS.md
   - .github/copilot-instructions.md
   - .github/agents/research-runner.agent.md
   - .github/skills/autonomous-iteration/SKILL.md
   - harness.yaml
4. If `workflows/harness-optimize/results/results.tsv` exists, read it for prior experiment history
5. If `workflows/harness-optimize/results/musings.md` exists, review it for accumulated learnings
6. If no baseline exists in results.tsv, establish one now by running the benchmark
7. Begin the autonomous-iteration loop as defined in the skill protocol

## Background Mode Overrides
- Self-assign run tags (do not ask the user)
- Skip all confirmation and setup steps
- Run indefinitely until the session ends or the context window fills
- Follow the "Between-Session Reflection" protocol if prior results exist
- Log every experiment to results.tsv and musings.md
- Use the Artisan's Triad (Additive/Reductive/Reformative) and don't repeat the same type 5+ times in a row

## Anti-Overfitting Guardrail
Changes must improve general agent effectiveness, not game specific benchmark tasks.
If a change cannot be justified on general-effectiveness grounds, discard it.
"""


def harness_bg_cmd(tag: str | None = None, resume: bool = False, refresh: bool = False) -> None:
    """Set up and print launch instructions for the background harness optimizer.

    Args:
        tag: Run tag (default: today's date as YYYYMMDD)
        resume: Reuse existing worktree if present
        refresh: Create fresh worktree from main, copy state from old worktree
    """
    import subprocess
    from datetime import datetime

    if tag is None:
        tag = datetime.now().strftime("%Y%m%d")

    repo_name = REPO_ROOT.name
    branch_name = f"autoiterate/harness-bg-{tag}"
    worktree_path = REPO_ROOT.parent / f"{repo_name}-harness-bg"
    state_dir = Path("workflows/harness-optimize/results")

    # Check for existing worktree
    if worktree_path.exists():
        if resume:
            print(f"Resuming existing worktree at {worktree_path}")
            _write_bootstrap(worktree_path, tag)
            _print_launch_instructions(worktree_path)
            return
        elif refresh:
            # Save state from old worktree
            old_state = {}
            for artifact in ["results.tsv", "musings.md", "ratchet_state.json"]:
                src = worktree_path / state_dir / artifact
                if src.exists():
                    old_state[artifact] = src.read_text(encoding="utf-8")
                    print(f"  Saved: {artifact}")

            # Remove old worktree
            subprocess.run(["git", "worktree", "remove", str(worktree_path), "--force"],
                           cwd=REPO_ROOT, capture_output=True)
            print(f"Removed old worktree")

            # Create fresh worktree from main
            _create_worktree(worktree_path, branch_name)

            # Restore state
            results_dir = worktree_path / state_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            for artifact, content in old_state.items():
                (results_dir / artifact).write_text(content, encoding="utf-8")
                print(f"  Restored: {artifact}")

            _write_bootstrap(worktree_path, tag)
            _print_launch_instructions(worktree_path)
            return
        else:
            print(f"Error: Worktree already exists at {worktree_path}", file=sys.stderr)
            print("  Use --resume to reuse it, or --refresh to recreate from main", file=sys.stderr)
            sys.exit(1)

    _create_worktree(worktree_path, branch_name)
    _write_bootstrap(worktree_path, tag)
    _print_launch_instructions(worktree_path)


def _create_worktree(worktree_path: Path, branch_name: str) -> None:
    """Create a git worktree with a new branch from the default branch."""
    import subprocess

    # Detect default branch (master or main)
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0:
        default_branch = result.stdout.strip().split("/")[-1]
    else:
        # Fallback: check if main or master exists
        for candidate in ["main", "master"]:
            check = subprocess.run(
                ["git", "rev-parse", "--verify", candidate],
                cwd=REPO_ROOT, capture_output=True,
            )
            if check.returncode == 0:
                default_branch = candidate
                break
        else:
            default_branch = "HEAD"

    # Create the branch
    subprocess.run(
        ["git", "branch", branch_name, default_branch],
        cwd=REPO_ROOT, capture_output=True,
    )

    # Create the worktree
    result = subprocess.run(
        ["git", "worktree", "add", str(worktree_path), branch_name],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error creating worktree: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"Created worktree at {worktree_path}")
    print(f"  Branch: {branch_name}")

    # Sync dependencies
    sync = subprocess.run(
        ["uv", "sync"],
        cwd=worktree_path, capture_output=True, text=True,
    )
    if sync.returncode == 0:
        print("  Dependencies synced")
    else:
        print(f"  Warning: uv sync failed: {sync.stderr[:200]}", file=sys.stderr)

    # Create results directory
    results_dir = worktree_path / "workflows" / "harness-optimize" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)


def _write_bootstrap(worktree_path: Path, tag: str) -> None:
    """Write the bootstrap prompt file into the worktree."""
    prompt = BOOTSTRAP_PROMPT.format(tag=tag, worktree_path=worktree_path)
    prompt_path = worktree_path / ".harness-bg-prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"  Bootstrap prompt: {prompt_path}")


def _print_launch_instructions(worktree_path: Path) -> None:
    """Print instructions for starting the Copilot CLI session."""
    prompt_path = worktree_path / ".harness-bg-prompt.md"
    print()
    print("=" * 60)
    print("  BACKGROUND HARNESS OPTIMIZER READY")
    print("=" * 60)
    print()
    print("To start the optimizer:")
    print()
    print("  Option A (VS Code Copilot CLI):")
    print("  1. Open VS Code")
    print("  2. Start a new Copilot CLI session (Worktree isolation)")
    print(f"  3. Paste the prompt from: {prompt_path}")
    print()
    print("  Option B (Terminal):")
    print(f"  1. cd {worktree_path}")
    print("  2. copilot")
    print(f"  3. Paste the prompt from: {prompt_path}")
    print()
    print("The optimizer will run autonomously until the session ends.")
    print("Results will be logged to workflows/harness-optimize/results/")
    print()


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

    bg = sub.add_parser("harness-bg", help="Set up background harness optimizer worktree")
    bg.add_argument("--tag", help="Run tag (default: today's date as YYYYMMDD)")
    bg.add_argument("--resume", action="store_true", help="Reuse existing worktree")
    bg.add_argument("--refresh", action="store_true", help="Fresh worktree from main, copy state")

    args = parser.parse_args()
    if args.command == "workflow":
        scaffold_workflow(args.name)
    elif args.command == "onboard":
        print(ONBOARDING_PROMPT)
    elif args.command == "generate":
        generate_workspace(args.harness_yaml, args.output_dir)
    elif args.command == "report":
        generate_report_cmd(args.workflow_name, args.open, getattr(args, 'design_system', None))
    elif args.command == "harness-bg":
        harness_bg_cmd(tag=args.tag, resume=args.resume, refresh=args.refresh)


if __name__ == "__main__":
    main()
