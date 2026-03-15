"""CLI entry point — click-based command-line interface.

Provides commands: init, validate, score, evaluate, serve, history, leaderboard.
"""

import os
import stat
import sys
import textwrap

import click

from autoanything.problem import load_problem, ValidationError


@click.group()
def main():
    """AutoAnything — autonomous optimization via AI agents."""


@main.command()
@click.argument("name")
@click.option("--dir", "parent_dir", default=".", help="Parent directory for the new problem.")
@click.option("--metric", default="score", help="Metric name (key in score.sh JSON output).")
@click.option("--direction", default="minimize", type=click.Choice(["minimize", "maximize"]),
              help="Score direction.")
def init(name, parent_dir, metric, direction):
    """Scaffold a new problem directory."""
    problem_dir = os.path.join(parent_dir, name)
    if os.path.exists(problem_dir):
        click.echo(f"Error: directory '{problem_dir}' already exists.", err=True)
        sys.exit(1)

    os.makedirs(problem_dir)
    os.makedirs(os.path.join(problem_dir, "state"))
    os.makedirs(os.path.join(problem_dir, "context"))
    os.makedirs(os.path.join(problem_dir, "scoring"))
    os.makedirs(os.path.join(problem_dir, ".autoanything"))

    # problem.yaml
    with open(os.path.join(problem_dir, "problem.yaml"), "w") as f:
        f.write(textwrap.dedent(f"""\
            name: {name}
            description: >
              Describe what agents are optimizing.

            state:
              - state/solution.py

            context: []

            score:
              name: {metric}
              direction: {direction}
              description: "Describe what this metric measures"
              timeout: 900

            git:
              base_branch: main
              proposal_pattern: "proposals/*"

            constraints:
              - "Must not modify files outside of state/"
        """))

    # state/solution.py
    with open(os.path.join(problem_dir, "state", "solution.py"), "w") as f:
        f.write("# Solution file — agents modify this.\n")

    # scoring/score.sh
    score_sh = os.path.join(problem_dir, "scoring", "score.sh")
    with open(score_sh, "w") as f:
        f.write(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            # Scoring script — outputs JSON on the last line.
            # The metric key must match score.name in problem.yaml.
            echo '{{"{ metric}": 0.0}}'
        """))
    os.chmod(score_sh, os.stat(score_sh).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # agent_instructions.md
    with open(os.path.join(problem_dir, "agent_instructions.md"), "w") as f:
        f.write(textwrap.dedent(f"""\
            # Agent Instructions

            ## Objective
            Optimize the metric `{metric}` ({direction}).

            ## Protocol
            1. Pull the latest main branch.
            2. Create a branch: `proposals/<your-name>/<description>`
            3. Modify only the files listed under `state:` in `problem.yaml`.
            4. Push your branch or open a PR targeting main.
            5. The evaluator will score your submission and update the leaderboard.

            ## Files
            - `problem.yaml` — problem definition and constraints
            - `state/` — files you can modify
            - `context/` — read-only background information
            - `leaderboard.md` — current scores and history
        """))

    # .gitignore
    with open(os.path.join(problem_dir, ".gitignore"), "w") as f:
        f.write(textwrap.dedent("""\
            # Private scoring code
            scoring/

            # Evaluator state
            .autoanything/
        """))

    click.echo(f"Created problem '{name}' in {problem_dir}")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory to validate.")
def validate(problem_dir):
    """Check that the problem directory is well-formed."""
    import subprocess

    errors = []
    warnings = []

    # Check problem.yaml
    try:
        config = load_problem(problem_dir)
    except FileNotFoundError:
        click.echo("Error: problem.yaml not found.", err=True)
        sys.exit(1)
    except ValidationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Check state files exist
    for f in config.state:
        if not os.path.exists(os.path.join(problem_dir, f)):
            errors.append(f"State file not found: {f}")

    # Check score script
    script_path = os.path.join(problem_dir, config.score.script)
    if not os.path.exists(script_path):
        errors.append(f"Score script not found: {config.score.script}")
    elif not os.access(script_path, os.X_OK):
        warnings.append(f"Score script not executable: {config.score.script}")

    # Check .gitignore
    gitignore_path = os.path.join(problem_dir, ".gitignore")
    if os.path.exists(gitignore_path):
        gitignore = open(gitignore_path).read()
        if "scoring" not in gitignore:
            warnings.append(".gitignore does not exclude scoring/")
    else:
        warnings.append("No .gitignore found")

    # Check if scoring/ is tracked by git
    try:
        result = subprocess.run(
            ["git", "ls-files", "scoring/"],
            capture_output=True, text=True, cwd=problem_dir,
        )
        if result.stdout.strip():
            warnings.append(f"scoring/ files are tracked by git: {result.stdout.strip()}")
    except Exception:
        pass

    if warnings:
        for w in warnings:
            click.echo(f"Warning: {w}")

    if errors:
        for e in errors:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo("Validation passed.")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
def score(problem_dir):
    """Run score.sh once and print the result."""
    from autoanything.scoring import run_score as _run_score

    try:
        config = load_problem(problem_dir)
    except (FileNotFoundError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    script_path = os.path.join(problem_dir, config.score.script)
    if not os.path.exists(script_path):
        click.echo(f"Error: Score script not found: {config.score.script}", err=True)
        sys.exit(1)

    score_val, metrics, duration, error = _run_score(
        script_path, score_name=config.score.name,
        timeout=config.score.timeout, cwd=problem_dir,
    )

    if error:
        click.echo(f"Error: {error}", err=True)
        sys.exit(1)

    click.echo(f"{config.score.name}: {score_val}")
    if metrics:
        for k, v in metrics.items():
            if k != config.score.name:
                click.echo(f"  {k}: {v}")
    click.echo(f"Duration: {duration:.1f}s")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
def history(problem_dir):
    """Print evaluation history."""
    from autoanything.history import init_db as _init_db

    db_path = os.path.join(problem_dir, ".autoanything", "history.db")
    if not os.path.exists(db_path):
        click.echo("No evaluation history yet.")
        return

    conn = _init_db(db_path)
    rows = conn.execute("""
        SELECT score, status, branch, description, evaluated_at
        FROM evaluations ORDER BY id DESC LIMIT 50
    """).fetchall()
    conn.close()

    if not rows:
        click.echo("No evaluations recorded.")
        return

    click.echo(f"{'Score':>12} {'Status':<10} {'Branch':<35} {'Description'}")
    click.echo("-" * 80)
    for s, status, branch, desc, when in rows:
        score_str = f"{s:.6f}" if s is not None else "crash"
        click.echo(f"{score_str:>12} {status:<10} {branch:<35} {desc or ''}")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
def leaderboard(problem_dir):
    """Regenerate leaderboard.md from history."""
    from autoanything.history import init_db as _init_db
    from autoanything.leaderboard import export_leaderboard as _export

    try:
        config = load_problem(problem_dir)
    except (FileNotFoundError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    db_path = os.path.join(problem_dir, ".autoanything", "history.db")
    if not os.path.exists(db_path):
        click.echo("No evaluation history yet.")
        return

    conn = _init_db(db_path)
    output_path = os.path.join(problem_dir, "leaderboard.md")
    _export(conn, output_path, direction=config.score.direction)
    conn.close()
    click.echo(f"Leaderboard written to {output_path}")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
@click.option("--baseline-only", is_flag=True, help="Establish baseline and exit.")
@click.option("--push", is_flag=True, help="Push results to origin.")
def evaluate(problem_dir, baseline_only, push):
    """Start the polling evaluator (watches for proposal branches)."""
    click.echo("Use 'python evaluator/evaluate.py' for now — full CLI integration in Phase 2.")
    sys.exit(1)


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
@click.option("--port", default=8000, help="Port (default: 8000).")
def serve(problem_dir, port):
    """Start the webhook server."""
    click.echo("Use 'python evaluator/server.py' for now — full CLI integration in Phase 2.")
    sys.exit(1)
