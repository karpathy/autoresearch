"""CLI entry point — click-based command-line interface.

Provides commands: init, validate, score, evaluate, serve, history, leaderboard.
"""

import os
import subprocess
import sys
from importlib import resources

import click

from autoanything.problem import load_problem, ValidationError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TEMPLATES_PKG = "autoanything.templates"


def _load_template(filename: str) -> str:
    """Load a template file from the autoanything.templates package."""
    return resources.files(_TEMPLATES_PKG).joinpath(filename).read_text()


def _render_template(filename: str, **kwargs: str) -> str:
    """Load a template and substitute {{key}} placeholders."""
    text = _load_template(filename)
    for key, value in kwargs.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _resolve_db_path(problem_dir: str, db: str | None) -> str:
    """Resolve the database path from --db flag or default location."""
    if db:
        return db
    new_path = os.path.join(problem_dir, ".autoanything", "history.db")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    return new_path


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main():
    """AutoAnything — autonomous optimization via AI agents."""


@main.command()
@click.argument("name")
@click.option("--dir", "parent_dir", default=".", help="Parent directory for the new problem.")
@click.option("--direction", default="minimize", type=click.Choice(["minimize", "maximize"]),
              help="Score direction.")
def init(name, parent_dir, direction):
    """Scaffold a new problem directory."""
    problem_dir = os.path.join(parent_dir, name)
    if os.path.exists(problem_dir):
        click.echo(f"Error: directory '{problem_dir}' already exists.", err=True)
        sys.exit(1)

    subs = {"name": name, "direction": direction}

    os.makedirs(problem_dir)
    os.makedirs(os.path.join(problem_dir, "state"))
    os.makedirs(os.path.join(problem_dir, "context"))
    os.makedirs(os.path.join(problem_dir, "scoring"))
    os.makedirs(os.path.join(problem_dir, ".autoanything"))

    # Write files from templates
    with open(os.path.join(problem_dir, "problem.yaml"), "w") as f:
        f.write(_render_template("problem.yaml", **subs))

    with open(os.path.join(problem_dir, "state", "solution.py"), "w") as f:
        f.write(_load_template("solution.py"))

    with open(os.path.join(problem_dir, "scoring", "score.py"), "w") as f:
        f.write(_load_template("score.py"))

    with open(os.path.join(problem_dir, "agent_instructions.md"), "w") as f:
        f.write(_render_template("agent_instructions.md", **subs))

    with open(os.path.join(problem_dir, ".gitignore"), "w") as f:
        f.write(_load_template("gitignore"))

    # Initialize a git repo
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=problem_dir, capture_output=True, check=True,
    )

    click.echo(f"Created problem '{name}' in {problem_dir}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  cd {problem_dir}")
    click.echo("  # Edit problem.yaml — describe the problem, set constraints")
    click.echo("  # Edit state/solution.py — set up the initial state")
    click.echo("  # Edit scoring/score.py — implement your score() function")
    click.echo("  autoanything validate    # check everything is wired up")
    click.echo("  autoanything score       # run scoring once as a sanity check")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory to validate.")
def validate(problem_dir):
    """Check that the problem directory is well-formed."""
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

    # Check state/ directory exists and is non-empty
    state_dir = os.path.join(problem_dir, "state")
    if not os.path.isdir(state_dir):
        errors.append("state/ directory not found")
    elif not os.listdir(state_dir):
        errors.append("state/ directory is empty")

    # Check scoring/score.py
    score_py = os.path.join(problem_dir, "scoring", "score.py")
    if not os.path.exists(score_py):
        errors.append("Score script not found: scoring/score.py")

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
    """Run scoring once and print the result."""
    from autoanything.scoring import run_score as _run_score

    try:
        config = load_problem(problem_dir)
    except (FileNotFoundError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    score_py = os.path.join(problem_dir, "scoring", "score.py")
    if not os.path.exists(score_py):
        click.echo("Error: Score script not found: scoring/score.py", err=True)
        sys.exit(1)

    score_val, metrics, duration, error = _run_score(
        problem_dir, score_name=config.score.name, timeout=config.score.timeout,
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
@click.option("--db", default=None, help="Path to history database.")
def history(problem_dir, db):
    """Print evaluation history."""
    from autoanything.history import init_db as _init_db

    db_path = _resolve_db_path(problem_dir, db)
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
@click.option("--db", default=None, help="Path to history database.")
def leaderboard(problem_dir, db):
    """Regenerate leaderboard.md from history."""
    from autoanything.history import init_db as _init_db
    from autoanything.leaderboard import export_leaderboard as _export

    try:
        config = load_problem(problem_dir)
    except (FileNotFoundError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    db_path = _resolve_db_path(problem_dir, db)
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
@click.option("--db", default=None, help="Path to history database.")
@click.option("-o", "--output", default=None, help="Output PNG path (default: <db_dir>/progress.png).")
@click.option("--title", default=None, help="Custom chart title.")
@click.option("--direction", default=None,
              type=click.Choice(["minimize", "maximize"]),
              help="Score direction (auto-detected from problem.yaml if available).")
@click.option("--score-label", default=None, help="Y-axis label (default: metric name or 'Score').")
def plot(problem_dir, db, output, title, direction, score_label):
    """Generate a progress chart from evaluation history."""
    from autoanything.plotting import generate_chart

    db_path = _resolve_db_path(problem_dir, db)
    if not os.path.exists(db_path):
        click.echo("No evaluation history yet.", err=True)
        sys.exit(1)

    # Auto-detect direction and score label from problem.yaml if available
    if direction is None or score_label is None:
        try:
            config = load_problem(problem_dir)
            if direction is None:
                direction = config.score.direction
            if score_label is None:
                score_label = config.score.description or config.score.name
        except Exception:
            if direction is None:
                direction = "minimize"
            if score_label is None:
                score_label = "Score"

    if output is None:
        output = os.path.join(os.path.dirname(db_path), "progress.png")

    try:
        generate_chart(db_path, output, title, direction, score_label)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"Chart saved to {output}")


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
@click.option("--baseline-only", is_flag=True, help="Establish baseline and exit.")
@click.option("--push", is_flag=True, help="Push results to origin.")
@click.option("--poll-interval", default=30, help="Seconds between polls (default: 30).")
@click.option("--db", default=None, help="Path to history database.")
def evaluate(problem_dir, baseline_only, push, poll_interval, db):
    """Start the polling evaluator (watches for proposal branches)."""
    from autoanything.evaluator import run_evaluator

    try:
        config = load_problem(problem_dir)
    except FileNotFoundError:
        click.echo("Error: No problem.yaml found. Set up a problem first.", err=True)
        sys.exit(1)
    except ValidationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    db_path = _resolve_db_path(problem_dir, db)

    run_evaluator(
        problem_dir=problem_dir,
        config=config,
        db_path=db_path,
        baseline_only=baseline_only,
        push=push,
        poll_interval=poll_interval,
    )


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
@click.option("--agent", "-a", required=True, help="Shell command to run as the agent.")
@click.option("--iterations", "-n", default=None, type=int,
              help="Max iterations (default: unlimited).")
@click.option("--max-crashes", default=5, help="Stop after N consecutive crashes (default: 5).")
@click.option("--db", default=None, help="Path to history database.")
def run(problem_dir, agent, iterations, max_crashes, db):
    """Run the local optimization loop with an agent command.

    The agent command runs in the problem directory and should modify only the
    state files. Scoring is hidden from the agent during execution. The
    framework handles branching, scoring, merging improvements, and updating
    the leaderboard.

    Examples:

        autoanything run -a "./optimize.sh"

        autoanything run -a "python my_agent.py" -n 50

        autoanything run -a "claude -p 'improve the solution'" -n 10
    """
    from autoanything.runner import run_local

    try:
        config = load_problem(problem_dir)
    except FileNotFoundError:
        click.echo("Error: No problem.yaml found. Set up a problem first.", err=True)
        sys.exit(1)
    except ValidationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    db_path = _resolve_db_path(problem_dir, db)

    run_local(
        problem_dir=problem_dir,
        config=config,
        db_path=db_path,
        agent_command=agent,
        max_iterations=iterations,
        max_consecutive_crashes=max_crashes,
    )


_CLAUDE_PROMPT = """
You are optimizing a problem.
Read problem.yaml, context/, and agent_instructions.md to understand the task.
Check leaderboard.md and history.md if they exist for context on past attempts and how they performed.
The ONLY files that you can modify are in state/. Do NOT modify anything outside of the state/ directory.
Be creative and try a different approach than previous attempts."""

_CLAUDE_DEFAULTS = {
    "rastrigin": 10,
    "tsp": 10,
    "packing": 10,
    "fib": 3,
}

_DEMO_DEFAULTS = {
    "rastrigin": 20,
    "tsp": 20,
    "packing": 20,
    "fib": 5,
}


@main.command(name="try")
@click.argument("problem", type=click.Choice(["rastrigin", "tsp", "packing", "fib"]))
@click.option("--dir", "target_dir", default=None,
              help="Target directory (default: /tmp/<problem>).")
@click.option("-n", "--iterations", default=None, type=int,
              help="Number of iterations (auto-selected if omitted).")
@click.option("-a", "--agent", "agent_override", default=None,
              help="Custom agent command.")
@click.option("--claude", "use_claude", is_flag=True,
              help="Use Claude as the agent.")
def try_problem(problem, target_dir, iterations, agent_override, use_claude):
    """Try an example problem with a built-in demo agent.

    Sets up a fresh copy of the example, runs optimization iterations,
    generates a progress chart, and opens it.

    Examples:

        autoanything try rastrigin

        autoanything try fib --claude

        autoanything try tsp -a "python my_agent.py" -n 10
    """
    import shutil
    import platform

    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "examples")

    # Resolve source — handle both dev (source tree) and installed (package resources)
    source_dir = os.path.join(examples_dir, problem)
    if not os.path.isdir(source_dir):
        click.echo(f"Error: example '{problem}' not found at {source_dir}", err=True)
        sys.exit(1)

    if target_dir is None:
        target_dir = os.path.join("/tmp", problem)

    # Clean up previous run
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Copy example
    shutil.copytree(source_dir, target_dir)

    # Write .gitignore
    with open(os.path.join(target_dir, ".gitignore"), "w") as f:
        f.write("scoring/\n.autoanything/\n__pycache__/\n*.pyc\n.DS_Store\n")

    # Init git repo
    subprocess.run(["git", "init", "-b", "main"], cwd=target_dir,
                   capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=target_dir,
                   capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "autoanything@example.com"],
                   cwd=target_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "autoanything"],
                   cwd=target_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target_dir,
                   capture_output=True, check=True)

    # Write demo agent script
    agent_script = os.path.join(target_dir, ".autoanything", "agent.py")
    os.makedirs(os.path.dirname(agent_script), exist_ok=True)

    agents = {
        "rastrigin": '''\
import random, json, os

# Read current best solution
try:
    exec(open("state/solution.py").read())
    current = x
except Exception:
    current = [0.0] * 10

# Perturb each value with decreasing step size
iteration = int(os.environ.get("AUTOANYTHING_ITERATION", "1"))
step = max(0.5, 3.0 / (1 + iteration * 0.1))
vals = [v + random.gauss(0, step) for v in current]
vals = [max(-5.12, min(5.12, v)) for v in vals]

with open("state/solution.py", "w") as f:
    f.write("x = " + repr(vals) + "\\n")
''',
        "tsp": '''\
import random

# Read current tour
exec(open("state/tour.py").read())
current = list(tour)

# Apply random 2-opt swap
n = len(current)
i, j = sorted(random.sample(range(n), 2))
current[i:j+1] = reversed(current[i:j+1])

with open("state/tour.py", "w") as f:
    f.write("tour = " + repr(current) + "\\n")
''',
        "packing": '''\
import random

# Read current placements
exec(open("state/packing.py").read())
current = list(placements)

# Pick a random rectangle and nudge its position or toggle rotation
i = random.randint(0, len(current) - 1)
x, y, rotated = current[i]
x = max(0, x + random.randint(-10, 10))
y = max(0, y + random.randint(-10, 10))
if random.random() < 0.3:
    rotated = not rotated
current[i] = (x, y, rotated)

with open("state/packing.py", "w") as f:
    f.write("placements = " + repr(current) + "\\n")
''',
        "fib": '''\
# Try memoization, then iterative
code = open("state/fib.py").read()

if "_cache" not in code and "a, b" not in code:
    # First optimization: memoization
    lines = [
        "def fib(n, _cache={0: 0, 1: 1}):",
        "    if n in _cache:",
        "        return _cache[n]",
        "    _cache[n] = fib(n - 1, _cache) + fib(n - 2, _cache)",
        "    return _cache[n]",
    ]
elif "_cache" in code:
    # Second optimization: iterative
    lines = [
        "def fib(n):",
        "    if n <= 1:",
        "        return n",
        "    a, b = 0, 1",
        "    for _ in range(n):",
        "        a, b = b, a + b",
        "    return a",
    ]
else:
    lines = None

if lines:
    with open("state/fib.py", "w") as f:
        f.write("\\n".join(lines) + "\\n")
''',
    }

    with open(agent_script, "w") as f:
        f.write(agents[problem])

    # Determine agent command
    if agent_override:
        agent_command = agent_override
        agent_label = agent_override.split()[0]
        if iterations is None:
            iterations = _DEMO_DEFAULTS[problem]
    elif use_claude:
        agent_command = f"claude -p '{_CLAUDE_PROMPT}' --dangerously-skip-permissions"
        agent_label = "claude"
        if iterations is None:
            iterations = _CLAUDE_DEFAULTS[problem]
    else:
        agent_command = f"python {agent_script}"
        agent_label = "demo agent"
        if iterations is None:
            iterations = _DEMO_DEFAULTS[problem]

    click.echo(f"Set up {problem} in {target_dir}")
    click.echo(f"Running {iterations} iterations with {agent_label}...\n")

    # Run the optimization loop
    try:
        config = load_problem(target_dir)
    except (FileNotFoundError, ValidationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    db_path = _resolve_db_path(target_dir, None)

    from autoanything.runner import run_local
    run_local(
        problem_dir=target_dir,
        config=config,
        db_path=db_path,
        agent_command=agent_command,
        max_iterations=iterations,
    )

    # Generate chart
    from autoanything.plotting import generate_chart
    chart_path = os.path.join(target_dir, ".autoanything", "progress.png")
    try:
        generate_chart(
            db_path, chart_path,
            title=f"{problem} — demo run",
            direction=config.score.direction,
            score_label=config.score.description or config.score.name,
        )
        click.echo(f"\nChart saved to {chart_path}")
    except Exception as e:
        click.echo(f"Chart generation failed: {e}", err=True)
        return

    # Try to open the chart
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", chart_path], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", chart_path], check=False)
        elif system == "Windows":
            os.startfile(chart_path)
    except Exception:
        pass


@main.command()
@click.option("--dir", "problem_dir", default=".", help="Problem directory.")
@click.option("--port", default=8000, help="Port (default: 8000).")
@click.option("--host", default="0.0.0.0", help="Host (default: 0.0.0.0).")
@click.option("--push", is_flag=True, help="Push leaderboard updates to origin.")
@click.option("--db", default=None, help="Path to history database.")
def serve(problem_dir, port, host, push, db):
    """Start the webhook server."""
    import logging

    from autoanything.history import init_db as _init_db, get_incumbent as _get_incumbent
    from autoanything.server import create_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    slogger = logging.getLogger("autoanything.server")

    webhook_secret = os.environ.get("WEBHOOK_SECRET")
    if not webhook_secret:
        slogger.warning(
            "WEBHOOK_SECRET not set — accepting all webhook requests without verification"
        )

    db_path = _resolve_db_path(problem_dir, db)

    # Require an existing baseline
    conn = _init_db(db_path)
    incumbent = _get_incumbent(conn)
    conn.close()
    if incumbent is None:
        click.echo(
            "Error: No baseline found. Run 'autoanything evaluate --baseline-only' first.",
            err=True,
        )
        sys.exit(1)

    try:
        config = load_problem(problem_dir)
    except Exception:
        config = None

    slogger.info("Incumbent: %.6f (%s)", incumbent["score"], incumbent["commit_sha"][:7])
    slogger.info("Push: %s", "enabled" if push else "disabled")
    if config:
        slogger.info("Base branch: %s", config.git.base_branch)
    slogger.info("Starting server on %s:%d", host, port)

    app = create_app(
        problem_dir=problem_dir,
        webhook_secret=webhook_secret,
        db_path=db_path,
        push=push,
    )

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
