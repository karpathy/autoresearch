"""Report data collector for autoresearch workflow reports.

Reads experiment artifacts and assembles them into structured data for Jinja2 templates.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class ExperimentEntry:
    """Single experiment result from results.tsv."""

    index: int  # experiment number (1-based)
    commit: str  # short commit hash
    metric_value: float  # the measured metric
    status: str  # "keep", "discard", or "crash"
    description: str  # one-line description
    delta: float | None = None  # change from previous kept value
    delta_pct: float | None = None  # percent change from previous kept


@dataclass
class MusingEntry:
    """Single musing from musings.md."""

    experiment_title: str
    rationale: str
    result: str
    learning: str


@dataclass
class ReportData:
    """Complete report data structure."""

    # Workflow metadata
    workflow_name: str
    workflow_description: str
    metric_name: str
    metric_direction: str  # "higher" or "lower"
    targets: list[dict[str, str]] = field(default_factory=list)  # [{path, description}]

    # Experiment data
    experiments: list[ExperimentEntry] = field(default_factory=list)
    baseline: ExperimentEntry | None = None
    best: ExperimentEntry | None = None

    # Aggregate stats
    total_experiments: int = 0
    kept_count: int = 0
    discarded_count: int = 0
    crash_count: int = 0
    improvement_pct: float = 0.0  # baseline to best, percentage

    # Musings
    musings: list[MusingEntry] = field(default_factory=list)

    # Timeline
    generated_at: str = ""  # ISO timestamp
    branch_name: str = ""  # git branch

    # Provenance
    git_log: list[dict[str, str]] = field(default_factory=list)  # [{hash, message, author, date}]


def load_workflow_yaml(workflow_dir: Path) -> dict[str, Any]:
    """Parse workflow.yaml and return workflow metadata.

    Args:
        workflow_dir: Path to workflow directory

    Returns:
        Dictionary with workflow metadata, or empty dict if file not found
    """
    yaml_path = workflow_dir / "workflow.yaml"

    if not yaml_path.exists():
        return {}

    if yaml is None:
        raise ImportError("PyYAML is required but not installed")

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except Exception as e:
        print(f"Warning: Failed to parse {yaml_path}: {e}")
        return {}


def load_results_tsv(workflow_dir: Path) -> list[ExperimentEntry]:
    """Parse results.tsv and compute deltas between experiments.

    Args:
        workflow_dir: Path to workflow directory

    Returns:
        List of ExperimentEntry objects, or empty list if file not found
    """
    results_path = workflow_dir / "results" / "results.tsv"

    if not results_path.exists():
        return []

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            # Find the metric column (first column that's not commit, status, description)
            if reader.fieldnames is None:
                return []

            metric_column = None
            for field in reader.fieldnames:
                if field not in ("commit", "status", "description"):
                    metric_column = field
                    break

            if metric_column is None:
                print(f"Warning: No metric column found in {results_path}")
                return []

            # Parse rows
            experiments: list[ExperimentEntry] = []
            last_kept_value: float | None = None

            for index, row in enumerate(reader, start=1):
                commit = row.get("commit", "")
                status = row.get("status", "")
                description = row.get("description", "")

                # Parse metric value
                metric_str = row.get(metric_column, "0")
                try:
                    metric_value = float(metric_str)
                except ValueError:
                    metric_value = 0.0

                # Compute delta from last kept value
                delta: float | None = None
                delta_pct: float | None = None

                if last_kept_value is not None and status == "keep":
                    delta = metric_value - last_kept_value
                    if last_kept_value != 0:
                        delta_pct = (delta / abs(last_kept_value)) * 100.0

                # Update last kept value
                if status == "keep":
                    last_kept_value = metric_value

                entry = ExperimentEntry(
                    index=index,
                    commit=commit,
                    metric_value=metric_value,
                    status=status,
                    description=description,
                    delta=delta,
                    delta_pct=delta_pct,
                )
                experiments.append(entry)

            return experiments

    except Exception as e:
        print(f"Warning: Failed to parse {results_path}: {e}")
        return []


def load_musings(workflow_dir: Path) -> list[MusingEntry]:
    """Parse musings.md and extract experiment sections.

    Args:
        workflow_dir: Path to workflow directory

    Returns:
        List of MusingEntry objects, or empty list if file not found
    """
    musings_path = workflow_dir / "results" / "musings.md"

    if not musings_path.exists():
        return []

    try:
        with open(musings_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by ## headers (experiment sections)
        sections = re.split(r"^## ", content, flags=re.MULTILINE)

        musings: list[MusingEntry] = []

        for section in sections[1:]:  # Skip first split (before first ##)
            lines = section.strip().split("\n")
            if not lines:
                continue

            # First line is the experiment title
            experiment_title = lines[0].strip()

            # Extract bold fields
            rationale_match = re.search(r"\*\*Rationale\*\*:\s*(.+?)(?=\n\*\*|\Z)", section, re.DOTALL)
            result_match = re.search(r"\*\*Result\*\*:\s*(.+?)(?=\n\*\*|\Z)", section, re.DOTALL)
            learning_match = re.search(r"\*\*Learning\*\*:\s*(.+?)(?=\n\*\*|\Z)", section, re.DOTALL)

            rationale = rationale_match.group(1).strip() if rationale_match else ""
            result = result_match.group(1).strip() if result_match else ""
            learning = learning_match.group(1).strip() if learning_match else ""

            # Clean up newlines within extracted text
            rationale = " ".join(rationale.split())
            result = " ".join(result.split())
            learning = " ".join(learning.split())

            entry = MusingEntry(
                experiment_title=experiment_title,
                rationale=rationale,
                result=result,
                learning=learning,
            )
            musings.append(entry)

        return musings

    except Exception as e:
        print(f"Warning: Failed to parse {musings_path}: {e}")
        return []


def load_git_log(workflow_dir: Path) -> list[dict[str, str]]:
    """Run git log for the workflow branch and return commit history.

    Args:
        workflow_dir: Path to workflow directory

    Returns:
        List of commit dictionaries, or empty list if git command fails
    """
    try:
        # Get current branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workflow_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        branch_name = branch_result.stdout.strip()

        # Get git log with custom format: hash|subject|author|date
        log_result = subprocess.run(
            ["git", "log", "--format=%H|%s|%an|%aI", "--", "."],
            cwd=workflow_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        commits: list[dict[str, str]] = []

        for line in log_result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|", maxsplit=3)
            if len(parts) != 4:
                continue

            commit_hash, message, author, date = parts
            commits.append(
                {
                    "hash": commit_hash[:7],  # Short hash
                    "full_hash": commit_hash,
                    "message": message,
                    "author": author,
                    "date": date,
                }
            )

        return commits

    except subprocess.CalledProcessError as e:
        print(f"Warning: Git command failed: {e}")
        return []
    except Exception as e:
        print(f"Warning: Failed to load git log: {e}")
        return []


def collect(workflow_dir: Path) -> ReportData:
    """Main entry point: assemble all data sources into ReportData.

    Args:
        workflow_dir: Path to workflow directory

    Returns:
        ReportData object with all collected data
    """
    workflow_dir = Path(workflow_dir)

    # Load workflow metadata
    workflow_yaml = load_workflow_yaml(workflow_dir)
    workflow_name = workflow_yaml.get("name", workflow_dir.name)
    workflow_description = workflow_yaml.get("description", "")

    metric = workflow_yaml.get("metric", {})
    metric_name = metric.get("name", "score")
    metric_direction = metric.get("direction", "higher")

    targets = workflow_yaml.get("targets", [])

    # Load experiments
    experiments = load_results_tsv(workflow_dir)

    # Find baseline (first entry) and best (highest/lowest based on direction)
    baseline = experiments[0] if experiments else None
    best: ExperimentEntry | None = None

    if experiments:
        if metric_direction == "higher":
            best = max(experiments, key=lambda e: e.metric_value)
        else:
            best = min(experiments, key=lambda e: e.metric_value)

    # Compute aggregate stats
    total_experiments = len(experiments)
    kept_count = sum(1 for e in experiments if e.status == "keep")
    discarded_count = sum(1 for e in experiments if e.status == "discard")
    crash_count = sum(1 for e in experiments if e.status == "crash")

    improvement_pct = 0.0
    if baseline and best and baseline.metric_value != 0:
        improvement = best.metric_value - baseline.metric_value
        improvement_pct = (improvement / abs(baseline.metric_value)) * 100.0

    # Load musings
    musings = load_musings(workflow_dir)

    # Load git log
    git_log = load_git_log(workflow_dir)

    # Get branch name
    branch_name = ""
    try:
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workflow_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        branch_name = branch_result.stdout.strip()
    except Exception:
        pass

    # Generate timestamp
    from datetime import timezone
    generated_at = datetime.now(timezone.utc).isoformat()

    return ReportData(
        workflow_name=workflow_name,
        workflow_description=workflow_description,
        metric_name=metric_name,
        metric_direction=metric_direction,
        targets=targets,
        experiments=experiments,
        baseline=baseline,
        best=best,
        total_experiments=total_experiments,
        kept_count=kept_count,
        discarded_count=discarded_count,
        crash_count=crash_count,
        improvement_pct=improvement_pct,
        musings=musings,
        generated_at=generated_at,
        branch_name=branch_name,
        git_log=git_log,
    )


def report_data_to_dict(data: ReportData) -> dict[str, Any]:
    """Convert ReportData to a dictionary for JSON serialization.

    Args:
        data: ReportData object

    Returns:
        Dictionary representation
    """
    return asdict(data)


if __name__ == "__main__":
    import json
    import sys

    # Test with a workflow directory argument
    if len(sys.argv) > 1:
        workflow_path = Path(sys.argv[1])
    else:
        # Default to exec-summarizer for testing
        workflow_path = Path(__file__).parent.parent.parent.parent / "workflows" / "exec-summarizer"

    if not workflow_path.exists():
        print(f"Error: Workflow directory not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Collecting report data from: {workflow_path}\n")

    # Collect data
    report_data = collect(workflow_path)

    # Convert to dict for JSON serialization
    data_dict = report_data_to_dict(report_data)

    # Print JSON
    print(json.dumps(data_dict, indent=2))
