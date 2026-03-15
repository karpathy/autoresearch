"""Problem configuration — load and validate problem.yaml.

State files are discovered from the state/ directory by convention.
Scoring always uses scoring/score.py (no script path in config).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ValidationError(Exception):
    """Raised when problem.yaml is missing required fields or has invalid values."""


# Files/dirs to skip when discovering state files
_STATE_IGNORE = {"__pycache__", ".pyc", ".pyo", "__init__.py"}


def get_state_files(problem_dir) -> list[str]:
    """Discover mutable files by listing the state/ directory.

    Returns paths relative to problem_dir, e.g. ["state/solution.py"].
    Excludes __pycache__, .pyc, .pyo files.
    """
    state_dir = Path(problem_dir) / "state"
    if not state_dir.is_dir():
        return []

    result = []
    for root, dirs, files in os.walk(state_dir):
        # Prune __pycache__ dirs
        dirs[:] = [d for d in dirs if d not in _STATE_IGNORE]
        for f in files:
            if f in _STATE_IGNORE or f.endswith((".pyc", ".pyo")):
                continue
            full = Path(root) / f
            rel = full.relative_to(problem_dir)
            result.append(str(rel))

    return sorted(result)


@dataclass
class ScoreConfig:
    """Scoring configuration from problem.yaml."""
    score_name: str
    direction: str
    description: str = ""
    timeout: int = 900
    bounded: bool = False

    @property
    def name(self) -> str:
        """The metric key in score output."""
        return self.score_name


@dataclass
class GitConfig:
    """Git configuration from problem.yaml."""
    base_branch: str = "main"
    proposal_pattern: str = "proposals/*"


@dataclass
class ProblemConfig:
    """Full problem configuration loaded from problem.yaml."""
    name: str
    description: str
    state: list[str]
    score: ScoreConfig
    git: GitConfig = field(default_factory=GitConfig)
    constraints: list[str] = field(default_factory=list)

    @property
    def mutable(self) -> list[str]:
        """Alias for state."""
        return self.state


def load_problem(path) -> ProblemConfig:
    """Load and validate problem.yaml from the given directory.

    Args:
        path: Path to the problem directory (string or Path).

    Returns:
        ProblemConfig with all fields populated (defaults applied).

    Raises:
        FileNotFoundError: If problem.yaml doesn't exist.
        ValidationError: If required fields are missing or invalid.
    """
    path = Path(path)
    yaml_path = path / "problem.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"No problem.yaml found in {path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValidationError("problem.yaml must be a YAML mapping")

    # Required: name
    if not data.get("name"):
        raise ValidationError("problem.yaml: 'name' is required")

    # State: from YAML if present, otherwise discover from state/ directory
    state = data.get("state") or data.get("mutable")
    if state is not None:
        if not isinstance(state, list):
            raise ValidationError("problem.yaml: 'state' must be a list")
    else:
        # Discover from state/ directory
        state = get_state_files(path)

    # Required: score section
    score_data = data.get("score")
    if not score_data or not isinstance(score_data, dict):
        raise ValidationError("problem.yaml: 'score' section is required")

    if not score_data.get("direction"):
        raise ValidationError("problem.yaml: 'score.direction' is required")

    direction = score_data["direction"]
    if direction not in ("minimize", "maximize"):
        raise ValidationError(
            f"problem.yaml: 'score.direction' must be 'minimize' or 'maximize', "
            f"got '{direction}'"
        )

    # score.name defaults to "score"
    score_name = score_data.get("name", "score")

    score_config = ScoreConfig(
        score_name=score_name,
        direction=direction,
        description=score_data.get("description", ""),
        timeout=score_data.get("timeout", 900),
        bounded=score_data.get("bounded", False),
    )

    # Git config
    git_data = data.get("git", {})
    if not isinstance(git_data, dict):
        git_data = {}
    git_config = GitConfig(
        base_branch=git_data.get("base_branch", "main"),
        proposal_pattern=git_data.get("proposal_pattern", "proposals/*"),
    )

    # Constraints
    constraints = data.get("constraints", [])

    return ProblemConfig(
        name=data["name"],
        description=data.get("description", ""),
        state=state,
        score=score_config,
        git=git_config,
        constraints=constraints,
    )
