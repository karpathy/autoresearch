"""Problem configuration — load and validate problem.yaml.

Replaces the old string-matching parsers (load_direction, load_score_name)
with proper PyYAML parsing and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ValidationError(Exception):
    """Raised when problem.yaml is missing required fields or has invalid values."""


@dataclass
class ScoreConfig:
    """Scoring configuration from problem.yaml."""
    score_name: str
    direction: str
    description: str = ""
    timeout: int = 900
    script: str = "scoring/score.sh"
    bounded: bool = False

    @property
    def name(self) -> str:
        """The metric key in score.sh JSON output."""
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
    context: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    @property
    def mutable(self) -> list[str]:
        """Backward compatibility alias for state."""
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

    # Required: state (with backward compat for 'mutable')
    state = data.get("state") or data.get("mutable")
    if not state or (isinstance(state, list) and len(state) == 0):
        raise ValidationError(
            "problem.yaml: 'state' is required (non-empty list of mutable file paths)"
        )
    if not isinstance(state, list):
        raise ValidationError("problem.yaml: 'state' must be a list")

    # Required: score section
    score_data = data.get("score")
    if not score_data or not isinstance(score_data, dict):
        raise ValidationError("problem.yaml: 'score' section is required")

    if not score_data.get("name"):
        raise ValidationError("problem.yaml: 'score.name' is required")

    if not score_data.get("direction"):
        raise ValidationError("problem.yaml: 'score.direction' is required")

    direction = score_data["direction"]
    if direction not in ("minimize", "maximize"):
        raise ValidationError(
            f"problem.yaml: 'score.direction' must be 'minimize' or 'maximize', "
            f"got '{direction}'"
        )

    # Determine score script with fallback
    script = score_data.get("script", "scoring/score.sh")
    if script == "scoring/score.sh" and not (path / script).exists():
        fallback = "evaluator/score.sh"
        if (path / fallback).exists():
            script = fallback

    score_config = ScoreConfig(
        score_name=score_data["name"],
        direction=direction,
        description=score_data.get("description", ""),
        timeout=score_data.get("timeout", 900),
        script=script,
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

    # Context (with backward compat for 'readonly')
    context = data.get("context") or data.get("readonly") or []

    # Constraints
    constraints = data.get("constraints", [])

    return ProblemConfig(
        name=data["name"],
        description=data.get("description", ""),
        state=state,
        score=score_config,
        git=git_config,
        context=context,
        constraints=constraints,
    )
