from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepoContext:
    framework: str
    package_manager: str
    scripts: dict[str, str]
    sensitive_paths: list[str]


def load_repo_context(repo_path: Path | str) -> RepoContext:
    repo_path = Path(repo_path)
    package_json = _load_package_json(repo_path / "package.json")
    scripts = _normalize_scripts(package_json.get("scripts", {}))
    framework = _detect_framework(package_json, repo_path)
    package_manager = _detect_package_manager(package_json, repo_path)
    sensitive_paths = [".env", ".env.local", ".env.production"]
    return RepoContext(
        framework=framework,
        package_manager=package_manager,
        scripts=scripts,
        sensitive_paths=sensitive_paths,
    )


def _load_package_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _normalize_scripts(raw_scripts: Any) -> dict[str, str]:
    if isinstance(raw_scripts, dict):
        return {k: str(v) for k, v in raw_scripts.items()}
    return {}


def _detect_framework(package_json: dict[str, Any], repo_path: Path) -> str:
    scripts = _normalize_scripts(package_json.get("scripts", {}))
    if any("next" in script.lower() for script in scripts.values()):
        return "nextjs"
    if (repo_path / "app").is_dir():
        return "nextjs"
    return "unknown"


def _detect_package_manager(package_json: dict[str, Any], repo_path: Path) -> str:
    if (repo_path / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (repo_path / "yarn.lock").exists():
        return "yarn"
    if (repo_path / "package-lock.json").exists():
        return "npm"
    package_manager_field = package_json.get("packageManager")
    if isinstance(package_manager_field, str):
        normalized = package_manager_field.split("@")[0].lower()
        if normalized in {"pnpm", "npm", "yarn"}:
            return normalized
    return "unknown"
