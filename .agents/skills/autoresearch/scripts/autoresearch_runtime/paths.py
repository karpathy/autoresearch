"""Where lab state lives. Never inside the project being measured."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def resolve_home() -> Path:
    """User state directory. AUTORESEARCH_HOME overrides the default ~/.autoresearch."""
    override = os.environ.get("AUTORESEARCH_HOME")
    path = Path(override) if override else Path.home() / ".autoresearch"
    return path.resolve()


def _path_key(path: Path | str) -> str:
    """Return a path identity using the current platform's case rules."""
    resolved = str(Path(path).resolve())
    # Preserve Windows keys created by older releases, which used casefold().
    # POSIX paths stay case-sensitive because normcase() is a no-op there.
    return resolved.casefold() if os.name == "nt" else os.path.normcase(resolved)


def project_id(project: Path) -> str:
    resolved = project.resolve()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", resolved.name).strip("-").lower() or "project"
    digest = hashlib.sha256(_path_key(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def brief_path(project: Path, home: Path | None = None) -> Path:
    """Canonical brief file for a measured project under AUTORESEARCH_HOME."""
    return (home or resolve_home()).resolve() / "briefs" / project_id(project) / "brief.md"


def home_root_id(home: Path | None = None) -> str:
    home = (home or resolve_home()).resolve()
    return hashlib.sha256(_path_key(home).encode("utf-8")).hexdigest()[:16]


def project_directory(root: Path, project: Path) -> Path:
    """Directory that groups the labs of one measured project under a storage root."""
    name = re.sub(r"[^a-zA-Z0-9]+", "-", project.name).strip("-").lower() or "project"
    identity = hashlib.sha256(str(project.resolve()).casefold().encode("utf-8")).hexdigest()[:10]
    return root / "projects" / f"{name}-{identity}"
