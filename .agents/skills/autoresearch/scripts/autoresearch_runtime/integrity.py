"""Skill file fingerprints for doctor and run manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_MANIFEST_NAME = "skill.manifest.json"
_MANIFEST_SCHEMA = 2
_GENERATED_DIRS = {".git", ".gitnexus", "__pycache__", ".pytest_cache", ".mypy_cache",
                   ".ruff_cache", ".cache", "cache"}


def skill_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tracked_files(root: Path) -> list[Path]:
    """List shipped package files, excluding generated caches and the manifest itself."""
    files = [root / "SKILL.md", root / "VERSION", root / "install.py", root / "LICENSE"]
    for directory in ("scripts", "references", "resources", "assets", "templates"):
        base = root / directory
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                relative_parts = path.relative_to(root).parts
                if (path.is_file() and not _GENERATED_DIRS.intersection(relative_parts)
                        and not any(part.endswith(".egg-info") for part in relative_parts)
                        and path.suffix != ".pyc"):
                    files.append(path)
    return [path for path in files if path.is_file()]


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def read_version(root: Path | None = None) -> str:
    path = (root or skill_root()) / "VERSION"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def manifest_payload(root: Path | None = None) -> dict[str, Any]:
    root = (root or skill_root()).resolve()
    tracked = _tracked_files(root)
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in tracked}
    return {
        "schema": _MANIFEST_SCHEMA,
        "version": read_version(root),
        "sha256": _digest(root, tracked),
        "files": files,
    }


def manifest_path(root: Path | None = None) -> Path:
    return (root or skill_root()) / _MANIFEST_NAME


def skill_stamp(root: Path | None = None) -> tuple[str, str]:
    """Fingerprint the skill files on disk. A stored manifest is never read."""
    payload = manifest_payload(root)
    return str(payload["version"]), str(payload["sha256"])


def check_skill(root: Path | None = None) -> dict[str, Any]:
    """Report whether the files match. Without a manifest the tree is an unmanaged source checkout."""
    root = (root or skill_root()).resolve()
    expected_path = manifest_path(root)
    if not expected_path.exists() and not expected_path.is_symlink():
        try:
            current = manifest_payload(root)
        except (OSError, ValueError) as error:
            return {"ok": False, "modified": True, "managed": False,
                    "error": f"cannot inspect skill package: {error}", "version": read_version(root)}
        return {"ok": True, "modified": False, "managed": False,
                "version": current["version"], "current_sha256": current["sha256"]}
    return {**_verify_manifest(root, expected_path), "managed": True}


def _verify_manifest(root: Path, expected_path: Path) -> dict[str, Any]:
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"ok": False, "modified": True, "error": f"invalid skill manifest: {error}",
                "version": read_version(root)}
    if (not isinstance(expected, dict) or expected.get("schema") != _MANIFEST_SCHEMA
            or not isinstance(expected.get("version"), str)
            or not isinstance(expected.get("sha256"), str)
            or not isinstance(expected.get("files"), dict)):
        return {"ok": False, "modified": True, "error": "invalid skill manifest schema",
                "version": read_version(root)}
    for relative, value in expected["files"].items():
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            return {"ok": False, "modified": True, "error": "invalid skill manifest file entry",
                    "version": read_version(root)}
    try:
        current = manifest_payload(root)
    except (OSError, ValueError) as error:
        return {"ok": False, "modified": True, "error": f"cannot inspect skill package: {error}",
                "version": read_version(root)}
    modified = (expected.get("version") != current["version"]
                or expected.get("sha256") != current["sha256"]
                or expected.get("files") != current["files"])
    return {
        "ok": not modified,
        "modified": modified,
        "version": current["version"],
        "expected_sha256": expected.get("sha256"),
        "current_sha256": current["sha256"],
    }
