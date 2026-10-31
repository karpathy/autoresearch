"""Install this skill into a harness directory.

    python install.py --target cursor
    python install.py --target all
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

TARGETS = {
    "agents": Path.home() / ".agents" / "skills" / "autoresearch",
    "cursor": Path.home() / ".cursor" / "skills" / "autoresearch",
    "claude": Path.home() / ".claude" / "skills" / "autoresearch",
    "codex": Path.home() / ".codex" / "skills" / "autoresearch",
}
_MANIFEST_NAME = "skill.manifest.json"
_CACHE_PARTS = {"__pycache__", ".pytest_cache"}


class InstallError(ValueError):
    """The package or destination is not safe to update automatically."""


def _integrity_module(source: Path):
    module_path = source / "scripts" / "autoresearch_runtime" / "integrity.py"
    spec = importlib.util.spec_from_file_location("_autoresearch_install_integrity", module_path)
    if spec is None or spec.loader is None:
        raise InstallError("package integrity verifier is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise InstallError("manifest contains an unsafe file path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InstallError("manifest contains an unsafe file path")
    return value


_REQUIRED_PACKAGE_FILES = (
    "SKILL.md",
    "VERSION",
    "scripts/autoresearch.py",
    "scripts/autoresearch_runtime/__init__.py",
    "scripts/autoresearch_runtime/integrity.py",
)


def _source_manifest(source: Path) -> tuple[dict[str, Any], Any]:
    for relative in _REQUIRED_PACKAGE_FILES:
        if not (source / relative).is_file():
            raise InstallError(f"source package is missing required file: {relative}")
    integrity = _integrity_module(source)
    return integrity.manifest_payload(source), integrity.check_skill


def _write_manifest(stage: Path, manifest: dict[str, Any]) -> None:
    (stage / _MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _installed_files(destination: Path) -> set[str]:
    """Validate existing ownership records and return the files they manage."""
    manifest_path = destination / _MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise InstallError("existing target has no trustworthy package manifest; preserve it and resolve manually")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InstallError(f"existing target manifest is invalid: {error}") from error
    if not isinstance(manifest, dict):
        raise InstallError("existing target manifest is invalid; preserve it and resolve manually")
    if manifest.get("schema") == 1:
        raise InstallError(
            "existing schema 1 manifest cannot verify installer and reference files; "
            "back up the target and use a clean install directory"
        )
    if (manifest.get("schema") != 2
            or not isinstance(manifest.get("files"), dict)
            or not isinstance(manifest.get("sha256"), str)):
        raise InstallError("existing target manifest is invalid; preserve it and resolve manually")

    digest = hashlib.sha256()
    managed: set[str] = set()
    for raw_relative, expected_hash in manifest["files"].items():
        relative = _safe_relative(raw_relative)
        target = destination.joinpath(*PurePosixPath(relative).parts)
        if target.is_symlink() or not target.is_file():
            raise InstallError(f"managed file is missing or linked: {relative}")
        content = target.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise InstallError(f"managed file was changed: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(actual_hash))
        managed.add(relative)
    if digest.hexdigest() != manifest["sha256"]:
        raise InstallError("existing target manifest digest is inconsistent")
    return managed


def _contains_link_parent(root: Path, relative: str) -> bool:
    current = root
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _generated_path(parent: Path, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def _remove_generated(path: Path, parent: Path, prefix: str) -> None:
    # Restrict recursive removal to a unique sibling created by this installer.
    if path.parent.resolve() != parent.resolve() or not path.name.startswith(prefix):
        raise InstallError("refusing to remove a path outside the install transaction")
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _prepare_install(source: Path, destination: Path, manifest: dict[str, Any], check_skill: Any,
                     *, link: bool = False) -> dict[str, Any]:
    source = source.resolve()
    requested_destination = Path(destination).expanduser()
    parent = requested_destination.parent.resolve()
    destination = parent / requested_destination.name
    if (destination == source or destination.is_relative_to(source) or
            source.is_relative_to(destination)):
        raise InstallError("install target overlaps the source package")
    parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if link:
            try:
                link_target = Path(os.readlink(destination))
                if not link_target.is_absolute():
                    link_target = destination.parent / link_target
                same_source = os.path.normcase(str(link_target.resolve())) == os.path.normcase(str(source))
            except (OSError, RuntimeError):
                same_source = False
            if same_source:
                return {"destination": destination, "stage": None, "backup": None, "already_linked": True}
        raise InstallError(f"target is a symbolic link; resolve manually: {destination}")
    if destination.exists() and not destination.is_dir():
        raise InstallError(f"target exists and is not a directory: {destination}")

    managed_new = {_safe_relative(name) for name in manifest["files"]}
    old_managed: set[str] = set()
    if destination.exists():
        old_managed = _installed_files(destination)
        managed_directories = {
            parent.as_posix()
            for relative in old_managed | managed_new
            for parent in PurePosixPath(relative).parents
            if parent.as_posix() != "."
        }
        for existing in destination.rglob("*"):
            if existing.is_symlink():
                relative = existing.relative_to(destination).as_posix()
                raise InstallError(f"unmanaged linked target content conflicts with update: {relative}")
            relative = existing.relative_to(destination).as_posix()
            parts = PurePosixPath(relative).parts
            if existing.is_dir():
                if _CACHE_PARTS.intersection(parts):
                    continue
                if relative not in managed_directories:
                    raise InstallError(f"unmanaged target directory conflicts with update: {relative}")
                continue
            if not existing.is_file():
                raise InstallError(f"non-file target content conflicts with update: {relative}")
            if relative == _MANIFEST_NAME or relative in old_managed:
                continue
            if _CACHE_PARTS.intersection(parts):
                continue
            raise InstallError(f"unmanaged target content conflicts with update: {relative}")
        for relative in managed_new:
            if _contains_link_parent(destination, relative):
                raise InstallError(f"target contains a linked parent for managed file: {relative}")
            existing = destination.joinpath(*PurePosixPath(relative).parts)
            if relative not in old_managed and existing.exists():
                source_file = source.joinpath(*PurePosixPath(relative).parts)
                if existing.is_symlink() or not existing.is_file() or existing.read_bytes() != source_file.read_bytes():
                    raise InstallError(f"unmanaged target file conflicts with package: {relative}")

    stage_prefix = f".{destination.name}.autoresearch-stage-"
    stage = _generated_path(parent, stage_prefix)
    try:
        if destination.exists():
            shutil.copytree(destination, stage, dirs_exist_ok=True, symlinks=True)
            for relative in old_managed - managed_new:
                obsolete = stage.joinpath(*PurePosixPath(relative).parts)
                if obsolete.is_file() and not obsolete.is_symlink():
                    obsolete.unlink()
        elif link:
            _remove_generated(stage, parent, stage_prefix)
            stage.symlink_to(source, target_is_directory=True)
            return {"destination": destination, "stage": stage, "backup": None,
                    "already_linked": False, "link": True}

        if not link:
            for relative in sorted(managed_new):
                source_file = source.joinpath(*PurePosixPath(relative).parts)
                staged_file = stage.joinpath(*PurePosixPath(relative).parts)
                staged_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, staged_file)
            _write_manifest(stage, manifest)
            result = check_skill(stage)
            if not result.get("ok"):
                raise InstallError(f"staged package failed integrity verification: {result.get('error', 'files differ')}")
            return {"destination": destination, "stage": stage, "backup": None,
                    "already_linked": False, "link": False}
        raise InstallError("cannot convert an existing managed installation into a link")
    except BaseException:
        if stage.exists() or stage.is_symlink():
            _remove_generated(stage, parent, stage_prefix)
        raise


def _activate(plan: dict[str, Any], check_skill: Any) -> None:
    if plan.get("already_linked"):
        return
    destination: Path = plan["destination"]
    stage: Path = plan["stage"]
    backup_prefix = f".{destination.name}.autoresearch-backup-"
    backup = _generated_path(destination.parent, backup_prefix)
    _remove_generated(backup, destination.parent, backup_prefix)
    plan["backup"] = backup
    had_destination = destination.exists()
    plan["had_destination"] = had_destination
    backed_up = False
    try:
        if had_destination:
            os.replace(destination, backup)
            backed_up = True
        os.replace(stage, destination)
        if plan.get("link"):
            integrity = _integrity_module(destination.resolve())
            result = integrity.check_skill(destination.resolve())
        else:
            result = check_skill(destination)
        if not result.get("ok"):
            raise InstallError("installed package failed integrity verification")
    except BaseException:
        failed_prefix = f".{destination.name}.autoresearch-failed-"
        failed: Path | None = None
        if destination.exists() or destination.is_symlink():
            if not backed_up and had_destination:
                raise
            failed = _generated_path(destination.parent, failed_prefix)
            _remove_generated(failed, destination.parent, failed_prefix)
            os.replace(destination, failed)
        if backed_up and backup.exists():
            os.replace(backup, destination)
        if failed is not None and (not had_destination or destination.exists()):
            _remove_generated(failed, destination.parent, failed_prefix)
        raise


def _cleanup_plan(plan: dict[str, Any], *, keep_backup: bool = False) -> None:
    destination: Path = plan["destination"]
    parent = destination.parent
    stage = plan.get("stage")
    if stage is not None and (stage.exists() or stage.is_symlink()):
        _remove_generated(stage, parent, f".{destination.name}.autoresearch-stage-")
    backup = plan.get("backup")
    if not keep_backup and backup is not None and backup.exists():
        _remove_generated(backup, parent, f".{destination.name}.autoresearch-backup-")


def install(target: str, *, link: bool = False) -> Path:
    source = Path(__file__).resolve().parent
    manifest, check_skill = _source_manifest(source)
    names = list(TARGETS) if target == "all" else [target]
    plans: list[dict[str, Any]] = []
    try:
        for name in names:
            plans.append(_prepare_install(source, TARGETS[name], manifest, check_skill, link=link))
        activated: list[dict[str, Any]] = []
        try:
            for plan in plans:
                _activate(plan, check_skill)
                if not plan.get("already_linked"):
                    activated.append(plan)
        except BaseException:
            for plan in reversed(activated):
                destination: Path = plan["destination"]
                backup: Path | None = plan.get("backup")
                had_destination = bool(plan.get("had_destination"))
                failed: Path | None = None
                failed_prefix = f".{destination.name}.autoresearch-failed-"
                if destination.exists() or destination.is_symlink():
                    failed = _generated_path(destination.parent, failed_prefix)
                    _remove_generated(failed, destination.parent, failed_prefix)
                    os.replace(destination, failed)
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                if failed is not None and (
                    not had_destination or destination.exists() or destination.is_symlink()
                ):
                    _remove_generated(failed, destination.parent, failed_prefix)
            raise
        for plan in plans:
            _cleanup_plan(plan)
        return plans[-1]["destination"]
    except BaseException:
        for plan in plans:
            _cleanup_plan(plan, keep_backup=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the autoresearch skill")
    parser.add_argument("--target", required=True, choices=[*TARGETS, "all"])
    parser.add_argument("--link", action="store_true")
    args = parser.parse_args(argv)
    try:
        path = install(args.target, link=args.link)
    except (InstallError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
