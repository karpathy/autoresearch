"""Read-only inspection and ZIP export for a legacy lab directory."""

from __future__ import annotations

import fnmatch
import hashlib
import heapq
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .errors import LabError


class LegacyArchiveError(LabError):
    """Safe, structured failure while reading or exporting a legacy lab."""


_REPARSE_POINT = 0x400
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_STATUSES = frozenset({
    "accepted", "aborted", "cancelled", "completed", "crash", "error",
    "failed", "incomparable", "invalid", "interrupted", "budget", "rejected",
    "running", "timeout",
})
_RUN_MANIFEST_WARNING = (
    "One or more legacy run manifests are unreadable or unsupported; details are omitted."
)
_MAX_RUN_MANIFEST_BYTES = 1024 * 1024
_MAX_RUN_SUMMARIES = 100
_CHUNK_SIZE = 1024 * 1024


def _error(code: str, message: str) -> LegacyArchiveError:
    return LegacyArchiveError(message, code=code)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _REPARSE_POINT)


def _absolute_lexical(path: Path | str) -> Path:
    try:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate
    except (OSError, TypeError, ValueError) as error:
        raise _error("E_CONFIG_INVALID", "Invalid path") from error


def _check_lexical_components(path: Path) -> None:
    """Reject links in the spelled path before any path resolution follows them."""
    anchor = Path(path.anchor)
    current = anchor
    if not anchor:
        raise _error("E_CONFIG_INVALID", "Path must be absolute")
    try:
        for part in path.parts[1:]:
            if part in {"", "."}:
                continue
            if part == "..":
                current = current.parent
                continue
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            if _is_link_or_reparse(info):
                raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Linked or reparse paths are not allowed")
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_CONFIG_INVALID", "Unable to inspect path") from error


def _safe_relative(name: str) -> str:
    """Return a portable ZIP name or reject names unsafe on common extractors."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(":" in part for part in path.parts)):
        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains an unsafe relative path")
    return path.as_posix()


def _file_signature(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(info.st_mode),
        int(info.st_size),
        int(getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000))),
        int(getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000))),
        int(getattr(info, "st_dev", 0)),
        int(getattr(info, "st_ino", 0)),
    )


def _hash_file(path: Path, relative: str) -> tuple[str, tuple[int, int, int, int, int, int]]:
    try:
        _check_lexical_components(path)
        before = path.lstat()
        if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or non-regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _file_signature(before) != _file_signature(opened):
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while True:
                    block = stream.read(_CHUNK_SIZE)
                    if not block:
                        break
                    digest.update(block)
            after_open = os.fstat(descriptor)
            after_path = path.lstat()
            if (_file_signature(opened) != _file_signature(after_open)
                    or _file_signature(after_open) != _file_signature(after_path)
                    or _is_link_or_reparse(after_path)):
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
            return digest.hexdigest(), _file_signature(after_path)
        finally:
            os.close(descriptor)
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed or could not be read") from error


def _snapshot_tree(root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    pending = [root]
    try:
        root_info = root.lstat()
        if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab root must be a regular directory")
        entries[""] = {
            "kind": "directory", "signature": _file_signature(root_info), "children_count": 0,
        }

        while pending:
            directory = pending.pop()
            _check_lexical_components(directory)
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
            directory_relative = directory.relative_to(root).as_posix() if directory != root else ""
            directory_relative = _safe_relative(directory_relative) if directory_relative else ""
            entries[directory_relative]["children_count"] = len(children)
            for child in children:
                _check_lexical_components(Path(child.path))
                relative = _safe_relative((Path(child.path).relative_to(root)).as_posix())
                info = Path(child.path).lstat()
                if _is_link_or_reparse(info):
                    raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
                if stat.S_ISDIR(info.st_mode):
                    signature = _file_signature(info)
                    entries[relative] = {
                        "kind": "directory", "signature": signature, "children_count": 0,
                    }
                    pending.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode):
                    digest, signature = _hash_file(Path(child.path), relative)
                    entries[relative] = {
                        "kind": "file",
                        "bytes": signature[1],
                        "sha256": digest,
                        "signature": signature,
                    }
                else:
                    raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a non-regular entry")

        after_root = root.lstat()
        if (_is_link_or_reparse(after_root)
                or _file_signature(after_root) != entries[""]["signature"]):
            raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
        for relative, entry in entries.items():
            if entry["kind"] != "directory":
                continue
            directory = root.joinpath(*PurePosixPath(relative).parts) if relative else root
            _check_lexical_components(directory)
            current_info = directory.lstat()
            if _is_link_or_reparse(current_info):
                raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
            if not stat.S_ISDIR(current_info.st_mode) or _file_signature(current_info) != entry["signature"]:
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
            observed_count = 0
            with os.scandir(directory) as iterator:
                for child in iterator:
                    observed_count += 1
                    child_path = Path(child.path)
                    _check_lexical_components(child_path)
                    child_relative = _safe_relative(child_path.relative_to(root).as_posix())
                    expected = entries.get(child_relative)
                    if expected is None:
                        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
                    child_info = child_path.lstat()
                    if _is_link_or_reparse(child_info):
                        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
                    if expected["kind"] == "directory":
                        if (not stat.S_ISDIR(child_info.st_mode)
                                or _file_signature(child_info)[4:] != expected["signature"][4:]):
                            raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
                    elif (not stat.S_ISREG(child_info.st_mode)
                          or _file_signature(child_info) != expected["signature"]):
                        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
            if observed_count != entry["children_count"]:
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being read")
        return entries
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed or could not be read") from error


def _parse_object(raw: bytes | bytearray, *, code: str, message: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as error:
        raise _error(code, message) from error
    if not isinstance(value, dict):
        raise _error(code, message)
    return value


def _state_directory(config: dict[str, Any], root: Path) -> Path:
    value = config.get("state_dir", ".")
    if not isinstance(value, str) or not value.strip():
        raise _error("E_CONFIG_INVALID", "Invalid state directory")
    if os.name != "nt" and PureWindowsPath(value).is_absolute():
        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Windows state paths cannot be represented on this platform")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = _absolute_lexical(candidate)
    _check_lexical_components(candidate)
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as error:
        raise _error("E_CONFIG_INVALID", "Unable to inspect state directory") from error
    if not resolved.is_relative_to(root):
        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "State directory must stay inside the lab root")
    if resolved.exists() and not resolved.is_dir():
        raise _error("E_CONFIG_INVALID", "Invalid state directory")
    return resolved


def _safe_run_value(value: Any, fallback: str) -> str:
    candidate = value if isinstance(value, str) else fallback
    if _SAFE_ID.fullmatch(candidate):
        return candidate
    if _SAFE_ID.fullmatch(fallback):
        return fallback
    return f"run-{hashlib.sha256(fallback.encode('utf-8', errors='replace')).hexdigest()[:12]}"


def _read_run_manifest(path: Path, expected_info: os.stat_result) -> dict[str, Any] | None:
    """Read at most one MiB, returning None for unsupported or unreadable manifests."""
    if int(expected_info.st_size) > _MAX_RUN_MANIFEST_BYTES:
        return None
    descriptor: int | None = None
    try:
        _check_lexical_components(path)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
        if not stat.S_ISREG(opened.st_mode):
            return None
        if (_file_signature(expected_info) != _file_signature(opened)
                or int(opened.st_size) > _MAX_RUN_MANIFEST_BYTES):
            return None

        data = bytearray()
        read_size = 0
        while read_size < _MAX_RUN_MANIFEST_BYTES:
            block = os.read(descriptor, min(_CHUNK_SIZE, _MAX_RUN_MANIFEST_BYTES - read_size))
            if not block:
                break
            data.extend(block)
            read_size += len(block)

        finished = os.fstat(descriptor)
        _check_lexical_components(path)
        current_path = path.lstat()
        if _is_link_or_reparse(current_path):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
        if (_file_signature(opened) != _file_signature(finished)
                or _file_signature(finished) != _file_signature(current_path)
                or read_size != int(finished.st_size)
                or int(finished.st_size) > _MAX_RUN_MANIFEST_BYTES):
            return None
        return _parse_object(data, code="E_CONFIG_INVALID", message="Invalid run manifest")
    except LegacyArchiveError as error:
        if error.code == "E_CONFIG_INVALID":
            return None
        raise
    except OSError:
        _check_lexical_components(path)
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _run_summaries(state_dir: Path) -> dict[str, Any]:
    run_root = state_dir / "runs"
    _check_lexical_components(run_root)
    try:
        run_root_info = run_root.lstat()
    except FileNotFoundError:
        return {
            "runs": [], "run_count": 0, "trial_count": 0,
            "running_run_count": 0, "truncated_run_count": 0, "warning_count": 0,
        }
    if _is_link_or_reparse(run_root_info):
        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
    if not stat.S_ISDIR(run_root_info.st_mode):
        raise _error("E_CONFIG_INVALID", "Invalid run directory")

    recent: list[tuple[int, str, dict[str, str | None]]] = []
    run_count = 0
    trial_count = 0
    running_run_count = 0
    warning_count = 0
    try:
        with os.scandir(run_root) as entries:
            for child in entries:
                run_dir = Path(child.path)
                _check_lexical_components(run_dir)
                try:
                    run_dir_info = run_dir.lstat()
                except FileNotFoundError:
                    continue
                if _is_link_or_reparse(run_dir_info):
                    raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
                if not stat.S_ISDIR(run_dir_info.st_mode):
                    continue

                run_count += 1
                manifest_path = run_dir / "manifest.json"
                _check_lexical_components(manifest_path)
                manifest: dict[str, Any] | None = None
                try:
                    manifest_info = manifest_path.lstat()
                except FileNotFoundError:
                    warning_count += 1
                except OSError:
                    warning_count += 1
                else:
                    if _is_link_or_reparse(manifest_info):
                        raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
                    if stat.S_ISREG(manifest_info.st_mode):
                        manifest = _read_run_manifest(manifest_path, manifest_info)
                    if manifest is None:
                        warning_count += 1

                if manifest is None:
                    raw_id = None
                    raw_status = None
                else:
                    raw_id = manifest.get("id")
                    raw_status = manifest.get("status")
                    if not isinstance(raw_status, str) or raw_status not in _RUN_STATUSES:
                        warning_count += 1
                run_id = _safe_run_value(raw_id, run_dir.name)
                status = raw_status if isinstance(raw_status, str) and raw_status in _RUN_STATUSES else None
                kind = "baseline" if run_id.startswith("baseline") else "trial" if run_id.startswith("trial-") else "other"
                if kind == "trial":
                    trial_count += 1
                if status == "running":
                    running_run_count += 1
                summary = {"id": run_id, "status": status, "kind": kind}
                mtime_ns = int(getattr(run_dir_info, "st_mtime_ns", int(run_dir_info.st_mtime * 1_000_000_000)))
                # Keep the newest runs by mtime; directory name breaks ties stably.
                entry = (mtime_ns, run_dir.name, summary)
                if len(recent) < _MAX_RUN_SUMMARIES:
                    heapq.heappush(recent, entry)
                elif entry[:2] > recent[0][:2]:
                    heapq.heapreplace(recent, entry)
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Run metadata could not be read") from error

    summaries = [
        entry[2] for entry in sorted(recent, key=lambda item: (item[0], item[1]), reverse=True)
    ]
    return {
        "runs": summaries,
        "run_count": run_count,
        "trial_count": trial_count,
        "running_run_count": running_run_count,
        "truncated_run_count": max(0, run_count - len(summaries)),
        "warning_count": warning_count,
    }


def _read_legacy_lab(
    config_path: Path | str, *, include_snapshot: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config_lexical = _absolute_lexical(config_path)
    _check_lexical_components(config_lexical)
    try:
        canonical = config_lexical.resolve(strict=True)
        if not canonical.is_file():
            raise _error("E_CONFIG_INVALID", "Config file is missing or invalid")
        root = canonical.parent.resolve(strict=True)
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_CONFIG_INVALID", "Config file is missing or invalid") from error

    snapshot = _snapshot_tree(root) if include_snapshot else {}
    config_entry: dict[str, Any] | None = None
    if include_snapshot:
        config_relative = _safe_relative(canonical.relative_to(root).as_posix())
        config_entry = snapshot.get(config_relative)
        if config_entry is None or config_entry["kind"] != "file":
            raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab config changed while it was being read")
    try:
        raw_bytes = canonical.read_bytes()
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab config changed while it was being read") from error
    if (config_entry is not None
            and (len(raw_bytes) != config_entry["bytes"]
                 or hashlib.sha256(raw_bytes).hexdigest() != config_entry["sha256"])):
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab config changed while it was being read")
    config = _parse_object(raw_bytes, code="E_CONFIG_INVALID", message="Invalid config")
    if "source_root" in config and config["source_root"] is not None and not isinstance(config["source_root"], str):
        raise _error("E_CONFIG_INVALID", "Invalid source root")
    state_dir = _state_directory(config, root)
    state_path = state_dir / "state.json"
    state: dict[str, Any] | None = None
    try:
        state_info = state_path.lstat()
    except FileNotFoundError:
        state_info = None
    if state_info is not None:
        if _is_link_or_reparse(state_info):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or reparse entry")
        if not stat.S_ISREG(state_info.st_mode):
            raise _error("E_CONFIG_INVALID", "Invalid lab state")
        try:
            state = _parse_object(state_path.read_bytes(), code="E_CONFIG_INVALID", message="Invalid lab state")
        except OSError as error:
            raise _error("E_CONFIG_INVALID", "Unable to read lab state") from error

    run_summary = _run_summaries(state_dir)
    run_manifest_warning_count = run_summary["warning_count"]
    lab = {
        "config_path": canonical,
        "root": root,
        "config_bytes": raw_bytes,
        "config": config,
        "state_dir": state_dir,
        "state": state,
        "runs": run_summary["runs"],
        "run_count": run_summary["run_count"],
        "trial_count": run_summary["trial_count"],
        "running_run_count": run_summary["running_run_count"],
        "truncated_run_count": run_summary["truncated_run_count"],
        "run_manifest_warning_count": run_manifest_warning_count,
        "run_manifest_warning": _RUN_MANIFEST_WARNING if run_manifest_warning_count else None,
    }
    return lab, snapshot


def read_legacy_lab(config_path: Path | str) -> dict[str, Any]:
    """Read a legacy lab without config validation, state repair, or writes."""
    lab, _ = _read_legacy_lab(config_path, include_snapshot=False)
    return lab


def _is_lock_artifact(name: str) -> bool:
    basename = PurePosixPath(name).name
    return basename in {".lock", ".lock.guard"} or fnmatch.fnmatchcase(basename, ".lock.*.tmp")


def _write_zip_entry(archive: zipfile.ZipFile, root: Path, relative: str,
                     expected_sha256: str) -> None:
    source = root.joinpath(*PurePosixPath(relative).parts)
    digest = hashlib.sha256()
    try:
        _check_lexical_components(source)
        info = source.lstat()
        if _is_link_or_reparse(info):
            raise _error("E_LEGACY_ARCHIVE_UNSAFE", "Lab contains a linked or non-regular file")
        if not stat.S_ISREG(info.st_mode):
            raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab entry type changed during archive export")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        try:
            opened = os.fstat(descriptor)
            if _file_signature(info) != _file_signature(opened):
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being archived")
            with os.fdopen(descriptor, "rb", closefd=False) as input_stream:
                with archive.open(f"lab/{relative}", "w") as output_stream:
                    while True:
                        block = input_stream.read(_CHUNK_SIZE)
                        if not block:
                            break
                        digest.update(block)
                        output_stream.write(block)
            final_info = os.fstat(descriptor)
            _check_lexical_components(source)
            final_path_info = source.lstat()
            if (_file_signature(opened) != _file_signature(final_info)
                    or _file_signature(final_info) != _file_signature(final_path_info)
                    or _is_link_or_reparse(final_path_info)):
                raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being archived")
        finally:
            os.close(descriptor)
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed or could not be archived") from error
    if digest.hexdigest() != expected_sha256:
        raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed while it was being archived")


def _same_snapshot(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> bool:
    if before.keys() != after.keys():
        return False
    for name, old in before.items():
        new = after[name]
        if old["kind"] != new["kind"]:
            return False
        if old["kind"] == "directory" and old["signature"][2:4] != new["signature"][2:4]:
            return False
        if old["kind"] == "file" and (old["bytes"] != new["bytes"] or old["sha256"] != new["sha256"]):
            return False
    return True


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _output_path(output: Path | str, root: Path) -> Path:
    try:
        spelled = _absolute_lexical(output)
        if spelled.name in {"", ".", ".."}:
            raise _error("E_LEGACY_ARCHIVE_OUTPUT", "Invalid output path")
        try:
            spelled.lstat()
        except FileNotFoundError:
            pass
        else:
            raise _error("E_OUTPUT_EXISTS", "Archive output already exists")
        parent = spelled.parent.resolve(strict=False)
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_OUTPUT", "Unable to inspect output path") from error
    target = parent / spelled.name
    if target.is_relative_to(root):
        raise _error("E_OUTPUT_CONTAINED", "Archive output must be outside the lab")
    if target.exists() or target.is_symlink():
        raise _error("E_OUTPUT_EXISTS", "Archive output already exists")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent = parent.resolve(strict=True)
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_OUTPUT", "Unable to prepare output directory") from error
    target = parent / spelled.name
    if target.is_relative_to(root):
        raise _error("E_OUTPUT_CONTAINED", "Archive output must be outside the lab")
    if target.exists() or target.is_symlink():
        raise _error("E_OUTPUT_EXISTS", "Archive output already exists")
    return target


def export_legacy_zip(config_path: Path | str, output: Path | str) -> dict[str, Any]:
    """Export all stable regular files as a native ZIP without mutating the lab."""
    lab, before = _read_legacy_lab(config_path, include_snapshot=True)
    root: Path = lab["root"]
    target = _output_path(output, root)
    staged_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".legacy-export-", suffix=".tmp",
                                         dir=target.parent, delete=False) as staged:
            staged_name = staged.name
        staged_path = Path(staged_name)
        included = [name for name, item in before.items()
                    if item["kind"] == "file" and not _is_lock_artifact(name)]
        excluded = sorted(name for name, item in before.items()
                          if item["kind"] == "file" and _is_lock_artifact(name))

        with zipfile.ZipFile(staged_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(included):
                _write_zip_entry(archive, root, relative, before[relative]["sha256"])
            archive_index = {
                "format": "autoresearch-legacy-archive-v1",
                "files": [
                    {"path": f"lab/{name}", "bytes": before[name]["bytes"],
                     "sha256": before[name]["sha256"]}
                    for name in sorted(included)
                ],
                "excluded_files": excluded,
            }
            archive.writestr(
                "archive-index.json",
                json.dumps(archive_index, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            )

        after = _snapshot_tree(root)
        if not _same_snapshot(before, after):
            raise _error("E_LEGACY_ARCHIVE_DRIFT", "Lab changed during archive export")

        zip_bytes = staged_path.stat().st_size
        zip_sha256 = _archive_sha256(staged_path)
        _check_lexical_components(target.parent)

        try:
            os.link(staged_path, target)
        except FileExistsError as error:
            raise _error("E_OUTPUT_EXISTS", "Archive output already exists") from error
        except OSError as error:
            raise _error("E_LEGACY_ARCHIVE_OUTPUT", "Unable to publish archive safely") from error

        file_index = [
            {"path": name, "bytes": before[name]["bytes"], "sha256": before[name]["sha256"]}
            for name in sorted(included)
        ]
        return {
            "output": str(target),
            "file_count": len(file_index),
            "byte_count": sum(item["bytes"] for item in file_index),
            "zip_bytes": zip_bytes,
            "sha256": zip_sha256,
            "files": file_index,
            "excluded_files": excluded,
        }
    except LegacyArchiveError:
        raise
    except OSError as error:
        raise _error("E_LEGACY_ARCHIVE_OUTPUT", "Unable to create archive safely") from error
    finally:
        if staged_name is not None:
            try:
                Path(staged_name).unlink(missing_ok=True)
            except OSError:
                pass
