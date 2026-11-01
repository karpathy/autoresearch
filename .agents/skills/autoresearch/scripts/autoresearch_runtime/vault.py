"""Stores and restores the vault and maps inputs to environment variables."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


from .errors import LabError
from .store import _read_json, _sha256, _write_json


def _store_vault(config: LabConfig) -> None:
    """Keep a byte copy of protected inputs and evaluator files outside candidate/."""
    from .config import LabConfig, MAX_UNCLASSIFIED_CANDIDATE_BYTES
    files: list[Path] = list(config.input_files)
    for arg in config.evaluator:
        path = Path(arg)
        if path.is_file():
            files.append(path.resolve())
    index_path = config.state_dir / "vault" / "index.json"
    index = _read_json(index_path, {"files": []})
    if not isinstance(index, dict):
        index = {"files": []}
    recorded = [item for item in index.get("files", []) if isinstance(item, dict)]
    by_path = {item.get("path"): item for item in recorded}
    for file in files:
        if not file.is_file():
            continue
        resolved = file.resolve()
        if resolved.is_relative_to((config.state_dir / "candidate").resolve()):
            continue
        digest = _sha256(resolved)
        size = resolved.stat().st_size
        entry = {"path": str(resolved), "sha256": digest, "bytes": size, "stored": size <= MAX_UNCLASSIFIED_CANDIDATE_BYTES}
        if entry["stored"]:
            blob = config.state_dir / "vault" / "blobs" / digest
            if not blob.is_file():
                blob.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resolved, blob)
        by_path[entry["path"]] = entry
    _write_json(index_path, {"files": list(by_path.values())})

def input_map_env(config: LabConfig) -> dict[str, str]:
    """Point the evaluator at vault copies of protected inputs. Missing blobs are omitted."""
    index = _read_json(config.state_dir / "vault" / "index.json", {"files": []})
    mapping = {}
    for item in (index or {}).get("files", []) if isinstance(index, dict) else []:
        if not isinstance(item, dict) or not item.get("stored"):
            continue
        digest = str(item.get("sha256") or "")
        original = str(item.get("path") or "")
        blob = config.state_dir / "vault" / "blobs" / digest
        if original and digest and blob.is_file():
            mapping[original] = str(blob)
    if not mapping:
        return {}
    return {"AUTORESEARCH_INPUT_MAP": json.dumps(mapping, sort_keys=True)}


def restore_vault(config: LabConfig, path: Path, *, allow_source_write: bool = False) -> dict[str, Any]:
    """Write a vaulted file back. Writing into the measured project requires an explicit flag."""
    from .config import LabConfig
    path = path.resolve()
    source = config.source_root.resolve()
    if not allow_source_write and (path == source or path.is_relative_to(source)):
        raise LabError(
            "Refusing to write a vault file into the source project",
            code="E_SOURCE_WRITE",
            hint="Re-run with --allow-source-write if this protected input should be replaced",
        )
    index = _read_json(config.state_dir / "vault" / "index.json", {"files": []})
    matches = [item for item in (index or {}).get("files", [])
               if isinstance(item, dict) and Path(str(item.get("path", ""))).resolve() == path]
    if not matches:
        raise LabError(f"No vault record for {path}")
    item = matches[-1]
    if not item.get("stored"):
        raise LabError(f"Vault has only a hash for {path}; the file is larger than 25 MiB")
    blob = config.state_dir / "vault" / "blobs" / str(item.get("sha256"))
    if not blob.is_file():
        raise LabError(f"Vault blob is missing for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(blob, path)
    if _sha256(path) != item.get("sha256"):
        raise LabError(f"Restored file hash does not match the vault: {path}")
    return {"path": str(path), "sha256": item["sha256"]}
