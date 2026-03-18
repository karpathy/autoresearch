from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from posixpath import normpath
from typing import Iterable, Set

from autosaas.task_slicer import TaskSlice


@dataclass
class ImplementationExecutor:
    slice_def: TaskSlice
    touched_files: Set[str] = field(default_factory=set)

    def _normalize_path(self, path: str) -> str:
        normalized = path.replace("\\", "/")
        if normalized.startswith("/"):
            raise ValueError("Absolute paths are not allowed in the execution boundary.")

        if len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha():
            raise ValueError("Absolute paths are not allowed in the execution boundary.")

        if normalized.startswith("//"):
            raise ValueError("Absolute paths are not allowed in the execution boundary.")

        segments = normalized.split("/")
        if any(segment == ".." for segment in segments):
            raise ValueError("Path traversal is not allowed within the execution boundary.")

        normalized = normpath(normalized)
        if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
            raise ValueError("Path traversal is not allowed within the execution boundary.")

        return normalized

    def is_path_allowed(self, path: str) -> bool:
        return any(fnmatch(path, pattern) for pattern in self.slice_def.allowed_file_patterns)

    def record_touch(self, path: str) -> None:
        normalized = self._normalize_path(path)
        if not self.is_path_allowed(normalized):
            allowed = ", ".join(self.slice_def.allowed_file_patterns)
            raise ValueError(
                f"Refusing to touch '{normalized}' because it lies outside allowed patterns ({allowed})"
            )
        self.touched_files.add(normalized)

    def run(self, paths: Iterable[str]) -> None:
        normalized_paths = [self._normalize_path(path) for path in paths]
        disallowed = [p for p in normalized_paths if not self.is_path_allowed(p)]
        if disallowed:
            allowed = ", ".join(self.slice_def.allowed_file_patterns)
            raise ValueError(
                f"Refusing to touch {disallowed} because it lies outside allowed patterns ({allowed})"
            )

        self.touched_files.update(normalized_paths)
