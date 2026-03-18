from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Iterable, Set

from autosaas.task_slicer import TaskSlice


@dataclass
class ImplementationExecutor:
    slice_def: TaskSlice
    touched_files: Set[str] = field(default_factory=set)

    def is_path_allowed(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        return any(fnmatch(normalized, pattern) for pattern in self.slice_def.allowed_file_patterns)

    def record_touch(self, path: str) -> None:
        if not self.is_path_allowed(path):
            allowed = ", ".join(self.slice_def.allowed_file_patterns)
            raise ValueError(
                f"Refusing to touch '{path}' because it lies outside allowed patterns ({allowed})"
            )
        self.touched_files.add(path)

    def run(self, paths: Iterable[str]) -> None:
        for path in paths:
            self.record_touch(path)
