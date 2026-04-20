#!/usr/bin/env python3
"""Validate AGENTS.md and SKILL.md structure indexes.

Checks:
- Every path referenced in AGENTS.md files exists
- Every directory with AGENTS.md is indexed in its parent
- All AGENTS.md and SKILL.md files are under 1000 characters

Exit code 0 on pass, 1 on any violation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHAR_CEILING = 1000
PIPE_PATH_RE = re.compile(r"\|\s*\[?`?([^|`\]\n]+?)`?\]?\s*\|")


def find_files(root: Path, name: str) -> list[Path]:
    return sorted(root.rglob(name))


def extract_paths(agents_file: Path) -> list[str]:
    """Extract path references from pipe-compressed tables."""
    paths: list[str] = []
    for line in agents_file.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        for match in PIPE_PATH_RE.finditer(line):
            candidate = match.group(1).strip()
            if candidate and not candidate.startswith("-") and "/" in candidate or candidate.endswith(".md"):
                paths.append(candidate)
    return paths


def check_char_ceiling(files: list[Path]) -> list[str]:
    violations = []
    for f in files:
        count = len(f.read_text(encoding="utf-8"))
        if count > CHAR_CEILING:
            violations.append(f"{f} -- {count} chars (ceiling: {CHAR_CEILING})")
    return violations


def check_paths_exist(agents_file: Path, root: Path) -> list[str]:
    violations = []
    base = agents_file.parent
    for ref in extract_paths(agents_file):
        resolved = (base / ref).resolve()
        alt = (root / ref).resolve()
        if not resolved.exists() and not alt.exists():
            violations.append(f"{agents_file}: dead link -> {ref}")
    return violations


def check_parent_index(agents_files: list[Path], root: Path) -> list[str]:
    violations = []
    indexed_dirs = set()
    for af in agents_files:
        content = af.read_text(encoding="utf-8")
        for match in PIPE_PATH_RE.finditer(content):
            indexed_dirs.add(match.group(1).strip())
    for af in agents_files:
        d = af.parent
        if d == root:
            continue
        rel = d.relative_to(root).as_posix()
        parent_agents = d.parent / "AGENTS.md"
        if parent_agents.exists():
            parent_text = parent_agents.read_text(encoding="utf-8")
            if rel not in parent_text and d.name not in parent_text:
                violations.append(f"{d} has AGENTS.md but is not indexed in {parent_agents}")
    return violations


def main() -> int:
    root = Path.cwd()
    agents_files = find_files(root, "AGENTS.md")
    skill_files = find_files(root, "SKILL.md")

    violations: list[str] = []
    violations.extend(check_char_ceiling(agents_files + skill_files))
    for af in agents_files:
        violations.extend(check_paths_exist(af, root))
    violations.extend(check_parent_index(agents_files, root))

    if violations:
        print(f"FAIL -- {len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"PASS -- {len(agents_files)} AGENTS.md, {len(skill_files)} SKILL.md checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
