"""AST-backed code graph utilities for ResearchFS."""

from __future__ import annotations

import ast
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from researchfs.db import get_connection

_CODEGRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_symbols (
    symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    docstring TEXT,
    code TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(file_path, symbol_name, symbol_kind, start_line)
);

CREATE VIRTUAL TABLE IF NOT EXISTS code_symbols_fts USING fts5(
    file_path,
    symbol_name,
    symbol_kind,
    docstring,
    code,
    content='code_symbols',
    content_rowid='symbol_id'
);

CREATE TABLE IF NOT EXISTS method_code_targets (
    method_id INTEGER NOT NULL REFERENCES methods(method_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_kind TEXT NOT NULL,
    rationale TEXT,
    PRIMARY KEY(method_id, file_path, symbol_name, symbol_kind)
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_file_path ON code_symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_code_symbols_name ON code_symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_method_code_targets_method_id ON method_code_targets(method_id);
"""

_KIND_ORDER = {
    "class": 0,
    "method": 1,
    "function": 2,
    "async_function": 2,
    "constant": 3,
    "assignment": 4,
    "loop": 5,
    "entrypoint": 6,
    "import": 7,
}


def _connect(conn: sqlite3.Connection | None) -> tuple[sqlite3.Connection, bool]:
    close = conn is None
    if conn is None:
        conn = get_connection()
    else:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
    return conn, close


def _ensure_codegraph_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_CODEGRAPH_SCHEMA)


def _rebuild_code_symbols_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO code_symbols_fts(code_symbols_fts) VALUES('rebuild')")


def _iter_path_args(paths: Iterable[str | Path] | str | Path | None) -> tuple[str | Path, ...]:
    if paths is None:
        return ()
    if isinstance(paths, (str, Path)):
        return (paths,)
    return tuple(paths)


def _path_key(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _expand_python_paths(paths: Iterable[str | Path] | str | Path | None) -> tuple[list[tuple[Path, str]], list[str]]:
    files: dict[str, Path] = {}
    delete_keys: dict[str, None] = {}
    for raw_path in _iter_path_args(paths):
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                key = _path_key(child)
                files.setdefault(key, child)
                delete_keys[key] = None
            continue

        key = _path_key(path)
        if path.suffix == ".py":
            delete_keys[key] = None
            if path.is_file():
                files.setdefault(key, path)

    ordered_files = [(path, key) for key, path in sorted(files.items())]
    return ordered_files, list(delete_keys)


def _node_start_line(node: ast.AST) -> int:
    lines = [getattr(node, "lineno", 1)]
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        lines.extend(getattr(decorator, "lineno", lines[0]) for decorator in decorators)
    return min(lines)


def _node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))


def _source_for(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line]).rstrip()


def _first_docstring_line(node: ast.AST) -> str:
    if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    docstring = ast.get_docstring(node, clean=True) or ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _assignment_targets(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_assignment_targets(element))
        return names
    return []


def _assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            names.extend(_assignment_targets(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return _assignment_targets(node.target)
    if isinstance(node, ast.AugAssign):
        return _assignment_targets(node.target)
    return []


def _import_name(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return ", ".join(alias.asname or alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = "." * node.level + (node.module or "")
        names = ", ".join(alias.asname or alias.name for alias in node.names)
        return f"{module} import {names}".strip()
    return ""


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _loop_symbol_name(node: ast.AST, file_path: str) -> str:
    del file_path
    if isinstance(node, ast.While):
        test = node.test
        if isinstance(test, ast.Constant) and test.value is True:
            return "infinite_loop"
        return f"while_loop_{getattr(node, 'lineno', 0)}"
    if isinstance(node, ast.AsyncFor):
        return f"async_for_loop_{getattr(node, 'lineno', 0)}"
    if isinstance(node, ast.For):
        return f"for_loop_{getattr(node, 'lineno', 0)}"
    return ""


def _symbol_dict(
    file_path: str,
    symbol_name: str,
    symbol_kind: str,
    start_line: int,
    end_line: int,
    docstring: str,
    code: str,
) -> dict:
    return {
        "file_path": file_path,
        "symbol_name": symbol_name,
        "symbol_kind": symbol_kind,
        "start_line": start_line,
        "end_line": end_line,
        "docstring": docstring,
        "code": code,
    }


def _add_symbol(
    symbols: list[dict],
    seen: set[tuple[str, str, str, int]],
    file_path: str,
    symbol_name: str,
    symbol_kind: str,
    start_line: int,
    end_line: int,
    docstring: str,
    code: str,
) -> None:
    symbol_name = symbol_name.strip()
    if not symbol_name or end_line < start_line or not code.strip():
        return
    key = (file_path, symbol_name, symbol_kind, start_line)
    if key in seen:
        return
    seen.add(key)
    symbols.append(
        _symbol_dict(
            file_path,
            symbol_name,
            symbol_kind,
            start_line,
            end_line,
            docstring,
            code,
        )
    )


def _extract_symbols(path: Path, file_path: str) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=file_path)
    lines = source.splitlines()
    symbols: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()

    for node in tree.body:
        start_line = _node_start_line(node)
        end_line = _node_end_line(node)
        code = _source_for(lines, start_line, end_line)

        if isinstance(node, ast.ClassDef):
            _add_symbol(
                symbols,
                seen,
                file_path,
                node.name,
                "class",
                start_line,
                end_line,
                _first_docstring_line(node),
                code,
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_start = _node_start_line(child)
                    child_end = _node_end_line(child)
                    child_kind = "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method"
                    _add_symbol(
                        symbols,
                        seen,
                        file_path,
                        f"{node.name}.{child.name}",
                        child_kind,
                        child_start,
                        child_end,
                        _first_docstring_line(child),
                        _source_for(lines, child_start, child_end),
                    )
            continue

        if isinstance(node, ast.AsyncFunctionDef):
            _add_symbol(
                symbols,
                seen,
                file_path,
                node.name,
                "async_function",
                start_line,
                end_line,
                _first_docstring_line(node),
                code,
            )
            continue

        if isinstance(node, ast.FunctionDef):
            _add_symbol(
                symbols,
                seen,
                file_path,
                node.name,
                "function",
                start_line,
                end_line,
                _first_docstring_line(node),
                code,
            )
            continue

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for name in _assignment_names(node):
                kind = "constant" if name.isupper() else "assignment"
                _add_symbol(symbols, seen, file_path, name, kind, start_line, end_line, "", code)
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _add_symbol(symbols, seen, file_path, _import_name(node), "import", start_line, end_line, "", code)
            continue

        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            symbol_name = _loop_symbol_name(node, file_path)
            if symbol_name:
                _add_symbol(symbols, seen, file_path, symbol_name, "loop", start_line, end_line, "", code)
            continue

        if isinstance(node, ast.If) and _is_main_guard(node):
            _add_symbol(symbols, seen, file_path, "__main__", "entrypoint", start_line, end_line, "", code)

    symbols.sort(
        key=lambda row: (
            row["file_path"],
            int(row["start_line"]),
            _KIND_ORDER.get(row["symbol_kind"], 99),
            row["symbol_name"],
        )
    )
    return symbols


def _delete_symbols_for_paths(conn: sqlite3.Connection, file_paths: list[str]) -> None:
    if not file_paths:
        return
    placeholders = ",".join("?" for _ in file_paths)
    conn.execute(f"DELETE FROM code_symbols WHERE file_path IN ({placeholders})", file_paths)


def _fetch_symbols_for_paths(conn: sqlite3.Connection, file_paths: list[str]) -> list[dict]:
    if not file_paths:
        return []
    placeholders = ",".join("?" for _ in file_paths)
    rows = conn.execute(
        f"""
        SELECT symbol_id, file_path, symbol_name, symbol_kind, start_line, end_line, docstring, code, updated_at
        FROM code_symbols
        WHERE file_path IN ({placeholders})
        ORDER BY file_path, start_line, symbol_name, symbol_kind
        """,
        file_paths,
    ).fetchall()
    return [dict(row) for row in rows]


def refresh_code_map(paths: Iterable[str | Path] | str | Path | None, conn=None) -> list[dict]:
    """Parse Python files into ``code_symbols`` and rebuild the code FTS index."""
    conn, close = _connect(conn)
    try:
        _ensure_codegraph_schema(conn)
        files, delete_keys = _expand_python_paths(paths)
        _delete_symbols_for_paths(conn, delete_keys)

        symbols: list[dict] = []
        for path, file_path in files:
            symbols.extend(_extract_symbols(path, file_path))

        if symbols:
            conn.executemany(
                """
                INSERT INTO code_symbols
                (file_path, symbol_name, symbol_kind, start_line, end_line, docstring, code)
                VALUES (:file_path, :symbol_name, :symbol_kind, :start_line, :end_line, :docstring, :code)
                """,
                symbols,
            )
        _rebuild_code_symbols_fts(conn)
        conn.commit()
        return _fetch_symbols_for_paths(conn, delete_keys)
    finally:
        if close:
            conn.close()

def _line_range(row: sqlite3.Row | dict) -> str:
    start = int(row["start_line"])
    end = int(row["end_line"])
    return f"L{start}" if start == end else f"L{start}-{end}"


def _clip(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = "\n... truncated"
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker

def _validate_optional_max_chars(max_chars: int | None) -> int | None:
    """Return max_chars when None or a nonnegative int, rejecting bools."""
    if max_chars is None:
        return None
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    return max_chars


def get_code_map(conn=None, max_chars: int | None = None) -> str:
    """Return a compact map of indexed code symbols grouped by file."""
    max_chars = _validate_optional_max_chars(max_chars)
    conn, close = _connect(conn)
    try:
        _ensure_codegraph_schema(conn)
        rows = conn.execute(
            """
            SELECT file_path, symbol_name, symbol_kind, start_line, end_line, docstring
            FROM code_symbols
            ORDER BY file_path, start_line,
                CASE symbol_kind
                    WHEN 'class' THEN 0
                    WHEN 'method' THEN 1
                    WHEN 'async_method' THEN 1
                    WHEN 'function' THEN 2
                    WHEN 'async_function' THEN 2
                    WHEN 'constant' THEN 3
                    WHEN 'assignment' THEN 4
                    WHEN 'loop' THEN 5
                    WHEN 'entrypoint' THEN 6
                    WHEN 'import' THEN 7
                    ELSE 99
                END,
                symbol_name
            """
        ).fetchall()
        if not rows:
            return "No code symbols indexed. Run refresh_code_map() first."

        lines: list[str] = []
        current_file = None
        for row in rows:
            file_path = row["file_path"]
            if file_path != current_file:
                if lines:
                    lines.append("")
                lines.append(f"# {file_path}")
                current_file = file_path
            doc = (row["docstring"] or "").strip()
            suffix = f" — {doc}" if doc else ""
            lines.append(
                f"- {row['symbol_kind']} {row['symbol_name']} {_line_range(row)}{suffix}"
            )
        return _clip("\n".join(lines), max_chars)
    finally:
        if close:
            conn.close()


def _fts_query(text: str | None, joiner: str = " OR ") -> str:
    terms = []
    seen = set()
    for term in re.findall(r"[A-Za-z0-9_]+", text or ""):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(f'"{term}"')
    return joiner.join(terms)


def _compact_snippet(snippet: str | None, fallback_code: str) -> str:
    text = snippet or ""
    if not text.strip():
        text = fallback_code.splitlines()[0] if fallback_code else ""
    return re.sub(r"\s+", " ", text).strip()


def search_code_symbols(query: str, conn=None, max_chars: int | None = None) -> str:
    """Search indexed code symbols with an FTS5 MATCH query built from quoted terms."""
    max_chars = _validate_optional_max_chars(max_chars)
    conn, close = _connect(conn)
    try:
        _ensure_codegraph_schema(conn)
        fts_query = _fts_query(query)
        if not fts_query:
            return "No code search query provided."

        rows = conn.execute(
            """
            SELECT
                cs.file_path,
                cs.symbol_name,
                cs.symbol_kind,
                cs.start_line,
                cs.end_line,
                cs.docstring,
                cs.code,
                snippet(code_symbols_fts, 4, '[', ']', ' … ', 18) AS snippet
            FROM code_symbols_fts
            JOIN code_symbols cs ON cs.symbol_id = code_symbols_fts.rowid
            WHERE code_symbols_fts MATCH ?
            ORDER BY bm25(code_symbols_fts), cs.file_path, cs.start_line
            LIMIT 25
            """,
            (fts_query,),
        ).fetchall()
        if not rows:
            return f"No code symbols matched: {query}"

        lines = [f"# Code symbol matches for: {query}"]
        for row in rows:
            doc = (row["docstring"] or "").strip()
            doc_suffix = f" — {doc}" if doc else ""
            snippet = _compact_snippet(row["snippet"], row["code"])
            lines.append(
                f"- {row['file_path']}:{_line_range(row)} {row['symbol_kind']} {row['symbol_name']}{doc_suffix}"
            )
            if snippet:
                lines.append(f"  {snippet}")
        return _clip("\n".join(lines), max_chars)
    finally:
        if close:
            conn.close()


def _target_path_key(value: str | Path) -> str:
    if isinstance(value, Path):
        return _path_key(value)
    if not isinstance(value, str):
        raise ValueError("file paths must be non-empty strings or Path values")
    path = value.strip()
    if not path:
        raise ValueError("file paths must be non-empty strings or Path values")
    return path


def replace_method_code_targets(
    file_paths: Iterable[str | Path] | str | Path,
    targets: Iterable[Mapping[str, object]],
    conn=None,
) -> int:
    """Replace explicit method-to-symbol links for a set of indexed files.

    ResearchFS owns generic code indexing and storage. Domain-specific target
    selection belongs to the caller, which supplies concrete method/symbol
    mappings after indexing its own source files.
    """
    if isinstance(file_paths, (str, Path)):
        raw_file_paths = (file_paths,)
    else:
        raw_file_paths = tuple(file_paths)
    normalized_paths = tuple(dict.fromkeys(_target_path_key(path) for path in raw_file_paths))
    if not normalized_paths:
        raise ValueError("file_paths must contain at least one indexed path")

    normalized_targets: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("targets must contain mapping objects")
        method_id = target.get("method_id")
        if isinstance(method_id, bool) or not isinstance(method_id, int) or method_id <= 0:
            raise ValueError("target method_id must be a positive integer")
        file_path = _target_path_key(target.get("file_path"))
        if file_path not in normalized_paths:
            raise ValueError(f"target file_path is outside file_paths: {file_path}")
        symbol_name = target.get("symbol_name")
        symbol_kind = target.get("symbol_kind")
        if not isinstance(symbol_name, str) or not symbol_name.strip():
            raise ValueError("target symbol_name must be a non-empty string")
        if not isinstance(symbol_kind, str) or not symbol_kind.strip():
            raise ValueError("target symbol_kind must be a non-empty string")
        rationale = target.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError("target rationale must be a string or None")
        row = {
            "method_id": method_id,
            "file_path": file_path,
            "symbol_name": symbol_name.strip(),
            "symbol_kind": symbol_kind.strip(),
            "rationale": rationale.strip() if isinstance(rationale, str) else None,
        }
        key = (method_id, file_path, row["symbol_name"], row["symbol_kind"])
        normalized_targets[key] = row

    conn, close = _connect(conn)
    try:
        _ensure_codegraph_schema(conn)
        for row in normalized_targets.values():
            if conn.execute(
                "SELECT 1 FROM methods WHERE method_id = ?",
                (row["method_id"],),
            ).fetchone() is None:
                raise ValueError(f"unknown method_id: {row['method_id']}")
            if conn.execute(
                """
                SELECT 1 FROM code_symbols
                WHERE file_path = ? AND symbol_name = ? AND symbol_kind = ?
                """,
                (row["file_path"], row["symbol_name"], row["symbol_kind"]),
            ).fetchone() is None:
                raise ValueError(
                    "unknown indexed symbol: "
                    f"{row['file_path']}:{row['symbol_kind']}:{row['symbol_name']}"
                )

        placeholders = ",".join("?" for _ in normalized_paths)
        conn.execute(
            f"DELETE FROM method_code_targets WHERE file_path IN ({placeholders})",
            normalized_paths,
        )
        if normalized_targets:
            conn.executemany(
                """
                INSERT INTO method_code_targets
                (method_id, file_path, symbol_name, symbol_kind, rationale)
                VALUES (:method_id, :file_path, :symbol_name, :symbol_kind, :rationale)
                """,
                normalized_targets.values(),
            )
        conn.commit()
        return len(normalized_targets)
    finally:
        if close:
            conn.close()
