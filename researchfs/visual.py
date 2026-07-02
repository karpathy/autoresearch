"""Optional PixelRAG visual indexing helpers for ResearchFS papers."""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import urlopen
from urllib.parse import unquote, urlparse


DEFAULT_VISUAL_INDEX_ROOT = Path("data") / "visual_indexes"
VISUAL_INDEX_ROOT_ENV = "RESEARCHFS_VISUAL_INDEX_ROOT"
PROVIDER = "pixelrag"
_INSTALL_MESSAGE = (
    "PixelRAG visual indexing requires the PixelRAG command-line tools "
    "(`pixelrag` and `pixelshot`). Install PixelRAG separately, for example "
    "with `uv tool install pixelrag` or `pip install 'pixelrag[index]'`."
)


Runner = Callable[..., Any]


class VisualIndexError(Exception):
    """Raised when an optional visual index cannot be planned or built."""


def _clean_text(value: Any) -> str | None:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _root_path(output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        return Path(output_root)
    configured = _clean_text(os.environ.get(VISUAL_INDEX_ROOT_ENV))
    if configured:
        return Path(configured)
    return DEFAULT_VISUAL_INDEX_ROOT

def _paper_paths(
    paper_id: int | str, output_root: str | Path | None = None
) -> tuple[Path, Path, Path, Path]:
    root = _root_path(output_root)
    paper_dir = root / f"paper-{paper_id}"
    tiles_dir = paper_dir / "tiles"
    embeddings_dir = paper_dir / "embeddings"
    index_dir = paper_dir / "index"
    return paper_dir, tiles_dir, embeddings_dir, index_dir


def _path_text(path: Path) -> str:
    return str(path)


def _arxiv_pdf_url(identifier: str) -> str:
    return f"https://arxiv.org/pdf/{identifier}.pdf"


def _arxiv_id_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] not in {"abs", "pdf"}:
        return None
    identifier = parts[1]
    if parts[0] == "pdf" and identifier.endswith(".pdf"):
        identifier = identifier[:-4]
    return identifier or None


def _looks_like_arxiv_id(value: str) -> bool:
    if "/" in value:
        prefix, _, suffix = value.partition("/")
        return bool(prefix and suffix and any(char.isdigit() for char in suffix))
    digits = value.replace(".", "")
    return "." in value and digits.isdigit() and 8 <= len(digits) <= 9


def infer_visual_source(meta: Mapping[str, Any]) -> str | None:
    """Infer the source PDF/path/URL to feed PixelRAG from paper metadata."""

    for key in ("visual_source", "visual_source_path", "visual_source_url"):
        value = _clean_text(meta.get(key))
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme == "file":
            return unquote(parsed.path)
        return value

    external_id = _clean_text(meta.get("external_id"))
    if external_id and _looks_like_arxiv_id(external_id):
        return _arxiv_pdf_url(external_id)

    url = _clean_text(meta.get("url"))
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        return _arxiv_pdf_url(arxiv_id)
    if parsed.scheme in {"http", "https"}:
        return url
    return None

def plan_pixelrag_visual_index(
    paper_id: int | str,
    meta: Mapping[str, Any],
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the planned PixelRAG artifact locations without running tools."""

    source = infer_visual_source(meta)
    _, tiles_dir, embeddings_dir, index_dir = _paper_paths(paper_id, output_root)
    return {
        "provider": PROVIDER,
        "status": "planned",
        "source": source,
        "tiles_path": _path_text(tiles_dir),
        "embeddings_path": _path_text(embeddings_dir),
        "index_path": _path_text(index_dir),
        "error_text": None,
    }


def _default_runner(
    command: Sequence[str], *, cwd: str | Path | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        tool = command[0] if command else "PixelRAG"
        raise VisualIndexError(f"Missing `{tool}` command. {_INSTALL_MESSAGE}") from exc
    except subprocess.CalledProcessError as exc:
        raise _command_error(command, exc) from exc


def _command_error(command: Sequence[str], exc: subprocess.CalledProcessError) -> VisualIndexError:
    output = "\n".join(part for part in (exc.stderr, exc.stdout) if part)
    detail = output.strip() or f"exit status {exc.returncode}"
    return VisualIndexError(f"PixelRAG command failed: {' '.join(command)}: {detail}")


def _yaml_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _download_remote_source(source: str, paper_dir: Path) -> Path:
    parsed = urlparse(source)
    suffix = Path(parsed.path).suffix or ".html"
    target = paper_dir / f"source{suffix}"
    with urlopen(source, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def _local_source_path(source: str, paper_dir: Path) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return _download_remote_source(source, paper_dir)
    return Path(source)


def _write_pixelrag_config(source_path: Path, paper_dir: Path, index_dir: Path) -> Path:
    config_path = paper_dir / "pixelrag.yaml"
    output_path = Path(os.path.relpath(index_dir, paper_dir))
    source_path = Path(os.path.relpath(source_path, paper_dir))
    config_path.write_text(
        "\n".join(
            (
                "source:",
                "  type: local",
                f"  path: {_yaml_single_quote(_path_text(source_path))}",
                "embed:",
                "  device: auto",
                f"output: {_yaml_single_quote(_path_text(output_path))}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return config_path


def _runner_supports_cwd(runner: Runner) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == "cwd"
        for parameter in signature.parameters.values()
    )


def _run_pixelrag_command(runner: Runner, command: Sequence[str], cwd: Path) -> None:
    try:
        result = runner(command, cwd=cwd) if _runner_supports_cwd(runner) else runner(command)
    except FileNotFoundError as exc:
        tool = exc.filename or (command[0] if command else "pixelrag")
        raise VisualIndexError(f"Missing `{tool}` command. {_INSTALL_MESSAGE}") from exc
    except subprocess.CalledProcessError as exc:
        raise _command_error(command, exc) from exc
    if getattr(result, "returncode", 0):
        raise VisualIndexError(
            f"PixelRAG command failed: {' '.join(command)}: exit status {result.returncode}"
        )


def build_pixelrag_visual_index(
    paper_id: int,
    meta: Mapping[str, Any],
    output_root: str | Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Build a PixelRAG visual index and return the indexed payload."""

    payload = plan_pixelrag_visual_index(paper_id, meta, output_root)
    source = payload["source"]
    if not source:
        raise VisualIndexError(
            "Cannot infer a visual source for PixelRAG indexing. Provide "
            "visual_source, visual_source_path, visual_source_url, an arXiv external_id, or a URL."
        )

    paper_dir, tiles_dir, embeddings_dir, index_dir = _paper_paths(paper_id, output_root)
    for directory in (paper_dir, tiles_dir, embeddings_dir, index_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_path = _local_source_path(source, paper_dir)
    _write_pixelrag_config(source_path, paper_dir, index_dir)

    command = ["pixelrag", "index", "build"]
    run = runner or _default_runner
    _run_pixelrag_command(run, command, paper_dir)

    payload["status"] = "indexed"
    return payload


def _row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    columns = (
        "visual_index_id",
        "paper_id",
        "provider",
        "status",
        "source",
        "tiles_path",
        "embeddings_path",
        "index_path",
        "error_text",
        "created_at",
        "updated_at",
    )
    return dict(zip(columns, row))


def _fetch_visual_index_row(conn: Any, paper_id: int, provider: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT visual_index_id, paper_id, provider, status, source, tiles_path,
               embeddings_path, index_path, error_text, created_at, updated_at
        FROM paper_visual_indexes
        WHERE paper_id = ? AND provider = ?
        """,
        (paper_id, provider),
    ).fetchone()
    if row is None:
        raise VisualIndexError("Visual index row was not stored")
    return _row_to_dict(row)


def upsert_visual_index(conn: Any, paper_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Insert or update the stored visual-index status for a paper."""

    provider = _clean_text(payload.get("provider")) or PROVIDER
    status = _clean_text(payload.get("status"))
    if status not in {"planned", "indexed", "failed"}:
        raise VisualIndexError("Visual index status must be one of: planned, indexed, failed")

    conn.execute(
        """
        INSERT INTO paper_visual_indexes (
            paper_id, provider, status, source, tiles_path, embeddings_path, index_path, error_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, provider) DO UPDATE SET
            status = excluded.status,
            source = excluded.source,
            tiles_path = excluded.tiles_path,
            embeddings_path = excluded.embeddings_path,
            index_path = excluded.index_path,
            error_text = excluded.error_text,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            paper_id,
            provider,
            status,
            _clean_text(payload.get("source")),
            _clean_text(payload.get("tiles_path")),
            _clean_text(payload.get("embeddings_path")),
            _clean_text(payload.get("index_path")),
            _clean_text(payload.get("error_text")),
        ),
    )
    return _fetch_visual_index_row(conn, paper_id, provider)


def build_and_store_pixelrag_visual_index(
    conn: Any,
    paper_id: int,
    meta: Mapping[str, Any],
    output_root: str | Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Build a PixelRAG visual index, storing indexed or failed status."""

    try:
        payload = build_pixelrag_visual_index(paper_id, meta, output_root, runner)
    except Exception as exc:
        planned = plan_pixelrag_visual_index(paper_id, meta, output_root)
        planned["status"] = "failed"
        planned["error_text"] = str(exc)
        return upsert_visual_index(conn, paper_id, planned)
    return upsert_visual_index(conn, paper_id, payload)
