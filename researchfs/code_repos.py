"""Paper-to-code repository discovery and persistence helpers.

Core discovery is deterministic and offline: it extracts explicit GitHub URLs
from stored paper metadata/chunks. Optional search enrichment only runs when a
caller supplies a search function.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from researchfs.db import get_connection


GITHUB_HOSTS = {"github.com", "www.github.com"}
GITHUB_URL_RE = re.compile(
    r"(?:(?:https?|git)://)?(?:www\.)?github\.com[:/][A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?(?:/[^\s<>'\")\]]*)?",
    re.IGNORECASE,
)
GIT_SSH_RE = re.compile(
    r"git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?(?:/[^\s<>'\")\]]*)?",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)([A-Za-z0-9.\-/]+)",
    re.IGNORECASE,
)
TITLE_STOPWORDS = {
    "about",
    "after",
    "along",
    "also",
    "analysis",
    "approach",
    "based",
    "between",
    "code",
    "data",
    "deep",
    "efficient",
    "from",
    "github",
    "into",
    "large",
    "learning",
    "method",
    "model",
    "models",
    "neural",
    "paper",
    "using",
    "with",
}

SearchFn = Callable[[list[str]], Any]


@dataclass(frozen=True, slots=True)
class GitHubRepository:
    """Normalized GitHub repository identity."""

    owner: str
    name: str

    @property
    def repository_key(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    def to_dict(self) -> dict[str, str]:
        return {
            "owner": self.owner,
            "name": self.name,
            "repository_key": self.repository_key,
            "repository_url": self.repository_url,
            "repo_key": self.repository_key,
            "repo_url": self.repository_url,
        }


@dataclass(frozen=True, slots=True)
class CodeRepositoryCandidate:
    """Repository candidate with provenance suitable for database upsert."""

    repository_url: str
    repository_key: str
    owner: str
    name: str
    source: str
    evidence: str | None = None
    confidence: float = 0.0
    selected: bool = False
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_url": self.repository_url,
            "repository_key": self.repository_key,
            "repo_url": self.repository_url,
            "repo_key": self.repository_key,
            "owner": self.owner,
            "name": self.name,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "selected": self.selected,
            "query": self.query,
        }


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row) if isinstance(row, sqlite3.Row) else dict(row)
    if "repo_url" in data and not data.get("repository_url"):
        data["repository_url"] = data["repo_url"]
    if "repo_key" in data and not data.get("repository_key"):
        data["repository_key"] = data["repo_key"]
    return data


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, limit: int = 700) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    if confidence != confidence:
        return default
    return max(0.0, min(1.0, confidence))


def _strip_repo_name_suffix(name: str) -> str:
    cleaned = name.strip().strip(".,;:!?)]}>")
    if cleaned.lower().endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned


def _repo_from_parts(owner: str, name: str) -> GitHubRepository | None:
    owner = owner.strip().strip(".,;:!?)]}>")
    name = _strip_repo_name_suffix(name)
    if not owner or not name:
        return None
    if owner in {".", ".."} or name in {".", ".."}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return None
    return GitHubRepository(owner=owner.lower(), name=name.lower())


def parse_github_repo_url(value: str | None) -> GitHubRepository | None:
    """Return normalized GitHub owner/repo identity, or None for non-repos."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().strip("<>'\"` ")
    raw = raw.rstrip(".,;:!?)]}")

    ssh_match = re.match(
        r"^(?:git@github\.com:|ssh://git@github\.com/)([^/]+)/([^/?#]+)",
        raw,
        re.IGNORECASE,
    )
    if ssh_match:
        return _repo_from_parts(ssh_match.group(1), ssh_match.group(2))

    candidate = raw
    if candidate.lower().startswith("github.com/") or candidate.lower().startswith("www.github.com/"):
        candidate = "https://" + candidate
    parsed = urlsplit(candidate)
    if not parsed.scheme or parsed.netloc.lower() not in GITHUB_HOSTS:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return _repo_from_parts(parts[0], parts[1])


def normalize_github_repo_url(value: str | None) -> str | None:
    """Normalize a GitHub URL or repo subpath to ``https://github.com/owner/repo``.

    Non-GitHub hosts and missing owner/repo paths are rejected. Branch, blob,
    query, and fragment suffixes are canonicalized to the repository root.
    """
    repo = parse_github_repo_url(value)
    if repo is None:
        return None
    return repo.repository_url


def _candidate_from_repo(
    repo: GitHubRepository,
    *,
    source: str,
    evidence: str | None,
    confidence: float,
    query: str | None = None,
) -> CodeRepositoryCandidate:
    return CodeRepositoryCandidate(
        repository_url=repo.repository_url,
        repository_key=repo.repository_key,
        owner=repo.owner,
        name=repo.name,
        source=source,
        evidence=_truncate(evidence) if evidence else None,
        confidence=_clamp_confidence(confidence),
        query=query,
    )


def extract_github_repo_urls(text: str | None) -> list[str]:
    """Extract normalized GitHub repository root URLs in first-seen order."""
    if not isinstance(text, str) or not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    matches = [
        (match.start(), match.group(0))
        for pattern in (GIT_SSH_RE, GITHUB_URL_RE)
        for match in pattern.finditer(text)
    ]
    for _position, raw_url in sorted(matches, key=lambda item: item[0]):
        normalized = normalize_github_repo_url(raw_url)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        found.append(normalized)
    return found


def extract_github_repository_candidates(
    text: str | None,
    *,
    source: str,
    confidence: float = 0.9,
) -> list[CodeRepositoryCandidate]:
    """Extract repository candidates from text with shared source metadata."""
    candidates: list[CodeRepositoryCandidate] = []
    for url in extract_github_repo_urls(text):
        repo = parse_github_repo_url(url)
        if repo is None:
            continue
        candidates.append(
            _candidate_from_repo(
                repo,
                source=source,
                evidence=text,
                confidence=confidence,
            )
        )
    return candidates


def load_paper_code_inputs(
    conn: sqlite3.Connection,
    paper_id: int,
    *,
    max_chunks: int = 8,
    max_chunk_chars: int = 1200,
) -> dict[str, Any]:
    """Load bounded stored paper metadata/chunks for code discovery."""
    if isinstance(paper_id, bool) or not isinstance(paper_id, int) or paper_id <= 0:
        raise ValueError("paper_id must be a positive integer")
    if max_chunks < 0:
        raise ValueError("max_chunks must be non-negative")
    if max_chunk_chars < 0:
        raise ValueError("max_chunk_chars must be non-negative")

    paper = conn.execute(
        """
        SELECT paper_id, title, authors, abstract, url, publish_date, created_at
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    if paper is None:
        raise LookupError(f"No paper found for id {paper_id}")

    chunks = conn.execute(
        """
        SELECT chunk_id, section, content, created_at
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        LIMIT ?
        """,
        (paper_id, max_chunks),
    ).fetchall()
    bounded_chunks: list[dict[str, Any]] = []
    for row in chunks:
        chunk = _row_to_dict(row)
        content = chunk.get("content") or ""
        chunk["content"] = content[:max_chunk_chars] if max_chunk_chars else ""
        bounded_chunks.append(chunk)
    return {"paper": _row_to_dict(paper), "chunks": bounded_chunks}


def _arxiv_ids_from_text(text: str | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in ARXIV_RE.finditer(text or ""):
        arxiv_id = match.group(1).strip().removesuffix(".pdf")
        arxiv_id = arxiv_id.rstrip(".,;:)]}")
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            ids.append(arxiv_id)
    return ids


def _prominent_terms(text: str, limit: int = 4) -> list[str]:
    counts: dict[str, tuple[str, int]] = {}
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9_-]{4,}", text or ""):
        if match.start() > 0 and text[match.start() - 1] == "\\":
            continue
        token = match.group(0)
        normalized = token.lower().strip("-_")
        if normalized in TITLE_STOPWORDS:
            continue
        has_camel_shape = bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", token))
        has_signal_shape = has_camel_shape or any(char.isdigit() for char in token)
        if any(char.isdigit() for char in token) and not has_camel_shape and token == token.lower():
            continue
        if not has_signal_shape and len(normalized) < 9:
            continue
        display, count = counts.get(normalized, (token, 0))
        counts[normalized] = (display, count + 1)
    ranked = sorted(
        counts.values(),
        key=lambda item: (
            0
            if any(char.isdigit() for char in item[0])
            or bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", item[0]))
            else 1,
            -item[1],
            item[0].lower(),
        ),
    )
    return [display for display, _count in ranked[:limit]]


def derive_code_repository_queries(
    paper: Mapping[str, Any],
    chunks: Iterable[Mapping[str, Any]] = (),
    *,
    max_queries: int = 8,
) -> list[str]:
    """Derive generic search queries from title, arXiv IDs, and paper text terms."""
    if max_queries <= 0:
        return []
    title = _normalize_text(paper.get("title"))
    abstract = _normalize_text(paper.get("abstract"))
    url = _normalize_text(paper.get("url"))
    chunk_text = " ".join(_normalize_text(chunk.get("content")) for chunk in chunks)
    combined = " ".join(part for part in (title, abstract, url, chunk_text) if part)

    queries: list[str] = []
    seen: set[str] = set()

    def add(query: str | None) -> None:
        query = _normalize_text(query)
        if not query:
            return
        key = query.lower()
        if key in seen or len(queries) >= max_queries:
            return
        seen.add(key)
        queries.append(query)

    if title:
        add(f"{title} GitHub")
        add(f"{title} code")
    for arxiv_id in _arxiv_ids_from_text(combined):
        add(f"arXiv {arxiv_id} GitHub")
        add(f"{arxiv_id} code")
    for term in _prominent_terms(combined):
        add(f"{term} GitHub")
    return queries[:max_queries]


def discover_explicit_code_repositories(
    paper: Mapping[str, Any],
    chunks: Iterable[Mapping[str, Any]] = (),
) -> list[CodeRepositoryCandidate]:
    """Discover GitHub repositories explicitly present in stored paper text."""
    candidates: list[CodeRepositoryCandidate] = []
    metadata_fields = (
        ("paper_url", paper.get("url"), 0.98),
        ("paper_title", paper.get("title"), 0.9),
        ("paper_authors", paper.get("authors"), 0.82),
        ("paper_abstract", paper.get("abstract"), 0.9),
    )
    for label, value, confidence in metadata_fields:
        candidates.extend(
            extract_github_repository_candidates(
                str(value or ""),
                source=f"explicit:{label}",
                confidence=confidence,
            )
        )
    for chunk in chunks:
        content = chunk.get("content") or ""
        chunk_id = chunk.get("chunk_id")
        section = chunk.get("section")
        source = f"explicit:chunk:{chunk_id}"
        if section:
            source += f":{_truncate(section, 80)}"
        candidates.extend(
            extract_github_repository_candidates(
                content,
                source=source,
                confidence=0.88,
            )
        )
    return _dedupe_candidates(candidates)


def _extract_url_from_result(result: Mapping[str, Any]) -> str | None:
    for key in ("repository_url", "html_url", "url", "clone_url", "ssh_url", "git_url", "svn_url"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            normalized = normalize_github_repo_url(value)
            if normalized is not None:
                return normalized
    for key in ("full_name", "repo", "repository", "name"):
        value = result.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value.strip()):
            normalized = normalize_github_repo_url("https://github.com/" + value.strip())
            if normalized is not None:
                return normalized
    return None


def _search_evidence_text(result: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "evidence",
        "readme",
        "README",
        "description",
        "summary",
        "full_name",
        "name",
        "url",
        "html_url",
    ):
        value = result.get(key)
        if value:
            parts.append(str(value))
    return _truncate(" ".join(parts), 1000)


def _score_search_result(
    result: Mapping[str, Any],
    *,
    query: str | None,
    title: str,
    arxiv_ids: list[str],
) -> float:
    if "confidence" in result:
        return _clamp_confidence(result.get("confidence"), default=0.55)
    evidence = _search_evidence_text(result).lower()
    score = 0.55
    title = title.lower()
    if title and title in evidence:
        score += 0.22
    for arxiv_id in arxiv_ids:
        if arxiv_id.lower() in evidence:
            score += 0.16
            break
    if query and query.lower() in evidence:
        score += 0.08
    if normalize_github_repo_url(str(result.get("url") or result.get("html_url") or "")):
        score += 0.04
    return _clamp_confidence(score)


def _iter_search_items(raw_results: Any) -> Iterable[tuple[str | None, Any]]:
    if raw_results is None:
        return []
    if isinstance(raw_results, Mapping):
        if any(key in raw_results for key in ("url", "html_url", "repository_url", "full_name")):
            return [(raw_results.get("query"), raw_results)]
        items: list[tuple[str | None, Any]] = []
        for key, value in raw_results.items():
            if key in {"items", "results", "repositories"} and isinstance(value, list):
                items.extend((None, item) for item in value)
            elif isinstance(value, list):
                items.extend((str(key), item) for item in value)
        return items
    if isinstance(raw_results, (str, CodeRepositoryCandidate)):
        return [(None, raw_results)]
    try:
        return [(None, item) for item in raw_results]
    except TypeError:
        return []


def search_results_to_candidates(
    raw_results: Any,
    *,
    paper: Mapping[str, Any],
) -> list[CodeRepositoryCandidate]:
    """Convert injected searcher results to normalized candidates."""
    title = _normalize_text(paper.get("title"))
    arxiv_ids = _arxiv_ids_from_text(
        " ".join(str(paper.get(key) or "") for key in ("title", "abstract", "url"))
    )
    candidates: list[CodeRepositoryCandidate] = []
    for query, item in _iter_search_items(raw_results):
        if isinstance(item, CodeRepositoryCandidate):
            candidates.append(item)
            continue
        if isinstance(item, str):
            repo = parse_github_repo_url(item)
            if repo is None:
                continue
            candidates.append(
                _candidate_from_repo(
                    repo,
                    source="search",
                    evidence=item,
                    confidence=0.62,
                    query=query,
                )
            )
            continue
        if not isinstance(item, Mapping):
            continue
        url = _extract_url_from_result(item)
        repo = parse_github_repo_url(url)
        if repo is None:
            continue
        item_query = str(item.get("query") or query or "").strip() or None
        confidence = _score_search_result(
            item,
            query=item_query,
            title=title,
            arxiv_ids=arxiv_ids,
        )
        candidates.append(
            _candidate_from_repo(
                repo,
                source=str(item.get("source") or "search"),
                evidence=_search_evidence_text(item),
                confidence=confidence,
                query=item_query,
            )
        )
    return _dedupe_candidates(candidates)


def _candidate_rank(candidate: CodeRepositoryCandidate) -> tuple[float, int, str]:
    source_priority = 1 if candidate.source.startswith("explicit:") else 0
    return (candidate.confidence, source_priority, candidate.repository_key)


def _dedupe_candidates(candidates: Iterable[CodeRepositoryCandidate]) -> list[CodeRepositoryCandidate]:
    best: dict[str, CodeRepositoryCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.repository_key)
        if current is None or _candidate_rank(candidate) > _candidate_rank(current):
            best[candidate.repository_key] = candidate
    return sorted(best.values(), key=lambda candidate: (-candidate.confidence, candidate.repository_key))


def _paper_code_table_exists(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_code_repositories'"
        ).fetchone()
        is not None
    )


def upsert_code_repository_candidate(
    conn: sqlite3.Connection,
    paper_id: int,
    candidate: CodeRepositoryCandidate | Mapping[str, Any],
) -> dict[str, Any]:
    """Insert or update one paper repository candidate and return the stored row."""
    if isinstance(candidate, Mapping):
        repo = parse_github_repo_url(str(candidate.get("repository_url") or candidate.get("repo_url") or ""))
        repo_key = candidate.get("repository_key") or candidate.get("repo_key")
        if repo is None and repo_key:
            repo = parse_github_repo_url("https://github.com/" + str(repo_key))
        if repo is None:
            raise ValueError("candidate must contain a GitHub repository URL or key")
        candidate = _candidate_from_repo(
            repo,
            source=str(candidate.get("source") or "manual"),
            evidence=candidate.get("evidence"),
            confidence=_clamp_confidence(candidate.get("confidence")),
            query=candidate.get("query"),
        )

    existing = get_paper_code_repository(conn, paper_id, candidate.repository_key)
    if existing is not None:
        existing_confidence = _clamp_confidence(existing.get("confidence"))
        use_new_provenance = candidate.confidence >= existing_confidence
        selected = 1 if candidate.selected else int(existing.get("selected") or 0)
        conn.execute(
            """
            UPDATE paper_code_repositories
            SET repo_url = ?,
                repo_key = ?,
                owner = ?,
                name = ?,
                source = ?,
                evidence = ?,
                confidence = ?,
                selected = ?,
                selected_at = CASE
                    WHEN ? = 1 THEN COALESCE(selected_at, CURRENT_TIMESTAMP)
                    ELSE selected_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id = ? AND repo_key = ?
            """,
            (
                candidate.repository_url,
                candidate.repository_key,
                candidate.owner,
                candidate.name,
                candidate.source if use_new_provenance else existing.get("source"),
                candidate.evidence if use_new_provenance else existing.get("evidence"),
                max(existing_confidence, candidate.confidence),
                selected,
                1 if candidate.selected else 0,
                paper_id,
                candidate.repository_key,
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO paper_code_repositories (
                paper_id, repo_url, repo_key, owner, name,
                source, evidence, confidence, selected, selected_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END, CURRENT_TIMESTAMP)
            """,
            (
                paper_id,
                candidate.repository_url,
                candidate.repository_key,
                candidate.owner,
                candidate.name,
                candidate.source,
                candidate.evidence,
                candidate.confidence,
                1 if candidate.selected else 0,
                1 if candidate.selected else 0,
            ),
        )
    return get_paper_code_repository(conn, paper_id, candidate.repository_key)


def get_paper_code_repository(
    conn: sqlite3.Connection, paper_id: int, repository_key_or_url: str
) -> dict[str, Any] | None:
    repo = parse_github_repo_url(repository_key_or_url)
    if repo is None:
        repo = parse_github_repo_url("https://github.com/" + str(repository_key_or_url))
    if repo is None or not _paper_code_table_exists(conn):
        return None
    row = conn.execute(
        """
        SELECT *
        FROM paper_code_repositories
        WHERE paper_id = ? AND repo_key = ?
        """,
        (paper_id, repo.repository_key),
    ).fetchone()
    return None if row is None else _row_to_dict(row)


def list_paper_code_repositories(
    conn: sqlite3.Connection,
    paper_id: int,
    *,
    selected_only: bool = False,
) -> list[dict[str, Any]]:
    """List stored repository candidates for a paper."""
    if not _paper_code_table_exists(conn):
        return []
    where = "WHERE paper_id = ?"
    params: tuple[Any, ...] = (paper_id,)
    if selected_only:
        where += " AND selected = 1"
    rows = conn.execute(
        f"""
        SELECT *
        FROM paper_code_repositories
        {where}
        ORDER BY selected DESC, confidence DESC, code_repository_id ASC
        """,
        params,
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def select_paper_code_repository(
    conn: sqlite3.Connection, paper_id: int, repository_key_or_url: str
) -> dict[str, Any] | None:
    """Mark exactly one stored repository as selected for a paper."""
    repo = parse_github_repo_url(repository_key_or_url)
    if repo is None:
        repo = parse_github_repo_url("https://github.com/" + str(repository_key_or_url))
    if repo is None:
        raise ValueError("repository_key_or_url must identify a GitHub repository")
    if get_paper_code_repository(conn, paper_id, repo.repository_key) is None:
        return None
    conn.execute(
        """
        UPDATE paper_code_repositories
        SET selected = 0, selected_at = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE paper_id = ?
        """,
        (paper_id,),
    )
    conn.execute(
        """
        UPDATE paper_code_repositories
        SET selected = 1, selected_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        WHERE paper_id = ? AND repo_key = ?
        """,
        (paper_id, repo.repository_key),
    )
    return get_paper_code_repository(conn, paper_id, repo.repository_key)


def enrich_paper_code_repositories(
    conn: sqlite3.Connection | None,
    paper_id: int,
    search_fn: SearchFn | None = None,
    *,
    searcher: SearchFn | None = None,
    max_chunks: int = 64,
    max_chunk_chars: int = 1600,
    max_queries: int = 8,
    select_best: bool = True,
) -> dict[str, Any]:
    """Discover, store, and optionally select code repositories for one paper.

    The deterministic path only uses stored database text. Network access can
    only happen inside a caller-supplied ``search_fn``/``searcher``.
    """
    if search_fn is not None and searcher is not None and search_fn is not searcher:
        raise ValueError("pass only one of search_fn or searcher")
    effective_search = search_fn or searcher

    close = False
    if conn is None:
        conn = get_connection()
        close = True
    try:
        loaded = load_paper_code_inputs(
            conn,
            paper_id,
            max_chunks=max_chunks,
            max_chunk_chars=max_chunk_chars,
        )
        paper = loaded["paper"]
        chunks = loaded["chunks"]
        explicit_candidates = discover_explicit_code_repositories(paper, chunks)
        queries = derive_code_repository_queries(paper, chunks, max_queries=max_queries)

        search_candidates: list[CodeRepositoryCandidate] = []
        search_enabled = effective_search is not None
        search_attempted = False
        if search_enabled and queries and not explicit_candidates:
            search_attempted = True
            raw_results = effective_search(queries)
            search_candidates = search_results_to_candidates(raw_results, paper=paper)

        candidates = _dedupe_candidates([*explicit_candidates, *search_candidates])
        stored = [upsert_code_repository_candidate(conn, paper_id, candidate) for candidate in candidates]

        selected = None
        if select_best and stored:
            best = max(
                stored,
                key=lambda row: (
                    float(row.get("confidence") or 0.0),
                    1 if str(row.get("source") or "").startswith("explicit:") else 0,
                    str(row.get("repository_key") or ""),
                ),
            )
            selected = select_paper_code_repository(conn, paper_id, best["repository_key"])
        else:
            selected_rows = list_paper_code_repositories(conn, paper_id, selected_only=True)
            selected = selected_rows[0] if selected_rows else None

        conn.commit()
        return {
            "paper_id": paper_id,
            "queries": queries,
            "search_enabled": search_enabled,
            "search_attempted": search_attempted,
            "candidates": [candidate.to_dict() for candidate in candidates],
            "stored": list_paper_code_repositories(conn, paper_id),
            "selected": selected,
        }
    finally:
        if close:
            conn.close()


__all__ = [
    "CodeRepositoryCandidate",
    "GitHubRepository",
    "derive_code_repository_queries",
    "discover_explicit_code_repositories",
    "enrich_paper_code_repositories",
    "extract_github_repo_urls",
    "extract_github_repository_candidates",
    "get_paper_code_repository",
    "list_paper_code_repositories",
    "load_paper_code_inputs",
    "normalize_github_repo_url",
    "parse_github_repo_url",
    "search_results_to_candidates",
    "select_paper_code_repository",
    "upsert_code_repository_candidate",
]
