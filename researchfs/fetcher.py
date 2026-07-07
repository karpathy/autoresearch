"""Fetchers for downloading arXiv LaTeX source papers and GitHub repository metadata."""

import gzip
import html
import itertools
import io
import os
import re
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests

from researchfs.pdf import extract_pdf_text
from researchfs.scraping import (
    ScrapingEngine,
    load_proxy_list,
    normalize_proxy_url,
    proxy_dict,
    retry_count,
    retry_delay_seconds,
)


_ARXIV_MODERN_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_OLD_RE = re.compile(r"^[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7}$")
_ARXIV_PROXY_URL_ENV = "RESEARCHFS_ARXIV_PROXY_URL"
_ARXIV_PROXY_LIST_PATH_ENV = "RESEARCHFS_ARXIV_PROXY_LIST_PATH"
_ARXIV_RETRIES_ENV = "RESEARCHFS_ARXIV_RETRIES"
_ARXIV_RETRY_DELAY_ENV = "RESEARCHFS_ARXIV_RETRY_DELAY_SECONDS"
_ARXIV_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_ARXIV_PROXY_ROTATION_COUNTER = itertools.count()
_ARXIV_ENGINE = ScrapingEngine(
    proxy_url_env=_ARXIV_PROXY_URL_ENV,
    proxy_list_path_env=_ARXIV_PROXY_LIST_PATH_ENV,
    retries_env=_ARXIV_RETRIES_ENV,
    retry_delay_env=_ARXIV_RETRY_DELAY_ENV,
    retry_status_codes=_ARXIV_RETRY_STATUS_CODES,
    rotation_counter=_ARXIV_PROXY_ROTATION_COUNTER,
)


def _arxiv_proxy_dict(proxy_url: str) -> dict[str, str]:
    """Return a requests proxy mapping for one proxy URL."""
    return proxy_dict(proxy_url)


def _normalize_arxiv_proxy_url(raw_proxy: str) -> str | None:
    """Normalize one proxy-list entry to a proxy URL, or skip invalid entries."""
    return normalize_proxy_url(raw_proxy)


def _arxiv_proxy_config() -> dict[str, str] | None:
    """Return explicit single-proxy settings for arXiv requests, if configured."""
    proxy_url = os.environ.get(_ARXIV_PROXY_URL_ENV, "").strip()
    if not proxy_url:
        return None
    return proxy_dict(proxy_url)


def _arxiv_proxy_list_configs() -> list[dict[str, str]]:
    """Return proxy settings loaded from a configured arXiv proxy-list file."""
    list_path = os.environ.get(_ARXIV_PROXY_LIST_PATH_ENV, "").strip()
    if not list_path:
        return []
    return load_proxy_list(list_path)


def _arxiv_retry_count() -> int:
    """Return bounded arXiv retry count from environment configuration."""
    return retry_count(_ARXIV_RETRIES_ENV)


def _arxiv_retry_delay_seconds() -> float:
    """Return bounded arXiv retry delay from environment configuration."""
    return retry_delay_seconds(_ARXIV_RETRY_DELAY_ENV)


def _arxiv_get(url: str, *, timeout: int, **kwargs):
    """GET an arXiv URL with optional proxy and retry behavior."""
    _ARXIV_ENGINE.rotation_counter = _ARXIV_PROXY_ROTATION_COUNTER
    _ARXIV_ENGINE.retry_status_codes = frozenset(_ARXIV_RETRY_STATUS_CODES)
    return _ARXIV_ENGINE.get(url, timeout=timeout, **kwargs)


def _arxiv_error_label(exc: Exception) -> str:
    """Return non-sensitive arXiv error text safe for logs."""
    return exc.__class__.__name__


def normalize_arxiv_id(value: str) -> str:
    """Normalize an arXiv identifier to its canonical bare form.

    Accepts bare IDs (``2305.13245``), versioned IDs (``2305.13245v1``),
    prefixed IDs (``arXiv:2305.13245``), and arXiv URLs
    (``https://arxiv.org/abs/2305.13245`` or ``https://arxiv.org/pdf/2305.13245.pdf``).

    Raises:
        ValueError: If the input is not a valid arXiv identifier.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Invalid arXiv identifier: input must be a non-empty string")

    normalized = value.strip()

    # Strip arXiv: / arxiv: prefix (case-insensitive).
    if normalized.lower().startswith("arxiv:"):
        normalized = normalized[6:].strip()

    # Parse URL form.
    parsed = urlparse(normalized)
    if parsed.scheme or parsed.netloc:
        if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
            raise ValueError(f"Invalid arXiv identifier: unsupported URL host {parsed.netloc!r}")
        path = parsed.path.strip("/")
        if path.endswith(".pdf"):
            path = path[:-4]
        parts = path.split("/")
        if len(parts) != 2 or parts[0] not in {"abs", "pdf"}:
            raise ValueError(f"Invalid arXiv identifier: unsupported URL path {parsed.path!r}")
        normalized = parts[1]

    # Strip version suffix to get canonical ID.
    normalized = re.sub(r"v\d+$", "", normalized)

    if _ARXIV_MODERN_RE.match(normalized) or _ARXIV_OLD_RE.match(normalized):
        return normalized

    raise ValueError(f"Invalid arXiv identifier: {value!r}")


def fetch_arxiv_metadata_api(arxiv_id: str) -> dict:
    """Fetch paper metadata from the arXiv API.

    Returns a dict with the same keys as :func:`fetch_arxiv_metadata`:
    ``title``, ``authors``, ``abstract``, ``publish_date``, ``url``.
    """
    normalized_id = normalize_arxiv_id(arxiv_id)
    url = f"http://export.arxiv.org/api/query?id_list={normalized_id}"
    res = _arxiv_get(url, timeout=30)
    res.raise_for_status()

    root = ET.fromstring(res.content)
    atom_ns = "{http://www.w3.org/2005/Atom}"

    entry = root.find(f"{atom_ns}entry")
    if entry is None:
        raise RuntimeError(f"arXiv API returned no entry for {normalized_id}")

    title_el = entry.find(f"{atom_ns}title")
    title = (
        " ".join(title_el.text.split())
        if title_el is not None and title_el.text
        else f"arXiv:{normalized_id}"
    )

    authors = []
    for author in entry.findall(f"{atom_ns}author"):
        name_el = author.find(f"{atom_ns}name")
        if name_el is not None and name_el.text:
            authors.append(" ".join(name_el.text.split()))
    authors_str = ", ".join(authors)

    summary_el = entry.find(f"{atom_ns}summary")
    abstract = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""

    published_el = entry.find(f"{atom_ns}published")
    publish_date = published_el.text[:10] if published_el is not None and published_el.text else "unknown"

    id_el = entry.find(f"{atom_ns}id")
    canonical_url = id_el.text.strip() if id_el is not None and id_el.text else f"https://arxiv.org/abs/{normalized_id}"

    return {
        "title": title,
        "authors": authors_str,
        "abstract": abstract,
        "publish_date": publish_date,
        "url": canonical_url,
    }


def _fetch_arxiv_metadata_html(arxiv_id: str) -> dict:
    """Fetch paper metadata by scraping the arXiv abstract HTML page."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    res = _arxiv_get(url, timeout=30)
    res.raise_for_status()
    html_text = res.text

    # Extract Title: <h1 class="title mathjax"><span class="descriptor">Title:</span>[...]</h1>
    title_match = re.search(
        r'<h1\s+[^>]*class="[^"]*title[^"]*"[^>]*>.*?<span\s+class="descriptor">Title:</span>\s*(.*?)\s*</h1>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    title = title_match.group(1).strip() if title_match else f"arXiv:{arxiv_id}"
    title = re.sub(r'<span\s+class="mathjax">.*?</span>', "", title, flags=re.DOTALL | re.IGNORECASE)
    title = re.sub(r"<.*?>", "", title)
    title = html.unescape(re.sub(r"\s+", " ", title)).strip()

    # Extract Authors: <div class="authors"><span class="descriptor">Authors:</span>[...]</div>
    authors_match = re.search(
        r'<div\s+[^>]*class="[^"]*authors[^"]*"[^>]*>.*?<span\s+class="descriptor">Authors:</span>\s*(.*?)\s*</div>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    authors = ""
    if authors_match:
        authors_html = authors_match.group(1)
        author_names = [
            html.unescape(re.sub(r"<.*?>", "", name)).strip()
            for name in re.findall(r'<a\s+href="[^"]*">(.*?)</a>', authors_html, flags=re.DOTALL | re.IGNORECASE)
        ]
        authors = ", ".join(name for name in author_names if name)
        if not authors:
            authors = re.sub(r"<.*?>", "", authors_html).strip()
            authors = html.unescape(re.sub(r"\s+", " ", authors)).strip()

    # Extract Abstract: <blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>[...]</blockquote>
    abstract_match = re.search(
        r'<blockquote\s+[^>]*class="[^"]*abstract[^"]*"[^>]*>.*?<span\s+class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    abstract = ""
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r"<.*?>", "", abstract)
        abstract = html.unescape(re.sub(r"\s+", " ", abstract)).strip()

    # Extract Date: [Submitted on Mon, 22 May 2023 ...]
    date_match = re.search(
        r'<div\s+[^>]*class="[^"]*dateline[^"]*"[^>]*>\s*\[\s*Submitted\s+on\s+([^,<>,\]\(]+)',
        html_text,
        re.DOTALL | re.IGNORECASE,
    )
    publish_date = date_match.group(1).strip() if date_match else "unknown"
    publish_date = re.sub(r"<.*?>", "", publish_date)
    publish_date = html.unescape(publish_date.split("(")[0].split(",")[0].strip())

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publish_date": publish_date,
        "url": url,
    }


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch title, authors, abstract, and publish date for an arXiv paper.

    Prefers the arXiv API and falls back to HTML scraping if the API fails.
    """
    normalized_id = normalize_arxiv_id(arxiv_id)
    try:
        return fetch_arxiv_metadata_api(normalized_id)
    except Exception as api_exc:
        print(
            "Warning: arXiv API metadata fetch failed, falling back to HTML scraping "
            f"({_arxiv_error_label(api_exc)})"
        )
        return _fetch_arxiv_metadata_html(normalized_id)


def clean_latex(latex_content: str) -> str:
    """Remove LaTeX comments and excessive blank lines to keep content clean."""
    # Remove single line comments starting with %
    # Matches % only if not escaped by \
    lines = []
    for line in latex_content.splitlines():
        # Strip comments
        clean_line = re.sub(r"(?<!\\)%.*$", "", line)
        if clean_line.strip():
            lines.append(clean_line)
    return "\n".join(lines)


def _extract_latex_from_source(content_bytes: bytes) -> str:
    """Extract ``.tex`` content from arXiv source archive bytes."""
    # Try extracting as tarfile (.tar.gz)
    try:
        with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as tar:
            tex_contents = []
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        tex_contents.append(f.read().decode("utf-8", errors="ignore"))
            latex_text = "\n".join(tex_contents)
            if latex_text:
                return latex_text
    except tarfile.TarError:
        pass

    # Try extracting as a single gzipped file (.gz)
    try:
        latex_text = gzip.decompress(content_bytes).decode("utf-8", errors="ignore")
        if latex_text:
            return latex_text
    except OSError:
        pass

    # Fallback: if it's not compressed, it might be raw text
    try:
        latex_text = content_bytes.decode("utf-8", errors="ignore")
        source_markers = ("\\documentclass", "\\begin{", "\\section", "\\input{", "\\usepackage")
        if latex_text and any(marker in latex_text for marker in source_markers):
            return latex_text
    except Exception:
        pass

    return ""


def fetch_arxiv_source(arxiv_id: str) -> dict:
    """Download content for an arXiv paper and return metadata plus extracted text.

    Attempts, in order:

    1. LaTeX source archive (``.tar.gz``) extraction.
    2. PDF fallback using :func:`researchfs.pdf.extract_pdf_text`.
    3. Abstract fallback from the paper metadata.
    """
    normalized_id = normalize_arxiv_id(arxiv_id)
    meta = fetch_arxiv_metadata(normalized_id)

    src_url = f"https://arxiv.org/src/{normalized_id}"
    print(f"Downloading LaTeX source from {src_url}...")
    latex_text = ""
    try:
        res = _arxiv_get(src_url, timeout=60)
        res.raise_for_status()
        latex_text = _extract_latex_from_source(res.content)
    except Exception as exc:
        print(f"Warning: Could not download LaTeX source ({_arxiv_error_label(exc)})")

    if not latex_text:
        pdf_url = f"https://arxiv.org/pdf/{normalized_id}.pdf"
        print(f"Downloading PDF fallback from {pdf_url}...")
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                pdf_res = _arxiv_get(pdf_url, timeout=60)
                pdf_res.raise_for_status()
                tmp.write(pdf_res.content)
                tmp_path = tmp.name
            latex_text = extract_pdf_text(Path(tmp_path))
        except Exception as pdf_exc:
            print(f"Warning: PDF fallback failed ({_arxiv_error_label(pdf_exc)})")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    if latex_text:
        meta["content"] = clean_latex(latex_text)
    else:
        meta["content"] = meta["abstract"]

    meta["external_id"] = normalized_id
    return meta


def fetch_arxiv_query(query: str, max_results: int = 10) -> list[str]:
    """Search arXiv and return a list of normalized arXiv IDs.

    Args:
        query: arXiv search query string (e.g. ``cat:cs.AI`` or ``all:transformers``).
        max_results: Maximum number of results to return (positive integer, capped at 100).

    Raises:
        ValueError: If ``query`` is empty or ``max_results`` is not a positive integer.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("arXiv query must be a non-empty string")

    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ValueError("max_results must be a positive integer")
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    max_results = min(max_results, 100)

    params = {
        "search_query": query.strip(),
        "max_results": max_results,
        "start": 0,
    }
    url = "http://export.arxiv.org/api/query"
    res = _arxiv_get(url, params=params, timeout=30)
    res.raise_for_status()

    root = ET.fromstring(res.content)
    atom_ns = "{http://www.w3.org/2005/Atom}"
    ids = []
    for entry in root.findall(f"{atom_ns}entry"):
        id_el = entry.find(f"{atom_ns}id")
        if id_el is None or not id_el.text:
            continue
        parsed = urlparse(id_el.text.strip())
        path = parsed.path.strip("/")
        parts = path.split("/")
        if len(parts) == 2 and parts[0] == "abs":
            try:
                ids.append(normalize_arxiv_id(parts[1]))
            except ValueError:
                pass
    return ids


def _parse_github_repo_path(repo_path: str) -> tuple[str, str]:
    """Return ``(owner, repo)`` for a bare owner/repo path or GitHub repo URL."""
    value = repo_path.strip()
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
        if parsed.query or parsed.fragment:
            raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
        clean_path = parsed.path.strip("/")
    else:
        if "?" in value or "#" in value:
            raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
        clean_path = value.strip("/")

    parts = clean_path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
    owner, repo = parts
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo:
        raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
    return owner, repo


def fetch_github_readme(repo_path: str) -> dict:
    """Download the README file from a GitHub repository."""
    owner, repo = _parse_github_repo_path(repo_path)
    clean_path = f"{owner}/{repo}"
    url = f"https://github.com/{clean_path}"

    # Try main branch first, then master
    readme_text = ""
    for branch in ["main", "master"]:
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
        print(f"Checking README at {readme_url}...")
        res = requests.get(readme_url, timeout=30)
        if res.status_code == 404:
            continue
        res.raise_for_status()
        readme_text = res.text
        break

    if not readme_text:
        raise RuntimeError(f"Could not find README.md for repo {clean_path} on main or master branch")

    # Extract a simple abstract/summary from the first few lines of README
    abstract_lines = []
    for line in readme_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        abstract_lines.append(line)
        if len(abstract_lines) >= 3:
            break
    abstract = " ".join(abstract_lines)

    return {
        "title": f"GitHub: {clean_path}",
        "authors": owner,
        "abstract": abstract,
        "publish_date": "n/a",
        "url": url,
        "content": readme_text,
    }


if __name__ == "__main__":
    # Test normalize_arxiv_id
    test_cases = [
        "2305.13245",
        "2305.13245v1",
        "arXiv:2305.13245",
        "arXiv:2305.13245v2",
        "https://arxiv.org/abs/2305.13245",
        "https://arxiv.org/pdf/2305.13245.pdf",
    ]
    for raw in test_cases:
        normalized = normalize_arxiv_id(raw)
        print(f"Normalized {raw!r} -> {normalized}")

    # Quick sanity check
    try:
        meta = fetch_arxiv_metadata("2305.13245")
        print("Metadata Fetch OK:", meta["title"])

        # Test cleaning latex
        test_latex = "Hello % comment here\nworld!"
        assert clean_latex(test_latex) == "Hello \nworld!"
        print("clean_latex OK")
    except Exception as e:
        print("Sanity check failed:", e)
