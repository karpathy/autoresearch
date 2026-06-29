"""Fetchers for downloading arXiv LaTeX source papers and GitHub repository metadata."""

import gzip
import html
import io
import re
import tarfile

import requests


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch title, authors, abstract, and publish date from arXiv HTML page."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    html_text = res.text

    # Extract Title: <h1 class="title mathjax"><span class="descriptor">Title:</span>[...]</h1>
    title_match = re.search(
        r'<h1 class="title mathjax"><span class="descriptor">Title:</span>\s*(.*?)</h1>',
        html_text,
        re.DOTALL,
    )
    title = title_match.group(1).strip() if title_match else f"arXiv:{arxiv_id}"
    title = re.sub(r'<span class="mathjax">.*?</span>', "", title)
    title = re.sub(r"<.*?>", "", title)
    title = html.unescape(re.sub(r"\s+", " ", title)).strip()

    # Extract Authors: <div class="authors"><span class="descriptor">Authors:</span>[...]</div>
    authors_match = re.search(
        r'<div class="authors"><span class="descriptor">Authors:</span>\s*(.*?)</div>',
        html_text,
        re.DOTALL,
    )
    authors = ""
    if authors_match:
        authors_html = authors_match.group(1)
        author_names = [
            html.unescape(re.sub(r"<.*?>", "", name)).strip()
            for name in re.findall(r'<a href=".*?">(.*?)</a>', authors_html)
        ]
        authors = ", ".join(name for name in author_names if name)
        if not authors:
            authors = re.sub(r"<.*?>", "", authors_html).strip()
            authors = html.unescape(re.sub(r"\s+", " ", authors))

    # Extract Abstract: <blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>[...]</blockquote>
    abstract_match = re.search(
        r'<blockquote class="abstract mathjax"><span class="descriptor">Abstract:</span>\s*(.*?)</blockquote>',
        html_text,
        re.DOTALL,
    )
    abstract = ""
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r"<.*?>", "", abstract)
        abstract = html.unescape(re.sub(r"\s+", " ", abstract)).strip()

    # Extract Date: [Submitted on Mon, 22 May 2023 ...]
    date_match = re.search(r'<div class="dateline">\s*\[Submitted on\s*([^,<>,\]]+)', html_text, re.DOTALL)
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


def fetch_arxiv_source(arxiv_id: str) -> dict:
    """Download LaTeX source for the arXiv paper, extract, and clean the text."""
    meta = fetch_arxiv_metadata(arxiv_id)

    src_url = f"https://arxiv.org/src/{arxiv_id}"
    print(f"Downloading LaTeX source from {src_url}...")
    res = requests.get(src_url, timeout=60)
    res.raise_for_status()

    latex_text = ""
    content_bytes = res.content

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
    except tarfile.TarError:
        # Try extracting as a single gzipped file (.gz)
        try:
            latex_text = gzip.decompress(content_bytes).decode("utf-8", errors="ignore")
        except OSError:
            # Fallback: if it's not compressed, it might be raw text
            try:
                latex_text = content_bytes.decode("utf-8", errors="ignore")
            except Exception:
                print("Warning: Could not parse arXiv source compression. Fallback to Abstract only.")
                latex_text = meta["abstract"]

    # If we got LaTeX text, clean it
    if latex_text:
        meta["content"] = clean_latex(latex_text)
    else:
        meta["content"] = meta["abstract"]

    return meta


def fetch_github_readme(repo_path: str) -> dict:
    """Download the README file from a GitHub repository."""
    repo_path = repo_path.strip()
    match = re.search(r"github\.com/([^/]+/[^/]+)", repo_path)
    clean_path = (match.group(1) if match else repo_path).strip("/")
    owner_repo = clean_path.split("/", 1)
    if len(owner_repo) != 2 or not owner_repo[0] or not owner_repo[1]:
        raise ValueError("repo_path must be 'owner/repo' or a GitHub repository URL")
    owner, repo = owner_repo
    if repo.endswith(".git"):
        repo = repo[:-4]
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
