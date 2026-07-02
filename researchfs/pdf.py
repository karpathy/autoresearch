"""PDF text extraction helpers for local ResearchFS ingestion."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_pdf_text(path: Path) -> str:
    """Extract text from a local PDF or raise an actionable ValueError."""
    try:
        reader = PdfReader(str(path))
    except (OSError, PdfReadError, ValueError) as exc:
        raise ValueError(f"Could not extract text from PDF {path}: unreadable PDF file") from exc

    if reader.is_encrypted:
        raise ValueError(f"Could not extract text from PDF {path}: encrypted PDFs are unsupported")

    pages: list[str] = []
    page_number = 0
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
    except (
        PdfReadError,
        KeyError,
        TypeError,
        ValueError,
        NotImplementedError,
    ) as exc:
        raise ValueError(
            f"Could not extract text from PDF {path}: failed while reading page {page_number}"
        ) from exc

    content = "\n\n".join(pages).strip()
    if not content:
        raise ValueError(f"PDF {path} does not contain extractable text")
    return content
