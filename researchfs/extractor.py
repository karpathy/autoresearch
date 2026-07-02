"""LLM-based Method Extractor for parsing research artifacts into ResearchFS methods."""

import json
import re
from typing import Callable, List, Dict, Any

EXTRACTION_WINDOW_SIZE = 30

PROMPT_VERSION = "researchfs-extractor-windowed-methods-v3"

METHOD_CATEGORY_TOKENS = (
    "attention",
    "mlp",
    "optimizer",
    "normalization",
    "precision",
    "regularization",
    "embeddings",
    "method",
    "architecture",
    "evaluation",
    "dataset",
    "instrumentation",
    "protocol",
    "implementation",
    "risk_mitigation",
)


def _validate_integer(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split clean research artifact text into overlapping chunks of approx chunk_size characters."""
    _validate_integer(chunk_size, "chunk_size")
    _validate_integer(overlap, "overlap")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to align end to a paragraph or sentence boundary
        if end < len(text):
            next_newline = text.find("\n\n", end - overlap, end + overlap)
            if next_newline != -1:
                end = next_newline + 2
            else:
                next_period = text.find(". ", end - overlap, end + overlap)
                if next_period != -1:
                    end = next_period + 2
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


def assemble_extractor_prompt(meta: Dict[str, Any], chunks: List[str]) -> str:
    """Assemble a prompt instructing the LLM to extract structured research methods."""
    chunks_str = ""
    for i, chunk in enumerate(chunks):
        chunks_str += f"=== CHUNK {i} ===\n{chunk}\n\n"

    title = meta.get("title") or "Untitled"
    authors = meta.get("authors") or "unknown"
    abstract = meta.get("abstract") or ""
    prompt = (
        "You are an expert research engineer and database curator.\n"
        f"Prompt contract version: {PROMPT_VERSION}\n"
        "The chunks below may be one bounded window from a longer research artifact. Extract only from the visible numbered chunks, and report source_chunk_indices using the local CHUNK labels shown in this prompt.\n"
        "Do not infer methods from title, authors, abstract, or outside knowledge unless a visible chunk supports them.\n"
        "Your task is to analyze the following research artifact, such as a paper, lab note, experiment report, or repository README, and extract specific, actionable methods, designs, protocols, or implementation lessons.\n\n"
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n\n"
        "Here are the numbered text chunks from the research artifact:\n"
        f"{chunks_str}\n"
        "Analyze the chunks and extract all distinct methods, architectures, evaluation setups, datasets, protocols, instrumentation details, implementation techniques, or risk mitigations described in the artifact.\n"
        "For each item, output a structured JSON schema. The JSON block must have this exact format:\n\n"
        "```json\n"
        "{\n"
        '  "methods": [\n'
        "    {\n"
        '      "name": "Method Name (e.g. SwiGLU MLP, evaluation protocol, dataset curation workflow)",\n'
        '      "description": "High-level description of what the item does and why it helps.",\n'
        '      "implementation_hint": "Concrete code snippet, configuration example, pseudo-code, protocol step, or operational instruction showing how to reproduce or apply it.",\n'
        '      "risks": "Warnings or tradeoffs (e.g., OOM risk, initialization sensitivity, hardware dependencies, protocol drift, or measurement bias).",\n'
        f'      "category": "one of: {", ".join(METHOD_CATEGORY_TOKENS)}",\n'
        '      "source_chunk_indices": [0, 1]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "For source_chunk_indices, list the integer indices of the CHUNKs that support each method.\n"
        "Only extract methods that have concrete description and implementation hints. Do not extract generic concepts.\n"
        "Reply ONLY with the raw JSON block inside the markdown ```json ``` markers, nothing else."
    )
    return prompt


def extract_json_block(text: str) -> str:
    """Extract json code block from markdown response."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback if raw JSON without block markers
    match_raw = re.search(r"^\s*(\{.*?\})\s*$", text, re.DOTALL)
    if match_raw:
        return match_raw.group(1).strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx : end_idx + 1].strip()
    return text.strip()


def _parse_extractor_methods(response: str) -> list:
    """Parse and validate one extractor response payload."""
    json_str = extract_json_block(response)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError("LLM response did not contain valid JSON schema.") from e

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object with a 'methods' list.")
    methods = data.get("methods", [])
    if methods is None:
        methods = []
    if not isinstance(methods, list):
        raise ValueError("LLM response JSON field 'methods' must be a list.")
    return methods


def _remap_method_source_indices(method: object, window_start: int, window_length: int) -> object:
    """Return a method with local source indices remapped to full-paper chunk indices."""
    if not isinstance(method, dict):
        return method
    source_indices = method.get("source_chunk_indices")
    if not isinstance(source_indices, list):
        return dict(method)

    remapped = [
        window_start + idx
        for idx in source_indices
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < window_length
    ]
    return {**method, "source_chunk_indices": remapped}


def _merge_extracted_methods(methods: list) -> list:
    """Deduplicate methods by valid name while unioning remapped source indices."""
    merged: list = []
    by_name: Dict[str, int] = {}
    for method in methods:
        if not isinstance(method, dict):
            merged.append(method)
            continue
        name = str(method.get("name") or "").strip()
        if not name:
            merged.append(method)
            continue
        key = name.casefold()
        if key not in by_name:
            by_name[key] = len(merged)
            merged.append(method)
            continue

        existing = merged[by_name[key]]
        if not isinstance(existing, dict):
            continue
        existing_indices = existing.get("source_chunk_indices")
        method_indices = method.get("source_chunk_indices")
        if isinstance(existing_indices, list) and isinstance(method_indices, list):
            unioned = list(dict.fromkeys(existing_indices + method_indices))
            existing["source_chunk_indices"] = unioned
    return merged


def extract_methods_from_chunks(
    meta: Dict[str, Any],
    chunks: List[str],
    llm_fn: Callable[[str], str],
    window_size: int = EXTRACTION_WINDOW_SIZE,
) -> Dict[str, Any]:
    """Extract methods from every chunk using bounded local-numbered windows."""
    _validate_integer(window_size, "window_size")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    methods = []
    for window_start in range(0, len(chunks), window_size):
        window_chunks = chunks[window_start : window_start + window_size]
        prompt = assemble_extractor_prompt(meta, window_chunks)
        response = llm_fn(prompt) or ""
        methods.extend(
            _remap_method_source_indices(method, window_start, len(window_chunks))
            for method in _parse_extractor_methods(response)
        )

    return {
        "paper": meta,
        "chunks": chunks,
        "methods": _merge_extracted_methods(methods),
    }


def extract_methods_from_paper(meta: Dict[str, Any], llm_fn: Callable[[str], str]) -> Dict[str, Any]:
    """Split artifact content, extract structured methods using LLM, and map back to source chunks."""
    content = meta.get("content", meta.get("abstract", ""))
    chunks = chunk_text(content)

    # If the artifact is very short, just use abstract as a single chunk
    if not chunks:
        chunks = [meta.get("abstract", "")]

    return extract_methods_from_chunks(meta, chunks, llm_fn)
