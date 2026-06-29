"""LLM-based Method Extractor for parsing deep learning literature into ResearchFS methods."""

import json
import re
from typing import Callable, List, Dict, Any


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split clean paper text into overlapping chunks of approx chunk_size characters."""
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
    """Assemble a prompt instructing the LLM to extract structured deep learning methods."""
    chunks_str = ""
    for i, chunk in enumerate(chunks):
        chunks_str += f"=== CHUNK {i} ===\n{chunk}\n\n"

    prompt = (
        "You are an expert machine learning researcher and database engineer.\n"
        "Your task is to analyze the following deep learning paper and extract specific, actionable optimization methods/architectures.\n\n"
        f"Title: {meta['title']}\n"
        f"Authors: {meta['authors']}\n"
        f"Abstract: {meta['abstract']}\n\n"
        "Here are the numbered text chunks from the paper:\n"
        f"{chunks_str}\n"
        "Analyze the chunks and extract all distinct optimization methods, architectures, or hyperparameter tricks described in the paper.\n"
        "For each method, output a structured JSON schema. The JSON block must have this exact format:\n\n"
        "```json\n"
        "{\n"
        '  "methods": [\n'
        "    {\n"
        '      "name": "Method Name (e.g. SwiGLU MLP)",\n'
        '      "description": "High-level description of what the method does and why it helps.",\n'
        '      "implementation_hint": "Concrete code snippet, configuration example, or pseudo-code showing how to write it in PyTorch.",\n'
        '      "risks": "Warnings or tradeoffs (e.g., OOM risk, initialization sensitivity, or hardware dependencies).",\n'
        '      "category": "one of: attention, mlp, optimizer, normalization, precision, regularization, embeddings",\n'
        '      "source_chunk_indices": [0, 1]  // List the integer indices of the CHUNKs that support this method\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "Only extract methods that have concrete description and implementation hints. Do not extract generic concepts.\n"
        "Reply ONLY with the raw JSON block inside the markdown ```json ``` markers, nothing else."
    )
    return prompt


def extract_json_block(text: str) -> str:
    """Extract json code block from markdown response."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback if raw JSON without block markers
    match_raw = re.search(r"^\s*(\{.*?\})\s*$", text, re.DOTALL)
    if match_raw:
        return match_raw.group(1).strip()
    return text.strip()


def extract_methods_from_paper(meta: Dict[str, Any], llm_fn: Callable[[str], str]) -> Dict[str, Any]:
    """Split the paper content, extract structured methods using LLM, and map back to source chunks."""
    content = meta.get("content", meta.get("abstract", ""))
    chunks = chunk_text(content)

    # If the paper is very short, just use abstract as a single chunk
    if not chunks:
        chunks = [meta.get("abstract", "")]

    prompt = assemble_extractor_prompt(meta, chunks[:30])  # Cap at 30 chunks to prevent context overflow

    print("Calling LLM to extract methods...")
    response = llm_fn(prompt)
    json_str = extract_json_block(response)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse LLM JSON response:", e)
        print("Raw JSON string attempted:", json_str)
        # Attempt minimal fallback parsing or raise
        raise ValueError("LLM response did not contain valid JSON schema.")

    return {
        "paper": meta,
        "chunks": chunks,
        "methods": data.get("methods", []),
    }
