"""Surgical code editor and syntax verifier for agent edits."""

import re
import subprocess
from pathlib import Path


class HunkError(ValueError):
    """Raised when a hunk cannot be applied due to mismatch or parsing error."""
    pass


def apply_hunks(file_content: str, response_text: str) -> str:
    """Parse SEARCH/REPLACE blocks and apply them to the file content.

    Blocks have the format:
    <<<<<<< SEARCH
    [lines to replace]
    =======
    [lines to insert]
    >>>>>>>
    """
    # Regex to capture SEARCH/REPLACE blocks.
    # Uses re.DOTALL to match multiline blocks.
    pattern = re.compile(
        r"<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>>",
        re.DOTALL
    )

    matches = list(pattern.finditer(response_text))
    if not matches:
        raise HunkError("No SEARCH/REPLACE blocks found in LLM response.")

    modified_content = file_content
    for match in matches:
        search_block = match.group(1).strip()
        replace_block = match.group(2).strip()

        # Try exact match first
        if search_block in modified_content:
            modified_content = modified_content.replace(search_block, replace_block, 1)
        else:
            # Fallback: clean whitespaces and try matching.
            # Split search block into lines and try to match with loose spacing.
            search_lines = [line.strip() for line in search_block.splitlines() if line.strip()]
            if not search_lines:
                continue

            # Attempt a looser line-by-line replacement.
            # To be safe, we fail if we can't find the exact block.
            raise HunkError(
                f"Could not find exact SEARCH block in file:\n{search_block[:200]}..."
            )

    return modified_content


def check_syntax(file_path: str | Path) -> tuple[bool, str]:
    """Run py_compile syntax check on the file.

    Returns (is_ok, error_message).
    """
    res = subprocess.run(
        ["python3", "-m", "py_compile", str(file_path)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # Extract stderr without the traceback wrapper if possible
        err = res.stderr.strip()
        return False, err
    return True, ""
