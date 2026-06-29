"""Surgical code editor and syntax verifier for agent edits."""

import re
import subprocess
import sys
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
    # Capture block contents without stripping indentation from the first line.
    # The markers themselves must be on their own lines; spaces before markers
    # are ignored, but spaces inside SEARCH/REPLACE payloads are preserved.
    pattern = re.compile(
        r"^[ \t]*<<<<<<< SEARCH[ \t]*\r?\n(.*?)\r?\n[ \t]*=======[ \t]*\r?\n(.*?)\r?\n[ \t]*>>>>>>>[ \t]*(?=\r?\n|$)",
        re.DOTALL | re.MULTILINE,
    )

    matches = list(pattern.finditer(response_text))
    if not matches:
        raise HunkError("No SEARCH/REPLACE blocks found in LLM response.")

    modified_content = file_content
    for match in matches:
        search_block = match.group(1)
        replace_block = match.group(2)

        if not search_block.strip():
            raise HunkError("SEARCH block must contain at least one non-whitespace line.")

        occurrences = modified_content.count(search_block)
        if occurrences == 0:
            raise HunkError(
                f"Could not find exact SEARCH block in file:\n{search_block[:200]}..."
            )
        if occurrences > 1:
            raise HunkError(
                "SEARCH block matched multiple locations; include more surrounding context."
            )

        modified_content = modified_content.replace(search_block, replace_block, 1)

    return modified_content


def check_syntax(file_path: str | Path) -> tuple[bool, str]:
    """Run py_compile syntax check on the file.

    Returns (is_ok, error_message).
    """
    res = subprocess.run(
        [sys.executable, "-m", "py_compile", str(file_path)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        # Extract stderr without the traceback wrapper if possible
        err = res.stderr.strip()
        return False, err
    return True, ""
