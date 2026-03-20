#!/usr/bin/env python3
"""Validate that the refactored pipeline produces identical diagnostic output.

Usage:
    python scripts/validate_refactor.py <baseline_file> <new_file>

Compares two --diagnose outputs, ignoring timing lines and warnings.
Reports any differences and exits with non-zero status on mismatch.
"""

import re
import sys


def clean_output(text: str) -> list[str]:
    """Remove non-deterministic lines (timing, warnings, cached data messages)."""
    lines = []
    for line in text.splitlines():
        # Skip warning lines, timing lines, and cache loading messages
        if line.startswith("warning:"):
            continue
        if "trained in" in line:
            # Normalize timing: "trained in 2.3s" -> "trained in Xs"
            line = re.sub(r"trained in \d+\.\d+s", "trained in Xs", line)
        if line.startswith("Loading cached"):
            continue
        lines.append(line)
    return lines


def extract_numbers(lines: list[str]) -> dict[str, float]:
    """Extract all numeric values with their context for comparison."""
    numbers = {}
    for i, line in enumerate(lines):
        # Find all floating point numbers in the line
        for match in re.finditer(r'[-+]?\d+\.?\d*%?', line):
            key = f"L{i}:{line.strip()[:60]}:{match.start()}"
            val = match.group().rstrip('%')
            try:
                numbers[key] = float(val)
            except ValueError:
                pass
    return numbers


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_file> <new_file>")
        sys.exit(1)

    baseline_path, new_path = sys.argv[1], sys.argv[2]

    with open(baseline_path) as f:
        baseline_text = f.read()
    with open(new_path) as f:
        new_text = f.read()

    baseline_lines = clean_output(baseline_text)
    new_lines = clean_output(new_text)

    # Line-by-line comparison
    mismatches = 0
    max_lines = max(len(baseline_lines), len(new_lines))
    for i in range(max_lines):
        bl = baseline_lines[i] if i < len(baseline_lines) else "<missing>"
        nl = new_lines[i] if i < len(new_lines) else "<missing>"
        if bl != nl:
            mismatches += 1
            print(f"MISMATCH at line {i + 1}:")
            print(f"  baseline: {bl}")
            print(f"  new:      {nl}")

    if mismatches == 0:
        print(f"PASS: All {len(baseline_lines)} lines match exactly.")
    else:
        print(f"\nFAIL: {mismatches} line(s) differ.")

    sys.exit(1 if mismatches > 0 else 0)


if __name__ == "__main__":
    main()
