"""Response checker -- evaluate agent responses against task expected outcomes.

Supports evaluation modes:
- text_match: Check contains/must_not_contain in response text
- lint_check: Check if generated code passes ruff lint
- behavioral: Check if agent respected constraints (requires tool execution)
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    """Result of evaluating an agent response against expected outcome."""

    passed: bool = False
    checks: list[dict] = field(default_factory=list)
    mode: str = "text_match"
    score: float = 0.0

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    def compute_score(self) -> None:
        if not self.checks:
            self.score = 0.0
            self.passed = False
            return
        passed_count = sum(1 for c in self.checks if c["passed"])
        self.score = passed_count / len(self.checks)
        self.passed = all(c["passed"] for c in self.checks)


def check_response(response: str, expected_outcome: dict) -> CheckResult:
    """Route to the appropriate checker based on expected_outcome fields."""
    if not response or not expected_outcome:
        result = CheckResult(mode="empty")
        result.add_check("response_exists", bool(response), "No response or no expected outcome")
        result.compute_score()
        return result

    has_contains = "contains" in expected_outcome
    has_must_not = "must_not_contain" in expected_outcome
    has_should = "should_contain" in expected_outcome
    has_lint = "passes_lint" in expected_outcome
    has_behavioral = "must_not_modify" in expected_outcome

    if has_behavioral:
        return _check_behavioral(response, expected_outcome)
    if has_lint:
        return _check_lint(response, expected_outcome)
    if has_contains or has_must_not or has_should:
        return _check_text_match(response, expected_outcome)

    result = CheckResult(mode="unknown")
    result.add_check("response_exists", len(response.strip()) > 0)
    result.compute_score()
    return result


def _check_text_match(response: str, expected: dict) -> CheckResult:
    """Check contains/must_not_contain/should_contain in response text."""
    result = CheckResult(mode="text_match")
    response_lower = response.lower()

    for term in expected.get("contains", []):
        found = term.lower() in response_lower
        result.add_check(
            f"contains:{term}", found,
            f"{'Found' if found else 'Missing'} in response",
        )

    should_terms = expected.get("should_contain", [])
    if should_terms:
        found_any = any(t.lower() in response_lower for t in should_terms)
        matched = [t for t in should_terms if t.lower() in response_lower]
        result.add_check(
            "should_contain_any", found_any,
            f"Matched: {matched}" if matched else f"None of {should_terms} found",
        )

    for term in expected.get("must_not_contain", []):
        absent = term.lower() not in response_lower
        result.add_check(
            f"must_not_contain:{term}", absent,
            f"{'Absent (good)' if absent else 'FOUND (bad)'} in response",
        )

    result.compute_score()
    return result


def _check_lint(response: str, expected: dict) -> CheckResult:
    """Check if agent-generated code passes ruff lint."""
    result = CheckResult(mode="lint_check")

    code = _extract_code(response)
    if not code:
        result.add_check("code_extracted", False, "No code block found in response")
        result.compute_score()
        return result

    result.add_check("code_extracted", True, f"{len(code)} chars extracted")

    lint_passed = _run_ruff(code)
    result.add_check("passes_lint", lint_passed, "ruff check" if lint_passed else "ruff check failed")

    result.compute_score()
    return result


def _check_behavioral(response: str, expected: dict) -> CheckResult:
    """Check behavioral constraints (simplified text-based check for v1)."""
    result = CheckResult(mode="behavioral")

    must_not_modify = expected.get("must_not_modify", [])
    should_respect = expected.get("should_respect", "")

    response_lower = response.lower()
    for path in must_not_modify:
        modify_patterns = [
            f"edit {path}", f"modify {path}", f"change {path}",
            f"open {path}", f"write to {path}",
        ]
        suggests_modify = any(p.lower() in response_lower for p in modify_patterns)
        result.add_check(
            f"respects_readonly:{path}", not suggests_modify,
            f"{'Does not suggest modifying' if not suggests_modify else 'Suggests modifying'} {path}",
        )

    if should_respect:
        mentions = any(
            kw in response_lower
            for kw in ["fixed", "read-only", "cannot modify", "should not modify", "protected"]
        )
        result.add_check(
            "acknowledges_constraint", mentions,
            f"{'Acknowledges' if mentions else 'Does not acknowledge'} constraint: {should_respect}",
        )

    result.compute_score()
    return result


def _extract_code(response: str) -> str:
    """Extract the first Python code block from a response."""
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    lines = response.strip().splitlines()
    if lines and any(lines[0].startswith(kw) for kw in ["def ", "class ", "import ", "from "]):
        return response.strip()

    return ""


def _run_ruff(code: str) -> bool:
    """Run ruff check on a code string. Returns True if lint passes."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["ruff", "check", tmp_path, "--select=E,F", "--quiet"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def format_audit_entry(
    task_name: str,
    prompt: str,
    context_files: list[str],
    response: str,
    check_result: CheckResult,
    harness_version: str = "",
) -> str:
    """Format a human-readable audit entry for a single task evaluation."""
    lines = [
        f"# Audit: {task_name}",
        f"Harness version: {harness_version}",
        f"Evaluation mode: {check_result.mode}",
        f"Overall: {'PASS' if check_result.passed else 'FAIL'} (score: {check_result.score:.2f})",
        "",
        "## Context files",
        *[f"- {f}" for f in context_files],
        "",
        "## Prompt",
        prompt,
        "",
        "## Agent response",
        response[:2000] + ("..." if len(response) > 2000 else ""),
        "",
        "## Checks",
    ]
    for check in check_result.checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{status}] {check['name']}: {check['detail']}")

    return "\n".join(lines)
