"""
Research Director — project-level strategic agent that scans the ENTIRE codebase
to determine what aspect of the platform should be tested/improved next.

This is NOT an ML hyperparameter tuner. It surveys all domains of the CRSAI platform
(financial tools, agent graphs, RAG, control plane, infrastructure, etc.) and produces
a prioritized research directive telling the orchestrator WHAT to focus on.

Flow:
    1. Scan codebase structure (file tree, recent git changes, existing tests)
    2. Build a domain coverage map (what has tests, what doesn't, what changed recently)
    3. Identify the highest-value next area to test/improve
    4. Produce a directive with specific hypothesis + acceptance criteria

Usage:
    from research_director import ResearchDirector
    director = ResearchDirector()
    directive = director.next_directive()
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_URL", "http://crsai-vllm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-cascade-2-nvfp4")
WORKSPACE_ROOT = Path(os.getenv("CRSAI_WORKSPACE_ROOT", "/workspace"))
AIQ_ROOT = Path(os.getenv("AIQ_WORKSPACE_ROOT", "/workspace-aiq"))

_LLM_RETRIES = int(os.getenv("RESEARCH_DIRECTOR_LLM_RETRIES", "3"))
_LLM_RETRY_DELAY = float(os.getenv("RESEARCH_DIRECTOR_LLM_RETRY_DELAY", "5.0"))

# ---------------------------------------------------------------------------
# Project domains — the universe of testable areas
# ---------------------------------------------------------------------------

DOMAINS: dict[str, dict] = {
    "financial_tools": {
        "description": "Financial statement builders, ratio analysis, accounting rules",
        "paths": ["cotton/tools/financial/"],
        "test_paths": ["tests/test_e2e_financial_statements.py", "tests/test_e2e_fs_parquet.py"],
        "metrics": ["cross-statement tie-out accuracy", "ratio calculation correctness"],
    },
    "fpa_agents": {
        "description": "FPA specialist nodes: variance, AR/AP, balance sheet, income statement, cash flow analysis",
        "paths": ["cotton/graphs/fpa/"],
        "test_paths": ["tests/test_fpa_agents.py", "tests/test_branches.py"],
        "metrics": ["agent output schema compliance", "analysis quality", "branch reconciliation"],
    },
    "finance_graph": {
        "description": "Month-end close, journal entry review, account reconciliation, AR/AP workers",
        "paths": ["cotton/graphs/finance/"],
        "test_paths": [],
        "metrics": ["GL integrity", "period closure completeness"],
    },
    "control_plane": {
        "description": "Agent lifecycle, event dispatch, resource governance, REST + WebSocket API",
        "paths": ["crsai_control_plane/"],
        "test_paths": ["tests/test_control_plane_decorator.py"],
        "metrics": ["event delivery latency", "resource limit enforcement", "API correctness"],
    },
    "swarm_routing": {
        "description": "CEO → CFO → FPA routing, swarm aggregation, graph edge connectivity",
        "paths": ["cotton/graphs/swarm/", "cotton/graphs/cfo/"],
        "test_paths": ["tests/test_node_graph.py"],
        "metrics": ["routing accuracy", "state propagation correctness"],
    },
    "enhancement_agent": {
        "description": "Scan → analyse → queue → review → execute improvement pipeline",
        "paths": ["cotton/graphs/enhancement_agent/"],
        "test_paths": [],
        "metrics": ["improvement detection rate", "artifact quality", "review gate accuracy"],
    },
    "collaborator_agent": {
        "description": "Multi-turn interview, synthesis, follow-up clarification agent",
        "paths": ["cotton/graphs/collaborator_agent/"],
        "test_paths": [
            "cotton/graphs/collaborator_agent/tests/test_interview.py",
            "cotton/graphs/collaborator_agent/tests/test_synthesis.py",
        ],
        "metrics": ["interview completion rate", "synthesis coherence"],
    },
    "databricks_connectors": {
        "description": "SQL warehouse bridge, incremental refresh, parquet queries, schema caching",
        "paths": ["cotton/databricks/"],
        "test_paths": [],
        "metrics": ["query correctness", "watermark incremental accuracy", "data freshness"],
    },
    "schemas": {
        "description": "Pydantic v2 output models enforcing LLM response structure",
        "paths": ["cotton/schemas/"],
        "test_paths": [],
        "metrics": ["validation error rate", "schema coverage", "serialization round-trip"],
    },
    "utilities": {
        "description": "DB pooling, Redis, MinIO, @control_plane_aware, temperature registry, notifications",
        "paths": ["cotton/utils/"],
        "test_paths": [
            "tests/test_control_plane_decorator.py",
            "tests/test_temperature.py",
            "tests/test_notify.py",
            "tests/test_graph_helpers_unit.py",
        ],
        "metrics": ["connection reliability", "retry correctness", "decorator lifecycle"],
    },
    "mcp_rag": {
        "description": "MCP server with ensemble RAG (3 models + RRF), code indexing, semantic search",
        "paths": ["mcp-rag/"],
        "test_paths": [],
        "metrics": ["search relevance (MRR)", "indexing throughput", "RRF ranking quality"],
    },
    "ml_training": {
        "description": "Autonomous ML pretraining experiments (train.py modifications via LLM)",
        "paths": ["autoresearch/"],
        "test_paths": [],
        "metrics": ["val_bpb (lower is better)", "VRAM usage", "experiment throughput"],
    },
    "ml_forecasting": {
        "description": "ML Forecasting Expert — model selection, experiment design, and results interpretation for time-series forecasting",
        "paths": ["cotton/graphs/ml_forecasting/"],
        "test_paths": [],
        "metrics": [
            "model recommendation quality",
            "experiment plan completeness",
            "forecast accuracy (MAPE, RMSE)",
        ],
    },
    "aiq_agents": {
        "description": "AIQ research agents: deep researcher, chat researcher, clarifier, intent classification",
        "paths": [],
        "aiq_paths": [
            "src/aiq_agent/agents/deep_researcher/",
            "src/aiq_agent/agents/chat_researcher/",
            "src/aiq_agent/agents/clarifier/",
        ],
        "test_paths": [],
        "aiq_test_paths": [
            "tests/aiq_agent/agents/deep_researcher/test_agent.py",
            "tests/aiq_agent/agents/deep_researcher/test_custom_middleware.py",
        ],
        "metrics": [
            "intent classification accuracy",
            "research depth quality",
            "citation accuracy",
        ],
    },
    "knowledge_layer": {
        "description": "Document ingestion, PDF extraction, embedding pipeline, Qdrant vector store",
        "paths": [],
        "aiq_paths": ["sources/knowledge_layer/"],
        "test_paths": [],
        "aiq_test_paths": ["tests/knowledge_layer_tests/test_summary_store.py"],
        "metrics": ["ingestion success rate", "retrieval relevance", "embedding quality"],
    },
    "infrastructure": {
        "description": "Container orchestration, DB migrations, Prometheus/Grafana, Loki logging",
        "paths": ["infra/", "docker-compose.yml"],
        "test_paths": ["tests/test_integration.py"],
        "metrics": ["container health", "service uptime", "migration success"],
    },
    "watchers": {
        "description": "Balance sheet integrity watcher, log watcher, heartbeat mechanism",
        "paths": ["cotton/graphs/finance/"],
        "test_paths": ["tests/test_bs_watcher.py", "tests/test_log_watcher.py"],
        "metrics": ["alarm detection accuracy", "false positive rate"],
    },
}


# ---------------------------------------------------------------------------
# Output schema — validated directive from the LLM
# ---------------------------------------------------------------------------


class DirectiveOutput(BaseModel):
    """Validated output from the Research Director LLM."""

    domain: str = Field(..., description="Domain ID from DOMAINS registry")
    analysis: str = Field(default="", description="2-3 sentence codebase health assessment")
    hypothesis: str = Field(..., description="Specific, testable hypothesis")
    test_approach: str = Field(default="", description="How to test this")
    acceptance_criteria: str = Field(default="", description="What success looks like")
    priority: Literal["critical", "high", "medium", "low"] = Field(default="medium")
    risk_if_untested: str = Field(default="", description="Risk if skipped")
    estimated_scope: Literal["small (1-2 files)", "medium (3-10 files)", "large (10+ files)"] = (
        Field(default="medium (3-10 files)")
    )

    @field_validator("domain")
    @classmethod
    def domain_must_be_known(cls, v: str) -> str:
        if v not in DOMAINS:
            raise ValueError(f"Unknown domain '{v}'. Valid domains: {sorted(DOMAINS.keys())}")
        return v


# ---------------------------------------------------------------------------
# Codebase scanner
# ---------------------------------------------------------------------------


@dataclass
class DomainScan:
    """Result of scanning a single project domain."""

    domain: str
    description: str
    file_count: int = 0
    test_count: int = 0
    recent_changes: int = 0
    recent_change_files: list[str] = field(default_factory=list)
    has_tests: bool = False
    test_coverage_gap: bool = False
    notes: str = ""


def _run(cmd: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Run a shell command and return stdout. Logs warnings on failure."""
    try:
        r = subprocess.run(  # noqa: S602
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        if r.returncode != 0 and r.stderr:
            log.debug("_run: non-zero exit for %r: %s", cmd[:80], r.stderr[:200])
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        log.warning("_run: command timed out after %ds: %r", timeout, cmd[:80])
        return ""
    except Exception as exc:
        log.warning("_run: unexpected error running %r: %s", cmd[:80], exc)
        return ""


def _count_files(root: Path, paths: list[str]) -> int:
    """Count Python files in given paths."""
    total = 0
    for p in paths:
        full = root / p
        if full.is_file():
            total += 1
        elif full.is_dir():
            total += sum(1 for _ in full.rglob("*.py"))
    return total


def _count_test_files(root: Path, paths: list[str]) -> int:
    """Count test files that actually exist."""
    count = 0
    for p in paths:
        full = root / p
        if full.is_file():
            count += 1
        elif full.is_dir():
            count += sum(1 for _ in full.rglob("test_*.py"))
    return count


def _recent_git_changes(root: Path, paths: list[str], days: int = 7) -> tuple[int, list[str]]:
    """Count files changed in the last N days via git log."""
    changed_files: list[str] = []
    for p in paths:
        full = root / p
        if not full.exists():
            continue
        output = _run(
            f'git log --since="{days} days ago" --name-only --pretty=format: -- {p}',
            cwd=str(root),
        )
        if output:
            files = [f for f in output.splitlines() if f.strip()]
            changed_files.extend(files)
    unique = list(set(changed_files))
    return len(unique), unique[:10]


def scan_domain(domain_id: str, config: dict) -> DomainScan:
    """Scan a single domain and return findings."""
    scan = DomainScan(
        domain=domain_id,
        description=config["description"],
    )

    paths = config.get("paths", [])
    aiq_paths = config.get("aiq_paths", [])
    scan.file_count = _count_files(WORKSPACE_ROOT, paths) + _count_files(AIQ_ROOT, aiq_paths)

    test_paths = config.get("test_paths", [])
    aiq_test_paths = config.get("aiq_test_paths", [])
    scan.test_count = _count_test_files(WORKSPACE_ROOT, test_paths) + _count_test_files(
        AIQ_ROOT, aiq_test_paths
    )
    scan.has_tests = scan.test_count > 0
    scan.test_coverage_gap = scan.file_count > 0 and not scan.has_tests

    changes_main, files_main = _recent_git_changes(WORKSPACE_ROOT, paths)
    changes_aiq, files_aiq = _recent_git_changes(AIQ_ROOT, aiq_paths)
    scan.recent_changes = changes_main + changes_aiq
    scan.recent_change_files = files_main + files_aiq

    return scan


def scan_all_domains() -> list[DomainScan]:
    """Scan every domain and return results."""
    scans = []
    for domain_id, config in DOMAINS.items():
        scans.append(scan_domain(domain_id, config))
    return scans


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def score_domain(scan: DomainScan) -> float:
    """Score a domain for testing priority. Higher = more urgent.

    Signals:
    - No tests + has code = high priority (coverage gap)
    - Recent changes + no tests = very high priority (regression risk)
    - Many files + no tests = structural risk
    - Has tests + recent changes = moderate (regression check)
    """
    if scan.file_count == 0:
        return 0.0

    score = 0.0

    if scan.test_coverage_gap:
        score += 50.0

    if scan.recent_changes > 0:
        score += min(scan.recent_changes * 5.0, 30.0)
        if scan.test_coverage_gap:
            score += 20.0

    if scan.file_count > 10:
        score += 10.0
    elif scan.file_count > 5:
        score += 5.0

    if scan.has_tests and scan.recent_changes > 0:
        score += 15.0

    return score


# ---------------------------------------------------------------------------
# LLM analysis — strategic decision
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM_PROMPT = """You are the Research Director for the CRSAI platform — a LangGraph-based
financial agent swarm for Cotton Holdings.

Your job: analyze codebase scan results and decide WHAT ASPECT of the platform
should be tested, validated, or improved next.

You receive a structured scan showing every domain, its file count, test count,
recent git changes, and coverage gaps. Use this to make a strategic decision.

DECISION CRITERIA (in priority order):
1. REGRESSION RISK: Code changed recently with no tests → must test before it breaks production
2. COVERAGE GAP: Large modules with zero tests → structural risk
3. QUALITY ASSURANCE: Existing tests need re-running after changes
4. IMPROVEMENT OPPORTUNITY: Underperforming areas where testing reveals optimization potential
5. NEW CAPABILITY: Newly added code that hasn't been validated

RESPONSE FORMAT — return ONLY a JSON object:
{
    "domain": "domain_id from the scan (e.g., 'financial_tools', 'control_plane')",
    "analysis": "2-3 sentence assessment of the overall codebase health",
    "hypothesis": "specific, testable hypothesis (e.g., 'Balance sheet auto-balance fails when a GL account has zero balance')",
    "test_approach": "how to test this (e.g., 'run test_e2e_financial_statements.py with edge-case fixtures')",
    "acceptance_criteria": "what success looks like (e.g., 'all tie-outs pass, no assertion errors')",
    "priority": "critical | high | medium | low",
    "risk_if_untested": "what could go wrong if we skip this",
    "estimated_scope": "small (1-2 files) | medium (3-10 files) | large (10+ files)"
}

Do NOT include markdown fences. Return raw JSON only."""


def _call_llm(system: str, user: str, max_tokens: int = 2048) -> str:
    """Call local vLLM endpoint with retry on transient failures."""
    import requests

    last_exc: Exception | None = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            resp = requests.post(
                f"{LLM_URL}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
                timeout=180,
            )
            resp.raise_for_status()
            content: str = resp.json()["choices"][0]["message"]["content"]

            # Strip chain-of-thought tags
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if "<think>" in content:
                idx = content.find("{")
                if idx >= 0:
                    content = content[idx:]

            content = re.sub(r"^```(?:json)?\n?", "", content.strip())
            content = re.sub(r"\n?```$", "", content.strip())
            return content

        except Exception as exc:
            last_exc = exc
            if attempt < _LLM_RETRIES:
                delay = _LLM_RETRY_DELAY * attempt
                log.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    _LLM_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                log.error("LLM call failed after %d attempts: %s", _LLM_RETRIES, exc)

    raise RuntimeError(f"LLM unavailable after {_LLM_RETRIES} attempts") from last_exc


def _parse_directive(raw: str) -> DirectiveOutput:
    """Parse and validate the LLM's JSON response into a DirectiveOutput."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Director returned unparseable response:\n{raw[:500]}")

    # Normalize estimated_scope to known literals
    scope_raw = data.get("estimated_scope", "")
    if scope_raw and not any(
        scope_raw == s for s in ("small (1-2 files)", "medium (3-10 files)", "large (10+ files)")
    ):
        if "small" in scope_raw.lower():
            data["estimated_scope"] = "small (1-2 files)"
        elif "large" in scope_raw.lower():
            data["estimated_scope"] = "large (10+ files)"
        else:
            data["estimated_scope"] = "medium (3-10 files)"

    return DirectiveOutput.model_validate(data)


class ResearchDirector:
    """Project-level strategic agent that scans the codebase and decides
    what to test next."""

    def scan(self) -> list[DomainScan]:
        """Scan all domains and return results."""
        return scan_all_domains()

    def next_directive(self, scans: list[DomainScan] | None = None) -> DirectiveOutput:
        """Analyze the codebase and produce the next research directive.

        Args:
            scans: Pre-computed domain scans. If None, runs a fresh scan.

        Returns:
            Validated DirectiveOutput with keys: domain, analysis, hypothesis,
            test_approach, acceptance_criteria, priority, risk_if_untested,
            estimated_scope.

        Raises:
            RuntimeError: If the LLM is unavailable after all retries.
            ValueError: If the LLM response cannot be parsed or fails validation.
        """
        if scans is None:
            scans = self.scan()

        scored = [(scan, score_domain(scan)) for scan in scans]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build scan report for LLM
        lines = ["## Codebase Domain Scan Results\n"]
        lines.append("| Domain | Files | Tests | Recent Changes | Coverage Gap | Score |")
        lines.append("|--------|-------|-------|----------------|-------------|-------|")
        for scan, score in scored:
            gap = "YES" if scan.test_coverage_gap else "no"
            lines.append(
                f"| {scan.domain} | {scan.file_count} | {scan.test_count} | "
                f"{scan.recent_changes} | {gap} | {score:.0f} |"
            )

        lines.append("\n## Domain Details\n")
        for scan, score in scored[:8]:
            lines.append(f"### {scan.domain} (score={score:.0f})")
            lines.append(f"- **Description:** {scan.description}")
            lines.append(f"- **Files:** {scan.file_count} | Tests: {scan.test_count}")
            lines.append(f"- **Coverage gap:** {scan.test_coverage_gap}")
            if scan.recent_change_files:
                lines.append(f"- **Recently changed:** {', '.join(scan.recent_change_files[:5])}")
            lines.append("")

        lines.append("## Available Domains")
        for domain_id, config in DOMAINS.items():
            lines.append(f"- `{domain_id}`: {config['description']}")
            if config.get("metrics"):
                lines.append(f"  Metrics: {', '.join(config['metrics'])}")

        user_msg = "\n".join(lines)
        raw = _call_llm(DIRECTOR_SYSTEM_PROMPT, user_msg)
        return _parse_directive(raw)

    def format_directive_for_orchestrator(self, directive: DirectiveOutput) -> str:
        """Convert a directive into a natural-language instruction."""
        lines = [
            f"RESEARCH DIRECTIVE (priority={directive.priority}):",
            f"  Domain: {directive.domain}",
            f"  Hypothesis: {directive.hypothesis}",
            f"  Test approach: {directive.test_approach}",
            f"  Acceptance criteria: {directive.acceptance_criteria}",
            f"  Risk if untested: {directive.risk_if_untested}",
        ]
        return "\n".join(lines)

    def scan_report(self, scans: list[DomainScan] | None = None) -> str:
        """Generate a human-readable scan report (no LLM needed)."""
        if scans is None:
            scans = self.scan()

        scored = [(scan, score_domain(scan)) for scan in scans]
        scored.sort(key=lambda x: x[1], reverse=True)

        lines = [
            "# CRSAI Research Director — Domain Scan Report",
            "",
            "| Rank | Domain | Files | Tests | Recent Changes | Gap | Score |",
            "|------|--------|-------|-------|----------------|-----|-------|",
        ]
        for i, (scan, score) in enumerate(scored, 1):
            gap = "YES" if scan.test_coverage_gap else "no"
            lines.append(
                f"| {i} | {scan.domain} | {scan.file_count} | "
                f"{scan.test_count} | {scan.recent_changes} | {gap} | {score:.0f} |"
            )

        lines.append("")
        lines.append("## Top Priorities")
        for scan, score in scored[:5]:
            if score > 0:
                lines.append(f"- **{scan.domain}** (score={score:.0f}): {scan.description}")
                if scan.test_coverage_gap:
                    lines.append(f"  -> Coverage gap: {scan.file_count} files with no tests")
                if scan.recent_changes > 0:
                    lines.append(f"  -> {scan.recent_changes} files changed recently")

        return "\n".join(lines)
