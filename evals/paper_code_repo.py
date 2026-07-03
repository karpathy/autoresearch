"""Deterministic eval for finding a paper's official code repository."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable
from urllib.parse import urlparse


GOLD_CODE_REPO_URL = "https://github.com/NVlabs/ProtoMotions"


@dataclass(frozen=True)
class PaperCodeRepoFixture:
    """Prompt-facing paper metadata for a code-repository lookup task."""

    title: str
    authors: tuple[str, ...]
    arxiv_id: str
    arxiv_url: str
    abstract: str


@dataclass(frozen=True)
class PaperCodeRepoCase:
    """Eval case data split into prompt metadata and scorer-only gold answer."""

    paper: PaperCodeRepoFixture
    expected_repo_url: str


@dataclass(frozen=True)
class PaperCodeRepoOfflineScenario:
    """Synthetic response fixture for the offline scorer harness."""

    name: str
    response: str
    expected_passed: bool
    description: str = ""


GPC_PAPER_FIXTURE = PaperCodeRepoFixture(
    title="GPC: Large-Scale Generative Pretraining for Transferable Motor Control",
    authors=("Yi Shi", "Yifeng Jiang", "Chen Tessler", "Xue Bin Peng"),
    arxiv_id="2606.29148",
    arxiv_url="https://arxiv.org/abs/2606.29148",
    abstract=(
        "Developing controllers capable of completing a wide range of tasks in a "
        "natural and life-like manner is a key challenge in enabling practical "
        "applications of physics-based character animation. In this work, we "
        "introduce Generative Pretrained Controllers (GPC), which leverage "
        "tokenization and next-token modeling to create general-purpose, reusable "
        "generative controllers from large-scale motion datasets. Our framework "
        "utilizes end-to-end reinforcement learning to jointly optimize a \"motion "
        "vocabulary\", modeled via Finite Scalar Quantization (FSQ), along with a "
        "corresponding control policy that can map the discrete codes to "
        "physics-based controls. After the \"codebook\" has been learned, the "
        "underlying structure of this large vocabulary is modeled by training a "
        "GPT-style autoregressive transformer, leading to a powerful generative "
        "controller that generates controls for a physically simulated character by "
        "performing next-token prediction. Once the generative controller has been "
        "trained, we propose a suite of adaptation techniques for finetuning the "
        "controller for new downstream tasks. Our proposed framework greatly "
        "simplifies the training process compared to previous tokenized methods, "
        "and achieves a 99.98% success rate in reproducing a vast corpus of motion "
        "clips. The generative controller exhibits a variety of natural emergent "
        "behaviors, such as responsive behaviors to perturbations and recovery "
        "behaviors after falling. This results in highly robust general purpose "
        "controllers for a variety of downstream applications."
    ),
)

GPC_PROTOMOTIONS_CASE = PaperCodeRepoCase(
    paper=GPC_PAPER_FIXTURE,
    expected_repo_url=GOLD_CODE_REPO_URL,
)


DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS = (
    PaperCodeRepoOfflineScenario(
        name="correct_root",
        response=(
            '{"code_repository_url": "https://github.com/NVlabs/ProtoMotions", '
            '"evidence": "official repository"}'
        ),
        expected_passed=True,
        description="Exact repository root.",
    ),
    PaperCodeRepoOfflineScenario(
        name="correct_subpath",
        response=(
            '{"code_repository_url": '
            '"https://github.com/NVlabs/ProtoMotions/tree/main/protomotions", '
            '"evidence": "code directory"}'
        ),
        expected_passed=True,
        description="Repository URL with an implementation subpath.",
    ),
    PaperCodeRepoOfflineScenario(
        name="correct_query_variant",
        response=(
            '{"code_repository_url": '
            '"https://github.com/NVlabs/ProtoMotions?tab=readme-ov-file", '
            '"evidence": "README"}'
        ),
        expected_passed=True,
        description="Repository root with a query string.",
    ),
    PaperCodeRepoOfflineScenario(
        name="docs_only",
        response=(
            '{"code_repository_url": '
            '"https://github.com/NVlabs/ProtoMotions/tree/main/docs", '
            '"evidence": "documentation only"}'
        ),
        expected_passed=False,
        description="Documentation subpath is not a code repository answer.",
    ),
    PaperCodeRepoOfflineScenario(
        name="sibling_repo",
        response=(
            '{"code_repository_url": "https://github.com/NVlabs/ProtoMotions-docs", '
            '"evidence": "nearby repository"}'
        ),
        expected_passed=False,
        description="Sibling repository is not the gold repository.",
    ),
    PaperCodeRepoOfflineScenario(
        name="ambiguous_correct_plus_wrong",
        response=(
            '{"code_repository_url": "https://github.com/NVlabs/ProtoMotions", '
            '"alternate": "https://github.com/NVlabs/ProtoMotions-docs"}'
        ),
        expected_passed=False,
        description="Correct answer mixed with a wrong GitHub repository.",
    ),
)

_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?)]}>'\""


def _paper_from_case(case: PaperCodeRepoCase | PaperCodeRepoFixture) -> PaperCodeRepoFixture:
    if isinstance(case, PaperCodeRepoCase):
        return case.paper
    return case


def _expected_repo_from_case(case: PaperCodeRepoCase | PaperCodeRepoFixture) -> str:
    if isinstance(case, PaperCodeRepoCase):
        return case.expected_repo_url
    return GOLD_CODE_REPO_URL


def build_paper_code_repo_prompt(
    case: PaperCodeRepoCase | PaperCodeRepoFixture = GPC_PAPER_FIXTURE,
) -> str:
    """Build the prompt without exposing the expected repository URL."""

    paper = _paper_from_case(case)
    response_contract = {
        "code_repository_url": "string URL for the official GitHub code repository, or null if none is found",
        "evidence": "brief explanation of the source or reasoning used",
    }
    return (
        "Identify the official GitHub code repository for the paper below. "
        "Return only JSON matching the response contract.\n\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"arXiv URL: {paper.arxiv_url}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        "Response contract:\n"
        f"{json.dumps(response_contract, indent=2, sort_keys=True)}"
    )


def normalize_github_repo_url(value: str | None) -> str | None:
    """Return a canonical github.com/owner/repo key, or None for non-repo URLs."""

    if not value:
        return None

    text = value.strip().strip(_TRAILING_PUNCTUATION)
    if "://" not in text and text.lower().startswith(("github.com/", "www.github.com/")):
        text = f"https://{text}"

    parsed = urlparse(text)
    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "github.com":
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None

    owner = parts[0].lower()
    repo = parts[1]
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo:
        return None

    return f"github.com/{owner}/{repo.lower()}"


def extract_github_repo_candidates(text: str | None) -> list[str]:
    """Extract unique normalized GitHub repository candidates from response text."""

    if not text:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for match in _URL_RE.finditer(text):
        candidate = normalize_github_repo_url(match.group(0))
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _github_repo_subpath_parts(value: str) -> list[str]:
    """Return path parts after owner/repo for a GitHub URL."""

    text = value.strip().strip(_TRAILING_PUNCTUATION)
    if "://" not in text and text.lower().startswith(("github.com/", "www.github.com/")):
        text = f"https://{text}"

    parsed = urlparse(text)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) <= 2:
        return []

    subpath = parts[2:]
    if len(subpath) >= 3 and subpath[0].lower() in {"blob", "tree"}:
        return subpath[2:]
    return subpath


def _is_docs_only_github_url(value: str) -> bool:
    """Return True when a GitHub URL points only at documentation content."""

    subpath = _github_repo_subpath_parts(value)
    return bool(subpath) and subpath[0].lower() in {"doc", "docs", "documentation", "site", "website"}


def evaluate_paper_code_repo_response(
    response: str,
    case: PaperCodeRepoCase | PaperCodeRepoFixture = GPC_PROTOMOTIONS_CASE,
) -> dict[str, object]:
    """Score an agent response for the exact expected GitHub repository."""

    expected = normalize_github_repo_url(_expected_repo_from_case(case))
    candidates = extract_github_repo_candidates(response)
    docs_only_candidates: list[str] = []
    has_code_expected_url = False
    for match in _URL_RE.finditer(response or ""):
        candidate = normalize_github_repo_url(match.group(0))
        if candidate == expected:
            if _is_docs_only_github_url(match.group(0)):
                if candidate not in docs_only_candidates:
                    docs_only_candidates.append(candidate)
            else:
                has_code_expected_url = True

    wrong_candidates = [candidate for candidate in candidates if candidate != expected]
    passed = has_code_expected_url and not wrong_candidates
    return {
        "passed": passed,
        "expected_repo": expected,
        "github_repo_candidates": candidates,
        "wrong_github_repo_candidates": wrong_candidates,
        "docs_only_github_repo_candidates": docs_only_candidates,
    }


score_paper_code_repo_response = evaluate_paper_code_repo_response


def run_paper_code_repo_eval(
    agent_fn: Callable[[str], str],
    case: PaperCodeRepoCase | PaperCodeRepoFixture = GPC_PROTOMOTIONS_CASE,
) -> dict[str, object]:
    """Run the eval with an injected agent callable and return response plus score."""

    prompt = build_paper_code_repo_prompt(_paper_from_case(case))
    response = agent_fn(prompt)
    score = evaluate_paper_code_repo_response(str(response), case)
    return {
        "prompt": prompt,
        "response": response,
        **score,
    }


def run_offline_paper_code_repo_eval_suite(
    case: PaperCodeRepoCase | PaperCodeRepoFixture = GPC_PROTOMOTIONS_CASE,
    scenarios: tuple[PaperCodeRepoOfflineScenario, ...] = DEFAULT_PAPER_CODE_REPO_OFFLINE_SCENARIOS,
) -> dict[str, object]:
    """Run deterministic scorer scenarios without fetching papers, GitHub, or agents."""

    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for scenario in scenarios:
        score = evaluate_paper_code_repo_response(scenario.response, case)
        actual_passed = bool(score["passed"])
        matched = actual_passed == scenario.expected_passed
        result = {
            "name": scenario.name,
            "description": scenario.description,
            "expected_passed": scenario.expected_passed,
            "actual_passed": actual_passed,
            "matched": matched,
            "response": scenario.response,
            "score": score,
        }
        results.append(result)
        if not matched:
            failures.append(
                {
                    "name": scenario.name,
                    "expected_passed": scenario.expected_passed,
                    "actual_passed": actual_passed,
                    "score": score,
                }
            )

    total = len(results)
    matched_count = total - len(failures)
    return {
        "passed": not failures,
        "counts": {
            "total": total,
            "matched": matched_count,
            "failed": len(failures),
        },
        "results": results,
        "failures": failures,
    }
