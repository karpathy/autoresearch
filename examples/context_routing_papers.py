"""Deterministic Context Routing / PaperServe example for ResearchFS.

The example intentionally has no model, vector database, network, or scheduler
dependency. It shows the contract between retrieval, routing, reading, and
verification: the reader receives only the selected evidence frontier, while the
workflow keeps the full candidate set available for audit.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchfs.context import ContextBudget, ContextCandidate, route_top_k_context


GOLD_PAPER_ID = "ps-gold-context-frontier"
RECOVERY_QUESTION_MARKERS = ("recover", "retry", "loop", "initial miss")
MISSING_GOLD_MARKERS = ("missing gold", "unavailable", "out of corpus", "not in corpus")

PAPER_CORPUS = {
    GOLD_PAPER_ID: {
        "id": GOLD_PAPER_ID,
        "title": "PaperServe: Context Frontiers for Evidence-Bounded Readers",
        "text": (
            "PaperServe routes a small context frontier to the reader instead "
            "of passing every retrieved paper. The gold result is selected by "
            "deterministic score, budget, and verifier feedback. A loop can "
            "retrieve again when verification reports that required evidence is "
            "missing. The reader must only see selected context."
        ),
        "source": "synthetic://paperserve/context-frontier",
        "score": 0.96,
        "token_estimate": 55,
        "metadata": {"kind": "gold", "year": 2026},
    },
    "ps-hard-negative-vector-db": {
        "id": "ps-hard-negative-vector-db",
        "title": "Vector Recall Is Not Context Routing",
        "text": (
            "A vector database may retrieve semantically similar papers, but it "
            "does not by itself enforce a reader context frontier or verifier "
            "recovery loop. This paper is a hard negative for PaperServe."
        ),
        "source": "synthetic://paperserve/hard-negative-vector-db",
        "score": 0.89,
        "token_estimate": 43,
        "metadata": {"kind": "hard_negative", "confuses": GOLD_PAPER_ID},
    },
    "ps-hard-negative-long-context": {
        "id": "ps-hard-negative-long-context",
        "title": "Long Context Windows Without Evidence Selection",
        "text": (
            "Long-context prompting increases available tokens but still risks "
            "attention dilution when irrelevant papers are included. It does not "
            "define a deterministic selected evidence frontier."
        ),
        "source": "synthetic://paperserve/hard-negative-long-context",
        "score": 0.83,
        "token_estimate": 38,
        "metadata": {"kind": "hard_negative", "confuses": GOLD_PAPER_ID},
    },
    "ps-distractor-scheduler": {
        "id": "ps-distractor-scheduler",
        "title": "Workflow Scheduling for Background Jobs",
        "text": (
            "A scheduler chooses which job runs next and handles retry timing. "
            "It says nothing about which evidence reaches a reader."
        ),
        "source": "synthetic://paperserve/distractor-scheduler",
        "score": 0.41,
        "token_estimate": 26,
        "metadata": {"kind": "distractor"},
    },
    "ps-distractor-gpu-kernel": {
        "id": "ps-distractor-gpu-kernel",
        "title": "Custom Attention Kernels for Throughput",
        "text": (
            "Custom kernels can speed model attention, but PaperServe in this "
            "example is dependency-free and does not modify model internals."
        ),
        "source": "synthetic://paperserve/distractor-gpu-kernel",
        "score": 0.34,
        "token_estimate": 28,
        "metadata": {"kind": "distractor"},
    },
}


def _question_mentions(question: str, markers: tuple[str, ...]) -> bool:
    normalized = question.lower()
    return any(marker in normalized for marker in markers)


def _paper_to_candidate(paper: dict[str, Any]) -> ContextCandidate:
    return ContextCandidate(
        id=paper["id"],
        title=paper["title"],
        text=paper["text"],
        source=paper["source"],
        score=paper["score"],
        token_estimate=paper["token_estimate"],
        metadata=dict(paper["metadata"]),
    )


def _candidate_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate["id"])
    return str(getattr(candidate, "id"))


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate["text"])
    return str(getattr(candidate, "text"))


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate.get("metadata") or {})
    return dict(getattr(candidate, "metadata", None) or {})


def _selection_candidates(selection: Any) -> list[Any]:
    """Return selected candidates across common public selection shapes."""

    if isinstance(selection, dict):
        selected = selection.get("selected") or selection.get("candidates") or []
    else:
        selected = getattr(selection, "selected", None)
        if selected is None:
            selected = getattr(selection, "candidates", None)
        if selected is None:
            selected = getattr(selection, "selected_candidates", None)
        if selected is None and isinstance(selection, (list, tuple)):
            selected = selection
    return list(selected or [])


def _candidate_to_payload(candidate: Any) -> dict[str, Any]:
    if is_dataclass(candidate):
        payload = asdict(candidate)
    elif isinstance(candidate, dict):
        payload = dict(candidate)
    else:
        payload = {
            "id": _candidate_id(candidate),
            "text": _candidate_text(candidate),
            "metadata": _candidate_metadata(candidate),
        }
        for field_name in ("score", "token_estimate"):
            if hasattr(candidate, field_name):
                payload[field_name] = getattr(candidate, field_name)
    metadata = dict(payload.get("metadata") or {})
    payload["metadata"] = metadata
    return payload


def retrieve_candidates(question: str, attempt: int) -> list[ContextCandidate]:
    """Retrieve deterministic paper candidates for a question and attempt.

    For loop-recovery questions, attempt one intentionally misses the gold paper
    to demonstrate verifier-driven retrieval recovery. Attempt two adds the gold
    paper with the highest score. Missing-gold questions never expose the gold
    paper, forcing an insufficient-evidence result.
    """

    if attempt < 1:
        raise ValueError("attempt must be >= 1")

    candidate_ids = [
        "ps-hard-negative-vector-db",
        "ps-hard-negative-long-context",
        "ps-distractor-scheduler",
        "ps-distractor-gpu-kernel",
    ]
    if not _question_mentions(question, MISSING_GOLD_MARKERS):
        if attempt > 1 or not _question_mentions(question, RECOVERY_QUESTION_MARKERS):
            candidate_ids.insert(0, GOLD_PAPER_ID)
    return [_paper_to_candidate(PAPER_CORPUS[paper_id]) for paper_id in candidate_ids]


def route_context(candidates: list[ContextCandidate], *, k: int, max_reader_tokens: int = 120) -> list[ContextCandidate]:
    """Select the bounded context frontier that the reader is allowed to see."""

    budget = ContextBudget(max_items=k, max_tokens=max_reader_tokens, overflow_policy="drop_lowest_score")
    selection = route_top_k_context(candidates, k=k, budget=budget)
    return _selection_candidates(selection)


def answer_from_selected_context(question: str, selected_context: list[ContextCandidate]) -> dict[str, Any]:
    """Reader node: answer using selected context only, never all candidates."""

    selected_ids = [_candidate_id(candidate) for candidate in selected_context]
    if GOLD_PAPER_ID not in selected_ids:
        return no_answer(
            "selected_context_missing_gold",
            selected_ids=selected_ids,
            question=question,
        )

    gold = next(candidate for candidate in selected_context if _candidate_id(candidate) == GOLD_PAPER_ID)
    return final_answer(
        answer=(
            "PaperServe keeps the reader grounded by routing only a selected "
            "evidence frontier, then using verifier feedback to retrieve again "
            "when that frontier lacks the required gold paper."
        ),
        citations=[_candidate_id(gold)],
        selected_ids=selected_ids,
    )


def verify_answer(answer_payload: dict[str, Any], selected_context: list[ContextCandidate]) -> dict[str, Any]:
    """Verifier node deciding whether to finish or request another retrieval."""

    selected_ids = {_candidate_id(candidate) for candidate in selected_context}
    citations = set(answer_payload.get("citations") or [])
    if GOLD_PAPER_ID not in selected_ids:
        return {
            "supported": False,
            "recoverable": True,
            "feedback": "Gold PaperServe evidence was not in the selected context frontier.",
            "missing_paper_ids": [GOLD_PAPER_ID],
        }
    if GOLD_PAPER_ID not in citations:
        return {
            "supported": False,
            "recoverable": False,
            "feedback": "Answer did not cite the selected gold evidence.",
            "missing_paper_ids": [],
        }
    return {
        "supported": True,
        "recoverable": False,
        "feedback": "Answer is supported by the selected PaperServe gold paper.",
        "missing_paper_ids": [],
    }


def final_answer(answer: str, citations: list[str], selected_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "answered",
        "answer": answer,
        "citations": list(citations),
        "selected_context_ids": list(selected_ids),
    }


def no_answer(reason: str, *, selected_ids: list[str] | None = None, question: str | None = None) -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "answer": None,
        "citations": [],
        "selected_context_ids": list(selected_ids or []),
        "reason": reason,
        "question": question,
    }


def run_paper_routing_workflow(question: str, *, k: int = 2, max_attempts: int = 2) -> dict[str, Any]:
    """Run a small deterministic paper-routing workflow.

    The returned audit fields prove the selected reader frontier is smaller than
    the retrieved candidate set:
    - ``all_candidate_ids_by_attempt`` records every retrieval result.
    - ``reader_context_ids_by_attempt`` records exactly what the reader saw.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    all_candidate_ids_by_attempt: list[list[str]] = []
    reader_context_ids_by_attempt: list[list[str]] = []
    attempts: list[dict[str, Any]] = []
    last_answer = no_answer("not_started", question=question)
    last_verification: dict[str, Any] = {
        "supported": False,
        "recoverable": True,
        "feedback": "Workflow has not attempted retrieval yet.",
        "missing_paper_ids": [GOLD_PAPER_ID],
    }

    for attempt in range(1, max_attempts + 1):
        candidates = retrieve_candidates(question, attempt)
        selected_context = route_context(candidates, k=k)
        candidate_ids = [_candidate_id(candidate) for candidate in candidates]
        selected_ids = [_candidate_id(candidate) for candidate in selected_context]

        all_candidate_ids_by_attempt.append(candidate_ids)
        reader_context_ids_by_attempt.append(selected_ids)

        last_answer = answer_from_selected_context(question, selected_context)
        last_verification = verify_answer(last_answer, selected_context)
        attempts.append(
            {
                "attempt": attempt,
                "candidate_ids": candidate_ids,
                "reader_context_ids": selected_ids,
                "reader_context": [_candidate_to_payload(candidate) for candidate in selected_context],
                "answer_status": last_answer["status"],
                "verifier": dict(last_verification),
            }
        )

        if last_verification["supported"]:
            return {
                **last_answer,
                "question": question,
                "attempts": attempts,
                "reader_context_ids_by_attempt": reader_context_ids_by_attempt,
                "all_candidate_ids_by_attempt": all_candidate_ids_by_attempt,
                "verifier": dict(last_verification),
            }
        if not last_verification["recoverable"]:
            break

    return {
        **no_answer("required_gold_evidence_not_selected", question=question),
        "attempts": attempts,
        "reader_context_ids_by_attempt": reader_context_ids_by_attempt,
        "all_candidate_ids_by_attempt": all_candidate_ids_by_attempt,
        "verifier": dict(last_verification),
    }



def _call_retriever(retriever: Any, query: str, attempt: int) -> list[Any]:
    try:
        return list(retriever(query, attempt=attempt))
    except TypeError:
        try:
            return list(retriever(query, attempt))
        except TypeError:
            return list(retriever(query))


def _call_reader(reader: Any, query: str, selected_context: list[Any]) -> dict[str, Any]:
    try:
        answer = reader(query, selected_context)
    except TypeError:
        answer = reader(selected_context)
    if isinstance(answer, dict):
        return dict(answer)
    return {"status": "answered", "answer": answer, "citations": []}


def _call_verifier(verifier: Any, query: str, answer: dict[str, Any], selected_context: list[Any]) -> dict[str, Any]:
    try:
        verification = verifier(query, answer, selected_context)
    except TypeError:
        try:
            verification = verifier(answer, selected_context)
        except TypeError:
            verification = verifier(answer)
    if isinstance(verification, dict):
        result = dict(verification)
    else:
        result = {"supported": bool(verification)}
    result.setdefault("supported", False)
    result.setdefault("recoverable", not result["supported"])
    result.setdefault("feedback", "")
    return result


def run_paperserve_workflow(
    query: str,
    retriever: Any,
    reader: Any,
    verifier: Any,
    *,
    k: int = 2,
    max_retries: int = 1,
    min_score: float | None = None,
    budget: ContextBudget | None = None,
) -> dict[str, Any]:
    """Generic PaperServe runner for tests and external toy components.

    ``retriever`` returns candidates, ``reader`` sees only routed candidates,
    and ``verifier`` decides whether to stop or retry. The audit payload keeps
    all candidate ids separate from reader-visible context ids.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    attempts: list[dict[str, Any]] = []
    all_candidate_ids_by_attempt: list[list[str]] = []
    reader_context_ids_by_attempt: list[list[str]] = []
    last_answer: dict[str, Any] = no_answer("not_started", question=query)
    last_verification: dict[str, Any] = {"supported": False, "recoverable": True, "feedback": ""}

    for retry_index in range(max_retries + 1):
        attempt = retry_index + 1
        candidates = _call_retriever(retriever, query, attempt)
        selection = route_top_k_context(candidates, k=k, min_score=min_score, budget=budget)
        selected_context = _selection_candidates(selection)
        candidate_ids = [_candidate_id(candidate) for candidate in candidates]
        selected_ids = [_candidate_id(candidate) for candidate in selected_context]

        all_candidate_ids_by_attempt.append(candidate_ids)
        reader_context_ids_by_attempt.append(selected_ids)

        last_answer = _call_reader(reader, query, selected_context)
        last_verification = _call_verifier(verifier, query, last_answer, selected_context)
        attempts.append(
            {
                "attempt": attempt,
                "candidate_ids": candidate_ids,
                "reader_context_ids": selected_ids,
                "answer_status": last_answer.get("status", "answered"),
                "verifier": dict(last_verification),
            }
        )
        if last_verification["supported"]:
            return {
                **last_answer,
                "status": last_answer.get("status", "answered"),
                "question": query,
                "attempts": attempts,
                "reader_context_ids_by_attempt": reader_context_ids_by_attempt,
                "all_candidate_ids_by_attempt": all_candidate_ids_by_attempt,
                "verifier": dict(last_verification),
            }
        if not last_verification["recoverable"]:
            break

    return {
        **no_answer("required_evidence_not_selected", question=query),
        "attempts": attempts,
        "reader_context_ids_by_attempt": reader_context_ids_by_attempt,
        "all_candidate_ids_by_attempt": all_candidate_ids_by_attempt,
        "verifier": dict(last_verification),
    }
if __name__ == "__main__":
    result = run_paper_routing_workflow(
        "How should a loop recover when initial miss retrieval omits PaperServe gold?"
    )
    print(result)
