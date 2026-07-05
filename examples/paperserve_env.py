"""Minimal deterministic PaperServe environment.

This is RL-ready in shape but framework-free: an external trainer can call
``reset()``, inspect observations, choose one of the named actions, and call
``step(action)``. The environment itself contains no training loop, policy,
model, vector store, scheduler, or provider integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_SELECT_CANDIDATE = "select_candidate"
ACTION_DROP_CANDIDATE = "drop_candidate"
ACTION_ASK_READER_NOW = "ask_reader_now"
ACTION_RETRIEVE_MORE = "retrieve_more"
ACTION_STOP_WITH_ANSWER = "stop_with_answer"

ACTIONS = (
    ACTION_SELECT_CANDIDATE,
    ACTION_DROP_CANDIDATE,
    ACTION_ASK_READER_NOW,
    ACTION_RETRIEVE_MORE,
    ACTION_STOP_WITH_ANSWER,
)

# Stable, prompt-facing reward component names. They mirror the requested
# paper-finding game terms so external policies can log or reweight them without
# binding to an RL library.
REWARD_COMPONENT_NAMES = (
    "correct_supported_answer",
    "gold_paper_selected",
    "valid_citation",
    "wrong_answer",
    "unsupported_citation",
    "too_many_selected_tokens",
    "unnecessary_retrieval_calls",
    "stopping_without_enough_evidence",
    "recovery_success",
)


@dataclass(frozen=True)
class PaperCandidate:
    id: str
    title: str
    summary: str
    score: float
    token_estimate: int
    is_gold: bool = False
    is_hard_negative: bool = False


DEFAULT_CANDIDATES = (
    PaperCandidate(
        id="ps-gold-context-frontier",
        title="PaperServe: Context Frontiers for Evidence-Bounded Readers",
        summary="Routes a selected evidence frontier to a reader and retries when verifier feedback says gold evidence is missing.",
        score=0.96,
        token_estimate=32,
        is_gold=True,
    ),
    PaperCandidate(
        id="ps-hard-negative-vector-db",
        title="Vector Recall Is Not Context Routing",
        summary="Retrieves similar papers but does not guarantee bounded reader context or verifier recovery.",
        score=0.88,
        token_estimate=27,
        is_hard_negative=True,
    ),
    PaperCandidate(
        id="ps-hard-negative-long-context",
        title="Long Context Is Not a Frontier",
        summary="Pastes more tokens into the reader and can dilute attention with irrelevant evidence.",
        score=0.79,
        token_estimate=24,
        is_hard_negative=True,
    ),
    PaperCandidate(
        id="ps-distractor-scheduler",
        title="Schedulers for Background Jobs",
        summary="Explains job retries, not evidence routing for readers.",
        score=0.36,
        token_estimate=18,
    ),
)

MORE_CANDIDATES = (
    PaperCandidate(
        id="ps-recovery-note",
        title="Verifier Feedback for Paper Retrieval Recovery",
        summary="A verifier can request another retrieval when the selected context lacks required evidence.",
        score=0.74,
        token_estimate=22,
    ),
)


class PaperServeEnv:
    """Tiny deterministic paper-finding game.

    Observation keys are stable and JSON-serializable:
    ``question``, ``candidate_summaries``, ``selected_ids``,
    ``budget_remaining``, and ``last_verifier_feedback``.
    """

    def __init__(self, question: str, *, max_selected: int = 2, reader_token_budget: int = 64) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(max_selected, int) or max_selected <= 0:
            raise ValueError("max_selected must be a positive integer")
        if not isinstance(reader_token_budget, int) or reader_token_budget <= 0:
            raise ValueError("reader_token_budget must be a positive integer")
        self.question = question
        self.max_selected = max_selected
        self.reader_token_budget = reader_token_budget
        self._initial_candidates = list(DEFAULT_CANDIDATES)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.candidates = list(self._initial_candidates)
        self.selected_ids: list[str] = []
        self.dropped_ids: set[str] = set()
        self.last_verifier_feedback = "No answer attempted yet."
        self.done = False
        self.retrieved_more = False
        return self.observation()

    def observation(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "candidate_summaries": [
                {
                    "id": candidate.id,
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "score": candidate.score,
                    "token_estimate": candidate.token_estimate,
                    "selected": candidate.id in self.selected_ids,
                    "dropped": candidate.id in self.dropped_ids,
                }
                for candidate in self.candidates
                if candidate.id not in self.dropped_ids
            ],
            "selected_ids": list(self.selected_ids),
            "budget_remaining": self._budget_remaining(),
            "last_verifier_feedback": self.last_verifier_feedback,
        }

    def step(self, action: str | dict[str, Any], candidate_id: str | None = None, answer: str | None = None) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is already done; call reset()")
        if isinstance(action, dict):
            candidate_id = action.get("candidate_id", candidate_id)
            answer = action.get("answer", answer)
            action = str(action.get("action") or action.get("name") or "")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")

        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}

        if action == ACTION_SELECT_CANDIDATE:
            candidate = self._candidate_by_id(candidate_id)
            if candidate.id not in self.selected_ids and self._budget_remaining() < candidate.token_estimate:
                self.last_verifier_feedback = "Selection would exceed the reader token budget."
                reward_components["too_many_selected_tokens"] = -1.0
            else:
                self._select_candidate(candidate.id)
                reward_components["gold_paper_selected"] = 1.0 if self._has_selected_gold() else 0.0
        elif action == ACTION_DROP_CANDIDATE:
            self._drop_candidate(candidate_id)
        elif action == ACTION_ASK_READER_NOW:
            supported = self._has_selected_gold()
            self.last_verifier_feedback = (
                "Reader answer is supported by selected gold evidence."
                if supported
                else "Reader lacks the gold paper; retrieve more or select different evidence."
            )
            if supported:
                reward_components["correct_supported_answer"] = 1.0
                reward_components["valid_citation"] = 1.0
            else:
                reward_components["wrong_answer"] = -1.0
                reward_components["unsupported_citation"] = -1.0
        elif action == ACTION_RETRIEVE_MORE:
            unnecessary = self.retrieved_more or self._has_available_gold()
            self._retrieve_more()
            reward_components["unnecessary_retrieval_calls"] = -0.25 if unnecessary else 0.0
            reward_components["recovery_success"] = 0.5 if self._has_available_gold() else 0.0
        elif action == ACTION_STOP_WITH_ANSWER:
            self.done = True
            supported = self._has_selected_gold() and bool(answer and answer.strip())
            if supported:
                self.last_verifier_feedback = "Stopped with a supported answer."
                reward_components["correct_supported_answer"] = 2.0
                reward_components["valid_citation"] = 1.0
                reward_components["gold_paper_selected"] = 1.0
            else:
                self.last_verifier_feedback = "Stopped without sufficient selected evidence."
                reward_components["stopping_without_enough_evidence"] = -1.0
                if answer:
                    reward_components["wrong_answer"] = -1.0
                    reward_components["unsupported_citation"] = -1.0
        reward = sum(reward_components.values())
        info = {
            "actions": list(ACTIONS),
            "reward_components": reward_components,
            "reward_component_names": list(REWARD_COMPONENT_NAMES),
        }
        return self.observation(), reward, self.done, info

    def _select_candidate(self, candidate_id: str | None) -> None:
        candidate = self._candidate_by_id(candidate_id)
        if candidate.id in self.dropped_ids:
            raise ValueError(f"candidate {candidate.id!r} has been dropped")
        if candidate.id not in self.selected_ids:
            if len(self.selected_ids) >= self.max_selected:
                raise ValueError("selected candidate limit exceeded")
            if self._budget_remaining() < candidate.token_estimate:
                raise ValueError("reader token budget exceeded")
            self.selected_ids.append(candidate.id)

    def _drop_candidate(self, candidate_id: str | None) -> None:
        candidate = self._candidate_by_id(candidate_id)
        self.dropped_ids.add(candidate.id)
        self.selected_ids = [selected_id for selected_id in self.selected_ids if selected_id != candidate.id]

    def _retrieve_more(self) -> None:
        if not self.retrieved_more:
            existing_ids = {candidate.id for candidate in self.candidates}
            self.candidates.extend(candidate for candidate in MORE_CANDIDATES if candidate.id not in existing_ids)
            self.retrieved_more = True
        self.last_verifier_feedback = "Retrieved one deterministic follow-up batch."

    def _candidate_by_id(self, candidate_id: str | None) -> PaperCandidate:
        if not candidate_id:
            raise ValueError("candidate_id is required for this action")
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        raise ValueError(f"unknown candidate_id {candidate_id!r}")

    def _budget_remaining(self) -> int:
        used = sum(candidate.token_estimate for candidate in self.candidates if candidate.id in self.selected_ids)
        return self.reader_token_budget - used

    def _budget_efficiency_reward(self) -> float:
        if not self._has_selected_gold():
            return 0.0
        return max(0.0, self._budget_remaining() / self.reader_token_budget)

    def _has_selected_gold(self) -> bool:
        return any(candidate.is_gold and candidate.id in self.selected_ids for candidate in self.candidates)

    def _has_available_gold(self) -> bool:
        return any(candidate.is_gold and candidate.id not in self.dropped_ids for candidate in self.candidates)

    def _selected_hard_negative_count(self) -> int:
        return sum(
            1
            for candidate in self.candidates
            if candidate.is_hard_negative and candidate.id in self.selected_ids
        )

    def _is_hard_negative(self, candidate_id: str | None) -> bool:
        if not candidate_id:
            return False
        return any(candidate.id == candidate_id and candidate.is_hard_negative for candidate in self.candidates)


def make_env(question: str, *, max_selected: int = 2, reader_token_budget: int = 64) -> PaperServeEnv:
    return PaperServeEnv(question, max_selected=max_selected, reader_token_budget=reader_token_budget)


if __name__ == "__main__":
    env = make_env("Which paper explains PaperServe context frontiers?")
    print(env.reset())
    print(env.step(ACTION_SELECT_CANDIDATE, "ps-gold-context-frontier"))
    print(env.step(ACTION_STOP_WITH_ANSWER, answer="PaperServe routes a selected evidence frontier."))
