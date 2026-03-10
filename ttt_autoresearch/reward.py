from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from ttt_autoresearch.config import BootstrapContext
from ttt_autoresearch.discover_compat import BaseRewardEvaluator
from ttt_autoresearch.runner import AutoResearchRunner, PatchCandidate, RunResult, parse_patch_candidate


_ARTIFACT_LOCK = threading.Lock()
_EVALUATION_SLOTS: threading.BoundedSemaphore | None = None


def reward_for_result(current_best_val_bpb: float, result: RunResult) -> tuple[float, float]:
    if result.status == "timeout":
        return -0.5, 0.0
    if result.status == "missing_metric":
        return -0.75, 0.0
    if result.status != "success" or result.val_bpb is None:
        return -1.0, 0.0
    reward = current_best_val_bpb - result.val_bpb
    correctness = 1.0 if reward > 0 else 0.0
    return reward, correctness


class AutoResearchRewardEvaluator(BaseRewardEvaluator):
    bootstrap: BootstrapContext | None = None
    runner: AutoResearchRunner | None = None

    @classmethod
    def configure(cls, bootstrap: BootstrapContext, runner: AutoResearchRunner) -> None:
        global _EVALUATION_SLOTS
        cls.bootstrap = bootstrap
        cls.runner = runner
        _EVALUATION_SLOTS = threading.BoundedSemaphore(bootstrap.config.max_concurrent_evaluations)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.problem_type = kwargs.get("problem_type", "autoresearch")
        self.log_dir = kwargs.get("log_dir")
        self.eval_timeout = kwargs.get("eval_timeout")
        self.num_cpus_per_task = kwargs.get("num_cpus_per_task")

    def get_reward(self, code: str, state: Any) -> dict[str, Any]:
        if self.bootstrap is None or self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")

        try:
            candidate = parse_patch_candidate(code)
        except ValueError as exc:
            return self._failure_payload(
                reward=-1.0,
                raw_score=self._current_best_from_state(state),
                msg=f"Invalid candidate payload: {exc}",
                status="invalid_candidate",
            )

        result = self._run_candidate(candidate, state)
        current_best = self._current_best_from_state(state)
        reward, correctness = reward_for_result(current_best, result)
        improved_global_best = False

        with _ARTIFACT_LOCK:
            if result.status == "success" and result.val_bpb is not None:
                improved_global_best = self.runner.update_best(
                    train_py_text=candidate.train_py,
                    result=result,
                    summary=candidate.summary,
                    rationale=candidate.rationale,
                )
            history_entry = {
                "step": getattr(state, "timestep", -1) + 1,
                "state_id": getattr(state, "id", "unknown"),
                "status": result.status,
                "summary": candidate.summary,
                "rationale": candidate.rationale,
                "reward": reward,
                "accepted": bool(correctness),
                "val_bpb": result.val_bpb,
                "parent_val_bpb": current_best,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "workspace_path": str(result.workspace_path),
                "improved_global_best": improved_global_best,
            }
            self.runner.append_history(history_entry)

        message = self._build_message(candidate, result, current_best, reward)
        stdout = self.runner.read_text(result.stdout_path)
        raw_score = result.val_bpb if result.val_bpb is not None else current_best
        return {
            "reward": float(reward),
            "msg": message,
            "correctness": float(correctness),
            "raw_score": float(raw_score),
            "result_construction": [],
            "stdout": stdout,
            "metrics": {
                "candidate_summary": candidate.summary,
                "candidate_rationale": candidate.rationale,
                "candidate_status": result.status,
                "candidate_val_bpb": result.val_bpb,
                "workspace_path": str(result.workspace_path),
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "improved_global_best": improved_global_best,
            },
        }

    def _run_candidate(self, candidate: PatchCandidate, state: Any) -> RunResult:
        if self.bootstrap is None or self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        if _EVALUATION_SLOTS is None:
            raise RuntimeError("AutoResearchRewardEvaluator evaluation slots are not configured.")

        # Grouped rollouts stay enabled for the upstream entropic advantage recipe,
        # but inner autoresearch training runs must be serialized on a single GPU.
        _EVALUATION_SLOTS.acquire()
        try:
            return self.runner.run_candidate(
                bootstrap=self.bootstrap,
                candidate=candidate,
                step=getattr(state, "timestep", -1) + 1,
                state_id=getattr(state, "id", "unknown"),
            )
        finally:
            _EVALUATION_SLOTS.release()

    @staticmethod
    def _build_message(candidate: PatchCandidate, result: RunResult, current_best: float, reward: float) -> str:
        val_bpb = "n/a" if result.val_bpb is None else f"{result.val_bpb:.6f}"
        return (
            f"{candidate.summary}\n"
            f"status={result.status} parent_val_bpb={current_best:.6f} "
            f"candidate_val_bpb={val_bpb} reward={reward:.6f}"
        )

    @staticmethod
    def _current_best_from_state(state: Any) -> float:
        current_best = getattr(state, "current_best_val_bpb", None)
        if current_best is not None:
            return float(current_best)
        value = getattr(state, "value", None)
        if value is None:
            raise RuntimeError("State is missing current_best_val_bpb and value.")
        return float(-value)

    @staticmethod
    def _failure_payload(reward: float, raw_score: float, msg: str, status: str) -> dict[str, Any]:
        return {
            "reward": float(reward),
            "msg": msg,
            "correctness": 0.0,
            "raw_score": float(raw_score),
            "result_construction": [],
            "stdout": "",
            "metrics": {
                "candidate_status": status,
            },
        }
