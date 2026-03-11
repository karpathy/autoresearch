from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Any

from ttt_autoresearch.config import BootstrapContext
from ttt_autoresearch.discover_compat import BaseRewardEvaluator
from ttt_autoresearch.prompt_builder import build_prompt_for_state
from ttt_autoresearch.runner import (
    AutoResearchRunner,
    PatchCandidate,
    PreflightResult,
    RunResult,
    parse_patch_candidate_for_state,
)


_ARTIFACT_LOCK = threading.Lock()
_EVALUATION_SLOTS: threading.BoundedSemaphore | None = None
_GPU_DEVICE_QUEUE: queue.Queue[str] | None = None
_REWARD_EPSILON = 1e-8
_FAIL_REWARD = 0.0
_FAIL_RAW_SCORE = 1e9


def reward_for_result(result: RunResult) -> tuple[float, float]:
    if result.status != "success" or result.val_bpb is None:
        return _FAIL_REWARD, 0.0
    reward = 1.0 / (_REWARD_EPSILON + result.val_bpb)
    correctness = 1.0
    return reward, correctness


class AutoResearchRewardEvaluator(BaseRewardEvaluator):
    bootstrap: BootstrapContext | None = None
    runner: AutoResearchRunner | None = None

    @classmethod
    def configure(cls, bootstrap: BootstrapContext, runner: AutoResearchRunner) -> None:
        global _EVALUATION_SLOTS, _GPU_DEVICE_QUEUE
        cls.bootstrap = bootstrap
        cls.runner = runner
        _EVALUATION_SLOTS = threading.BoundedSemaphore(bootstrap.config.max_concurrent_evaluations)
        _GPU_DEVICE_QUEUE = None
        if bootstrap.config.execution_backend == "runpod":
            return
        gpu_devices = bootstrap.config.gpu_devices or []
        if bootstrap.config.max_concurrent_evaluations > 1:
            if not gpu_devices:
                raise ValueError(
                    "max_concurrent_evaluations > 1 requires gpu_devices to be set so candidate runs can be pinned to distinct GPUs."
                )
            if bootstrap.config.max_concurrent_evaluations > len(gpu_devices):
                raise ValueError(
                    "max_concurrent_evaluations cannot exceed the number of configured gpu_devices."
                )
        if gpu_devices:
            _GPU_DEVICE_QUEUE = queue.Queue()
            for gpu_device in gpu_devices:
                _GPU_DEVICE_QUEUE.put(gpu_device)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.problem_type = kwargs.get("problem_type", "autoresearch")
        self.log_dir = kwargs.get("log_dir")
        self.eval_timeout = kwargs.get("eval_timeout")
        self.num_cpus_per_task = kwargs.get("num_cpus_per_task")

    def get_reward(self, code: str, state: Any) -> dict[str, Any]:
        if self.bootstrap is None or self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        prompt = self._prompt_for_state(state)
        step = getattr(state, "timestep", -1) + 1
        state_id = getattr(state, "id", "unknown")

        try:
            candidate = parse_patch_candidate_for_state(code, state.current_train_py)
        except ValueError as exc:
            return self._persist_invalid_candidate(
                code=code,
                prompt=prompt,
                state=state,
                error_message=f"Invalid candidate payload: {exc}",
            )

        workspace = self.runner.prepare_candidate_workspace(candidate, step=step)
        (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
        (workspace / "response.txt").write_text(code, encoding="utf-8")
        preflight = self.runner.preflight_candidate(workspace, candidate)
        self.runner.write_json_artifact(workspace / "preflight.json", preflight.to_dict())
        if not preflight.ok:
            return self._persist_preflight_failed_candidate(
                candidate=candidate,
                code=code,
                prompt=prompt,
                state=state,
                workspace=workspace,
                preflight=preflight,
            )

        result = self._run_candidate(candidate, state, workspace)
        current_best = self._current_best_from_state(state)
        reward, correctness = reward_for_result(result)
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
                "step": step,
                "state_id": state_id,
                "status": result.status,
                "summary": candidate.summary,
                "rationale": candidate.rationale,
                "candidate_format": candidate.candidate_format,
                "patch_block_count": candidate.patch_block_count,
                "lines_changed": candidate.lines_changed,
                "reward": reward,
                "accepted": bool(result.status == "success" and result.val_bpb is not None),
                "val_bpb": result.val_bpb,
                "parent_val_bpb": current_best,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "workspace_path": str(result.workspace_path),
                "improved_global_best": improved_global_best,
                "prompt_path": str(result.workspace_path / "prompt.txt"),
                "preflight_path": str(result.workspace_path / "preflight.json"),
                "failure_stage": "runtime" if result.status != "success" else "",
                "failure_reason": result.status if result.status != "success" else "",
            }
            self.runner.append_history(history_entry)
            self.runner.write_rollout_manifest(
                result.workspace_path,
                {
                    "step": step,
                    "starting_state": state.to_dict() if hasattr(state, "to_dict") else {
                        "id": state_id,
                        "timestep": getattr(state, "timestep", -1),
                    },
                    "candidate": {
                        "summary": candidate.summary,
                        "rationale": candidate.rationale,
                        "train_py": candidate.train_py,
                        "candidate_format": candidate.candidate_format,
                        "patch_block_count": candidate.patch_block_count,
                        "lines_changed": candidate.lines_changed,
                    },
                    "prompt": prompt,
                    "prompt_path": str(result.workspace_path / "prompt.txt"),
                    "raw_response": code,
                    "raw_response_path": str(result.workspace_path / "response.txt"),
                    "preflight": preflight.to_dict(),
                    "preflight_path": str(result.workspace_path / "preflight.json"),
                    "evaluation": result.to_dict(),
                    "reward": reward,
                    "correctness": correctness,
                    "message": self._build_message(candidate, result, current_best, reward),
                    "improved_global_best": improved_global_best,
                    "failure_stage": "runtime" if result.status != "success" else "",
                    "failure_reason": result.status if result.status != "success" else "",
                },
            )

        message = self._build_message(candidate, result, current_best, reward)
        stdout = self.runner.read_text(result.stdout_path)
        raw_score = result.val_bpb if result.val_bpb is not None else _FAIL_RAW_SCORE
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
                "candidate_format": candidate.candidate_format,
                "patch_block_count": candidate.patch_block_count,
                "lines_changed": candidate.lines_changed,
                "candidate_status": result.status,
                "candidate_val_bpb": result.val_bpb,
                "workspace_path": str(result.workspace_path),
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "improved_global_best": improved_global_best,
                "prompt": prompt,
                "preflight": preflight.to_dict(),
            },
        }

    def _persist_invalid_candidate(self, code: str, prompt: str, state: Any, error_message: str) -> dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        step = getattr(state, "timestep", -1) + 1
        state_id = getattr(state, "id", "unknown")
        current_best = self._current_best_from_state(state)
        with _ARTIFACT_LOCK:
            workspace = self.runner.create_candidate_artifact_dir(step=step, prefix="invalid")
            response_path = workspace / "response.txt"
            response_path.write_text(code, encoding="utf-8")
            prompt_path = workspace / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            preflight_path = workspace / "preflight.json"
            self.runner.write_json_artifact(
                preflight_path,
                {
                    "ok": False,
                    "stage": "invalid_format",
                    "reason": error_message,
                    "details": {},
                },
            )
            metrics_path = workspace / "metrics.json"
            self.runner.write_json_artifact(
                metrics_path,
                {
                    "candidate_status": "invalid_candidate",
                    "error": error_message,
                    "prompt_path": str(prompt_path),
                    "candidate_format": "invalid_format",
                },
            )
            history_entry = {
                "step": step,
                "state_id": state_id,
                "status": "invalid_candidate",
                "summary": "",
                "rationale": "",
                "candidate_format": "invalid_format",
                "patch_block_count": 0,
                "lines_changed": 0,
                "reward": _FAIL_REWARD,
                "accepted": False,
                "val_bpb": None,
                "parent_val_bpb": current_best,
                "stdout_path": "",
                "stderr_path": "",
                "workspace_path": str(workspace),
                "improved_global_best": False,
                "error": error_message,
                "prompt_path": str(prompt_path),
                "preflight_path": str(preflight_path),
                "failure_stage": "invalid_format",
                "failure_reason": error_message,
            }
            self.runner.append_history(history_entry)
            self.runner.write_rollout_manifest(
                workspace,
                {
                    "step": step,
                    "starting_state": state.to_dict() if hasattr(state, "to_dict") else {
                        "id": state_id,
                        "timestep": getattr(state, "timestep", -1),
                    },
                    "candidate": None,
                    "candidate_format": "invalid_format",
                    "patch_block_count": 0,
                    "lines_changed": 0,
                    "prompt": prompt,
                    "prompt_path": str(prompt_path),
                    "raw_response_path": str(response_path),
                    "raw_response": code,
                    "preflight": {
                        "ok": False,
                        "stage": "invalid_format",
                        "reason": error_message,
                        "details": {},
                    },
                    "preflight_path": str(preflight_path),
                    "evaluation": {
                        "status": "invalid_candidate",
                        "workspace_path": str(workspace),
                        "metrics_path": str(metrics_path),
                    },
                    "reward": _FAIL_REWARD,
                    "correctness": 0.0,
                    "message": error_message,
                    "improved_global_best": False,
                    "failure_stage": "invalid_format",
                    "failure_reason": error_message,
                },
            )
            return self._failure_payload(
                reward=_FAIL_REWARD,
                raw_score=_FAIL_RAW_SCORE,
                msg=error_message,
                status="invalid_candidate",
            )

    def _persist_preflight_failed_candidate(
        self,
        *,
        candidate: PatchCandidate,
        code: str,
        prompt: str,
        state: Any,
        workspace: Path,
        preflight: PreflightResult,
    ) -> dict[str, Any]:
        if self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        step = getattr(state, "timestep", -1) + 1
        state_id = getattr(state, "id", "unknown")
        current_best = self._current_best_from_state(state)
        metrics_path = workspace / "metrics.json"
        with _ARTIFACT_LOCK:
            self.runner.write_json_artifact(
                metrics_path,
                {
                    "candidate_status": "preflight_failed",
                    "error": preflight.reason,
                    "candidate_format": candidate.candidate_format,
                    "patch_block_count": candidate.patch_block_count,
                    "lines_changed": candidate.lines_changed,
                    "preflight_path": str(workspace / "preflight.json"),
                },
            )
            history_entry = {
                "step": step,
                "state_id": state_id,
                "status": "preflight_failed",
                "summary": candidate.summary,
                "rationale": candidate.rationale,
                "candidate_format": candidate.candidate_format,
                "patch_block_count": candidate.patch_block_count,
                "lines_changed": candidate.lines_changed,
                "reward": _FAIL_REWARD,
                "accepted": False,
                "val_bpb": None,
                "parent_val_bpb": current_best,
                "stdout_path": "",
                "stderr_path": "",
                "workspace_path": str(workspace),
                "improved_global_best": False,
                "error": preflight.reason,
                "prompt_path": str(workspace / "prompt.txt"),
                "preflight_path": str(workspace / "preflight.json"),
                "failure_stage": preflight.stage,
                "failure_reason": preflight.reason,
            }
            self.runner.append_history(history_entry)
            self.runner.write_rollout_manifest(
                workspace,
                {
                    "step": step,
                    "starting_state": state.to_dict() if hasattr(state, "to_dict") else {
                        "id": state_id,
                        "timestep": getattr(state, "timestep", -1),
                    },
                    "candidate": {
                        "summary": candidate.summary,
                        "rationale": candidate.rationale,
                        "train_py": candidate.train_py,
                        "candidate_format": candidate.candidate_format,
                        "patch_block_count": candidate.patch_block_count,
                        "lines_changed": candidate.lines_changed,
                    },
                    "prompt": prompt,
                    "prompt_path": str(workspace / "prompt.txt"),
                    "raw_response": code,
                    "raw_response_path": str(workspace / "response.txt"),
                    "preflight": preflight.to_dict(),
                    "preflight_path": str(workspace / "preflight.json"),
                    "evaluation": {
                        "status": "preflight_failed",
                        "workspace_path": str(workspace),
                        "metrics_path": str(metrics_path),
                    },
                    "reward": _FAIL_REWARD,
                    "correctness": 0.0,
                    "message": preflight.reason,
                    "improved_global_best": False,
                    "failure_stage": preflight.stage,
                    "failure_reason": preflight.reason,
                },
            )
        return self._failure_payload(
            reward=_FAIL_REWARD,
            raw_score=_FAIL_RAW_SCORE,
            msg=preflight.reason,
            status="preflight_failed",
        )

    def _run_candidate(self, candidate: PatchCandidate, state: Any, workspace: Path) -> RunResult:
        if self.bootstrap is None or self.runner is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        if _EVALUATION_SLOTS is None:
            raise RuntimeError("AutoResearchRewardEvaluator evaluation slots are not configured.")

        # Grouped rollouts stay enabled for the upstream entropic advantage recipe,
        # but inner autoresearch training runs must be serialized on a single GPU.
        _EVALUATION_SLOTS.acquire()
        gpu_device: str | None = None
        try:
            if _GPU_DEVICE_QUEUE is not None:
                gpu_device = _GPU_DEVICE_QUEUE.get()
            return self.runner.run_candidate(
                bootstrap=self.bootstrap,
                workspace=workspace,
                step=getattr(state, "timestep", -1) + 1,
                state_id=getattr(state, "id", "unknown"),
                gpu_device=gpu_device,
            )
        finally:
            if _GPU_DEVICE_QUEUE is not None and gpu_device is not None:
                _GPU_DEVICE_QUEUE.put(gpu_device)
            _EVALUATION_SLOTS.release()

    @staticmethod
    def _build_message(candidate: PatchCandidate, result: RunResult, current_best: float, reward: float) -> str:
        val_bpb = "n/a" if result.val_bpb is None else f"{result.val_bpb:.6f}"
        return (
            f"{candidate.summary}\n"
            f"status={result.status} parent_val_bpb={current_best:.6f} "
            f"candidate_val_bpb={val_bpb} reward={reward:.6f}"
        )

    def _prompt_for_state(self, state: Any) -> str:
        if self.bootstrap is None:
            raise RuntimeError("AutoResearchRewardEvaluator is not configured.")
        target = self.bootstrap.config.target_val_bpb
        if target is None:
            target = self._current_best_from_state(state)
        return build_prompt_for_state(state, target)

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
