from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import json
from typing import Any, ClassVar

from ttt_autoresearch.config import BootstrapContext
from ttt_autoresearch.discover_compat import Environment, State, VerifyResult
from ttt_autoresearch.reward import AutoResearchRewardEvaluator
from ttt_autoresearch.runner import parse_patch_candidate


def read_recent_history(history_path: Path, limit: int) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:]


class AutoResearchState(State):
    def __init__(
        self,
        timestep: int,
        construction: list[Any] | None,
        code: str,
        value: float | None = None,
        parent_values: list[float] | None = None,
        parents: list[dict[str, Any]] | None = None,
        id: str | None = None,
        observation: str = "",
        baseline_val_bpb: float | None = None,
        current_best_val_bpb: float | None = None,
        history: list[dict[str, Any]] | None = None,
        summary: str = "",
        rationale: str = "",
        raw_score: float | None = None,
    ) -> None:
        super().__init__(
            timestep=timestep,
            construction=construction or [],
            code=code,
            value=value,
            parent_values=parent_values,
            parents=parents,
            id=id,
            observation=observation,
        )
        self.baseline_val_bpb = baseline_val_bpb
        self.current_best_val_bpb = current_best_val_bpb
        self.history = history or []
        self.summary = summary
        self.rationale = rationale
        self.raw_score = raw_score

    @property
    def step(self) -> int:
        return self.timestep

    @property
    def current_train_py(self) -> str:
        return self.code

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "type": self.__class__.__name__,
                "baseline_val_bpb": self.baseline_val_bpb,
                "current_best_val_bpb": self.current_best_val_bpb,
                "history": self.history,
                "summary": self.summary,
                "rationale": self.rationale,
                "raw_score": self.raw_score,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoResearchState":
        return cls(
            timestep=data["timestep"],
            construction=data.get("construction"),
            code=data["code"],
            value=data.get("value"),
            parent_values=data.get("parent_values"),
            parents=data.get("parents"),
            id=data.get("id"),
            observation=data.get("observation", ""),
            baseline_val_bpb=data.get("baseline_val_bpb"),
            current_best_val_bpb=data.get("current_best_val_bpb"),
            history=data.get("history"),
            summary=data.get("summary", ""),
            rationale=data.get("rationale", ""),
            raw_score=data.get("raw_score"),
        )


class AutoResearchDiscoverEnv(Environment):
    reward_function = AutoResearchRewardEvaluator
    state_type = AutoResearchState
    bootstrap: ClassVar[BootstrapContext | None] = None

    @classmethod
    def configure(cls, bootstrap: BootstrapContext) -> None:
        cls.bootstrap = bootstrap

    @classmethod
    def create_initial_state(cls, problem_type: str) -> AutoResearchState:
        if cls.bootstrap is None:
            raise RuntimeError("AutoResearchDiscoverEnv is not configured.")
        baseline_stdout = ""
        baseline_stdout_path = cls.bootstrap.run_dir / "baseline" / "workspace" / "stdout.log"
        if baseline_stdout_path.exists():
            baseline_stdout = baseline_stdout_path.read_text(encoding="utf-8")[:4000]
        # Read the actual current best from disk — it may have improved
        # across prior RL steps since the baseline was established.
        current_best = cls.bootstrap.baseline_val_bpb
        best_metrics = cls.bootstrap.best_dir / "metrics.json"
        if best_metrics.exists():
            try:
                stored = json.loads(best_metrics.read_text(encoding="utf-8"))
                if stored.get("val_bpb") is not None:
                    current_best = float(stored["val_bpb"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        # Read the best train.py if it has been updated.
        best_train_py = cls.bootstrap.baseline_train_py
        best_train_path = cls.bootstrap.best_dir / "train.py"
        if best_train_path.exists():
            best_train_py = best_train_path.read_text(encoding="utf-8")
        return AutoResearchState(
            timestep=-1,
            construction=[],
            code=best_train_py,
            value=-current_best,
            observation=baseline_stdout,
            baseline_val_bpb=cls.bootstrap.baseline_val_bpb,
            current_best_val_bpb=current_best,
            history=[],
            summary="baseline",
            rationale="seed state from the original autoresearch train.py",
            raw_score=current_best,
        )

    def is_maximize(self) -> bool:
        return False

    def _get_code_languages(self) -> list[str]:
        return ["json"]

    def _should_keep_code_separators(self) -> bool:
        return False

    def get_question(self) -> str:
        if self.bootstrap is None:
            raise RuntimeError("AutoResearchDiscoverEnv is not configured.")

        state = self.initial_state
        history = read_recent_history(self.bootstrap.history_path, self.bootstrap.config.keep_history)
        if history:
            history_text = "\n".join(
                f"- [{entry['status']}] reward={entry['reward']:.6f} val_bpb={entry.get('val_bpb')} summary={entry['summary']}"
                for entry in history
            )
        else:
            history_text = "- No prior candidate evaluations yet."

        return f"""{self.bootstrap.program_text}

You are the outer autoresearch agent. Your only job is to improve train.py.
You may edit train.py only. Do not modify prepare.py, program.md, or any other file.
The reward comes from running train.py and measuring val_bpb. Lower val_bpb is better.

Current best val_bpb: {state.current_best_val_bpb:.6f}
Baseline val_bpb: {state.baseline_val_bpb:.6f}

Current best train.py:
```python
{state.current_train_py}
```

Recent accepted and rejected edits:
{history_text}

Return exactly one ```json``` block with this schema:
{{
  "summary": "short description of the change",
  "rationale": "why this should improve val_bpb",
  "train_py": "the full replacement contents of train.py"
}}

Rules:
- train_py must be the full file, not a diff.
- Only edit train.py.
- Keep the file runnable as a standalone script.
- Optimize for the lowest val_bpb under the existing time budget.
"""

    def check_format(self, parsed_code: str) -> bool:
        try:
            parse_patch_candidate(parsed_code)
        except ValueError:
            return False
        return True

    async def check_answer(self, parsed_code: str, step: int) -> VerifyResult:
        if not self.check_format(parsed_code):
            return VerifyResult(
                reward=-1.0,
                msg="Invalid candidate JSON.",
                correctness=0.0,
                raw_score=float(self.initial_state.current_best_val_bpb),
                result_construction=[],
                stdout="",
                metrics={"candidate_status": "invalid_candidate"},
            )

        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, self._run_reward, parsed_code)
        return VerifyResult(
            reward=out["reward"],
            msg=out["msg"],
            correctness=out["correctness"],
            raw_score=out["raw_score"],
            result_construction=out.get("result_construction", []),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
        )

    def _create_next_state(self, step_idx: int, parsed_code: str, outs: VerifyResult) -> AutoResearchState:
        candidate = parse_patch_candidate(parsed_code)
        history_entry = {
            "step": step_idx,
            "summary": candidate.summary,
            "rationale": candidate.rationale,
            "reward": outs.reward,
            "val_bpb": outs.raw_score,
            "status": outs.metrics.get("candidate_status", "unknown"),
        }
        prior_history = list(getattr(self.initial_state, "history", []))
        next_history = (prior_history + [history_entry])[-10:]
        parent_best = self.initial_state.current_best_val_bpb
        new_best = min(parent_best, outs.raw_score) if outs.raw_score is not None else parent_best
        return AutoResearchState(
            timestep=step_idx,
            construction=[],
            code=candidate.train_py,
            value=-outs.raw_score,
            observation=outs.stdout,
            baseline_val_bpb=self.initial_state.baseline_val_bpb,
            current_best_val_bpb=new_best,
            history=next_history,
            summary=candidate.summary,
            rationale=candidate.rationale,
            raw_score=outs.raw_score,
        )

    def _build_metrics(
        self,
        outs: VerifyResult,
        correct_format: bool,
        message: dict[str, Any],
        parsed_code: str,
    ) -> dict[str, Any]:
        metrics = {
            "format": correct_format,
            "reward": outs.reward,
            "correctness": outs.correctness,
            "raw_score": outs.raw_score,
            "prompt": self.get_question(),
            "response": message["content"],
            "parsed_code": parsed_code,
            "msg": outs.msg,
        }
        metrics.update(outs.metrics)
        return metrics

    def _run_reward(self, parsed_code: str) -> dict[str, Any]:
        evaluator = self.reward_function(
            problem_type=self.problem_type,
            log_dir=self.log_path,
            eval_timeout=self.eval_timeout,
            num_cpus_per_task=self.num_cpus_per_task,
        )
        return evaluator.get_reward(parsed_code, state=self.initial_state)
