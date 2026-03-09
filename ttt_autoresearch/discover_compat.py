from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    from ttt_discover import BaseRewardEvaluator, Environment, State
    from ttt_discover.tinker_utils.dataset_builder import VerifyResult
except ImportError:
    class BaseRewardEvaluator(ABC):
        @abstractmethod
        def get_reward(self, code: str, state: Any) -> dict[str, Any]:
            raise NotImplementedError

    class State:
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
        ) -> None:
            self.id = id or str(uuid.uuid4())
            self.timestep = timestep
            self.construction = construction or []
            self.code = code
            self.value = value
            self.parent_values = parent_values or []
            self.parents = parents or []
            self.observation = observation

        def to_dict(self) -> dict[str, Any]:
            return {
                "type": self.__class__.__name__,
                "id": self.id,
                "timestep": self.timestep,
                "value": self.value,
                "construction": self.construction,
                "code": self.code,
                "parent_values": self.parent_values,
                "parents": self.parents,
                "observation": self.observation,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> "State":
            return cls(
                timestep=data["timestep"],
                construction=data.get("construction"),
                code=data["code"],
                value=data.get("value"),
                parent_values=data.get("parent_values"),
                parents=data.get("parents"),
                id=data.get("id"),
                observation=data.get("observation", ""),
            )

    @dataclass
    class VerifyResult:
        reward: float
        msg: str
        correctness: float
        raw_score: float
        result_construction: Any
        stdout: str
        metrics: dict[str, Any] = field(default_factory=dict)

    class Environment:
        reward_function: type[BaseRewardEvaluator]
        state_type: type[State]

        def __init__(self, renderer: Any, initial_state: State, sampler: Any, config: Any) -> None:
            self.renderer = renderer
            self.initial_state = initial_state
            self.state = initial_state
            self.sampler = sampler
            self.config = config
            self.problem_type = getattr(config, "problem_type", "autoresearch")
            self.log_path = getattr(config, "log_path", "")
            self.eval_timeout = getattr(config, "eval_timeout", 0)
            self.num_cpus_per_task = getattr(config, "num_cpus_per_task", 0)

