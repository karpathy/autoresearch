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


def patch_ttt_discover_no_wandb_bug() -> None:
    """Pad discover's multiplex logger so W&B-optional runs do not crash.

    Upstream do_sync_training incorrectly checks ``len(loggers) >= 2`` and then
    indexes ``loggers[2]``. When W&B is disabled, setup_logging only creates two
    loggers, so the first train step crashes. We pad the logger list with no-op
    loggers until index 2 is always safe.
    """

    try:
        from ttt_discover.tinker_utils import ml_log
    except ImportError:
        return

    if getattr(ml_log, "_autoresearch_no_wandb_patch", False):
        return

    class _NullLogger(ml_log.Logger):
        def log_hparams(self, config: Any) -> None:
            return None

        def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
            return None

    original_setup_logging = ml_log.setup_logging

    def patched_setup_logging(*args: Any, **kwargs: Any):
        logger = original_setup_logging(*args, **kwargs)
        if hasattr(logger, "loggers"):
            while len(logger.loggers) < 3:
                logger.loggers.append(_NullLogger())
        return logger

    ml_log.setup_logging = patched_setup_logging
    ml_log._autoresearch_no_wandb_patch = True


def patch_ttt_discover_kimi_tokenizer() -> None:
    """Teach upstream discover tokenizers to trust_remote_code for Kimi K2.5.

    Upstream currently special-cases only ``moonshotai/Kimi-K2-Thinking`` in some
    tokenizer paths. ``moonshotai/Kimi-K2.5`` requires the same trust_remote_code
    handling, otherwise detached runs die on an interactive prompt.
    """

    def _wrap_get_tokenizer(module: Any, sentinel_name: str) -> None:
        if getattr(module, sentinel_name, False):
            return

        original_get_tokenizer = module.get_tokenizer

        def patched_get_tokenizer(model_name: str):
            if model_name == "moonshotai/Kimi-K2.5":
                import os
                from transformers.models.auto.tokenization_auto import AutoTokenizer

                if os.path.isdir(model_name):
                    return AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=True,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                return AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                    trust_remote_code=True,
                )
            return original_get_tokenizer(model_name)

        module.get_tokenizer = patched_get_tokenizer
        setattr(module, sentinel_name, True)

    try:
        from ttt_discover.tinker_utils import misc_utils
    except ImportError:
        misc_utils = None
    if misc_utils is not None:
        _wrap_get_tokenizer(misc_utils, "_autoresearch_kimi_patch")

    try:
        from ttt_discover.tinker_utils import renderers
    except ImportError:
        renderers = None
    if renderers is not None:
        _wrap_get_tokenizer(renderers, "_autoresearch_kimi_patch")

    try:
        from ttt_discover.tinker_utils import dataset_builder
    except ImportError:
        dataset_builder = None
    if dataset_builder is not None and misc_utils is not None:
        dataset_builder.get_tokenizer = misc_utils.get_tokenizer


def patch_transformers_kimi_trust_remote_code() -> None:
    """Force trust_remote_code=True for Kimi K2.5 tokenizer loads.

    This catches code paths that bypass discover's helper and call the
    Transformers auto-tokenizer directly.
    """

    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except ImportError:
        return

    if getattr(AutoTokenizer, "_autoresearch_kimi_trust_patch", False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any):
        model_name = str(pretrained_model_name_or_path)
        if model_name == "moonshotai/Kimi-K2.5":
            kwargs.setdefault("trust_remote_code", True)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = patched_from_pretrained
    AutoTokenizer._autoresearch_kimi_trust_patch = True
