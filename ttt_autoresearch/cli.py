from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from ttt_autoresearch.config import TTTAutoResearchConfig, load_config, write_resolved_config
from ttt_autoresearch.env import AutoResearchDiscoverEnv
from ttt_autoresearch.reward import AutoResearchRewardEvaluator
from ttt_autoresearch.runner import AutoResearchRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AutoResearch with TTT-Discover outer-loop RL.")
    parser.add_argument("--config", default="configs/ttt_discover_autoresearch.yaml", help="Path to the YAML config file.")
    parser.add_argument("--model-name", help="Override the outer-agent model name.")
    parser.add_argument("--provider", help="Override the provider identifier passed via environment variables.")
    parser.add_argument("--api-base", help="Override the API base URL passed via environment variables.")
    parser.add_argument("--run-dir", help="Override the run output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    config_path = _resolve_config_path(args.config, repo_root)
    config = load_config(config_path, repo_root=repo_root)
    config = _apply_overrides(config, args)

    try:
        from ttt_discover.rl.train import Config as RLConfig, main as discover_main
        from ttt_discover.tinker_utils.dataset_builder import DatasetConfig, get_single_problem_dataset_builder
    except ImportError as exc:
        parser.error(
            "ttt-discover is not installed. Run `uv sync` after updating dependencies, "
            "or install the pinned git dependency from pyproject.toml."
        )
        raise AssertionError from exc

    run_dir = Path(config.run_dir)
    runner = AutoResearchRunner(repo_root=repo_root, config=config, run_dir=run_dir)
    bootstrap = runner.build_bootstrap(baseline_val_bpb=float("inf"))
    baseline_result = runner.run_baseline(bootstrap=bootstrap)
    if baseline_result.val_bpb is None:
        parser.error(f"Baseline run failed with status={baseline_result.status}. Check {baseline_result.stdout_path} and {baseline_result.stderr_path}.")

    bootstrap = runner.build_bootstrap(baseline_val_bpb=baseline_result.val_bpb)
    runner.initialize_best_from_baseline(baseline_result, bootstrap.baseline_train_py)
    AutoResearchDiscoverEnv.configure(bootstrap)
    AutoResearchRewardEvaluator.configure(bootstrap, runner)
    write_resolved_config(run_dir / "resolved_config.json", config)

    dataset_config = DatasetConfig(
        env_type=AutoResearchDiscoverEnv,
        problem_type="autoresearch",
        batch_size=1,
        group_size=config.samples_per_step,
        model_name_for_tokenizer=config.local_model_path or config.model_name,
        renderer_name=config.renderer_name,
        num_cpus_per_task=config.num_cpus_per_task,
        eval_timeout=config.eval_timeout,
        log_path=str(bootstrap.discover_log_dir),
    )
    dataset_builder = get_single_problem_dataset_builder(dataset_config)
    # Keep discover's RL recipe unchanged and only swap in the autoresearch task surface.
    rl_config = RLConfig(
        env_type=AutoResearchDiscoverEnv,
        problem_type="autoresearch",
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        num_epochs=config.max_steps,
        temperature=config.temperature,
        lora_rank=config.lora_rank,
        wandb_project=config.wandb_project,
        wandb_name=config.experiment_name,
        log_path=str(bootstrap.discover_log_dir),
        kl_penalty_coef=config.kl_penalty_coef,
        save_every=config.save_every,
        remove_constant_reward_groups=True,
        phase1_max_tokens=config.phase1_max_tokens,
        local_model_path=config.local_model_path,
    )
    asyncio.run(discover_main(rl_config))
    return 0


def _resolve_config_path(config_arg: str, repo_root: Path) -> Path:
    candidate = Path(config_arg).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _apply_overrides(config: TTTAutoResearchConfig, args: argparse.Namespace) -> TTTAutoResearchConfig:
    updated = config.to_dict()
    if args.model_name:
        updated["model_name"] = args.model_name
        updated["renderer_name"] = None
    if args.provider:
        updated["provider"] = args.provider
    if args.api_base:
        updated["api_base"] = args.api_base
    if args.run_dir:
        updated["run_dir"] = args.run_dir
    return TTTAutoResearchConfig(**updated).normalized(Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
