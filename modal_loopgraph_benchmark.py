"""Run LoopGraph benchmarks on Modal.

This wrapper copies the current repo into a Modal container, prepares a small
Autoresearch cache on a persistent Modal volume, and runs either the baseline vs
LoopGraph comparison or one ASEP benchmark mode.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

import modal

APP_NAME = "loopgraph-benchmark"
REMOTE_REPO = "/root/loopgraph"
CACHE_PATH = "/root/.cache/autoresearch"

_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "VENICE_API_KEY",
)

_secret_values = {
    key: value
    for key in _PROVIDER_ENV_KEYS
    if (value := os.environ.get(key))
}
_provider_secrets = [modal.Secret.from_dict(_secret_values)] if _secret_values else []

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("loopgraph-autoresearch-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        Path(__file__).resolve().parent,
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            "run_*.log",
            "run_baseline.log",
            "results.tsv",
            "researchfs.db",
        ],
    )
)


def _run(cmd: list[str], *, env: dict[str, str]) -> dict[str, object]:
    proc = subprocess.run(
        cmd,
        cwd=REMOTE_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _ensure_git_repo(env: dict[str, str]) -> dict[str, object]:
    """Create a disposable git repo so the agent loop can branch and commit."""
    steps = []
    if (Path(REMOTE_REPO) / ".git").exists():
        return {"cmd": ["git", "status"], "returncode": 0, "stdout": "existing git repo", "stderr": ""}

    for cmd in (
        ["git", "init"],
        ["git", "config", "user.name", env["GIT_AUTHOR_NAME"]],
        ["git", "config", "user.email", env["GIT_AUTHOR_EMAIL"]],
        ["git", "add", "."],
        ["git", "commit", "-m", "modal benchmark base"],
    ):
        step = _run(cmd, env=env)
        steps.append(step)
        if step["returncode"] != 0:
            return {
                "cmd": ["git", "bootstrap"],
                "returncode": step["returncode"],
                "stdout": "\n".join(str(s.get("stdout") or "") for s in steps),
                "stderr": "\n".join(str(s.get("stderr") or "") for s in steps),
            }
    return {
        "cmd": ["git", "bootstrap"],
        "returncode": 0,
        "stdout": "\n".join(str(s.get("stdout") or "") for s in steps),
        "stderr": "\n".join(str(s.get("stderr") or "") for s in steps),
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={CACHE_PATH: cache_volume},
    secrets=_provider_secrets,
)
def run_benchmark(
    mode: Literal["comparison", "baseline", "loopgraph"] = "comparison",
    provider: Literal["openai", "openrouter", "venice"] = "openai",
    model: str = "gpt-4o-mini",
    max_experiments: int = 1,
    time_budget: int = 30,
    steps_budget: int = 20,
    prepare_shards: int = 2,
    max_method_chars: int = 6000,
) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "Autoresearch Agent")
    env.setdefault("GIT_AUTHOR_EMAIL", "agent@autoresearch.local")
    env.setdefault("GIT_COMMITTER_NAME", "Autoresearch Agent")
    env.setdefault("GIT_COMMITTER_EMAIL", "agent@autoresearch.local")

    # Fast benchmark settings. prepare.py selects cuda automatically on Modal GPU.
    env.setdefault("MAX_SEQ_LEN", "512")
    env.setdefault("DEPTH", "4")
    env.setdefault("DEVICE_BATCH_SIZE", "16")
    env.setdefault("TOTAL_BATCH_SIZE", "65536")
    env.setdefault("EVAL_TOKENS", "524288")
    env.setdefault("TIME_BUDGET", str(time_budget))
    env.setdefault("STEPS_BUDGET", str(steps_budget))

    steps: list[dict[str, object]] = []
    git_bootstrap = _ensure_git_repo(env)
    steps.append(git_bootstrap)
    if git_bootstrap["returncode"] != 0:
        return {"ok": False, "failed_step": "git bootstrap", "steps": steps}


    sync = _run(["uv", "sync", "--group", "dev"], env=env)
    steps.append(sync)
    if sync["returncode"] != 0:
        return {"ok": False, "failed_step": "uv sync", "steps": steps}

    if prepare_shards > 0:
        prepare = _run(
            ["uv", "run", "python", "prepare.py", "--num-shards", str(prepare_shards)],
            env=env,
        )
        steps.append(prepare)
        if prepare["returncode"] != 0:
            return {"ok": False, "failed_step": "prepare", "steps": steps}
        cache_volume.commit()

    if mode == "comparison":
        cmd = [
            "uv",
            "run",
            "python",
            "run_comparison.py",
            "--model",
            model,
            "--max-experiments",
            str(max_experiments),
            "--time-budget",
            str(time_budget),
            "--steps-budget",
            str(steps_budget),
            "--api-provider",
            provider,
        ]
    else:
        cmd = [
            "uv",
            "run",
            "python",
            "asep_loopgraph_benchmark.py",
            "--mode",
            mode,
            "--model",
            model,
            "--max-experiments",
            str(max_experiments),
            "--time-budget",
            str(time_budget),
            "--steps-budget",
            str(steps_budget),
            "--api-provider",
            provider,
            "--device",
            "cuda",
            "--max-method-chars",
            str(max_method_chars),
        ]

    benchmark = _run(cmd, env=env)
    steps.append(benchmark)
    return {
        "ok": benchmark["returncode"] == 0,
        "failed_step": None if benchmark["returncode"] == 0 else "benchmark",
        "steps": steps,
    }


@app.local_entrypoint()
def main(
    mode: Literal["comparison", "baseline", "loopgraph"] = "comparison",
    provider: Literal["openai", "openrouter", "venice"] = "openai",
    model: str = "gpt-4o-mini",
    max_experiments: int = 1,
    time_budget: int = 30,
    steps_budget: int = 20,
    prepare_shards: int = 2,
    max_method_chars: int = 6000,
) -> None:
    result = run_benchmark.remote(
        mode=mode,
        provider=provider,
        model=model,
        max_experiments=max_experiments,
        time_budget=time_budget,
        steps_budget=steps_budget,
        prepare_shards=prepare_shards,
        max_method_chars=max_method_chars,
    )
    print(f"ok={result['ok']} failed_step={result['failed_step']}")
    for index, step in enumerate(result["steps"], start=1):
        print("\n" + "=" * 80)
        print(f"step {index}: {' '.join(step['cmd'])}")
        print(f"returncode={step['returncode']}")
        stdout = str(step.get("stdout") or "")
        stderr = str(step.get("stderr") or "")
        if stdout:
            print("--- stdout ---")
            print(stdout)
        if stderr:
            print("--- stderr ---")
            print(stderr)
