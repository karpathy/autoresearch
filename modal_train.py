#!/usr/bin/env python3
"""Modal wrapper that runs autoresearch train.py on a remote GPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import modal
from modal_contract import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VOLUME_NAME,
    EVENT_PREFIX,
    RESULT_PREFIX,
)

MINUTES = 60
APP_NAME = "autoresearch-local"
REPO_ROOT = Path(__file__).resolve().parent
WORKDIR = "/root"
DATA_DIR = f"{WORKDIR}/.cache/autoresearch"
TRAIN_PATH = Path(f"{WORKDIR}/train.py")
PREPARE_PATH = Path(f"{WORKDIR}/prepare.py")
SUMMARY_PATTERNS = {
    "val_bpb": re.compile(r"^val_bpb:\s+([0-9.]+)\s*$"),
    "training_seconds": re.compile(r"^training_seconds:\s+([0-9.]+)\s*$"),
    "total_seconds": re.compile(r"^total_seconds:\s+([0-9.]+)\s*$"),
    "peak_vram_mb": re.compile(r"^peak_vram_mb:\s+([0-9.]+)\s*$"),
}
TRAIN_CONFIG_PATTERNS = {
    "embedding_lr": re.compile(r"^EMBEDDING_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "unembedding_lr": re.compile(r"^UNEMBEDDING_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "matrix_lr": re.compile(r"^MATRIX_LR\s*=\s*([0-9.]+)", re.MULTILINE),
    "scalar_lr": re.compile(r"^SCALAR_LR\s*=\s*([0-9.]+)", re.MULTILINE),
}


def emit(event: str, **data: Any) -> None:
    print(f"{EVENT_PREFIX}{json.dumps({'event': event, **data}, sort_keys=True)}", flush=True)


def emit_result(data: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(data, sort_keys=True)}", flush=True)


def get_data_volume(volume_name: str) -> modal.Volume:
    return modal.Volume.from_name(volume_name, create_if_missing=True)


def parse_float_lines(line: str, summary_metrics: dict[str, float | None]) -> None:
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.match(line)
        if match:
            summary_metrics[key] = float(match.group(1))


def read_train_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    train_config: dict[str, float] = {}
    for key, pattern in TRAIN_CONFIG_PATTERNS.items():
        match = pattern.search(text)
        if match:
            train_config[key] = float(match.group(1))
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "config": train_config,
    }


def stream_command(
    command: list[str],
    *,
    cwd: str,
    phase: str,
    summary_metrics: dict[str, float | None] | None = None,
) -> tuple[int, float]:
    start = time.time()
    emit(f"{phase}_start", command=command, cwd=cwd)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if summary_metrics is not None:
            parse_float_lines(line.rstrip("\n"), summary_metrics)
    proc.wait()
    elapsed = time.time() - start
    emit(f"{phase}_complete", exit_code=proc.returncode, seconds=round(elapsed, 3))
    return proc.returncode, elapsed


def ensure_cuda_ready(*, cwd: str, max_attempts: int = 3, retry_delay_seconds: int = 5) -> None:
    probe = [
        "python",
        "-u",
        "-c",
        "import torch; torch.cuda.init(); print('CUDA_READY', torch.cuda.get_device_name(0), flush=True)",
    ]
    for attempt in range(1, max_attempts + 1):
        code, _ = stream_command(probe, cwd=cwd, phase=f"gpu_probe_attempt_{attempt}")
        if code == 0:
            emit("gpu_probe_ready", attempt=attempt)
            return
        if attempt < max_attempts:
            emit("gpu_probe_retrying", attempt=attempt, sleep_seconds=retry_delay_seconds)
            time.sleep(retry_delay_seconds)
    raise RuntimeError("GPU probe failed; CUDA never became ready for train.py")


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128",
        "uv pip install --system 'kernels>=0.11.7' 'tiktoken>=0.11.0' 'rustbpe>=0.1.0' "
        "'pyarrow>=21.0.0' 'pandas>=2.3.3' 'numpy>=2.2.6' 'requests>=2.32.0' 'matplotlib>=3.10.8'",
    )
    .add_local_file(str(REPO_ROOT / "train.py"), remote_path=str(TRAIN_PATH), copy=False)
    .add_local_file(str(REPO_ROOT / "prepare.py"), remote_path=str(PREPARE_PATH), copy=False)
    .add_local_file(str(REPO_ROOT / "modal_contract.py"), remote_path=f"{WORKDIR}/modal_contract.py", copy=False)
)

app = modal.App(APP_NAME)


@app.cls(image=image)
class TrainRunner:
    @modal.method()
    def train(self, skip_prepare: bool, gpu: str, volume_name: str, timeout_minutes: int) -> dict[str, Any]:
        run_start = time.time()
        prepare_seconds = 0.0
        train_seconds = 0.0
        prepare_exit_code = 0
        train_exit_code = 0
        summary_metrics: dict[str, float | None] = {
            "val_bpb": None,
            "training_seconds": None,
            "total_seconds": None,
            "peak_vram_mb": None,
        }
        train_manifest = read_train_manifest(Path(REPO_ROOT / "train.py"))
        data_volume = get_data_volume(volume_name)
        result: dict[str, Any] = {
            "schema_version": 1,
            "app_name": APP_NAME,
            "app_id": os.environ.get("MODAL_APP_ID"),
            "run_url": os.environ.get("MODAL_RUN_URL"),
            "logs_url": os.environ.get("MODAL_LOGS_URL"),
            "modal_environment": os.environ.get("MODAL_ENVIRONMENT"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "cwd": WORKDIR,
            "data_dir": DATA_DIR,
            "gpu": gpu,
            "volume_name": volume_name,
            "timeout_minutes": timeout_minutes,
            "skip_prepare": skip_prepare,
            "train_sha256": train_manifest["sha256"],
            "train_bytes": train_manifest["bytes"],
            "train_config": train_manifest["config"],
            "val_bpb": None,
            "peak_vram_mb": None,
            "training_seconds": None,
            "total_seconds": None,
            "prepare_seconds": None,
            "remote_train_seconds": None,
            "wrapper_total_seconds": None,
            "exit_status": 0,
            "crashed": False,
            "error_type": None,
            "error_message": None,
        }

        data_volume.reload()
        emit(
            "run_start",
            app=APP_NAME,
            cwd=WORKDIR,
            data_dir=DATA_DIR,
            data_volume=volume_name,
            gpu=gpu,
            skip_prepare=skip_prepare,
            timeout_minutes=timeout_minutes,
        )
        emit(
            "train_manifest",
            path=str(TRAIN_PATH),
            sha256=train_manifest["sha256"],
            bytes=train_manifest["bytes"],
            train_config=train_manifest["config"],
        )

        try:
            if not skip_prepare:
                prepare_exit_code, prepare_seconds = stream_command(
                    ["python", "-u", "prepare.py"],
                    cwd=WORKDIR,
                    phase="prepare",
                )
                if prepare_exit_code != 0:
                    raise RuntimeError(f"prepare.py failed with exit code {prepare_exit_code}")
                data_volume.commit()
            else:
                emit("prepare_skipped", reason="skip_prepare flag set")

            ensure_cuda_ready(cwd=WORKDIR)
            train_exit_code, train_seconds = stream_command(
                ["python", "-u", "train.py"],
                cwd=WORKDIR,
                phase="train",
                summary_metrics=summary_metrics,
            )
            if train_exit_code != 0:
                raise RuntimeError(f"train.py failed with exit code {train_exit_code}")
            data_volume.commit()
        except Exception as exc:
            result["crashed"] = True
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)
            result["exit_status"] = train_exit_code or prepare_exit_code or 1

        result["val_bpb"] = summary_metrics["val_bpb"]
        result["peak_vram_mb"] = summary_metrics["peak_vram_mb"]
        result["training_seconds"] = summary_metrics["training_seconds"]
        result["total_seconds"] = summary_metrics["total_seconds"]
        result["prepare_seconds"] = round(prepare_seconds, 3)
        result["remote_train_seconds"] = round(train_seconds, 3)
        result["wrapper_total_seconds"] = round(time.time() - run_start, 3)

        if (
            result["exit_status"] == 0
            and result["val_bpb"] is None
            and result["peak_vram_mb"] is None
        ):
            result["crashed"] = True
            result["exit_status"] = 1
            result["error_type"] = "MissingMetricsError"
            result["error_message"] = "train.py finished without emitting summary metrics"

        emit(
            "run_complete",
            exit_code=result["exit_status"],
            crashed=result["crashed"],
            prepare_seconds=result["prepare_seconds"],
            train_seconds=result["remote_train_seconds"],
            total_seconds=result["wrapper_total_seconds"],
            val_bpb=result["val_bpb"],
            peak_vram_mb=result["peak_vram_mb"],
            training_seconds=result["training_seconds"],
            train_total_seconds=result["total_seconds"],
            train_sha256=result["train_sha256"],
            train_config=result["train_config"],
        )
        return result


@app.local_entrypoint()
def main(
    skip_prepare: bool = False,
    gpu: str = DEFAULT_GPU,
    volume_name: str = DEFAULT_VOLUME_NAME,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
) -> None:
    normalized_volume_name = volume_name.strip()
    if not normalized_volume_name:
        raise ValueError("volume_name must not be empty")
    if timeout_minutes <= 0:
        raise ValueError("timeout_minutes must be positive")

    data_volume = get_data_volume(normalized_volume_name)
    runner = TrainRunner.with_options(
        gpu=gpu,
        timeout=timeout_minutes * MINUTES,
        volumes={DATA_DIR: data_volume},
    )
    function_call = runner().train.spawn(
        skip_prepare=skip_prepare,
        gpu=gpu,
        volume_name=normalized_volume_name,
        timeout_minutes=timeout_minutes,
    )
    emit(
        "call_spawned",
        function_call_id=function_call.object_id,
        gpu=gpu,
        skip_prepare=skip_prepare,
        timeout_minutes=timeout_minutes,
        volume_name=normalized_volume_name,
    )

    try:
        result = function_call.get()
    except Exception as exc:
        result = {
            "schema_version": 1,
            "app_name": APP_NAME,
            "app_id": None,
            "run_url": None,
            "logs_url": None,
            "modal_environment": os.environ.get("MODAL_ENVIRONMENT"),
            "modal_task_id": None,
            "cwd": WORKDIR,
            "data_dir": DATA_DIR,
            "gpu": gpu,
            "volume_name": normalized_volume_name,
            "timeout_minutes": timeout_minutes,
            "skip_prepare": skip_prepare,
            "train_sha256": None,
            "train_bytes": None,
            "train_config": {},
            "val_bpb": None,
            "peak_vram_mb": None,
            "training_seconds": None,
            "total_seconds": None,
            "prepare_seconds": None,
            "remote_train_seconds": None,
            "wrapper_total_seconds": None,
            "exit_status": 1,
            "crashed": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    emit("local_result", exit_status=result.get("exit_status"), skip_prepare=skip_prepare)
    emit_result(result)


if __name__ == "__main__":
    main()
