#!/usr/bin/env python3
"""harness_ane.py — ANE training orchestrator for autoresearch-ane.

Compiles the ANE training binary, runs an experiment with a wall-time budget,
parses JSON output, and logs results. The agent drives the experiment loop
via program_ane.md — this script handles a single experiment cycle.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ANE_DIR = Path(__file__).parent / "ane"
BINARY = ANE_DIR / "train_ane"
CONFIG_FILE = ANE_DIR / "experiment_config.h"
CKPT_FILE = ANE_DIR / "ane_stories110M_ckpt.bin"
DATA_FILE = ANE_DIR / "tinystories_data00.bin"
RESULTS_FILE = Path(__file__).parent / "results.tsv"

ARCH_DEFINES = {"DIM", "HIDDEN", "HEADS", "SEQ", "NLAYERS"}


def parse_config(path: Path = CONFIG_FILE) -> dict[str, str]:
    """Parse #define values from experiment_config.h."""
    defines = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"#define\s+(\w+)\s+(.+?)(?:\s*//.*)?$", line.strip())
            if m:
                defines[m.group(1)] = m.group(2).strip()
    return defines


def hash_arch_config(path: Path = CONFIG_FILE) -> str:
    """Hash architecture defines to detect arch changes."""
    defines = parse_config(path)
    arch_vals = sorted((k, defines[k]) for k in ARCH_DEFINES if k in defines)
    return hashlib.sha256(str(arch_vals).encode()).hexdigest()[:16]


def validate_config(path: Path = CONFIG_FILE) -> list[str]:
    """Validate experiment_config.h for obvious errors."""
    defines = parse_config(path)
    errors = []
    try:
        dim = int(defines.get("DIM", "0"))
        hidden = int(defines.get("HIDDEN", "0"))
        heads = int(defines.get("HEADS", "0"))
        if dim <= 0:
            errors.append("DIM must be positive")
        if hidden <= 0:
            errors.append("HIDDEN must be positive")
        if heads <= 0:
            errors.append("HEADS must be positive")
        if dim > 0 and heads > 0 and dim % heads != 0:
            errors.append(f"DIM ({dim}) must be divisible by HEADS ({heads})")
        if hidden > 0 and dim > 0 and hidden <= dim:
            errors.append(f"HIDDEN ({hidden}) should be > DIM ({dim})")
    except ValueError as e:
        errors.append(f"Invalid numeric value: {e}")
    return errors


def compile_ane() -> tuple[bool, str]:
    """Compile the ANE training binary. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["make", "-C", str(ANE_DIR), "clean", "train_ane"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out (120s)"
    except Exception as e:
        return False, str(e)


def run_experiment(wall_time: int = 300) -> dict:
    """Run a single training experiment. Returns parsed metrics dict."""
    if not BINARY.exists():
        return {"status": "error", "error": "Binary not found. Run compile first."}

    if not DATA_FILE.exists():
        return {"status": "error", "error": f"Data not found at {DATA_FILE}. Run: bash ane/download_data.sh"}

    # Check if architecture changed → delete checkpoint
    arch_hash_file = ANE_DIR / ".arch_hash"
    current_hash = hash_arch_config()
    if CKPT_FILE.exists():
        old_hash = ""
        if arch_hash_file.exists():
            old_hash = arch_hash_file.read_text().strip()
        if old_hash != current_hash:
            print(f"Architecture changed (hash {old_hash[:8]}→{current_hash[:8]}), deleting checkpoint")
            CKPT_FILE.unlink()

    arch_hash_file.write_text(current_hash)

    cmd = [
        str(BINARY),
        "--wall-time", str(wall_time),
        "--data", str(DATA_FILE),
        "--fresh",  # always random init (no pretrained model file needed)
    ]
    if CKPT_FILE.exists():
        cmd = [
            str(BINARY),
            "--resume",
            "--reset-timing",
            "--ckpt", str(CKPT_FILE),
            "--wall-time", str(wall_time),
            "--data", str(DATA_FILE),
        ]

    print(f"Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=wall_time + 120,  # backstop
            cwd=str(ANE_DIR),
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr

        # Parse JSON after ###JSON### marker
        marker = "###JSON###"
        if marker in stdout:
            json_str = stdout[stdout.index(marker) + len(marker):].strip()
            # Take first line (the JSON object)
            json_line = json_str.split("\n")[0].strip()
            try:
                metrics = json.loads(json_line)
                metrics["raw_stdout"] = stdout
                metrics["elapsed_s"] = elapsed
                return metrics
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "error": f"JSON parse error: {e}",
                    "raw_stdout": stdout,
                    "raw_stderr": stderr,
                }
        else:
            return {
                "status": "crash",
                "error": "No ###JSON### marker in output",
                "raw_stdout": stdout[-2000:],
                "raw_stderr": stderr[-2000:],
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        return {"status": "crash", "error": f"Timed out after {wall_time + 120}s"}
    except Exception as e:
        return {"status": "crash", "error": str(e)}


def log_result(commit: str, metrics: dict, status: str, description: str):
    """Append result to results.tsv."""
    header = "commit\tval_loss\tane_util_pct\tstatus\tdescription\n"
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(header)

    val_loss = metrics.get("val_loss", 0.0)
    ane_util = metrics.get("ane_util_pct", 0.0)

    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{ane_util:.1f}\t{status}\t{description}\n")


def main():
    """Run a full experiment cycle: validate config → compile → train → parse → log."""
    # Validate config
    errors = validate_config()
    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Compile
    print("=== Compiling ANE binary ===")
    ok, output = compile_ane()
    if not ok:
        print(f"Compile failed:\n{output}")
        sys.exit(1)
    print("Compile OK")

    # Run experiment
    wall_time = int(os.environ.get("ANE_WALL_TIME", "300"))
    print(f"\n=== Running experiment (wall_time={wall_time}s) ===")
    metrics = run_experiment(wall_time)

    # Print summary in same format as karpathy's train.py
    print("\n---")
    if metrics.get("status") == "ok":
        print(f"val_loss:         {metrics['val_loss']:.6f}")
        print(f"train_loss:       {metrics['train_loss']:.6f}")
        print(f"steps:            {metrics['steps']}")
        print(f"ms_per_step:      {metrics['ms_per_step']:.1f}")
        print(f"wall_time_s:      {metrics['wall_time_s']:.1f}")
        print(f"compile_time_s:   {metrics['compile_time_s']:.1f}")
        print(f"ane_util_pct:     {metrics['ane_util_pct']:.1f}")
    else:
        print(f"status:           {metrics.get('status', 'unknown')}")
        print(f"error:            {metrics.get('error', 'unknown')}")

    return metrics


if __name__ == "__main__":
    metrics = main()
    sys.exit(0 if metrics.get("status") == "ok" else 1)
