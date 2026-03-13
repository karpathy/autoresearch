"""
Parametrized wrapper for train.py - modifies in-place and runs from root.
"""

import os
import time
import subprocess
import shutil

def run_training_with_params(
    depth=4,
    device_batch_size=4,
    total_batch_size=2**16,
    embedding_lr=0.6,
    unembedding_lr=0.004,
    matrix_lr=0.04,
    scalar_lr=0.5,
    weight_decay=0.2,
    warmdown_ratio=0.5,
    window_pattern="L",
    output_dir="optuna_runs",
    trial_id=None,
):
    """Run training with specified hyperparameters by modifying train.py in-place."""

    # Backup train.py
    shutil.copy("train.py", "train.py.backup")

    try:
        # Read train.py
        with open("train.py", "r") as f:
            code = f.read()

        # Replace hyperparameters
        replacements = {
            'TOTAL_BATCH_SIZE = 2**16': f'TOTAL_BATCH_SIZE = {total_batch_size}',
            'EMBEDDING_LR = 0.6': f'EMBEDDING_LR = {embedding_lr}',
            'UNEMBEDDING_LR = 0.004': f'UNEMBEDDING_LR = {unembedding_lr}',
            'MATRIX_LR = 0.04': f'MATRIX_LR = {matrix_lr}',
            'SCALAR_LR = 0.5': f'SCALAR_LR = {scalar_lr}',
            'WEIGHT_DECAY = 0.2': f'WEIGHT_DECAY = {weight_decay}',
            'WARMDOWN_RATIO = 0.5': f'WARMDOWN_RATIO = {warmdown_ratio}',
            'DEPTH = 4': f'DEPTH = {depth}',
            'DEVICE_BATCH_SIZE = 4': f'DEVICE_BATCH_SIZE = {device_batch_size}',
            'WINDOW_PATTERN = "L"': f'WINDOW_PATTERN = "{window_pattern}"',
        }

        for old, new in replacements.items():
            code = code.replace(old, new, 1)

        # Write modified train.py
        with open("train.py", "w") as f:
            f.write(code)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        trial_name = f"trial_{trial_id}" if trial_id is not None else f"trial_{int(time.time())}"
        log_file = os.path.join(output_dir, f"{trial_name}.log")

        # Run training
        start_time = time.time()
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        end_time = time.time()

        # Save log
        with open(log_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Parse results
        output = result.stdout + result.stderr
        val_bpb = None
        peak_vram_mb = 0.0
        training_seconds = 0.0
        num_params_m = 0.0

        for line in output.split('\n'):
            if line.startswith('val_bpb:'):
                val_bpb = float(line.split(':')[1].strip())
            elif line.startswith('peak_vram_mb:'):
                peak_vram_mb = float(line.split(':')[1].strip())
            elif line.startswith('training_seconds:'):
                training_seconds = float(line.split(':')[1].strip())
            elif line.startswith('num_params_M:'):
                num_params_m = float(line.split(':')[1].strip())

        if val_bpb is None or result.returncode != 0:
            return {
                'val_bpb': float('inf'),
                'peak_vram_mb': peak_vram_mb,
                'wall_time_s': end_time - start_time,
                'training_time_s': training_seconds,
                'num_params_m': num_params_m,
                'status': 'failed',
                'error': result.stderr[-500:] if result.stderr else 'Unknown error',
            }

        return {
            'val_bpb': val_bpb,
            'peak_vram_mb': peak_vram_mb,
            'wall_time_s': end_time - start_time,
            'training_time_s': training_seconds,
            'num_params_m': num_params_m,
            'status': 'success',
        }

    except subprocess.TimeoutExpired:
        return {
            'val_bpb': float('inf'),
            'peak_vram_mb': 0.0,
            'wall_time_s': 600,
            'status': 'timeout',
        }
    except Exception as e:
        return {
            'val_bpb': float('inf'),
            'peak_vram_mb': 0.0,
            'wall_time_s': time.time() - start_time if 'start_time' in locals() else 0,
            'status': 'error',
            'error': str(e),
        }
    finally:
        # Restore train.py
        shutil.move("train.py.backup", "train.py")
