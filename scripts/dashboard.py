"""
Training dashboard for autoresearch.
Launches train.py and displays live metrics with hardware protection.

Usage: uv run scripts/dashboard.py
"""

import os
import sys
import re
import time
import subprocess
import threading
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.console import Console

# ---------------------------------------------------------------------------
# Hardware monitoring
# ---------------------------------------------------------------------------

_nvml_available = False
_nvml_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
except Exception:
    pass


def get_gpu_stats():
    """Returns dict with GPU temp, vram_used_mb, vram_total_mb, gpu_util."""
    if not _nvml_available:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": 0, "gpu_util": 0}
    try:
        temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        return {
            "temp": temp,
            "vram_used_mb": mem.used / 1024 / 1024,
            "vram_total_mb": mem.total / 1024 / 1024,
            "gpu_util": util.gpu,
        }
    except Exception:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": 0, "gpu_util": 0}


# ---------------------------------------------------------------------------
# Safety thresholds (match train.py)
# ---------------------------------------------------------------------------

GPU_TEMP_WARN = 75
GPU_TEMP_PAUSE = 80
GPU_TEMP_ABORT = 90

# Auto-detect GPU VRAM (leave 500MB headroom)
if _nvml_available:
    _gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    _gpu_vram_total = _gpu_mem.total // (1024 * 1024)
    VRAM_LIMIT_MB = _gpu_vram_total - 500
    GPU_NAME = pynvml.nvmlDeviceGetName(_nvml_handle)
    if isinstance(GPU_NAME, bytes):
        GPU_NAME = GPU_NAME.decode()
else:
    VRAM_LIMIT_MB = 7500
    GPU_NAME = "Unknown GPU"
    _gpu_vram_total = 8192

# ---------------------------------------------------------------------------
# Training output parser
# ---------------------------------------------------------------------------

# Pattern: step 00005 (0.0%) | loss: 11.217838 | lrm: 1.00 | dt: 1844ms | tok/sec: 284,322 | mfu: 0.3% | epoch: 0 | remaining: 300s
STEP_RE = re.compile(
    r"step\s+(\d+)\s+\((\d+\.\d+)%\)\s*\|\s*loss:\s*([\d.]+)\s*\|\s*lrm:\s*([\d.]+)\s*\|\s*"
    r"dt:\s*(\d+)ms\s*\|\s*tok/sec:\s*([\d,]+)\s*\|\s*mfu:\s*([\d.]+)%\s*\|\s*"
    r"epoch:\s*(\d+)\s*\|\s*remaining:\s*(\d+)s"
)


def parse_step_line(line):
    """Parse a training step output line into a dict."""
    m = STEP_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "progress": float(m.group(2)),
        "loss": float(m.group(3)),
        "lrm": float(m.group(4)),
        "dt_ms": int(m.group(5)),
        "tok_sec": int(m.group(6).replace(",", "")),
        "mfu": float(m.group(7)),
        "epoch": int(m.group(8)),
        "remaining": int(m.group(9)),
    }


# ---------------------------------------------------------------------------
# ASCII sparkline for loss history
# ---------------------------------------------------------------------------

SPARK_CHARS = " ._-~=*#"


def sparkline(values, width=50):
    """Render a sparkline string from a list of floats."""
    if not values:
        return ""
    recent = list(values)[-width:]
    lo, hi = min(recent), max(recent)
    rng = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in recent)


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------

def progress_bar(pct, width=30, fill="#", empty="-"):
    filled = int(width * pct / 100)
    return f"{fill * filled}{empty * (width - filled)}"


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def build_dashboard(metrics, gpu, loss_history, log_lines, status):
    """Build a rich Layout for the dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=5),
    )
    layout["body"].split_row(
        Layout(name="metrics", ratio=2),
        Layout(name="gpu", ratio=1),
    )

    # Header
    header_text = Text("  AUTORESEARCH TRAINING DASHBOARD", style="bold cyan")
    header_text.append(f"  [{status}]", style="bold yellow" if status == "RUNNING" else "bold red" if "ABORT" in status else "bold green")
    layout["header"].update(Panel(header_text, style="cyan"))

    # Metrics panel
    if metrics:
        step = metrics["step"]
        pct = metrics["progress"]
        loss = metrics["loss"]
        lrm = metrics["lrm"]
        dt = metrics["dt_ms"]
        tok_s = metrics["tok_sec"]
        mfu = metrics["mfu"]
        remaining = metrics["remaining"]
        epoch = metrics["epoch"]

        mins, secs = divmod(remaining, 60)

        lines = []
        lines.append(f"  Step:       {step:,}")
        lines.append(f"  Progress:   {progress_bar(pct)} {pct:.1f}%")
        lines.append(f"  Loss:       {loss:.6f}")
        lines.append(f"  LR mult:    {lrm:.3f}")
        lines.append(f"  Step time:  {dt}ms")
        lines.append(f"  Tok/sec:    {tok_s:,}")
        lines.append(f"  MFU:        {mfu:.1f}%")
        lines.append(f"  Epoch:      {epoch}")
        lines.append(f"  Remaining:  {mins:.0f}m {secs:.0f}s")
        lines.append(f"")
        lines.append(f"  Loss trend: {sparkline(loss_history)}")

        metrics_text = Text("\n".join(lines))
    else:
        metrics_text = Text("  Waiting for training to start...", style="dim")

    layout["metrics"].update(Panel(metrics_text, title="Training Metrics", border_style="green"))

    # GPU panel
    temp = gpu.get("temp")
    vram_used = gpu.get("vram_used_mb", 0)
    vram_total = gpu.get("vram_total_mb", _gpu_vram_total)
    util = gpu.get("gpu_util", 0)
    vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0

    # Color coding for temperature
    if temp is None:
        temp_str = "N/A (no monitoring!)"
        temp_style = "bold red"
    elif temp >= GPU_TEMP_ABORT:
        temp_str = f"{temp}C  CRITICAL!"
        temp_style = "bold red"
    elif temp >= GPU_TEMP_PAUSE:
        temp_str = f"{temp}C  HIGH"
        temp_style = "bold yellow"
    elif temp >= GPU_TEMP_WARN:
        temp_str = f"{temp}C  Warm"
        temp_style = "yellow"
    else:
        temp_str = f"{temp}C  OK"
        temp_style = "green"

    # Color coding for VRAM
    if vram_used >= VRAM_LIMIT_MB:
        vram_style = "bold red"
    elif vram_used >= VRAM_LIMIT_MB * 0.9:
        vram_style = "yellow"
    else:
        vram_style = "green"

    gpu_lines = Text()
    gpu_lines.append(f"\n  {GPU_NAME}\n\n", style="bold")
    gpu_lines.append(f"  Temp:  ")
    gpu_lines.append(f"{temp_str}\n", style=temp_style)
    gpu_lines.append(f"  VRAM:  ")
    gpu_lines.append(f"{vram_used:.0f}/{vram_total:.0f}MB\n", style=vram_style)
    gpu_lines.append(f"  ")
    gpu_lines.append(f"{progress_bar(vram_pct, width=20)}")
    gpu_lines.append(f" {vram_pct:.0f}%\n", style=vram_style)
    gpu_lines.append(f"  Load:  {util}%\n")
    gpu_lines.append(f"\n  Limits:\n")
    gpu_lines.append(f"  Pause:  {GPU_TEMP_PAUSE}C\n", style="dim")
    gpu_lines.append(f"  Abort:  {GPU_TEMP_ABORT}C\n", style="dim")
    gpu_lines.append(f"  VRAM:   {VRAM_LIMIT_MB}MB\n", style="dim")

    layout["gpu"].update(Panel(gpu_lines, title="GPU Health", border_style="blue"))

    # Footer - recent log lines
    footer_text = Text("\n".join(list(log_lines)[-3:]) if log_lines else "  No output yet...", style="dim")
    layout["footer"].update(Panel(footer_text, title="Log", border_style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Main: launch training and display dashboard
# ---------------------------------------------------------------------------

def main():
    console = Console()
    console.clear()

    metrics = {}
    loss_history = deque(maxlen=200)
    log_lines = deque(maxlen=50)
    status = "STARTING"
    gpu_stats = get_gpu_stats()

    # Pre-flight GPU check
    if gpu_stats["temp"] is not None and gpu_stats["temp"] >= GPU_TEMP_PAUSE:
        console.print(f"[bold red]GPU is already at {gpu_stats['temp']}C! Let it cool below {GPU_TEMP_WARN}C before training.[/]")
        return

    if gpu_stats["vram_used_mb"] > VRAM_LIMIT_MB * 0.8:
        console.print(f"[bold red]VRAM already at {gpu_stats['vram_used_mb']:.0f}MB. Close other GPU apps first.[/]")
        return

    # Launch train.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, os.path.join(project_root, "train.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_root,
        env=env,
        bufsize=0,
    )

    status = "RUNNING"

    # Background thread to read process output
    output_buffer = []
    lock = threading.Lock()
    done_event = threading.Event()

    def read_output():
        buf = b""
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk == b"\r" or chunk == b"\n":
                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    if line:
                        with lock:
                            output_buffer.append(line)
                    buf = b""
            else:
                buf += chunk
        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                with lock:
                    output_buffer.append(line)
        done_event.set()

    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()

    # Dashboard loop
    try:
        with Live(build_dashboard(metrics, gpu_stats, loss_history, log_lines, status),
                  console=console, refresh_per_second=2, screen=True) as live:
            while proc.poll() is None or not done_event.is_set():
                # Process buffered output
                with lock:
                    new_lines = list(output_buffer)
                    output_buffer.clear()

                for line in new_lines:
                    log_lines.append(line)
                    parsed = parse_step_line(line)
                    if parsed:
                        metrics = parsed
                        loss_history.append(parsed["loss"])

                # Update GPU stats every cycle
                gpu_stats = get_gpu_stats()

                # Dashboard-level safety: kill process if GPU is critical
                if gpu_stats["temp"] is not None and gpu_stats["temp"] >= GPU_TEMP_ABORT:
                    status = "ABORTED (GPU TEMP)"
                    proc.terminate()
                    break

                if gpu_stats["vram_used_mb"] > VRAM_LIMIT_MB:
                    status = "ABORTED (VRAM)"
                    proc.terminate()
                    break

                live.update(build_dashboard(metrics, gpu_stats, loss_history, log_lines, status))
                time.sleep(0.5)

            # Final status
            rc = proc.wait()
            if "ABORT" not in status:
                status = "COMPLETED" if rc == 0 else f"FAILED (exit {rc})"

            # Final render
            with lock:
                for line in output_buffer:
                    log_lines.append(line)
                    parsed = parse_step_line(line)
                    if parsed:
                        metrics = parsed
                        loss_history.append(parsed["loss"])
                output_buffer.clear()

            live.update(build_dashboard(metrics, gpu_stats, loss_history, log_lines, status))
            time.sleep(1)

    except KeyboardInterrupt:
        status = "STOPPED (user)"
        proc.terminate()
        proc.wait()

    # Print final summary outside of Live
    console.print()
    if metrics:
        console.print(f"[bold]Final step:[/] {metrics.get('step', '?')} | [bold]Loss:[/] {metrics.get('loss', '?')}")
    console.print(f"[bold]Status:[/] {status}")
    console.print(f"[bold]GPU temp:[/] {gpu_stats.get('temp', 'N/A')}C | [bold]VRAM:[/] {gpu_stats.get('vram_used_mb', 0):.0f}MB")


if __name__ == "__main__":
    main()
