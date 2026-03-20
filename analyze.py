"""
Post-run analysis and bookkeeping for autoresearch experiments.
Runs after each training run to archive data, compute baseline stats,
and produce analysis.md for the agent to build on.

Usage: uv run analyze.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
LOG_DIR = ROOT / ".auto-log-research"
RESULTS_TSV = LOG_DIR / "results.tsv"
RUN_SUMMARY = ROOT / "run_summary.json"
RUN_LOG = ROOT / "run.log"
METRICS_JSONL = ROOT / "metrics.jsonl"

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(cmd):
    try:
        return subprocess.check_output(
            ["git"] + cmd.split(), stderr=subprocess.DEVNULL, cwd=str(ROOT)
        ).decode().strip()
    except Exception:
        return None

def get_commit_hash(short=False):
    flag = "--short" if short else ""
    return git(f"rev-parse {flag} HEAD".strip())

def get_commit_message():
    return git("log -1 --pretty=%s")

# ---------------------------------------------------------------------------
# Load run data
# ---------------------------------------------------------------------------

def load_run_summary():
    if not RUN_SUMMARY.exists():
        print("ERROR: run_summary.json not found. Did train.py finish?", file=sys.stderr)
        sys.exit(1)
    with open(RUN_SUMMARY) as f:
        return json.load(f)

def load_metrics():
    """Load metrics.jsonl written directly by train.py."""
    if not METRICS_JSONL.exists():
        return []
    history = []
    with open(METRICS_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return history

# ---------------------------------------------------------------------------
# Results TSV management
# ---------------------------------------------------------------------------

def load_results():
    if not RESULTS_TSV.exists():
        return []
    results = []
    with open(RESULTS_TSV) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                results.append(dict(zip(header, parts)))
    return results

def append_result(commit, val_bpb, memory_gb, status, description):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_loss_trajectory_stats(history):
    losses = [h.get("train/loss") for h in history if h.get("train/loss") is not None]
    smooth_losses = [h.get("train/loss_smooth") for h in history if h.get("train/loss_smooth") is not None]

    if not losses:
        return {}

    stats = {
        "num_steps": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "min_loss_step": losses.index(min(losses)),
        "max_loss": max(losses),
    }

    if smooth_losses:
        stats["first_smooth_loss"] = smooth_losses[0]
        stats["final_smooth_loss"] = smooth_losses[-1]
        stats["min_smooth_loss"] = min(smooth_losses)

    # End-of-run slope (last 10% of steps)
    if len(smooth_losses) > 20:
        tail_start = int(len(smooth_losses) * 0.9)
        tail = smooth_losses[tail_start:]
        slope = (tail[-1] - tail[0]) / len(tail) if len(tail) > 1 else 0
        stats["end_slope"] = slope
        stats["still_improving"] = slope < -1e-6

    # MFU
    mfus = [h.get("perf/mfu_percent") for h in history if h.get("perf/mfu_percent") is not None]
    if mfus:
        steady = mfus[min(10, len(mfus)):]
        if steady:
            stats["mfu_mean"] = sum(steady) / len(steady)

    return stats

def find_best_run(results):
    best = None
    for r in results:
        try:
            bpb = float(r["val_bpb"])
            if bpb > 0 and (best is None or bpb < best[1]):
                best = (r, bpb)
        except (ValueError, KeyError):
            continue
    return best

def config_diff(current, best_summary_path):
    if not best_summary_path or not os.path.exists(best_summary_path):
        return []
    with open(best_summary_path) as f:
        best = json.load(f)
    diffs = []
    config_keys = [
        "aspect_ratio", "head_dim", "window_pattern", "total_batch_size",
        "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
        "weight_decay", "warmup_ratio", "warmdown_ratio", "final_lr_frac",
        "depth", "device_batch_size",
    ]
    for key in config_keys:
        cur_val = current.get(key)
        best_val = best.get(key)
        if cur_val != best_val:
            diffs.append((key, best_val, cur_val))
    return diffs

# ---------------------------------------------------------------------------
# Archive run
# ---------------------------------------------------------------------------

def archive_run(commit_short, run_summary, history):
    run_dir = LOG_DIR / commit_short
    run_dir.mkdir(parents=True, exist_ok=True)
    if RUN_SUMMARY.exists():
        shutil.copy2(RUN_SUMMARY, run_dir / "run_summary.json")
    if RUN_LOG.exists():
        shutil.copy2(RUN_LOG, run_dir / "run.log")
    if METRICS_JSONL.exists():
        shutil.copy2(METRICS_JSONL, run_dir / "metrics.jsonl")
    return run_dir

# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _extract(history, key):
    return [(h.get("step", i), h[key]) for i, h in enumerate(history) if key in h and h[key] is not None]

def _load_prev_history(results):
    for r in reversed(results):
        path = LOG_DIR / r["commit"] / "metrics.jsonl"
        if path.exists():
            with open(path) as f:
                return r["commit"], [json.loads(l) for l in f if l.strip()]
    return None, []

def generate_plots(history, run_dir, results):
    if not history:
        return []

    prev_commit, prev_history = _load_prev_history(results)
    generated = []

    def _save(fig, name):
        path = run_dir / name
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        generated.append(name)

    datasets = [("current", history)]
    if prev_history:
        datasets.append((f"prev ({prev_commit})", prev_history))

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in datasets:
        smooth = _extract(data, "train/loss_smooth")
        if smooth:
            ax.plot([s for s, _ in smooth], [v for _, v in smooth], label=f"{label} (smooth)", linewidth=1.5)
        raw = _extract(data, "train/loss")
        if raw:
            ax.plot([s for s, _ in raw], [v for _, v in raw], alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_curve.png")

    # 2. LR schedule, weight decay, momentum
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, data in datasets:
        lr = _extract(data, "train/lr_multiplier")
        if lr:
            axes[0].plot([s for s, _ in lr], [v for _, v in lr], label=label, linewidth=1.2)
        wd = _extract(data, "train/weight_decay")
        if wd:
            axes[1].plot([s for s, _ in wd], [v for _, v in wd], label=label, linewidth=1.2)
        mom = _extract(data, "train/muon_momentum")
        if mom:
            axes[2].plot([s for s, _ in mom], [v for _, v in mom], label=label, linewidth=1.2)
    for ax, title in zip(axes, ["LR Multiplier", "Weight Decay", "Muon Momentum"]):
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "lr_and_schedule.png")

    # 3. GPU perf (MFU + step time)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for label, data in datasets:
        mfu = _extract(data, "perf/mfu_percent")
        if mfu:
            ax1.plot([s for s, _ in mfu], [v for _, v in mfu], label=label, linewidth=1)
        stime = _extract(data, "perf/step_time_ms")
        if stime:
            ax2.plot([s for s, _ in stime], [v for _, v in stime], label=label, linewidth=1)
    ax1.set_xlabel("Step"); ax1.set_ylabel("MFU %"); ax1.set_title("Model FLOPs Utilization")
    ax2.set_xlabel("Step"); ax2.set_ylabel("ms"); ax2.set_title("Step Time")
    ax1.legend(); ax2.legend()
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "gpu_perf.png")

    return generated

# ---------------------------------------------------------------------------
# Generate analysis.md
# ---------------------------------------------------------------------------

def generate_analysis(run_summary, traj_stats, results, run_dir, plot_files, commit_short):
    val_bpb = run_summary.get("val_bpb", 0)

    lines = []
    lines.append(f"# Run Analysis: {commit_short}")
    lines.append("")
    lines.append(f"**val_bpb: {val_bpb:.6f}**")
    lines.append("")

    # Config summary
    lines.append("## Configuration")
    lines.append("```")
    for key in ["depth", "aspect_ratio", "head_dim", "window_pattern",
                 "total_batch_size", "device_batch_size", "embedding_lr",
                 "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
                 "warmup_ratio", "warmdown_ratio", "final_lr_frac"]:
        val = run_summary.get(key, "?")
        lines.append(f"  {key}: {val}")
    lines.append("```")
    lines.append("")

    # Performance
    lines.append("## Performance")
    lines.append(f"- Training time: {run_summary.get('training_seconds', 0):.1f}s")
    lines.append(f"- Startup time: {run_summary.get('startup_seconds', 0):.1f}s")
    lines.append(f"- Total steps: {run_summary.get('num_steps', 0)}")
    lines.append(f"- Parameters: {run_summary.get('num_params_M', 0):.1f}M")
    lines.append(f"- Peak VRAM: {run_summary.get('peak_vram_mb', 0):.0f} MB")
    lines.append(f"- MFU: {run_summary.get('mfu_percent', 0):.1f}%")
    lines.append(f"- Tokens trained: {run_summary.get('total_tokens_M', 0):.1f}M")
    lines.append("")

    # Loss trajectory
    if traj_stats:
        lines.append("## Loss Trajectory")
        lines.append(f"- First loss: {traj_stats.get('first_loss', '?')}")
        lines.append(f"- Final loss: {traj_stats.get('final_loss', '?')}")
        lines.append(f"- Min loss: {traj_stats.get('min_loss', '?')} (step {traj_stats.get('min_loss_step', '?')})")
        if "end_slope" in traj_stats:
            slope = traj_stats["end_slope"]
            improving = traj_stats.get("still_improving", False)
            lines.append(f"- End-of-run slope: {slope:.8f} ({'still improving' if improving else 'converged/plateaued'})")
        if "mfu_mean" in traj_stats:
            lines.append(f"- Steady-state MFU: {traj_stats['mfu_mean']:.1f}%")
        lines.append("")

    # Auto-generated plots (relative paths only)
    if plot_files:
        lines.append("## Plots")
        for pf in plot_files:
            lines.append(f"- `{pf}`")
        lines.append("")

    # Cross-run comparison
    if results:
        lines.append("## Cross-Run Summary")
        best = find_best_run(results)
        if best:
            best_row, best_bpb = best
            lines.append(f"- Best val_bpb so far: {best_bpb:.6f} (commit {best_row['commit']})")
            if val_bpb > 0:
                delta = val_bpb - best_bpb
                lines.append(f"- This run vs best: {'+' if delta > 0 else ''}{delta:.6f}")

        recent = results[-5:]
        lines.append(f"- Recent {len(recent)} runs:")
        for r in recent:
            status = r.get("status", "?")
            bpb = r.get("val_bpb", "?")
            desc = r.get("description", "?")
            lines.append(f"  - [{status}] bpb={bpb} | {desc}")
        lines.append("")

        kept = [r for r in results if r.get("status") == "keep"]
        if kept:
            lines.append("## Improvement Lineage (all kept runs)")
            for r in kept:
                lines.append(f"  - {r['commit']} bpb={r['val_bpb']} | {r.get('description', '?')}")
            lines.append("")

    # Config diff with best
    if results:
        best = find_best_run(results)
        if best:
            best_row = best[0]
            best_dir = LOG_DIR / best_row["commit"]
            diffs = config_diff(run_summary, best_dir / "run_summary.json")
            if diffs:
                lines.append("## Config Diff vs Best Run")
                for key, best_val, cur_val in diffs:
                    lines.append(f"  - {key}: {best_val} (best) -> {cur_val} (this)")
                lines.append("")

    # How to load metrics for custom analysis
    lines.append("## Metrics")
    lines.append(f"Per-step metrics: `.auto-log-research/{commit_short}/metrics.jsonl`")
    lines.append("Keys: `step`, `train/loss`, `train/loss_smooth`, `train/lr_multiplier`, `train/muon_momentum`, `train/weight_decay`, `train/progress`, `perf/step_time_ms`, `perf/tokens_per_sec`, `perf/mfu_percent`")
    lines.append("")

    # Agent section
    lines.append("## Agent Investigation Notes")
    lines.append("")
    lines.append("**YOU MUST write your analysis findings below this line before moving on.**")
    lines.append("Look at the plots above. Load metrics.jsonl and compare with previous runs.")
    lines.append("What patterns do you see? What was surprising? What should you try next?")
    lines.append("")

    analysis_text = "\n".join(lines)
    analysis_path = run_dir / "analysis.md"
    with open(analysis_path, "w") as f:
        f.write(analysis_text)

    return analysis_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("analyze.py: Post-run analysis")

    run_summary = load_run_summary()
    commit = get_commit_hash(short=True) or "unknown"
    commit_msg = get_commit_message() or "no message"
    val_bpb = run_summary.get("val_bpb", 0)
    peak_vram_mb = run_summary.get("peak_vram_mb", 0)

    print(f"  Commit: {commit} ({commit_msg})")
    print(f"  val_bpb: {val_bpb:.6f}")
    print(f"  Peak VRAM: {peak_vram_mb:.0f} MB")

    # Load metrics directly from train.py output
    print("  Loading metrics.jsonl...")
    history = load_metrics()
    traj_stats = compute_loss_trajectory_stats(history) if history else {}
    if traj_stats:
        print(f"  Trajectory: {traj_stats.get('num_steps', 0)} steps, final_loss={traj_stats.get('final_loss', '?')}")
    else:
        print("  Warning: No step-level metrics found in metrics.jsonl")

    # Load existing results (before appending current)
    results = load_results()

    # Archive (copies metrics.jsonl, run_summary.json, run.log)
    run_dir = archive_run(commit, run_summary, history)
    print(f"  Archived to: {run_dir}")

    # Generate plots
    print("  Generating plots...")
    plot_files = generate_plots(history, run_dir, results)
    if plot_files:
        print(f"  Generated: {', '.join(plot_files)}")

    # Append to results.tsv
    append_result(commit, val_bpb, peak_vram_mb / 1024, "pending", commit_msg)
    print(f"  Appended to results.tsv (status=pending)")

    # Generate analysis
    analysis_path = generate_analysis(run_summary, traj_stats, results, run_dir, plot_files, commit)
    print(f"  Analysis: {analysis_path}")

    print("  Done.")

if __name__ == "__main__":
    main()