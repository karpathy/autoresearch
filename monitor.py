"""
Experiment monitoring and observability for autoresearch.

Inspired by mdemg's internal/metrics/ package, this module provides:
- Per-experiment and per-step metrics collection
- Prometheus-compatible text exposition format
- JSON export for external dashboards (e.g. Grafana)
- Real-time alerting hooks with configurable thresholds
- Aggregate statistics across experiment sessions

The monitor is designed to be non-intrusive: it collects data passively
and never interferes with the training loop. All state persists to disk
so metrics survive process restarts and can be analyzed across sessions.

Architecture note: Uses only stdlib (json, time, statistics) to avoid
adding dependencies beyond what autoresearch already requires.
"""

import json
import os
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Metrics captured at each training step."""
    step: int
    timestamp: float
    train_loss: float
    learning_rate: float
    vram_mb: float = 0.0
    tokens_per_sec: float = 0.0
    grad_norm: float = 0.0
    progress: float = 0.0  # fraction of TIME_BUDGET elapsed


@dataclass
class ExperimentRecord:
    """Complete record of a single experiment run."""
    experiment_id: int
    commit: str
    description: str
    start_time: float
    end_time: float = 0.0
    val_bpb: float = 0.0
    peak_vram_mb: float = 0.0
    status: str = "running"  # running | keep | discard | crash
    num_steps: int = 0
    total_tokens_m: float = 0.0
    mfu_percent: float = 0.0
    training_seconds: float = 0.0
    # Per-step loss curve (sampled every N steps to bound memory)
    loss_curve: list = field(default_factory=list)
    # Smoothed loss at end of training (pre-eval)
    final_train_loss: float = 0.0
    # Change category tags for this experiment (used by memory.py)
    change_tags: list = field(default_factory=list)


@dataclass
class SessionStats:
    """Aggregate statistics for the current monitoring session."""
    session_start: float = 0.0
    total_experiments: int = 0
    kept: int = 0
    discarded: int = 0
    crashed: int = 0
    best_val_bpb: float = float("inf")
    best_commit: str = ""
    total_training_seconds: float = 0.0
    # Rolling improvement velocity (BPB improvement per hour)
    improvement_velocity: float = 0.0


# ---------------------------------------------------------------------------
# Alert thresholds (configurable via environment variables)
# ---------------------------------------------------------------------------

@dataclass
class AlertThresholds:
    """Configurable alerting thresholds.

    Set via environment variables prefixed with AUTORESEARCH_ALERT_.
    For example: AUTORESEARCH_ALERT_LOSS_SPIKE=5.0
    """
    # If training loss exceeds this multiple of the smoothed loss, flag it
    loss_spike_ratio: float = 3.0
    # If VRAM exceeds this (MB), warn about potential OOM
    vram_warning_mb: float = 75_000.0
    # If N consecutive experiments crash, trigger critical alert
    consecutive_crash_limit: int = 3
    # If no improvement in N experiments, flag plateau
    plateau_window: int = 15
    # Minimum tokens/sec to consider training healthy
    min_tokens_per_sec: float = 1000.0

    @classmethod
    def from_env(cls) -> "AlertThresholds":
        """Load thresholds from environment variables, falling back to defaults."""
        t = cls()
        prefix = "AUTORESEARCH_ALERT_"
        for fld in ["loss_spike_ratio", "vram_warning_mb",
                     "consecutive_crash_limit", "plateau_window",
                     "min_tokens_per_sec"]:
            env_key = prefix + fld.upper()
            val = os.environ.get(env_key)
            if val is not None:
                cast_fn = type(getattr(t, fld))
                setattr(t, fld, cast_fn(val))
        return t


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A monitoring alert emitted when a threshold is breached."""
    severity: str  # info | warning | critical
    category: str  # loss_spike | vram_pressure | crash_streak | plateau | throughput
    message: str
    timestamp: float = field(default_factory=time.time)
    experiment_id: int = -1
    value: float = 0.0
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# ExperimentTracker - the main monitoring class
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Tracks experiment metrics across an autonomous research session.

    Collects per-step training metrics, per-experiment summaries, and
    session-level aggregate statistics. Exports to JSON and Prometheus
    text format for integration with external dashboards.

    Inspired by mdemg's Prometheus metrics pipeline which provides
    10-panel overview dashboards with request rate, latency percentiles,
    error rate, and cache hit tracking. Here we adapt that pattern for
    experiment-level observability.

    Usage:
        tracker = ExperimentTracker()
        tracker.start_experiment("a1b2c3d", "increase LR to 0.06")
        for step in training_loop:
            tracker.record_step(step, loss, lr, vram_mb, tokens_per_sec)
        tracker.end_experiment(val_bpb=0.995, status="keep", peak_vram_mb=44000)
        tracker.export_json("metrics.json")
    """

    def __init__(self, metrics_dir: str = ".autoresearch/metrics",
                 thresholds: Optional[AlertThresholds] = None,
                 loss_sample_interval: int = 10):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or AlertThresholds.from_env()
        self.loss_sample_interval = loss_sample_interval

        self.session = SessionStats(session_start=time.time())
        self.experiments: list[ExperimentRecord] = []
        self.current: Optional[ExperimentRecord] = None
        self.alerts: list[Alert] = []

        # Track consecutive crashes for circuit-breaker alerting
        self._consecutive_crashes = 0
        # Smoothed training loss for spike detection
        self._smooth_loss = 0.0
        self._smooth_alpha = 0.05

        # Load prior session data if available
        self._load_session()

    # -- Experiment lifecycle -----------------------------------------------

    def start_experiment(self, commit: str, description: str,
                         change_tags: Optional[list[str]] = None) -> int:
        """Begin tracking a new experiment. Returns the experiment ID."""
        exp_id = self.session.total_experiments
        self.current = ExperimentRecord(
            experiment_id=exp_id,
            commit=commit,
            description=description,
            start_time=time.time(),
            change_tags=change_tags or [],
        )
        self._smooth_loss = 0.0
        return exp_id

    def record_step(self, step: int, train_loss: float,
                    learning_rate: float = 0.0, vram_mb: float = 0.0,
                    tokens_per_sec: float = 0.0, grad_norm: float = 0.0,
                    progress: float = 0.0):
        """Record metrics for a single training step.

        Called from within the training loop. Only stores every Nth step
        to the loss curve to bound memory usage.
        """
        if self.current is None:
            return

        # Update smoothed loss for spike detection
        if self._smooth_loss == 0.0:
            self._smooth_loss = train_loss
        else:
            self._smooth_loss = (self._smooth_alpha * train_loss +
                                 (1 - self._smooth_alpha) * self._smooth_loss)

        # Check for loss spike
        if (self._smooth_loss > 0 and
                train_loss > self.thresholds.loss_spike_ratio * self._smooth_loss):
            self._emit_alert("warning", "loss_spike",
                             f"Loss spike: {train_loss:.4f} vs smoothed {self._smooth_loss:.4f}",
                             value=train_loss, threshold=self._smooth_loss)

        # Check VRAM pressure
        if vram_mb > self.thresholds.vram_warning_mb:
            self._emit_alert("warning", "vram_pressure",
                             f"VRAM {vram_mb:.0f}MB exceeds threshold {self.thresholds.vram_warning_mb:.0f}MB",
                             value=vram_mb, threshold=self.thresholds.vram_warning_mb)

        # Check throughput
        if tokens_per_sec > 0 and tokens_per_sec < self.thresholds.min_tokens_per_sec:
            self._emit_alert("info", "throughput",
                             f"Low throughput: {tokens_per_sec:.0f} tok/s",
                             value=tokens_per_sec,
                             threshold=self.thresholds.min_tokens_per_sec)

        # Sample loss curve at intervals
        if step % self.loss_sample_interval == 0:
            self.current.loss_curve.append(
                StepMetrics(
                    step=step, timestamp=time.time(),
                    train_loss=train_loss, learning_rate=learning_rate,
                    vram_mb=vram_mb, tokens_per_sec=tokens_per_sec,
                    grad_norm=grad_norm, progress=progress,
                )
            )

        self.current.final_train_loss = self._smooth_loss
        self.current.num_steps = step + 1

    def end_experiment(self, val_bpb: float = 0.0, status: str = "discard",
                       peak_vram_mb: float = 0.0, training_seconds: float = 0.0,
                       total_tokens_m: float = 0.0, mfu_percent: float = 0.0):
        """Finalize the current experiment and update session stats."""
        if self.current is None:
            return

        self.current.end_time = time.time()
        self.current.val_bpb = val_bpb
        self.current.status = status
        self.current.peak_vram_mb = peak_vram_mb
        self.current.training_seconds = training_seconds
        self.current.total_tokens_m = total_tokens_m
        self.current.mfu_percent = mfu_percent

        # Convert StepMetrics in loss_curve to dicts for serialization
        self.current.loss_curve = [asdict(s) if isinstance(s, StepMetrics) else s
                                   for s in self.current.loss_curve]

        self.experiments.append(self.current)

        # Update session stats
        self.session.total_experiments += 1
        self.session.total_training_seconds += training_seconds
        if status == "keep":
            self.session.kept += 1
            self._consecutive_crashes = 0
            if val_bpb < self.session.best_val_bpb:
                self.session.best_val_bpb = val_bpb
                self.session.best_commit = self.current.commit
        elif status == "discard":
            self.session.discarded += 1
            self._consecutive_crashes = 0
        elif status == "crash":
            self.session.crashed += 1
            self._consecutive_crashes += 1
            if self._consecutive_crashes >= self.thresholds.consecutive_crash_limit:
                self._emit_alert(
                    "critical", "crash_streak",
                    f"{self._consecutive_crashes} consecutive crashes detected",
                    value=self._consecutive_crashes,
                    threshold=self.thresholds.consecutive_crash_limit)

        # Check for plateau
        self._check_plateau()

        # Update improvement velocity
        self._update_velocity()

        self.current = None
        self._save_session()

    # -- Analysis helpers ---------------------------------------------------

    def get_summary(self) -> dict:
        """Return a summary dict of the current session."""
        s = self.session
        return {
            "session_duration_hours": (time.time() - s.session_start) / 3600,
            "total_experiments": s.total_experiments,
            "kept": s.kept,
            "discarded": s.discarded,
            "crashed": s.crashed,
            "keep_rate": s.kept / max(s.kept + s.discarded, 1),
            "best_val_bpb": s.best_val_bpb if s.best_val_bpb < float("inf") else None,
            "best_commit": s.best_commit,
            "improvement_velocity_per_hour": s.improvement_velocity,
            "total_training_hours": s.total_training_seconds / 3600,
            "active_alerts": len([a for a in self.alerts
                                  if a.severity in ("warning", "critical")]),
        }

    def get_recent_alerts(self, n: int = 10) -> list[dict]:
        """Return the N most recent alerts."""
        return [asdict(a) for a in self.alerts[-n:]]

    def get_loss_curves(self, last_n: int = 5) -> dict:
        """Return loss curves for the last N experiments.

        Useful for comparing training dynamics across experiments.
        """
        curves = {}
        for exp in self.experiments[-last_n:]:
            curves[exp.commit] = {
                "description": exp.description,
                "status": exp.status,
                "val_bpb": exp.val_bpb,
                "loss_curve": exp.loss_curve,
            }
        return curves

    # -- Export formats ------------------------------------------------------

    def export_json(self, path: str = ".autoresearch/metrics/session.json"):
        """Export full session data to JSON for dashboard consumption."""
        data = {
            "session": asdict(self.session),
            "experiments": [asdict(e) for e in self.experiments],
            "alerts": [asdict(a) for a in self.alerts],
            "exported_at": time.time(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_prometheus_text(self) -> str:
        """Generate Prometheus text exposition format.

        Compatible with Prometheus /metrics endpoint or node_exporter
        textfile collector. Inspired by mdemg's 10-panel Grafana dashboard
        which tracks request rate, latency percentiles, error rate, and
        cache hit ratios. Here we adapt those patterns for experiment metrics.

        Metrics exposed:
        - autoresearch_experiments_total (counter, by status)
        - autoresearch_best_val_bpb (gauge)
        - autoresearch_improvement_velocity (gauge, BPB/hour)
        - autoresearch_session_duration_seconds (gauge)
        - autoresearch_alerts_total (counter, by severity)
        - autoresearch_last_experiment_val_bpb (gauge)
        - autoresearch_last_experiment_vram_mb (gauge)
        - autoresearch_last_experiment_steps (gauge)
        """
        s = self.session
        lines = [
            "# HELP autoresearch_experiments_total Total experiments by status",
            "# TYPE autoresearch_experiments_total counter",
            f'autoresearch_experiments_total{{status="keep"}} {s.kept}',
            f'autoresearch_experiments_total{{status="discard"}} {s.discarded}',
            f'autoresearch_experiments_total{{status="crash"}} {s.crashed}',
            "",
            "# HELP autoresearch_best_val_bpb Best validation BPB achieved",
            "# TYPE autoresearch_best_val_bpb gauge",
            f"autoresearch_best_val_bpb {s.best_val_bpb if s.best_val_bpb < float('inf') else 0}",
            "",
            "# HELP autoresearch_improvement_velocity BPB improvement per hour",
            "# TYPE autoresearch_improvement_velocity gauge",
            f"autoresearch_improvement_velocity {s.improvement_velocity:.8f}",
            "",
            "# HELP autoresearch_session_duration_seconds Session uptime",
            "# TYPE autoresearch_session_duration_seconds gauge",
            f"autoresearch_session_duration_seconds {time.time() - s.session_start:.1f}",
            "",
            "# HELP autoresearch_alerts_total Alerts by severity",
            "# TYPE autoresearch_alerts_total counter",
        ]
        for sev in ("info", "warning", "critical"):
            count = sum(1 for a in self.alerts if a.severity == sev)
            lines.append(f'autoresearch_alerts_total{{severity="{sev}"}} {count}')

        # Last experiment metrics
        if self.experiments:
            last = self.experiments[-1]
            lines += [
                "",
                "# HELP autoresearch_last_experiment_val_bpb Last experiment val BPB",
                "# TYPE autoresearch_last_experiment_val_bpb gauge",
                f"autoresearch_last_experiment_val_bpb {last.val_bpb}",
                "",
                "# HELP autoresearch_last_experiment_vram_mb Last experiment peak VRAM",
                "# TYPE autoresearch_last_experiment_vram_mb gauge",
                f"autoresearch_last_experiment_vram_mb {last.peak_vram_mb}",
                "",
                "# HELP autoresearch_last_experiment_steps Last experiment step count",
                "# TYPE autoresearch_last_experiment_steps gauge",
                f"autoresearch_last_experiment_steps {last.num_steps}",
            ]

        lines.append("")
        return "\n".join(lines)

    # -- Internal helpers ---------------------------------------------------

    def _emit_alert(self, severity: str, category: str, message: str,
                    value: float = 0.0, threshold: float = 0.0):
        """Create and store an alert."""
        alert = Alert(
            severity=severity, category=category, message=message,
            experiment_id=self.current.experiment_id if self.current else -1,
            value=value, threshold=threshold,
        )
        self.alerts.append(alert)

    def _check_plateau(self):
        """Detect if we're on an improvement plateau.

        A plateau is defined as N consecutive non-keep experiments
        (excluding crashes). Inspired by mdemg's anomaly detection
        which flags empty-recall and stale-node patterns.
        """
        window = self.thresholds.plateau_window
        if len(self.experiments) < window:
            return
        recent = self.experiments[-window:]
        if all(e.status != "keep" for e in recent):
            self._emit_alert(
                "warning", "plateau",
                f"No improvements in last {window} experiments — consider radical changes",
                value=window, threshold=window)

    def _update_velocity(self):
        """Calculate improvement velocity (BPB improvement per hour).

        Uses the kept experiments to compute the rate of improvement,
        weighted toward recent results. Inspired by mdemg's learning
        rate scheduling which adapts based on maturity.
        """
        kept_exps = [e for e in self.experiments if e.status == "keep"]
        if len(kept_exps) < 2:
            self.session.improvement_velocity = 0.0
            return

        first_kept = kept_exps[0]
        last_kept = kept_exps[-1]
        bpb_delta = first_kept.val_bpb - last_kept.val_bpb  # positive = improvement
        time_delta = last_kept.end_time - first_kept.end_time
        if time_delta > 0:
            self.session.improvement_velocity = bpb_delta / (time_delta / 3600)

    def _save_session(self):
        """Persist session state to disk for crash recovery."""
        path = self.metrics_dir / "session_state.json"
        data = {
            "session": asdict(self.session),
            "experiment_count": len(self.experiments),
            "consecutive_crashes": self._consecutive_crashes,
            "saved_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_session(self):
        """Restore session state from disk if available."""
        path = self.metrics_dir / "session_state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            # Restore crash streak counter
            self._consecutive_crashes = data.get("consecutive_crashes", 0)
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted state file — start fresh


# ---------------------------------------------------------------------------
# Convenience: format a quick text dashboard for terminal output
# ---------------------------------------------------------------------------

def format_dashboard(tracker: ExperimentTracker) -> str:
    """Format a compact text dashboard for terminal display.

    Provides a quick-glance summary similar to mdemg's per-space
    statistics view showing nodes, edges, layers, and health scores.
    Here we show experiment counts, keep rate, best BPB, velocity,
    and active alerts.
    """
    s = tracker.get_summary()
    lines = [
        "=" * 60,
        "  AUTORESEARCH MONITOR",
        "=" * 60,
        f"  Experiments: {s['total_experiments']:>5}  "
        f"(keep={s['kept']} discard={s['discarded']} crash={s['crashed']})",
        f"  Keep rate:   {s['keep_rate']:>5.1%}",
        f"  Best BPB:    {s['best_val_bpb'] or 'N/A':>10}  "
        f"(commit: {s['best_commit'] or 'N/A'})",
        f"  Velocity:    {s['improvement_velocity_per_hour']:>10.6f} BPB/hour",
        f"  Duration:    {s['session_duration_hours']:>5.1f} hours  "
        f"({s['total_training_hours']:.1f}h training)",
    ]

    alerts = tracker.get_recent_alerts(3)
    if alerts:
        lines.append(f"  Alerts:      {s['active_alerts']} active")
        for a in alerts:
            icon = {"info": "[i]", "warning": "[!]", "critical": "[X]"}.get(
                a["severity"], "[?]")
            lines.append(f"    {icon} {a['message']}")

    lines.append("=" * 60)
    return "\n".join(lines)
