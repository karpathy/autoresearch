"""
Circuit breakers and anomaly detection for autoresearch.

Inspired by mdemg's resilience infrastructure:
- internal/circuitbreaker/: Per-endpoint circuit breakers with half-open
  recovery and exponential backoff
- internal/anomaly/: Multi-dimensional anomaly detection (empty-resume,
  empty-recall, stale-node patterns)
- internal/backpressure/: Memory pressure handling with graceful degradation
- internal/ratelimit/: Provider-specific rate limiting

This module adapts those patterns for autonomous experiment management:

1. CircuitBreaker: Prevents wasting GPU time on repeated failures.
   After N consecutive crashes, enters OPEN state (experiments blocked).
   Periodically allows a single HALF_OPEN probe to test recovery.

2. AnomalyDetector: Detects problematic patterns across experiments:
   - Loss plateaus (no improvement in N experiments)
   - VRAM creep (monotonically increasing memory usage)
   - Systematic regression (each experiment worse than the last)
   - Crash clustering (crashes concentrated in recent experiments)

3. BackpressureMonitor: Tracks VRAM usage trends and warns when
   the system is approaching GPU memory limits.

4. ExperimentGuard: Wraps experiment execution with all safety checks,
   providing a single pre/post-experiment interface.

Usage:
    guard = ExperimentGuard()

    # Before running an experiment
    verdict = guard.pre_experiment(description="double model width")
    if verdict.blocked:
        print(f"Blocked: {verdict.reason}")
        # Skip this experiment

    # After running an experiment
    guard.post_experiment(
        val_bpb=0.995, status="keep", peak_vram_mb=44000
    )
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Circuit Breaker (inspired by mdemg's internal/circuitbreaker/)
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    """Circuit breaker states.

    CLOSED: Normal operation, experiments proceed freely.
    OPEN: Too many failures, experiments are blocked.
    HALF_OPEN: Probing — allow one experiment to test recovery.

    State transitions:
      CLOSED -> OPEN: after N consecutive failures
      OPEN -> HALF_OPEN: after cooldown period
      HALF_OPEN -> CLOSED: if probe experiment succeeds
      HALF_OPEN -> OPEN: if probe experiment fails
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration.

    Mirrors mdemg's per-endpoint circuit breaker configuration which
    allows different thresholds for different failure modes.
    """
    # Number of consecutive failures to trip the breaker
    failure_threshold: int = 5
    # Seconds to wait before allowing a probe (half-open)
    cooldown_seconds: float = 120.0
    # Max consecutive probes that can fail before extending cooldown
    max_probe_failures: int = 2
    # Cooldown multiplier after probe failure (exponential backoff)
    backoff_multiplier: float = 2.0
    # Maximum cooldown duration (cap for exponential backoff)
    max_cooldown_seconds: float = 600.0


class CircuitBreaker:
    """Prevents wasting GPU time on repeated experiment failures.

    When experiments crash repeatedly, the circuit breaker OPENS to
    prevent further wasted runs. After a cooldown period, it enters
    HALF_OPEN state and allows a single probe experiment. If the probe
    succeeds, the breaker closes. If it fails, the cooldown is extended
    with exponential backoff.

    Adapted from mdemg's circuit breaker pattern which protects embedding
    provider endpoints from cascading failures with half-open recovery.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.current_cooldown = self.config.cooldown_seconds
        self.probe_failures = 0
        self.total_trips = 0

    def record_success(self):
        """Record a successful experiment (keep or discard with valid BPB)."""
        self.consecutive_failures = 0
        self.probe_failures = 0
        self.current_cooldown = self.config.cooldown_seconds
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED

    def record_failure(self):
        """Record a failed experiment (crash, OOM, timeout)."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — re-open with extended cooldown
            self.probe_failures += 1
            self.current_cooldown = min(
                self.current_cooldown * self.config.backoff_multiplier,
                self.config.max_cooldown_seconds
            )
            self.state = CircuitState.OPEN
            return

        if self.consecutive_failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.total_trips += 1

    def allow_experiment(self) -> tuple[bool, str]:
        """Check if an experiment is allowed to run.

        Returns (allowed, reason) tuple. If the circuit is OPEN and the
        cooldown has elapsed, transitions to HALF_OPEN and allows one probe.
        """
        if self.state == CircuitState.CLOSED:
            return True, "circuit closed — normal operation"

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.current_cooldown:
                self.state = CircuitState.HALF_OPEN
                return True, (f"circuit half-open — probe experiment "
                              f"(cooldown was {self.current_cooldown:.0f}s)")
            remaining = self.current_cooldown - elapsed
            return False, (f"circuit OPEN — {self.consecutive_failures} consecutive "
                           f"failures, {remaining:.0f}s until probe")

        if self.state == CircuitState.HALF_OPEN:
            return True, "circuit half-open — probe in progress"

        return True, "unknown state — allowing"

    def get_status(self) -> dict:
        """Return circuit breaker status."""
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_trips": self.total_trips,
            "current_cooldown": self.current_cooldown,
            "probe_failures": self.probe_failures,
        }


# ---------------------------------------------------------------------------
# Anomaly Detector (inspired by mdemg's internal/anomaly/)
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A detected anomaly in the experiment stream."""
    anomaly_type: str  # plateau | vram_creep | systematic_regression | crash_cluster
    severity: str  # info | warning | critical
    message: str
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


class AnomalyDetector:
    """Detects problematic patterns across experiments.

    Inspired by mdemg's anomaly detection system which watches for:
    - Empty resume events (no prior memory found)
    - Empty recall results (retrieval returning nothing)
    - Stale nodes (nodes not accessed in a long time)

    Here we adapt those patterns for experiment research:
    - Plateau: no improvements in a window of experiments
    - VRAM creep: memory usage increasing monotonically
    - Systematic regression: each experiment worse than the last
    - Crash clustering: high crash rate in recent experiments
    """

    def __init__(self, plateau_window: int = 15,
                 vram_creep_window: int = 5,
                 regression_window: int = 5,
                 crash_rate_threshold: float = 0.5):
        self.plateau_window = plateau_window
        self.vram_creep_window = vram_creep_window
        self.regression_window = regression_window
        self.crash_rate_threshold = crash_rate_threshold
        self.anomalies: list[Anomaly] = []

    def analyze(self, experiments: list[dict]) -> list[Anomaly]:
        """Run all anomaly detectors on the experiment history.

        Args:
            experiments: List of experiment dicts with keys:
                val_bpb, status, peak_vram_mb, description

        Returns:
            List of newly detected anomalies.
        """
        new_anomalies = []

        if len(experiments) >= self.plateau_window:
            anomaly = self._check_plateau(experiments)
            if anomaly:
                new_anomalies.append(anomaly)

        if len(experiments) >= self.vram_creep_window:
            anomaly = self._check_vram_creep(experiments)
            if anomaly:
                new_anomalies.append(anomaly)

        if len(experiments) >= self.regression_window:
            anomaly = self._check_regression(experiments)
            if anomaly:
                new_anomalies.append(anomaly)

        if len(experiments) >= 5:
            anomaly = self._check_crash_cluster(experiments)
            if anomaly:
                new_anomalies.append(anomaly)

        self.anomalies.extend(new_anomalies)
        return new_anomalies

    def _check_plateau(self, experiments: list[dict]) -> Optional[Anomaly]:
        """Detect improvement plateaus."""
        recent = experiments[-self.plateau_window:]
        kept = [e for e in recent if e.get("status") == "keep"]
        if len(kept) == 0:
            return Anomaly(
                anomaly_type="plateau",
                severity="warning",
                message=f"No improvements in last {self.plateau_window} experiments",
                data={"window": self.plateau_window, "experiments_checked": len(recent)},
            )
        return None

    def _check_vram_creep(self, experiments: list[dict]) -> Optional[Anomaly]:
        """Detect monotonically increasing VRAM usage.

        Inspired by mdemg's backpressure monitoring which tracks memory
        pressure and triggers graceful degradation.
        """
        recent = [e for e in experiments[-self.vram_creep_window:]
                  if e.get("peak_vram_mb", 0) > 0]
        if len(recent) < self.vram_creep_window:
            return None

        vram_values = [e["peak_vram_mb"] for e in recent]
        # Check if strictly increasing
        if all(vram_values[i] < vram_values[i + 1]
               for i in range(len(vram_values) - 1)):
            increase_pct = ((vram_values[-1] - vram_values[0]) /
                            max(vram_values[0], 1)) * 100
            return Anomaly(
                anomaly_type="vram_creep",
                severity="warning" if increase_pct < 20 else "critical",
                message=(f"VRAM increasing monotonically: "
                         f"{vram_values[0]:.0f} -> {vram_values[-1]:.0f} MB "
                         f"(+{increase_pct:.1f}%)"),
                data={"vram_values": vram_values, "increase_pct": increase_pct},
            )
        return None

    def _check_regression(self, experiments: list[dict]) -> Optional[Anomaly]:
        """Detect systematic regression (each result worse than the last)."""
        recent = [e for e in experiments[-self.regression_window:]
                  if e.get("status") != "crash" and e.get("val_bpb", 0) > 0]
        if len(recent) < self.regression_window:
            return None

        bpb_values = [e["val_bpb"] for e in recent]
        # Check if strictly increasing (worse BPB)
        if all(bpb_values[i] < bpb_values[i + 1]
               for i in range(len(bpb_values) - 1)):
            return Anomaly(
                anomaly_type="systematic_regression",
                severity="warning",
                message=(f"BPB worsening over last {self.regression_window} "
                         f"experiments: {bpb_values[0]:.6f} -> {bpb_values[-1]:.6f}"),
                data={"bpb_values": bpb_values},
            )
        return None

    def _check_crash_cluster(self, experiments: list[dict]) -> Optional[Anomaly]:
        """Detect high crash rate in recent experiments."""
        recent = experiments[-10:]
        crash_count = sum(1 for e in recent if e.get("status") == "crash")
        crash_rate = crash_count / len(recent)
        if crash_rate >= self.crash_rate_threshold:
            return Anomaly(
                anomaly_type="crash_cluster",
                severity="critical",
                message=f"High crash rate: {crash_count}/{len(recent)} ({crash_rate:.0%}) in recent experiments",
                data={"crash_count": crash_count, "window": len(recent),
                      "crash_rate": crash_rate},
            )
        return None

    def get_all_anomalies(self) -> list[dict]:
        """Return all detected anomalies."""
        return [asdict(a) for a in self.anomalies]


# ---------------------------------------------------------------------------
# Backpressure Monitor (inspired by mdemg's internal/backpressure/)
# ---------------------------------------------------------------------------

class BackpressureMonitor:
    """Monitors VRAM pressure and suggests adjustments.

    Tracks VRAM usage across experiments and provides warnings when
    approaching GPU memory limits. Can suggest batch size or model
    size reductions to stay within safe limits.

    Inspired by mdemg's backpressure system which monitors Go heap
    memory and triggers graceful degradation (reducing cache sizes,
    deferring background tasks) when pressure exceeds thresholds.
    """

    def __init__(self, gpu_vram_mb: float = 81_920.0,  # H100 80GB
                 warning_threshold: float = 0.85,
                 critical_threshold: float = 0.95):
        self.gpu_vram_mb = gpu_vram_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.vram_history: list[float] = []

    def record_vram(self, peak_vram_mb: float):
        """Record peak VRAM from an experiment."""
        self.vram_history.append(peak_vram_mb)

    def get_pressure(self) -> dict:
        """Calculate current VRAM pressure level.

        Returns pressure analysis with:
        - level: ok | warning | critical
        - utilization: fraction of GPU VRAM used
        - trend: increasing | stable | decreasing
        - suggestion: actionable recommendation
        """
        if not self.vram_history:
            return {"level": "ok", "utilization": 0.0,
                    "trend": "unknown", "suggestion": None}

        latest = self.vram_history[-1]
        utilization = latest / self.gpu_vram_mb

        # Determine trend from last 5 measurements
        trend = "stable"
        if len(self.vram_history) >= 3:
            recent = self.vram_history[-3:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                trend = "increasing"
            elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                trend = "decreasing"

        # Determine level and suggestion
        level = "ok"
        suggestion = None
        if utilization >= self.critical_threshold:
            level = "critical"
            suggestion = ("VRAM critically high — reduce DEVICE_BATCH_SIZE "
                          "or model depth immediately")
        elif utilization >= self.warning_threshold:
            level = "warning"
            if trend == "increasing":
                suggestion = ("VRAM approaching limit with increasing trend — "
                              "avoid changes that increase model size")
            else:
                suggestion = "VRAM high but stable — monitor closely"

        return {
            "level": level,
            "utilization": round(utilization, 3),
            "peak_vram_mb": latest,
            "gpu_vram_mb": self.gpu_vram_mb,
            "trend": trend,
            "suggestion": suggestion,
        }


# ---------------------------------------------------------------------------
# ExperimentGuard - unified pre/post experiment safety
# ---------------------------------------------------------------------------

@dataclass
class PreExperimentVerdict:
    """Result of pre-experiment safety checks."""
    allowed: bool
    blocked: bool  # True if experiment should be skipped
    reason: str
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class ExperimentGuard:
    """Unified pre/post experiment safety wrapper.

    Combines circuit breaker, anomaly detection, and backpressure
    monitoring into a single interface. Call pre_experiment() before
    running and post_experiment() after.

    This is the primary integration point for resilience features.
    Inspired by mdemg's guardrail system which validates MCP constraints
    and provides server-side enforcement.

    Usage:
        guard = ExperimentGuard()

        # Before each experiment:
        verdict = guard.pre_experiment("try larger model")
        if verdict.blocked:
            # Skip this experiment, log the reason
            ...

        # After each experiment:
        guard.post_experiment(val_bpb=0.995, status="keep", peak_vram_mb=44000)
    """

    def __init__(self, state_dir: str = ".autoresearch/resilience",
                 gpu_vram_mb: float = 81_920.0):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.circuit_breaker = CircuitBreaker()
        self.anomaly_detector = AnomalyDetector()
        self.backpressure = BackpressureMonitor(gpu_vram_mb=gpu_vram_mb)

        self._experiment_history: list[dict] = []
        self._load_state()

    def pre_experiment(self, description: str = "") -> PreExperimentVerdict:
        """Run all pre-experiment safety checks.

        Returns a verdict indicating whether the experiment should proceed,
        along with any warnings or suggestions.
        """
        warnings = []
        suggestions = []

        # Check circuit breaker
        cb_allowed, cb_reason = self.circuit_breaker.allow_experiment()
        if not cb_allowed:
            return PreExperimentVerdict(
                allowed=False, blocked=True,
                reason=f"Circuit breaker: {cb_reason}",
                warnings=[cb_reason],
                suggestions=["Wait for cooldown or fix the root cause of crashes"],
            )

        # Check VRAM pressure
        pressure = self.backpressure.get_pressure()
        if pressure["level"] == "critical":
            warnings.append(f"VRAM critical: {pressure['utilization']:.0%} utilized")
            suggestions.append(pressure["suggestion"] or "Reduce model size")
        elif pressure["level"] == "warning":
            warnings.append(f"VRAM warning: {pressure['utilization']:.0%} utilized")
            if pressure["suggestion"]:
                suggestions.append(pressure["suggestion"])

        # Run anomaly detection
        new_anomalies = self.anomaly_detector.analyze(self._experiment_history)
        for anomaly in new_anomalies:
            if anomaly.severity == "critical":
                warnings.append(f"[CRITICAL] {anomaly.message}")
            elif anomaly.severity == "warning":
                warnings.append(anomaly.message)

        return PreExperimentVerdict(
            allowed=True, blocked=False,
            reason=cb_reason,
            warnings=warnings,
            suggestions=suggestions,
        )

    def post_experiment(self, val_bpb: float = 0.0, status: str = "discard",
                        peak_vram_mb: float = 0.0, description: str = ""):
        """Record experiment outcome and update all safety systems."""
        # Record in history
        self._experiment_history.append({
            "val_bpb": val_bpb,
            "status": status,
            "peak_vram_mb": peak_vram_mb,
            "description": description,
            "timestamp": time.time(),
        })

        # Update circuit breaker
        if status == "crash":
            self.circuit_breaker.record_failure()
        else:
            self.circuit_breaker.record_success()

        # Update backpressure
        if peak_vram_mb > 0:
            self.backpressure.record_vram(peak_vram_mb)

        self._save_state()

    def get_status(self) -> dict:
        """Return comprehensive resilience status."""
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "backpressure": self.backpressure.get_pressure(),
            "anomalies": len(self.anomaly_detector.anomalies),
            "recent_anomalies": [
                asdict(a) for a in self.anomaly_detector.anomalies[-3:]
            ],
            "total_experiments_tracked": len(self._experiment_history),
        }

    def _save_state(self):
        """Persist resilience state to disk."""
        path = self.state_dir / "state.json"
        data = {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "experiment_history": self._experiment_history[-100:],  # Keep last 100
            "anomalies": [asdict(a) for a in self.anomaly_detector.anomalies],
            "vram_history": self.backpressure.vram_history[-50:],
            "saved_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_state(self):
        """Restore resilience state from disk."""
        path = self.state_dir / "state.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._experiment_history = data.get("experiment_history", [])
            self.backpressure.vram_history = data.get("vram_history", [])

            # Restore circuit breaker state
            cb = data.get("circuit_breaker", {})
            self.circuit_breaker.consecutive_failures = cb.get("consecutive_failures", 0)
            self.circuit_breaker.total_trips = cb.get("total_trips", 0)
            state_str = cb.get("state", "closed")
            self.circuit_breaker.state = CircuitState(state_str)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            pass  # Corrupted state — start fresh
