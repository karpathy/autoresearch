"""Evaluate harness candidates against environment scenarios."""

import hashlib
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# Option B: try-import castor with fallback to simulated evaluation.
# The real run_skill_eval requires AgentHarness + SkillLoader instances
# which are tightly coupled to castor internals. We import to detect presence,
# but use simulated scoring for the research pipeline. When castor's eval API
# gains a standalone mode, flip USE_CASTOR_EVAL to True.
try:
    from castor.eval_harness import run_skill_eval  # type: ignore[import-not-found]

    CASTOR_AVAILABLE = True
except ImportError:
    CASTOR_AVAILABLE = False

# Until castor exposes a standalone eval API, always use simulation
USE_CASTOR_EVAL = False

ENVIRONMENTS_DIR = Path(__file__).parent / "environments"


@dataclass
class ScenarioResult:
    scenario_id: str
    environment: str
    success: bool
    p66_compliant: bool
    tokens_used: int
    latency_ms: float


@dataclass
class EvalResults:
    candidate_id: str
    config: dict
    description: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.scenario_results:
            return 0.0
        return sum(1 for r in self.scenario_results if r.success) / len(self.scenario_results)

    @property
    def p66_rate(self) -> float:
        if not self.scenario_results:
            return 0.0
        return sum(1 for r in self.scenario_results if r.p66_compliant) / len(self.scenario_results)

    @property
    def token_efficiency(self) -> float:
        """Lower token usage = higher efficiency, normalized 0-1."""
        if not self.scenario_results:
            return 0.0
        avg_tokens = sum(r.tokens_used for r in self.scenario_results) / len(self.scenario_results)
        # Normalize: 0 tokens = 1.0, 8000+ tokens = 0.0
        return max(0.0, 1.0 - avg_tokens / 8000.0)

    @property
    def latency_score(self) -> float:
        """Lower latency = higher score, normalized 0-1."""
        if not self.scenario_results:
            return 0.0
        avg_latency = sum(r.latency_ms for r in self.scenario_results) / len(self.scenario_results)
        # Normalize: 0ms = 1.0, 5000ms+ = 0.0
        return max(0.0, 1.0 - avg_latency / 5000.0)

    def env_aggregate(self, env: str) -> dict:
        """Get aggregated scores for a specific environment."""
        env_results = [r for r in self.scenario_results if r.environment == env]
        if not env_results:
            return {"success_rate": 0.0, "p66_rate": 0.0, "count": 0}
        return {
            "success_rate": sum(1 for r in env_results if r.success) / len(env_results),
            "p66_rate": sum(1 for r in env_results if r.p66_compliant) / len(env_results),
            "count": len(env_results),
        }


def _load_scenarios() -> list[dict]:
    """Load all scenarios from environment YAML files."""
    scenarios = []
    for yaml_file in sorted(ENVIRONMENTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text())
        env_name = yaml_file.stem
        for s in data.get("scenarios", []):
            s["environment"] = env_name
            scenarios.append(s)
    return scenarios


def _deterministic_seed(candidate_id: str, scenario_id: str) -> int:
    """Create a deterministic seed from candidate + scenario IDs."""
    h = hashlib.sha256(f"{candidate_id}:{scenario_id}".encode()).hexdigest()
    return int(h[:8], 16)


def _simulate_scenario(config: dict, scenario: dict, seed: int) -> ScenarioResult:
    """Simulate evaluation based on harness parameter heuristics.

    Higher max_iterations → better success (more retries).
    Higher thinking_budget → better success (more reasoning).
    Higher cost_gate → slightly worse efficiency (allows more spending).
    drift_detection → better p66 compliance.
    """
    rng = random.Random(seed)

    max_iter = config.get("max_iterations", 6)
    think_budget = config.get("thinking_budget", 1024)
    cost_gate = config.get("cost_gate_usd", 0.05)
    drift = config.get("drift_detection", True)
    retry = config.get("retry_on_error", True)
    p66_threshold = config.get("p66_consent_threshold", "physical")

    # Success probability: base 0.6, boosted by iterations and thinking
    base_success = 0.60
    base_success += min(max_iter / 20.0, 0.20)  # up to +0.20
    base_success += min(think_budget / 8192.0, 0.10)  # up to +0.10
    if retry:
        base_success += 0.05
    # Safety scenarios are harder
    if scenario.get("scope") == "safety":
        base_success -= 0.05

    success = rng.random() < min(base_success, 0.95)

    # P66 compliance: drift detection and threshold help
    p66_base = 0.70
    if drift:
        p66_base += 0.15
    if p66_threshold == "physical":
        p66_base += 0.10
    elif p66_threshold == "verbal":
        p66_base += 0.05
    p66_compliant = rng.random() < min(p66_base, 0.98)

    # Token usage: proportional to budgets
    tokens_used = int(
        think_budget * rng.uniform(0.3, 0.8)
        + config.get("context_budget", 8192) * rng.uniform(0.1, 0.3)
    )

    # Latency: proportional to iterations and budget
    latency_ms = (
        max_iter * rng.uniform(100, 300)
        + think_budget * rng.uniform(0.5, 1.5)
        + cost_gate * rng.uniform(1000, 3000)
    )

    return ScenarioResult(
        scenario_id=scenario["id"],
        environment=scenario["environment"],
        success=success,
        p66_compliant=p66_compliant,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
    )


def _real_eval(config: dict, scenario: dict) -> ScenarioResult:
    """Run real evaluation using castor eval harness."""
    result = run_skill_eval(
        harness_config=config,
        instruction=scenario["instruction"],
        expected_tools=scenario.get("expected_tools", []),
        expected_checks=scenario.get("expected_checks", []),
        max_tokens=scenario.get("max_tokens", 500),
        dry_run=True,
    )
    return ScenarioResult(
        scenario_id=scenario["id"],
        environment=scenario["environment"],
        success=result.get("success", False),
        p66_compliant=result.get("p66_compliant", True),
        tokens_used=result.get("tokens_used", 0),
        latency_ms=result.get("latency_ms", 0.0),
    )


def evaluate_candidate(candidate: dict) -> EvalResults:
    """Evaluate a single candidate against all 30 scenarios."""
    candidate_id = candidate["id"]
    config = candidate["config"]
    scenarios = _load_scenarios()

    eval_results = EvalResults(
        candidate_id=candidate_id,
        config=config,
        description=candidate.get("description", ""),
    )

    mode = "castor" if USE_CASTOR_EVAL else "simulated"
    log.info("Evaluating candidate '%s' against %d scenarios [%s]", candidate_id, len(scenarios), mode)

    for scenario in scenarios:
        if USE_CASTOR_EVAL:
            result = _real_eval(config, scenario)
        else:
            seed = _deterministic_seed(candidate_id, scenario["id"])
            result = _simulate_scenario(config, scenario, seed)
        eval_results.scenario_results.append(result)

    log.info(
        "Candidate '%s': success=%.2f p66=%.2f efficiency=%.2f latency=%.2f",
        candidate_id,
        eval_results.success_rate,
        eval_results.p66_rate,
        eval_results.token_efficiency,
        eval_results.latency_score,
    )
    return eval_results


def evaluate_all(candidates: list[dict]) -> list[EvalResults]:
    """Evaluate all candidates and return results."""
    return [evaluate_candidate(c) for c in candidates]
