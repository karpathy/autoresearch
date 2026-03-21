"""Generate harness YAML variations using Gemini, optionally keyed by hardware tier."""

import json
import logging
import re as _re
from pathlib import Path

log = logging.getLogger(__name__)

# ─── Known hardware tiers ────────────────────────────────────────────────────

HARDWARE_TIERS = [
    "pi5-hailo8l",
    "pi5-8gb",
    "pi4-8gb",
    "server",
    "waveshare-alpha",
    "unitree-go2",
]

RESEARCH_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "anthropic/claude-haiku-3-5",
    "anthropic/claude-sonnet-4-5",
    "ollama/llama3.2",  # local option
]

# Per-tier hints that steer the Gemini prompt toward hardware-aware candidates
TIER_HINTS: dict[str, str] = {
    "pi5-hailo8l": (
        "This robot has a Raspberry Pi 5 with a Hailo-8L NPU (26 TOPS). "
        "The NPU offloads on-device inference, so thinking_budget can be lower than usual "
        "because the model reasons faster. Explore aggressive cost_gate_usd reduction and "
        "lower thinking_budget (512-1024). nexa_enabled is available. Prioritise latency."
    ),
    "pi5-8gb": (
        "Raspberry Pi 5 with 8 GB RAM, no NPU. CPU-only inference. "
        "context_budget is more flexible than pi4, but thinking_budget should stay moderate (1024-2048). "
        "Explore mid-range cost_gate_usd (0.02-0.05) and drift_detection trade-offs."
    ),
    "pi4-8gb": (
        "Raspberry Pi 4 with 8 GB RAM. Constrained CPU and memory bandwidth. "
        "Prioritise minimal cost_gate_usd (0.01-0.03), smaller context_budget (4096-8192), "
        "and lower max_iterations (3-5). Keeping cloud costs low is critical."
    ),
    "server": (
        "Server-class hardware (Jetson, x86 workstation, or cloud VM) with ample RAM and GPU/VRAM. "
        "Explore higher context_budget (16384-32768), higher thinking_budget (2048-4096), "
        "and more max_iterations (8-12). Cost sensitivity is lower; capability is the priority."
    ),
    "waveshare-alpha": (
        "Waveshare ALPHA humanoid robot. Physical manipulation means HITL gates and "
        "P66 consent are critical. Explore p66_consent_threshold variations carefully — "
        "prefer 'physical' or 'verbal'. Retry policy matters: a failed arm motion needs "
        "careful retry logic. Latency is important for smooth joint control."
    ),
    "unitree-go2": (
        "Unitree Go2 quadruped robot. Locomotion stack handles gait; the OpenCastor harness "
        "handles task-level reasoning. Drift detection is important (gait state drift). "
        "Explore drift_detection=True with varying thresholds and lower context_budget "
        "since the task space is more structured than general manipulation."
    ),
}

# ─── Default seed harness ─────────────────────────────────────────────────────

DEFAULT_SEED = {
    "enabled": True,
    "max_iterations": 6,
    "thinking_budget": 1024,
    "context_budget": 8192,
    "p66_consent_threshold": "physical",
    "retry_on_error": True,
    "drift_detection": True,
    "cost_gate_usd": 0.01,  # current champion value
}

HARNESS_SOURCE = Path.home() / "OpenCastor" / "castor" / "harness" / "default_harness.yaml"

# ─── Prompts ─────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are a robotics harness configuration researcher. Given the current seed harness \
configuration below, propose {n} variations that might improve agent performance.

Current seed config:
{seed_json}

{tier_section}
Each variation should tweak 1-3 parameters. Consider:
- max_iterations: 3-12 (higher = more retries, more cost)
- thinking_budget: 512-4096 (token budget for reasoning)
- context_budget: 4096-32768 (token budget for context)
- p66_consent_threshold: "physical", "verbal", "none"
- retry_on_error: true/false
- drift_detection: true/false
- cost_gate_usd: 0.01-0.20

Return a JSON array of {n} objects, each with:
- "id": short snake_case identifier (e.g. "high_think_low_cost")
- "config": the full harness config dict with your tweaks applied
- "description": one-line explanation of the hypothesis

Return ONLY the JSON array, no markdown fences or extra text.
"""

TIER_SECTION_TEMPLATE = """\
Hardware profile: {tier}
Hardware context: {hint}

"""


# ─── Seed loading ─────────────────────────────────────────────────────────────

def _load_seed(hardware_tier: str | None = None) -> dict:
    """Load seed harness from OpenCastor if available, else use built-in default.

    If hardware_tier is given and a profile champion exists for it, seeds from there.
    """
    import os
    ops_repo = Path(os.environ.get("OPENCASTOR_OPS_DIR", Path.home() / "opencastor-ops"))
    profile_champion = ops_repo / "harness-research" / "profiles" / f"{hardware_tier}.yaml"

    if hardware_tier and profile_champion.exists():
        import yaml
        data = yaml.safe_load(profile_champion.read_text())
        if data and data.get("config"):
            log.info("Seeding from profile champion: %s (score=%.4f)", hardware_tier, data.get("score", 0))
            return data["config"]

    if HARNESS_SOURCE.exists():
        import yaml
        raw = yaml.safe_load(HARNESS_SOURCE.read_text())
        if isinstance(raw, dict):
            seed = {}
            for key in DEFAULT_SEED:
                if key in raw:
                    seed[key] = raw[key]
            if seed:
                return seed

    return DEFAULT_SEED.copy()


# ─── Gemini client ────────────────────────────────────────────────────────────

def _get_genai_client():
    """Create Gemini client using Google ADC."""
    import google.auth
    import google.auth.transport.requests
    from google import genai

    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    if not project:
        adc = Path.home() / ".config/gcloud/application_default_credentials.json"
        if adc.exists():
            project = json.loads(adc.read_text()).get("quota_project_id")

    return genai.Client(
        vertexai=True,
        project=project,
        location="us-central1",
        credentials=creds,
    )


# ─── Synthetic generation (dry-run / CI) ─────────────────────────────────────

_SYNTHETIC_VARIATIONS: dict[str | None, list[tuple]] = {
    None: [
        ("high_think", {"thinking_budget": 2048}, "Double thinking budget"),
        ("low_cost", {"cost_gate_usd": 0.02}, "Halve cost gate"),
        ("max_iter_10", {"max_iterations": 10}, "Increase max iterations to 10"),
        ("no_drift", {"drift_detection": False}, "Disable drift detection"),
        ("verbal_consent", {"p66_consent_threshold": "verbal"}, "Lower consent to verbal"),
        ("big_context", {"context_budget": 16384}, "Double context budget"),
        ("min_think", {"thinking_budget": 512}, "Minimize thinking budget"),
        ("no_retry", {"retry_on_error": False}, "Disable retry on error"),
    ],
    "pi5-hailo8l": [
        ("hailo_low_think", {"thinking_budget": 512, "cost_gate_usd": 0.01}, "NPU offloads — cut thinking budget"),
        ("hailo_min_cost", {"cost_gate_usd": 0.005, "thinking_budget": 768}, "Ultra-low cost gate for NPU robot"),
        ("hailo_fast_iter", {"max_iterations": 4, "thinking_budget": 512}, "Fast iterations for NPU inference"),
        ("hailo_no_drift", {"drift_detection": False, "cost_gate_usd": 0.01}, "Disable drift to reduce latency"),
        ("hailo_verbal", {"p66_consent_threshold": "verbal", "thinking_budget": 512}, "Verbal consent + NPU speed"),
        ("hailo_small_ctx", {"context_budget": 4096, "thinking_budget": 512}, "Minimal context for speed"),
        ("hailo_mid_think", {"thinking_budget": 1024, "cost_gate_usd": 0.008}, "Balanced NPU config"),
        ("hailo_no_retry", {"retry_on_error": False, "thinking_budget": 512}, "Fail-fast on NPU"),
    ],
    "pi5-8gb": [
        ("pi5_moderate", {"thinking_budget": 1536, "cost_gate_usd": 0.03}, "Moderate CPU-only config"),
        ("pi5_big_ctx", {"context_budget": 12288, "thinking_budget": 1024}, "Larger context on 8GB"),
        ("pi5_low_cost", {"cost_gate_usd": 0.015, "max_iterations": 5}, "Cost-conscious Pi 5"),
        ("pi5_high_think", {"thinking_budget": 2048, "cost_gate_usd": 0.04}, "Higher reasoning, controlled cost"),
        ("pi5_no_drift", {"drift_detection": False}, "Skip drift detection"),
        ("pi5_max_iter", {"max_iterations": 8}, "More retries for complex tasks"),
        ("pi5_verbal", {"p66_consent_threshold": "verbal"}, "Verbal consent threshold"),
        ("pi5_min_config", {"thinking_budget": 768, "context_budget": 6144, "cost_gate_usd": 0.01}, "Minimal Pi 5"),
    ],
    "pi4-8gb": [
        ("pi4_minimal", {"thinking_budget": 512, "context_budget": 4096, "cost_gate_usd": 0.01}, "Bare minimum"),
        ("pi4_tiny_cost", {"cost_gate_usd": 0.005, "max_iterations": 3}, "Ultra-low cost for Pi 4"),
        ("pi4_small_ctx", {"context_budget": 4096, "thinking_budget": 768}, "Small context to save memory"),
        ("pi4_no_retry", {"retry_on_error": False, "cost_gate_usd": 0.01}, "Fail-fast to save cost"),
        ("pi4_no_drift", {"drift_detection": False, "thinking_budget": 512}, "No drift, low think"),
        ("pi4_fast", {"max_iterations": 3, "thinking_budget": 512, "cost_gate_usd": 0.008}, "Fastest Pi 4 config"),
        ("pi4_verbal", {"p66_consent_threshold": "verbal", "cost_gate_usd": 0.01}, "Verbal + minimal cost"),
        ("pi4_mid", {"thinking_budget": 768, "context_budget": 6144, "cost_gate_usd": 0.015}, "Mid-range Pi 4"),
    ],
    "server": [
        ("server_full", {"thinking_budget": 4096, "context_budget": 32768, "max_iterations": 12}, "Full server power"),
        ("server_high_ctx", {"context_budget": 24576, "thinking_budget": 3072}, "Maximise context on GPU RAM"),
        ("server_high_iter", {"max_iterations": 10, "thinking_budget": 2048}, "Many retries"),
        ("server_big_think", {"thinking_budget": 4096, "cost_gate_usd": 0.10}, "Max reasoning"),
        ("server_parallel", {"max_iterations": 8, "context_budget": 16384}, "Balance for parallel workloads"),
        ("server_relaxed_cost", {"cost_gate_usd": 0.20, "thinking_budget": 3072}, "Relax cost gate on server"),
        ("server_no_drift", {"drift_detection": False, "thinking_budget": 3072}, "Skip drift on stable server"),
        ("server_verbal", {"p66_consent_threshold": "verbal", "thinking_budget": 2048}, "Verbal consent server"),
    ],
    "waveshare-alpha": [
        ("alpha_physical", {"p66_consent_threshold": "physical", "retry_on_error": True}, "Max safety for manipulation"),
        ("alpha_careful", {"p66_consent_threshold": "physical", "max_iterations": 4}, "Fewer retries, high consent"),
        ("alpha_verbal_retry", {"p66_consent_threshold": "verbal", "retry_on_error": True}, "Verbal + retry"),
        ("alpha_no_retry", {"p66_consent_threshold": "physical", "retry_on_error": False}, "Physical consent, no retry"),
        ("alpha_low_think", {"thinking_budget": 768, "p66_consent_threshold": "physical"}, "Fast + safe"),
        ("alpha_drift_on", {"drift_detection": True, "p66_consent_threshold": "physical"}, "Drift detect for safety"),
        ("alpha_low_cost", {"cost_gate_usd": 0.02, "p66_consent_threshold": "physical"}, "Cost-conscious + safe"),
        ("alpha_big_ctx", {"context_budget": 12288, "p66_consent_threshold": "physical"}, "More context for manipulation"),
    ],
    "unitree-go2": [
        ("go2_drift_on", {"drift_detection": True, "thinking_budget": 1024}, "Drift detection for locomotion"),
        ("go2_low_ctx", {"context_budget": 4096, "drift_detection": True}, "Small context, drift on"),
        ("go2_fast", {"thinking_budget": 512, "max_iterations": 4, "drift_detection": True}, "Fast gait-level decisions"),
        ("go2_no_drift", {"drift_detection": False, "thinking_budget": 1024}, "Trust locomotion stack"),
        ("go2_verbal", {"p66_consent_threshold": "verbal", "drift_detection": True}, "Verbal + drift detect"),
        ("go2_low_cost", {"cost_gate_usd": 0.015, "drift_detection": True}, "Cost-aware quadruped"),
        ("go2_high_iter", {"max_iterations": 8, "drift_detection": True}, "Retry on locomotion error"),
        ("go2_min", {"thinking_budget": 512, "context_budget": 4096, "cost_gate_usd": 0.01}, "Minimal Go2"),
    ],
}


def _generate_synthetic(n: int, seed: dict, hardware_tier: str | None = None) -> list[dict]:
    """Generate synthetic candidates without Gemini (for CI dry-run)."""
    import random
    variations = _SYNTHETIC_VARIATIONS.get(hardware_tier) or _SYNTHETIC_VARIATIONS[None]
    selected = random.sample(variations, min(n, len(variations)))
    candidates = []
    for var_id, tweaks, desc in selected:
        config = seed.copy()
        config.update(tweaks)
        candidates.append({"id": var_id, "config": config, "description": desc})
    return candidates


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_candidates(
    n: int = 5,
    dry_run: bool = False,
    hardware_tier: str | None = None,
    model_research: bool = False,
) -> list[dict]:
    """Generate N candidate harness variations.

    Args:
        n: Number of candidates to generate.
        dry_run: Use synthetic variations instead of Gemini.
        hardware_tier: Optional hardware profile key (e.g. "pi5-hailo8l").
            When set, the seed and Gemini prompt are tailored to that tier.
        model_research: If True, produce variants of each candidate for each
            model in RESEARCH_MODELS, adding a ``model_id`` field to each.

    Returns:
        List of dicts with keys: id, config, description[, model_id].
    """
    seed = _load_seed(hardware_tier=hardware_tier)
    log.info("Seed harness (tier=%s): %s", hardware_tier or "generic", seed)

    if dry_run:
        log.info("Dry-run mode: generating synthetic candidates for tier=%s", hardware_tier)
        candidates = _generate_synthetic(n, seed, hardware_tier=hardware_tier)
        log.info("Generated %d synthetic candidates", len(candidates))
        if model_research:
            candidates = _expand_model_variants(candidates)
            log.info("Expanded to %d model-variant candidates", len(candidates))
        return candidates

    client = _get_genai_client()

    tier_section = ""
    if hardware_tier:
        hint = TIER_HINTS.get(hardware_tier, f"Hardware tier: {hardware_tier}")
        tier_section = TIER_SECTION_TEMPLATE.format(tier=hardware_tier, hint=hint)

    prompt = GENERATION_PROMPT.format(
        n=n,
        seed_json=json.dumps(seed, indent=2),
        tier_section=tier_section,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Gemini occasionally emits trailing commas
    text = _re.sub(r",(\s*[}\]])", r"\1", text)

    candidates = json.loads(text)
    if not isinstance(candidates, list):
        raise ValueError(f"Expected JSON array from Gemini, got {type(candidates)}")

    log.info("Generated %d candidate configs (tier=%s)", len(candidates), hardware_tier)
    if model_research:
        candidates = _expand_model_variants(candidates)
        log.info("Expanded to %d model-variant candidates", len(candidates))
    return candidates


def _expand_model_variants(candidates: list[dict]) -> list[dict]:
    """For each candidate, produce one variant per model in RESEARCH_MODELS."""
    variants = []
    for candidate in candidates:
        for model in RESEARCH_MODELS:
            sanitized = model.replace("/", "_")
            variant = {**candidate, "model_id": model}
            variant["id"] = f"{candidate['id']}__{sanitized}"
            variants.append(variant)
    return variants
