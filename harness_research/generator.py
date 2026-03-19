"""Generate harness YAML variations using Gemini."""

import json
import logging
from pathlib import Path

import google.auth
import google.auth.transport.requests
from google import genai

log = logging.getLogger(__name__)

DEFAULT_SEED = {
    "enabled": True,
    "max_iterations": 6,
    "thinking_budget": 1024,
    "context_budget": 8192,
    "p66_consent_threshold": "physical",
    "retry_on_error": True,
    "drift_detection": True,
    "cost_gate_usd": 0.05,
}

HARNESS_SOURCE = Path.home() / "OpenCastor" / "castor" / "harness" / "default_harness.yaml"

GENERATION_PROMPT = """\
You are a robotics harness configuration researcher. Given the current seed harness \
configuration below, propose {n} variations that might improve agent performance.

Current seed config:
{seed_json}

Each variation should tweak 1-3 parameters. Consider:
- max_iterations: 3-12 (higher = more retries, more cost)
- thinking_budget: 512-4096 (token budget for reasoning)
- context_budget: 4096-16384 (token budget for context)
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


def _load_seed() -> dict:
    """Load seed harness from OpenCastor if available, else use built-in default."""
    if HARNESS_SOURCE.exists():
        import yaml

        raw = yaml.safe_load(HARNESS_SOURCE.read_text())
        # Extract relevant keys if nested
        if isinstance(raw, dict):
            seed = {}
            for key in DEFAULT_SEED:
                if key in raw:
                    seed[key] = raw[key]
            return seed if seed else DEFAULT_SEED
    return DEFAULT_SEED.copy()


def _get_genai_client() -> genai.Client:
    """Create Gemini client using Google ADC (same pattern as run_agent.py)."""
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


def generate_candidates(n: int = 5) -> list[dict]:
    """Generate N candidate harness variations using Gemini.

    Returns list of dicts with keys: id, config, description.
    """
    seed = _load_seed()
    log.info("Seed harness: %s", seed)

    client = _get_genai_client()
    prompt = GENERATION_PROMPT.format(n=n, seed_json=json.dumps(seed, indent=2))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    candidates = json.loads(text)
    if not isinstance(candidates, list):
        raise ValueError(f"Expected JSON array from Gemini, got {type(candidates)}")

    log.info("Generated %d candidate harness configs", len(candidates))
    return candidates
