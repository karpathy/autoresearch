"""
Collaborative autoresearch coordinator.

Bridges the autoresearch experiment loop with the Ensue memory network,
enabling SETI@home-style distributed research across multiple GPU participants.

Uses `requests` (already in pyproject.toml) for JSON-RPC calls. Zero new deps.

Usage:
    from coordinator import Coordinator
    coord = Coordinator()  # reads ENSUE_API_KEY or .autoresearch-key
    coord.join_hub()
    coord.claim_experiment("increase LR to 0.04")
    coord.publish_result(exp_key, result_dict, open("train.py").read())
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HUB_ORG = "autoresearch-at-home"
API_URL = "https://api.ensue-network.ai/"
KEY_FILE = ".autoresearch-key"

CLAIM_TTL = 900              # 15 min soft expiry (3x expected 5-min experiment)
VERIFY_DELAY = 2             # seconds between claim and verify
SEMANTIC_THRESHOLD = 0.92    # block if active claim is this similar
MAX_CLAIM_ATTEMPTS = 5       # alternatives before giving up
SYNC_EVERY_N = 5             # pull global best every N experiments

# ---------------------------------------------------------------------------
# Base JSON-RPC
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Read API key from env var or key file."""
    key = os.environ.get("ENSUE_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    return None


def ensue_rpc(api_key: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON-RPC call to the Ensue MCP API."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1,
    }
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()

    # Response may have SSE "data: " prefix
    text = resp.text.strip()
    if text.startswith("data: "):
        text = text[len("data: "):]

    data = json.loads(text)

    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")

    # Extract text content from result
    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return json.loads(first["text"])
    return result


def _experiment_hash(description: str) -> str:
    """Hash an experiment description for dedup keying."""
    return hashlib.sha256(description.lower().strip().encode()).hexdigest()[:12]


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn text into a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug[:max_len].rstrip('-')


def _experiment_key(agent_id: str, description: str) -> str:
    """
    Human-readable experiment key: <agent>--<slug>--<short_hash>

    Example: gpu0--increase-lr-to-004--a7f3b2
    """
    slug = _slugify(description)
    short_hash = _experiment_hash(description)[:6]
    agent = _slugify(agent_id, max_len=20) or "unknown"
    return f"{agent}--{slug}--{short_hash}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_remote_url() -> Optional[str]:
    """Get the GitHub HTTPS URL for the current repo."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Convert SSH to HTTPS: git@github.com:user/repo.git -> https://github.com/user/repo
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except Exception:
        return None


def _git_branch() -> Optional[str]:
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_commit_short() -> Optional[str]:
    """Get the short commit hash of HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """
    Synchronous coordinator for collaborative autoresearch.

    All methods catch exceptions and return gracefully so the training loop
    never crashes due to network issues.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_api_key()
        self.agent_id: Optional[str] = None
        self.experiment_count = 0

    def _log(self, msg: str) -> None:
        """Print with agent identity prefix."""
        tag = self.agent_id or "coordinator"
        print(f"[{tag}] {msg}")

    @property
    def connected(self) -> bool:
        return self.api_key is not None

    def _rpc(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """RPC call with the stored API key."""
        if not self.api_key:
            raise RuntimeError("No API key configured")
        return ensue_rpc(self.api_key, tool_name, arguments)

    def _make_key(self, description: str) -> str:
        """Create a human-readable experiment key using agent_id."""
        return _experiment_key(self.agent_id or "unknown", description)

    # --- Onboarding ---

    def join_hub(self, invite_token: str) -> dict[str, Any]:
        """Claim the hub invite to join autoresearch-community."""
        try:
            result = self._rpc("claim_invite", {"token": invite_token})
            self._log(f"Joined hub: {result}")
            return result
        except Exception as e:
            self._log(f"join_hub failed: {e}")
            return {"error": str(e)}

    def test_connectivity(self) -> bool:
        """Test if the API key works."""
        try:
            self._rpc("list_keys", {"limit": 1})
            return True
        except Exception:
            return False

    def announce(self) -> None:
        """Print a startup banner with swarm state."""
        try:
            tag = self.agent_id or "unknown"
            # Get global best
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            best_line = "no results yet"
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                best_bpb = current.get("val_bpb", "?")
                best_by = current.get("agent_id", "?")
                best_line = f"val_bpb={best_bpb} (by {best_by})"

            # Count results
            result_list = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/results/",
                "limit": 200,
            })
            result_keys = result_list.get("keys", [])
            total = len(result_keys)

            # Count active claims
            claim_list = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/claims/",
                "limit": 50,
            })
            active_claims = len(claim_list.get("keys", []))

            banner = f"""
{'=' * 54}
  AUTORESEARCH AGENT: {tag}
  Swarm: {HUB_ORG}
  Global best: {best_line}
  Experiments completed: {total}
  Active claims: {active_claims}
{'=' * 54}"""
            print(banner)
        except Exception as e:
            self._log(f"announce error (non-fatal): {e}")
            print(f"\n  AUTORESEARCH AGENT: {self.agent_id or 'unknown'}\n")

    # --- Work Claiming ---

    def check_claimed(self, experiment_key: str) -> bool:
        """Check if an experiment is already claimed (active) or completed."""
        try:
            # Check if result already exists (try new key format)
            result_key = f"@{HUB_ORG}/results/{experiment_key}"
            result = self._rpc("get_memory", {"key_names": [result_key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                return True  # already done

            # Fallback: check old hash-only format if key contains '--'
            if "--" in experiment_key:
                old_hash = experiment_key.rsplit("--", 1)[-1]
                if len(old_hash) <= 12:
                    old_result_key = f"@{HUB_ORG}/results/{old_hash}"
                    old_result = self._rpc("get_memory", {"key_names": [old_result_key]})
                    old_results = old_result.get("results", [])
                    if old_results and old_results[0].get("status") == "success":
                        return True

            # Check for active claim
            claim_key = f"@{HUB_ORG}/claims/{experiment_key}"
            claim = self._rpc("get_memory", {"key_names": [claim_key]})
            claims = claim.get("results", [])
            if claims and claims[0].get("status") == "success":
                value = json.loads(claims[0].get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                # Check if claim is stale (> CLAIM_TTL seconds old)
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            return True  # fresh claim, someone's on it
                    except (ValueError, TypeError):
                        pass
            return False
        except Exception as e:
            self._log(f"check_claimed error: {e}")
            return False  # assume not claimed on error, let training proceed

    def check_similar_claimed(self, description: str) -> list[dict]:
        """Semantic search for similar in-progress work."""
        try:
            result = self._rpc("search_memories", {
                "query": description,
                "limit": 5,
                "prefix": f"@{HUB_ORG}/claims/",
            })
            matches = result.get("results", [])
            # Filter to fresh claims above threshold
            similar = []
            for match in matches:
                score = match.get("score", 0)
                if score < SEMANTIC_THRESHOLD:
                    continue
                value = json.loads(match.get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            similar.append({"description": value.get("description", ""), "score": score, "agent": value.get("agent_id", "")})
                    except (ValueError, TypeError):
                        pass
            return similar
        except Exception as e:
            self._log(f"check_similar_claimed error: {e}")
            return []

    def claim_experiment(self, description: str) -> Optional[str]:
        """
        Attempt to claim an experiment. Returns the experiment key if claimed,
        or None if already taken / similar work in progress.

        The key is human-readable: <agent>--<slug>--<short_hash>
        """
        exp_key = self._make_key(description)

        try:
            # 1. CHECK exact
            if self.check_claimed(exp_key):
                self._log(f"Experiment already claimed/completed: {exp_key}")
                return None

            # 2. CHECK semantic
            similar = self.check_similar_claimed(description)
            if similar:
                self._log(f"Similar work in progress: {similar[0]['description']} (score={similar[0]['score']:.3f} by {similar[0]['agent']})")
                return None

            # 3. CLAIM
            claim_key = f"@{HUB_ORG}/claims/{exp_key}"
            claim_data = {
                "agent_id": self.agent_id or "unknown",
                "description": description,
                "experiment_key": exp_key,
                "claimed_at": _now_iso(),
                "expected_duration_seconds": 300,
                "status": "claimed",
            }
            value_b64 = base64.b64encode(json.dumps(claim_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": claim_key,
                "description": f"[autoresearch] Claim: {description}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            # 4. VERIFY (wait for race resolution)
            time.sleep(VERIFY_DELAY)
            verify = self._rpc("get_memory", {"key_names": [claim_key]})
            verify_results = verify.get("results", [])
            if verify_results and verify_results[0].get("status") == "success":
                value = json.loads(verify_results[0].get("value", "{}"))
                if value.get("agent_id") == (self.agent_id or "unknown"):
                    self._log(f"Claimed experiment: {exp_key}")
                    return exp_key

            self._log(f"Lost claim race for: {exp_key}")
            return None

        except Exception as e:
            self._log(f"claim_experiment error: {e}")
            # On error, return the key anyway so training can proceed locally
            return exp_key

    # --- Results ---

    def publish_result(
        self,
        experiment_key: str,
        val_bpb: float,
        memory_gb: float,
        status: str,
        description: str,
        train_py_source: str,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Publish an experiment result to the hub with full train.py source."""
        try:
            repo_url = _git_remote_url()
            branch = _git_branch()
            commit = _git_commit_short()

            # Get current global best for delta calculation
            global_best_bpb = self._get_global_best_bpb()
            delta_vs_best = val_bpb - global_best_bpb if global_best_bpb is not None else None

            result_data = {
                "agent_id": self.agent_id or "unknown",
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "commit": commit,
                "description": description,
                "train_py": train_py_source,
                "repo_url": repo_url,
                "branch": branch,
                "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else None,
                "completed_at": _now_iso(),
                "delta_vs_best": delta_vs_best,
                "global_best_at_publish": global_best_bpb,
                **(extra_metrics or {}),
            }

            result_key = f"@{HUB_ORG}/results/{experiment_key}"
            value_b64 = base64.b64encode(json.dumps(result_data).encode()).decode()

            agent = self.agent_id or "unknown"
            desc_prefix = f"[{agent} {status.upper()}] val_bpb={val_bpb:.6f}"
            if delta_vs_best is not None:
                desc_prefix += f" (delta={delta_vs_best:+.4f})"
            desc_prefix += f" | {description}"

            self._rpc("create_memory", {"items": [{
                "key_name": result_key,
                "description": f"[autoresearch] Result: {desc_prefix}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            # Print with explicit metrics
            delta_str = ""
            if delta_vs_best is not None:
                delta_str = f" (delta={delta_vs_best:+.4f} vs global best {global_best_bpb:.6f})"
            self._log(f"RESULT: val_bpb={val_bpb:.6f}{delta_str} ({status})")

            # Update global best if this is an improvement
            if status == "keep":
                self.maybe_update_best(val_bpb, result_data, train_py_source)

        except Exception as e:
            self._log(f"publish_result error: {e}")

    def _get_global_best_bpb(self) -> Optional[float]:
        """Read the current global best val_bpb. Returns None if unavailable."""
        try:
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                return current.get("val_bpb")
        except Exception:
            pass
        return None

    def maybe_update_best(
        self,
        val_bpb: float,
        result_data: dict,
        train_py_source: str,
    ) -> bool:
        """
        Update the global best if this result beats it. Returns True if updated.

        Safety rules:
        - Sanity check: reject val_bpb <= 0 or suspiciously low (< 0.5) — likely a bug
        - Read-compare-write: re-read current best right before writing to minimize race window
        - Previous best is always preserved in the new record for recovery
        """
        try:
            # Sanity: reject obviously bogus values
            if val_bpb <= 0:
                self._log(f"REJECTED best update: val_bpb={val_bpb} <= 0 (likely a crash/bug)")
                return False
            if val_bpb < 0.5:
                self._log(f"REJECTED best update: val_bpb={val_bpb} < 0.5 (suspiciously low, likely a bug)")
                return False

            # Read current best
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])

            previous_best_bpb = None
            previous_best_by = None
            previous_best_description = None
            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                previous_best_bpb = current.get("val_bpb")
                previous_best_by = current.get("agent_id")
                previous_best_description = current.get("description")
                if previous_best_bpb is not None and val_bpb >= previous_best_bpb:
                    return False  # not better

            # Sanity: improvement shouldn't be impossibly large (> 0.1 in one step)
            if previous_best_bpb is not None:
                improvement = previous_best_bpb - val_bpb
                if improvement > 0.1:
                    self._log(f"REJECTED best update: improvement of {improvement:.4f} is suspiciously large (> 0.1 in one step)")
                    return False

            # Re-read right before write to minimize race window
            meta2 = self._rpc("get_memory", {"key_names": [meta_key]})
            meta2_results = meta2.get("results", [])
            if meta2_results and meta2_results[0].get("status") == "success":
                current2 = json.loads(meta2_results[0].get("value", "{}"))
                current2_bpb = current2.get("val_bpb")
                if current2_bpb is not None and val_bpb >= current2_bpb:
                    self._log(f"Lost best update race: someone posted {current2_bpb:.6f} while we were checking")
                    return False

            # Enrich metadata — always preserve previous best for recovery
            best_data = {
                **result_data,
                "best_val_bpb": val_bpb,
                "achieved_by": self.agent_id or "unknown",
                "achieved_at": _now_iso(),
                "previous_best_val_bpb": previous_best_bpb,
                "previous_best_by": previous_best_by,
                "previous_best_description": previous_best_description,
                "improvement_over_previous": (previous_best_bpb - val_bpb) if previous_best_bpb is not None else None,
            }

            # Upsert best/train_py (create if missing, update if exists)
            code_key = f"@{HUB_ORG}/best/train_py"
            code_b64 = base64.b64encode(train_py_source.encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": code_key,
                    "value": code_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": code_key,
                    "description": "[autoresearch] Current best train.py source code",
                    "value": code_b64,
                    "base64": True,
                }]})

            # Upsert best/metadata
            meta_b64 = base64.b64encode(json.dumps(best_data).encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": meta_key,
                    "value": meta_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": meta_key,
                    "description": "[autoresearch] Metadata for current best train.py",
                    "value": meta_b64,
                    "base64": True,
                }]})

            improvement = (previous_best_bpb - val_bpb) if previous_best_bpb else 0
            prev_info = f" (improved {improvement:.4f} over {previous_best_by}'s {previous_best_bpb:.6f})" if previous_best_bpb else ""
            self._log(f"NEW GLOBAL BEST! val_bpb={val_bpb:.6f}{prev_info}")
            return True

        except Exception as e:
            self._log(f"maybe_update_best error: {e}")
            return False

    # --- Config Sharing ---

    def pull_best_config(self) -> Optional[tuple[str, dict]]:
        """
        Pull the current global best train.py and metadata.
        Returns (source_code, metadata_dict) or None.
        """
        try:
            meta_key = f"@{HUB_ORG}/best/metadata"
            code_key = f"@{HUB_ORG}/best/train_py"

            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if not meta_results or meta_results[0].get("status") != "success":
                return None

            code = self._rpc("get_memory", {"key_names": [code_key]})
            code_results = code.get("results", [])
            if not code_results or code_results[0].get("status") != "success":
                return None

            metadata = json.loads(meta_results[0]["value"])
            source = code_results[0]["value"]

            self._log(f"Pulled best config: val_bpb={metadata.get('val_bpb', '?')} (by {metadata.get('agent_id', '?')})")
            return source, metadata

        except Exception as e:
            self._log(f"pull_best_config error: {e}")
            return None

    def should_sync(self) -> bool:
        """Check if it's time to sync with the global best (every N experiments)."""
        self.experiment_count += 1
        return self.experiment_count % SYNC_EVERY_N == 0

    # --- Collective Intelligence ---

    def ask_swarm(self, question: str, namespace: str = None) -> dict:
        """
        Ask the swarm a natural language question and get structured answers.

        Args:
            question: e.g. "what learning rates have been tried and which worked best?"
            namespace: scope the search — "results", "insights", "hypotheses", "claims", or None for all

        Returns dict with: relevant_results, best_match, namespace_searched, summary
        """
        try:
            prefix = f"@{HUB_ORG}/"
            if namespace:
                prefix = f"@{HUB_ORG}/{namespace}/"

            result = self._rpc("search_memories", {
                "query": question,
                "limit": 20,
                "prefix": prefix,
            })

            matches = result.get("results", [])
            relevant = []
            for match in matches:
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    relevant.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Sort by relevance score
            relevant.sort(key=lambda x: x.get("_score", 0), reverse=True)

            best_match = relevant[0] if relevant else None

            # Build summary
            lines = [f"Swarm answer for: {question}"]
            lines.append(f"Namespace: {namespace or 'all'} | {len(relevant)} results")
            lines.append("")
            for r in relevant[:5]:
                agent = r.get("agent_id", "?")
                bpb = r.get("val_bpb")
                status = r.get("status", "")
                desc = r.get("description", r.get("title", r.get("insight", "?")))
                score = r.get("_score", 0)
                if bpb is not None:
                    lines.append(f"  [{agent}] val_bpb={bpb:.6f} ({status}) — {desc} (relevance={score:.2f})")
                else:
                    lines.append(f"  [{agent}] {desc} (relevance={score:.2f})")

            return {
                "relevant_results": relevant,
                "best_match": best_match,
                "namespace_searched": namespace or "all",
                "summary": "\n".join(lines),
            }

        except Exception as e:
            self._log(f"ask_swarm error: {e}")
            return {"relevant_results": [], "best_match": None, "namespace_searched": namespace or "all", "summary": f"Error: {e}"}

    def list_namespace(self, namespace: str, limit: int = 50) -> list[dict]:
        """
        List all keys under a namespace prefix (results, claims, insights, hypotheses).

        Returns list of dicts with key_name and any available metadata.
        """
        try:
            result = self._rpc("list_keys", {
                "prefix": f"@{HUB_ORG}/{namespace}/",
                "limit": limit,
            })
            keys = result.get("keys", [])
            entries = []
            for k in keys:
                if isinstance(k, dict):
                    entries.append(k)
                elif isinstance(k, str):
                    entries.append({"key_name": k})
            return entries

        except Exception as e:
            self._log(f"list_namespace error: {e}")
            return []

    def analyze_swarm(self) -> dict:
        """
        Comprehensive swarm state analysis.

        Returns dict with: global_best, recent_keeps, recent_failures,
        active_claims, unclaimed_hypotheses, improvement_trend, summary
        """
        try:
            # Global best
            global_best = None
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if meta_results and meta_results[0].get("status") == "success":
                global_best = json.loads(meta_results[0].get("value", "{}"))

            # Recent results
            result_search = self._rpc("search_memories", {
                "query": "experiment result val_bpb",
                "limit": 30,
                "prefix": f"@{HUB_ORG}/results/",
            })
            all_results = []
            for match in result_search.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_key"] = match.get("key_name", "")
                    all_results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass

            recent_keeps = sorted(
                [r for r in all_results if r.get("status") == "keep"],
                key=lambda x: x.get("val_bpb", float("inf")),
            )
            recent_failures = [r for r in all_results if r.get("status") in ("discard", "crash")]

            # Active claims
            claim_search = self._rpc("search_memories", {
                "query": "autoresearch claim experiment",
                "limit": 20,
                "prefix": f"@{HUB_ORG}/claims/",
            })
            active_claims = []
            for match in claim_search.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    claimed_at = data.get("claimed_at", "")
                    if claimed_at:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            active_claims.append(data)
                except Exception:
                    pass

            # Unclaimed hypotheses
            unclaimed = self.get_unclaimed_hypotheses(limit=5)

            # Improvement trend (compare recent keeps)
            trend = "unknown"
            if len(recent_keeps) >= 3:
                recent_bpbs = [r["val_bpb"] for r in recent_keeps[:5] if "val_bpb" in r]
                older_bpbs = [r["val_bpb"] for r in recent_keeps[5:10] if "val_bpb" in r]
                if recent_bpbs and older_bpbs:
                    if min(recent_bpbs) < min(older_bpbs):
                        trend = "improving"
                    elif min(recent_bpbs) == min(older_bpbs):
                        trend = "plateaued"
                    else:
                        trend = "regressing"

            # Build summary
            lines = ["=" * 50, "SWARM ANALYSIS", "=" * 50]
            if global_best:
                lines.append(f"Global best: val_bpb={global_best.get('val_bpb', '?'):.6f} by {global_best.get('agent_id', '?')}")
                if global_best.get("achieved_at"):
                    lines.append(f"  Achieved at: {global_best['achieved_at']}")
            else:
                lines.append("Global best: none yet")

            lines.append(f"\nKeeps ({len(recent_keeps)}):")
            for r in recent_keeps[:5]:
                lines.append(f"  [{r.get('agent_id', '?')}] val_bpb={r.get('val_bpb', 0):.6f} — {r.get('description', '?')}")

            lines.append(f"\nFailures ({len(recent_failures)}):")
            for r in recent_failures[:5]:
                lines.append(f"  [{r.get('agent_id', '?')}] {r.get('status', '?')} — {r.get('description', '?')}")

            lines.append(f"\nActive claims ({len(active_claims)}):")
            for c in active_claims:
                lines.append(f"  [{c.get('agent_id', '?')}] {c.get('description', '?')}")

            lines.append(f"\nUnclaimed hypotheses ({len(unclaimed)}):")
            for h in unclaimed[:3]:
                lines.append(f"  {h.get('title', '?')} (priority={h.get('priority', '?')})")

            lines.append(f"\nTrend: {trend}")
            lines.append("=" * 50)

            return {
                "global_best": global_best,
                "recent_keeps": recent_keeps,
                "recent_failures": recent_failures,
                "active_claims": active_claims,
                "unclaimed_hypotheses": unclaimed,
                "improvement_trend": trend,
                "summary": "\n".join(lines),
            }

        except Exception as e:
            self._log(f"analyze_swarm error: {e}")
            return {
                "global_best": None, "recent_keeps": [], "recent_failures": [],
                "active_claims": [], "unclaimed_hypotheses": [],
                "improvement_trend": "unknown", "summary": f"Error: {e}",
            }

    def post_insight(self, insight: str, evidence_keys: list[str] = None) -> None:
        """
        Post an observation/learning to the collective.

        Not a hypothesis with a config — an *insight* about what has been observed.
        Example: "LR above 0.08 consistently causes instability — 3 agents tried, all regressed"
        """
        try:
            slug = _slugify(insight)
            agent = _slugify(self.agent_id or "unknown", max_len=20)
            short_hash = hashlib.sha256(insight.encode()).hexdigest()[:6]
            insight_key = f"@{HUB_ORG}/insights/{agent}--{slug}--{short_hash}"

            insight_data = {
                "agent_id": self.agent_id or "unknown",
                "insight": insight,
                "evidence_keys": evidence_keys or [],
                "posted_at": _now_iso(),
            }

            value_b64 = base64.b64encode(json.dumps(insight_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": insight_key,
                "description": f"[autoresearch] Insight by {self.agent_id or 'unknown'}: {insight}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            self._log(f"Published insight: {insight}")

        except Exception as e:
            self._log(f"post_insight error: {e}")

    def get_swarm_insights(self, topic: str) -> list[dict]:
        """Search insights by topic to see what the group has learned."""
        try:
            result = self._rpc("search_memories", {
                "query": topic,
                "limit": 10,
                "prefix": f"@{HUB_ORG}/insights/",
            })
            insights = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    insights.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return insights

        except Exception as e:
            self._log(f"get_swarm_insights error: {e}")
            return []

    # --- Thinking Phase ---

    def get_recent_results(self, limit: int = 20) -> list[dict]:
        """Get recent experiment results from the swarm."""
        try:
            result = self._rpc("search_memories", {
                "query": "autoresearch experiment result val_bpb",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            self._log(f"get_recent_results error: {e}")
            return []

    def get_unclaimed_hypotheses(self, limit: int = 10) -> list[dict]:
        """Get hypotheses that haven't been claimed/tested yet."""
        try:
            result = self._rpc("search_memories", {
                "query": "autoresearch hypothesis experiment suggestion",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/hypotheses/",
            })
            hypotheses = []
            for match in result.get("results", []):
                try:
                    hyp = json.loads(match.get("value", "{}"))
                    hypotheses.append(hyp)
                except (json.JSONDecodeError, KeyError):
                    pass
            return hypotheses

        except Exception as e:
            self._log(f"get_unclaimed_hypotheses error: {e}")
            return []

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        suggested_config: Optional[dict] = None,
        evidence_keys: Optional[list[str]] = None,
        priority: int = 3,
    ) -> None:
        """Publish a research hypothesis for other agents to consider."""
        try:
            slug = _slugify(title)
            agent = _slugify(self.agent_id or "unknown", max_len=20)
            short_hash = hashlib.sha256(title.encode()).hexdigest()[:6]
            hyp_key = f"@{HUB_ORG}/hypotheses/{agent}--{slug}--{short_hash}"

            hyp_data = {
                "agent_id": self.agent_id or "unknown",
                "title": title,
                "hypothesis": hypothesis,
                "suggested_config": suggested_config,
                "evidence_keys": evidence_keys or [],
                "priority": priority,
                "created_at": _now_iso(),
            }

            value_b64 = base64.b64encode(json.dumps(hyp_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": hyp_key,
                "description": f"[autoresearch] Hypothesis: {title}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            self._log(f"Published hypothesis: {title}")

        except Exception as e:
            self._log(f"publish_hypothesis error: {e}")

    def search_experiments(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over past experiment results."""
        try:
            result = self._rpc("search_memories", {
                "query": query,
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    data["_key"] = match.get("key_name", "")
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            self._log(f"search_experiments error: {e}")
            return []

    def get_leaderboard(self) -> list[dict]:
        """Get the current global leaderboard."""
        try:
            result = self._rpc("get_memory", {
                "key_names": [f"@{HUB_ORG}/leaderboard"],
            })
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                data = json.loads(results[0]["value"])
                return data.get("entries", [])
            return []
        except Exception as e:
            self._log(f"get_leaderboard error: {e}")
            return []
