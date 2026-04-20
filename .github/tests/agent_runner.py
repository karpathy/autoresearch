"""Agent runner -- invoke an LLM with harness context and a task prompt.

Supports multiple backends:
- copilot: Uses the Copilot SDK (CopilotClient) for full agent sessions
- mock: Returns empty responses for dry-run/structural validation

Usage:
    runner = AgentRunner(backend="copilot", model="claude-sonnet-4-6")
    result = await runner.run_task(prompt, context_texts, timeout=60)
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONTEXT_SEARCH_PATHS = [
    REPO_ROOT,
    REPO_ROOT / ".github",
]


@dataclass
class TaskResult:
    """Result of running a single task against an agent."""

    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    error: str | None = None
    model: str = ""
    prompt_hash: str = ""
    harness_version: str = ""


def resolve_context_file(relative_path: str) -> Path | None:
    """Resolve a context_files entry to an absolute path."""
    for base in CONTEXT_SEARCH_PATHS:
        candidate = base / relative_path
        if candidate.exists():
            return candidate
    return None


def load_context_files(paths: list[str]) -> dict[str, str]:
    """Load context files from short paths. Returns {path: content}."""
    result = {}
    for p in paths:
        resolved = resolve_context_file(p)
        if resolved:
            result[p] = resolved.read_text(encoding="utf-8")
    return result


def build_system_prompt(context_texts: dict[str, str]) -> str:
    """Build a system prompt from loaded context files."""
    if not context_texts:
        return "You are an AI agent operating in a development workspace."

    sections = ["You are an AI agent operating in the autoresearch workspace.",
                "The following files provide your instructions and context:\n"]
    for path, content in context_texts.items():
        sections.append(f"--- {path} ---\n{content}\n")
    return "\n".join(sections)


def get_harness_version() -> str:
    """Return the current git HEAD SHA for provenance tracking."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def hash_prompt(text: str) -> str:
    """SHA256 hash of a prompt for reproducibility tracking."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AgentRunner:
    """Invoke an LLM with harness context for benchmark tasks."""

    def __init__(
        self,
        backend: Literal["copilot", "mock"] = "copilot",
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.backend = backend
        self.model = model
        self._client = None
        self._harness_version = get_harness_version()

    async def start(self) -> None:
        """Initialize the backend client."""
        if self.backend == "copilot":
            try:
                from copilot import CopilotClient
                self._client = CopilotClient()
                await self._client.start()
            except ImportError:
                raise RuntimeError(
                    "Copilot SDK not installed. Run: uv pip install github-copilot-sdk"
                )

    async def stop(self) -> None:
        """Shut down the backend client."""
        if self._client:
            await self._client.stop()
            self._client = None

    async def run_task(
        self,
        prompt: str,
        context_texts: dict[str, str],
        timeout: float = 120,
    ) -> TaskResult:
        """Run a single task and return the result."""
        system_prompt = build_system_prompt(context_texts)
        full_prompt = f"{system_prompt}\n\n---\n\nTask: {prompt}"
        p_hash = hash_prompt(full_prompt)

        if self.backend == "mock":
            return TaskResult(
                response="[mock response]",
                prompt_hash=p_hash,
                harness_version=self._harness_version,
                model="mock",
            )

        return await self._run_copilot(full_prompt, p_hash, timeout)

    async def _run_copilot(
        self, full_prompt: str, p_hash: str, timeout: float,
    ) -> TaskResult:
        """Execute via Copilot SDK."""
        if not self._client:
            raise RuntimeError("Client not started. Call start() first.")

        def approve_all(request, context):
            return {"kind": "approved", "rules": []}

        start = time.time()
        response_parts: list[str] = []
        token_info: dict = {}
        error = None

        try:
            session = await self._client.create_session({
                "model": self.model,
                "on_permission_request": approve_all,
            })

            done = asyncio.Event()

            def on_event(event):
                nonlocal token_info
                etype = event.type.value if hasattr(event.type, "value") else str(event.type)
                if etype == "assistant.message":
                    if hasattr(event.data, "content") and event.data.content:
                        response_parts.append(event.data.content)
                elif etype == "session.idle":
                    done.set()
                if hasattr(event.data, "usage") and event.data.usage:
                    u = event.data.usage
                    token_info["input"] = getattr(u, "input_tokens", 0)
                    token_info["output"] = getattr(u, "output_tokens", 0)

            session.on(on_event)
            await session.send({"prompt": full_prompt})
            await asyncio.wait_for(done.wait(), timeout=timeout)

        except asyncio.TimeoutError:
            error = f"Timeout after {timeout}s"
        except Exception as exc:
            error = str(exc)

        end = time.time()
        return TaskResult(
            response="".join(response_parts),
            input_tokens=token_info.get("input", 0),
            output_tokens=token_info.get("output", 0),
            start_time=start,
            end_time=end,
            error=error,
            model=self.model,
            prompt_hash=p_hash,
            harness_version=self._harness_version,
        )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()
