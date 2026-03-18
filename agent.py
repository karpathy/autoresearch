"""
Autonomous ML experiment agent for autoresearch.
Wraps the experiment loop with LLM tool-calling: any provider, any model.

Usage:
    python agent.py --provider anthropic --model claude-sonnet-4-20250514
    python agent.py --provider openai --model gpt-4o
    python agent.py --provider gemini --model gemini-2.0-flash
    python agent.py --provider anthropic --model claude-sonnet-4-20250514 --tag mar18
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
TRAIN_PY = PROJECT_DIR / "train.py"
RESULTS_TSV = PROJECT_DIR / "results.tsv"
RUN_LOG = PROJECT_DIR / "run.log"
RESULTS_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
RUN_TIMEOUT = 600  # 10 min hard kill
SECTION_DELIM = re.compile(r"^# -{10,}")

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _find_section(lines, section_name):
    """Find start/end line indices for a named section in train.py."""
    sections = {
        "model": "GPT Model",
        "optimizer": "Optimizer",
        "hyperparameters": "Hyperparameters",
        "setup": "Setup",
        "training_loop": "Training loop",
    }
    target = sections.get(section_name, section_name)
    start = end = None
    for i, line in enumerate(lines):
        if target.lower() in line.lower() and (i > 0 and SECTION_DELIM.match(lines[i - 1])):
            start = i - 1
        elif start is not None and SECTION_DELIM.match(line) and i > start + 2:
            # Next section delimiter pair: the line before it is the end
            # Actually, look for the next section header pattern
            if i + 1 < len(lines) and not SECTION_DELIM.match(lines[i + 1]):
                continue
            end = i
            break
    if start is not None and end is None:
        end = len(lines)
    return start, end


def _parse_hyperparams():
    """Parse hyperparameters from train.py. Returns {name: (value_str, comment)}."""
    text = TRAIN_PY.read_text(encoding="utf-8")
    lines = text.splitlines()
    start, end = _find_section(lines, "hyperparameters")
    if start is None:
        return {}
    params = {}
    pattern = re.compile(r"^([A-Z_]+)\s*=\s*(.+?)(\s*#.*)?$")
    for line in lines[start:end]:
        m = pattern.match(line)
        if m:
            name, val_str, comment = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
            params[name] = (val_str, comment)
    return params


def tool_get_config(**kwargs):
    """Read current hyperparameters from train.py."""
    params = _parse_hyperparams()
    result = {}
    for name, (val_str, comment) in params.items():
        try:
            val = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            try:
                val = eval(val_str, {"__builtins__": {}})
            except Exception:
                val = val_str
        result[name] = {"value": val, "raw": val_str, "comment": comment}
    return json.dumps(result, indent=2)


def tool_set_hyperparams(changes: dict, **kwargs):
    """Modify hyperparameters in train.py. changes is {name: new_value_str}."""
    text = TRAIN_PY.read_text(encoding="utf-8")
    lines = text.splitlines()
    modified = []
    for name, new_val in changes.items():
        pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(.+?)(\s*#.*)$")
        found = False
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                lines[i] = f"{m.group(1)}{new_val}{m.group(3)}"
                modified.append(f"  {name}: {m.group(2).strip()} -> {new_val}")
                found = True
                break
        if not found:
            return json.dumps({"error": f"Hyperparameter '{name}' not found in train.py"})
    TRAIN_PY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json.dumps({"modified": modified})


def tool_edit_code(section: str, new_code: str, **kwargs):
    """Replace an entire named section of train.py."""
    text = TRAIN_PY.read_text(encoding="utf-8")
    lines = text.splitlines()
    start, end = _find_section(lines, section)
    if start is None:
        return json.dumps({"error": f"Section '{section}' not found. Valid: model, optimizer, hyperparameters, setup, training_loop"})

    # Syntax check the new code
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        return json.dumps({"error": f"Syntax error in new code: {e}"})

    # Replace section content (keep the delimiter header)
    header_end = start + 3  # delimiter + title + delimiter
    new_lines = lines[:header_end] + [""] + new_code.splitlines() + [""] + lines[end:]
    new_text = "\n".join(new_lines) + "\n"

    # Full file syntax check
    try:
        ast.parse(new_text)
    except SyntaxError as e:
        return json.dumps({"error": f"Resulting file has syntax error: {e}"})

    TRAIN_PY.write_text(new_text, encoding="utf-8")
    return json.dumps({"ok": True, "section": section, "lines_replaced": end - header_end, "lines_new": len(new_code.splitlines())})


def tool_run_experiment(**kwargs):
    """Run uv run train.py and return metrics."""
    try:
        with open(RUN_LOG, "w") as log:
            result = subprocess.run(
                ["uv", "run", "train.py"],
                stdout=log, stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT, cwd=PROJECT_DIR,
            )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "timeout", "error": "Exceeded 10 minute timeout"})

    log_text = RUN_LOG.read_text()

    if result.returncode != 0:
        last_lines = "\n".join(log_text.splitlines()[-50:])
        if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
            return json.dumps({"status": "oom", "error": last_lines})
        return json.dumps({"status": "crash", "exit_code": result.returncode, "error": last_lines})

    # Parse metrics after the --- separator
    metrics = {}
    in_summary = False
    for line in log_text.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if in_summary:
            m = re.match(r"^(\w+):\s+(.+)$", line.strip())
            if m:
                key, val = m.group(1), m.group(2).strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val

    if "val_bpb" not in metrics:
        last_lines = "\n".join(log_text.splitlines()[-30:])
        return json.dumps({"status": "no_metrics", "error": last_lines})

    metrics["status"] = "ok"
    return json.dumps(metrics)


def tool_get_history(**kwargs):
    """Read results.tsv."""
    if RESULTS_TSV.exists():
        return RESULTS_TSV.read_text()
    return "No experiments yet. results.tsv does not exist."


def tool_keep(description: str, **kwargs):
    """Git commit current train.py and log success to results.tsv."""
    # Git commit
    subprocess.run(["git", "add", "train.py"], cwd=PROJECT_DIR, capture_output=True)
    subprocess.run(["git", "commit", "-m", description], cwd=PROJECT_DIR, capture_output=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_DIR, capture_output=True, text=True
    ).stdout.strip()

    # Parse last run metrics
    val_bpb = "0.000000"
    mem_gb = "0.0"
    if RUN_LOG.exists():
        log_text = RUN_LOG.read_text()
        for line in log_text.splitlines():
            m = re.match(r"^val_bpb:\s+(.+)$", line.strip())
            if m:
                val_bpb = m.group(1).strip()
            m = re.match(r"^peak_vram_mb:\s+(.+)$", line.strip())
            if m:
                try:
                    mem_gb = f"{float(m.group(1).strip()) / 1024:.1f}"
                except ValueError:
                    pass

    # Log to results.tsv
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb}\t{mem_gb}\tkeep\t{description}\n")

    return json.dumps({"commit": commit, "val_bpb": val_bpb, "memory_gb": mem_gb, "status": "kept"})


def tool_discard(reason: str = "", **kwargs):
    """Discard uncommitted changes to train.py and log to results.tsv."""
    # Parse metrics before discarding
    val_bpb = "0.000000"
    mem_gb = "0.0"
    status = "discard"
    if RUN_LOG.exists():
        log_text = RUN_LOG.read_text()
        for line in log_text.splitlines():
            m = re.match(r"^val_bpb:\s+(.+)$", line.strip())
            if m:
                val_bpb = m.group(1).strip()
            m = re.match(r"^peak_vram_mb:\s+(.+)$", line.strip())
            if m:
                try:
                    mem_gb = f"{float(m.group(1).strip()) / 1024:.1f}"
                except ValueError:
                    pass
        if "CUDA out of memory" in log_text or "FAIL" in log_text:
            status = "crash"

    # Revert train.py
    subprocess.run(["git", "checkout", "--", "train.py"], cwd=PROJECT_DIR, capture_output=True)

    # Log
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_DIR, capture_output=True, text=True
    ).stdout.strip()
    desc = reason or "discarded"
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb}\t{mem_gb}\t{status}\t{desc}\n")

    return json.dumps({"status": "discarded", "reverted_to": commit})


def tool_read_code(start_line: int = 1, end_line: int = 50, **kwargs):
    """Read a range of lines from train.py (1-indexed)."""
    lines = TRAIN_PY.read_text(encoding="utf-8").splitlines()
    start_line = max(1, start_line)
    end_line = min(len(lines), end_line)
    result = []
    for i in range(start_line - 1, end_line):
        result.append(f"{i + 1:4d}  {lines[i]}")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {
    "get_config": {
        "fn": tool_get_config,
        "description": "Read current hyperparameters from train.py. Returns name, value, and comment for each.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "set_hyperparams": {
        "fn": tool_set_hyperparams,
        "description": "Modify hyperparameters in train.py. Pass a dict of {name: new_value} where values are Python expressions as strings (e.g. '2**17', '0.04', '\"SSSSL\"').",
        "parameters": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "object",
                    "description": "Dict of hyperparameter names to new values (as strings)",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["changes"],
        },
    },
    "edit_code": {
        "fn": tool_edit_code,
        "description": "Replace an entire named section of train.py with new code. Sections: model, optimizer, hyperparameters, setup, training_loop. Use for architectural changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "Section name: model, optimizer, hyperparameters, setup, training_loop"},
                "new_code": {"type": "string", "description": "Complete Python code to replace the section content"},
            },
            "required": ["section", "new_code"],
        },
    },
    "run_experiment": {
        "fn": tool_run_experiment,
        "description": "Run 'uv run train.py' (5-min training budget). Returns val_bpb and other metrics, or crash/OOM/timeout info.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_history": {
        "fn": tool_get_history,
        "description": "Read results.tsv — the log of all past experiments (commit, val_bpb, memory, status, description).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "keep": {
        "fn": tool_keep,
        "description": "Keep the current changes: git commit train.py and log success to results.tsv. Call after a successful experiment that improved val_bpb.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Short description of what this experiment tried"},
            },
            "required": ["description"],
        },
    },
    "discard": {
        "fn": tool_discard,
        "description": "Discard current changes: revert train.py to last commit and log to results.tsv. Call after a failed or regressed experiment.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why this experiment was discarded"},
            },
            "required": [],
        },
    },
    "read_code": {
        "fn": tool_read_code,
        "description": "Read a range of lines from train.py (1-indexed). Use to inspect model architecture, optimizer, or training loop code.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to read (1-indexed)"},
            },
            "required": ["start_line", "end_line"],
        },
    },
}


def execute_tool(name, arguments):
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return TOOLS[name]["fn"](**arguments)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class AgentResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"


class Provider(ABC):
    @abstractmethod
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> AgentResponse:
        pass


class AnthropicProvider(Provider):
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, system, messages, tools):
        tool_defs = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        # Convert messages to Anthropic format
        msgs = []
        for m in messages:
            if m["role"] == "tool":
                msgs.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}],
                })
            elif m["role"] == "assistant" and m.get("tool_calls"):
                content = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                msgs.append({"role": "assistant", "content": content})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})

        resp = self.client.messages.create(
            model=self.model, max_tokens=4096, system=system,
            messages=msgs, tools=tool_defs,
        )
        text_parts = [b.text for b in resp.content if b.type == "text"]
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content if b.type == "tool_use"
        ]
        return AgentResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
        )


class OpenAIProvider(Provider):
    def __init__(self, model: str):
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def chat(self, system, messages, tools):
        tool_defs = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                tc_list = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in m["tool_calls"]
                ]
                msgs.append({"role": "assistant", "content": m.get("text") or "", "tool_calls": tc_list})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})

        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs, tools=tool_defs, max_tokens=4096,
        )
        choice = resp.choices[0]
        text = choice.message.content
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id, name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
        return AgentResponse(text=text, tool_calls=tool_calls, stop_reason=choice.finish_reason)


class GeminiProvider(Provider):
    def __init__(self, model: str):
        from google import genai
        self.client = genai.Client()
        self.model = model

    def chat(self, system, messages, tools):
        from google.genai import types

        tool_defs = types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["parameters"])
            for t in tools
        ])

        contents = []
        for m in messages:
            if m["role"] == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=m["name"], response=json.loads(m["content"]),
                    ))],
                ))
            elif m["role"] == "assistant" and m.get("tool_calls"):
                parts = []
                if m.get("text"):
                    parts.append(types.Part(text=m["text"]))
                for tc in m["tool_calls"]:
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=tc["name"], args=tc["arguments"],
                    )))
                contents.append(types.Content(role="model", parts=parts))
            elif m["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))
            else:
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system, tools=[tool_defs], max_output_tokens=4096,
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        text_parts = [p.text for p in resp.candidates[0].content.parts if hasattr(p, "text") and p.text]
        tool_calls = []
        for p in resp.candidates[0].content.parts:
            if hasattr(p, "function_call") and p.function_call:
                fc = p.function_call
                tool_calls.append(ToolCall(
                    id=f"gemini_{int(time.time()*1000)}", name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                ))
        return AgentResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous ML researcher optimizing a neural network training script.

## Goal
Minimize val_bpb (validation bits per byte) — lower is better. Each training run has a fixed 5-minute time budget. You modify train.py: architecture, optimizer, hyperparameters, batch size, model size — everything is fair game.

## Workflow
1. Check current config and experiment history
2. Decide what to try next (one change at a time for attribution)
3. Make the change via set_hyperparams or edit_code
4. Run the experiment (takes ~5-7 min including startup)
5. Compare val_bpb to the current best
6. If improved: keep. If equal or worse: discard.
7. Repeat forever.

## Tools
- get_config: see current hyperparameters
- set_hyperparams: tweak values (pass Python expressions as strings)
- edit_code: replace entire sections for architectural changes
- run_experiment: execute training, get metrics
- get_history: see all past experiments
- keep: commit improvement + log to results.tsv
- discard: revert failed experiment + log to results.tsv
- read_code: inspect specific lines of train.py

## Strategy
- Time-constrained optimization: more gradient steps in 5 min often beats larger models/batches
- Regularization is often under-explored (weight decay on embeddings, value embeddings, lm_head)
- Initialization scale matters — narrow optima are real
- Try one thing at a time so you know what worked
- If 5+ experiments fail to improve, try something more radical (architectural change, not just hyperparameter tuning)
- Simpler is better: a tiny improvement that adds complexity is not worth it

## Constraints
- Only modify train.py — prepare.py is read-only
- No new packages or dependencies
- Peak VRAM should stay reasonable (some increase is OK for meaningful val_bpb gains)
- NEVER stop experimenting. Loop until interrupted.

## Important
After each experiment, you MUST call either keep() or discard(). Do not leave changes uncommitted.
When you first start, check get_history to see what's been tried, and get_config for current state.
If no experiments exist yet, run the baseline first (no changes) to establish a reference point.
"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def make_tool_list():
    """Build tool definitions in provider-agnostic format."""
    return [
        {"name": name, "description": spec["description"], "parameters": spec["parameters"]}
        for name, spec in TOOLS.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML experiment agent")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-experiments", type=int, default=0, help="Stop after N experiments (0 = unlimited)")
    parser.add_argument("--tag", type=str, default=None, help="Git branch tag (creates autoresearch/<tag>)")
    args = parser.parse_args()

    # Branch setup
    if args.tag:
        branch = f"autoresearch/{args.tag}"
        subprocess.run(["git", "checkout", "-b", branch], cwd=PROJECT_DIR, capture_output=True)
        print(f"Branch: {branch}")

    # Provider setup
    if args.provider == "anthropic":
        provider = AnthropicProvider(args.model)
    elif args.provider == "openai":
        provider = OpenAIProvider(args.model)
    elif args.provider == "gemini":
        provider = GeminiProvider(args.model)
    print(f"Provider: {args.provider} / {args.model}")

    tools = make_tool_list()
    messages = [{"role": "user", "content": "Begin experimenting. Start by checking the current config and history."}]
    experiment_count = 0

    print("\n--- Agent loop started (Ctrl+C to stop) ---\n")

    while True:
        try:
            response = provider.chat(SYSTEM_PROMPT, messages, tools)
        except KeyboardInterrupt:
            print("\n--- Interrupted ---")
            break
        except Exception as e:
            print(f"API error: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        # Print any text from the model
        if response.text:
            print(f"\n[agent] {response.text}\n")

        # Build assistant message for history
        assistant_msg = {"role": "assistant", "text": response.text, "content": response.text or ""}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # Execute tool calls
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  -> {tc.name}({json.dumps(tc.arguments, indent=None)[:200]})")
                result = execute_tool(tc.name, tc.arguments)
                print(f"  <- {result[:200]}{'...' if len(result) > 200 else ''}")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": tc.name, "content": result,
                })

                # Track experiment count
                if tc.name in ("keep", "discard"):
                    experiment_count += 1
                    print(f"\n  === Experiment #{experiment_count} complete ===\n")
                    if args.max_experiments and experiment_count >= args.max_experiments:
                        print(f"Reached --max-experiments {args.max_experiments}. Stopping.")
                        return
        else:
            # Model didn't call a tool — nudge it
            messages.append({"role": "user", "content": "Continue experimenting. Use the tools to make changes and run experiments."})

        # Context window management: compress after 60 messages
        if len(messages) > 60:
            print("  [compressing context]")
            history = tool_get_history()
            config = tool_get_config()
            messages = [
                {"role": "user", "content": f"Context reset. Here is the current state:\n\nExperiment history:\n{history}\n\nCurrent config:\n{config}\n\nContinue experimenting. Build on what worked, avoid what didn't."},
            ]


if __name__ == "__main__":
    main()
