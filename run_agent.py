#!/usr/bin/env python3
"""
OpenCastor AutoResearcher orchestrator.

Loop:
  1. Pick ONE specific target (function/class for B, test file for A, etc.)
  2. Ask on-device model to draft an improvement for that ONE target
  3. Ask Gemini to review the draft
  4. If approved: apply, run metric, keep or revert
  5. Log to results.tsv
  6. Repeat

Tracks:
  A = Write new pytest tests for ONE untested function
  B = Add a Google-style docstring to ONE function/class missing it
  C = Generate ONE new RCAN preset YAML for a hardware combination not yet covered
  D = Improve ONE failing skill eval test case in castor/skills/builtin/
  E = Write ONE new pytest test for harness/P66 code paths
  F = Mine trajectory DB for patterns (read-only, runs 2-4am)
"""

import ast
import os
import re
import shlex
import subprocess
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import google.auth
import google.auth.transport.requests
from google import genai
import ollama

# ── Auth ──────────────────────────────────────────────────────────────────────

_creds, _project = google.auth.default()
_auth_req = google.auth.transport.requests.Request()
_creds.refresh(_auth_req)
if not _project:
    import json as _json
    _adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    if _adc.exists():
        _project = _json.loads(_adc.read_text()).get("quota_project_id")

_genai_client = genai.Client(
    vertexai=True, project=_project, location="us-central1", credentials=_creds,
)

# ── Config ────────────────────────────────────────────────────────────────────

OPENCASTOR_REPO = Path(os.environ["OPENCASTOR_REPO_PATH"])
TODAY_TRACK = os.environ.get("TODAY_TRACK", "A")
REVIEWER_MODEL = "gemini-2.0-flash"
# REVIEWER=rcan → route review to Alex via RCAN send_rcan_message
# REVIEWER=gemini (default) → local Gemini ADC
# REVIEWER_RRN → Alex's RRN (default RRN-000000000005)
REVIEWER = os.environ.get("REVIEWER", "gemini")
REVIEWER_RRN = os.environ.get("REVIEWER_RRN", "RRN-000000000005")
REVIEWER_URL = os.environ.get("REVIEWER_URL", "http://alex.local:8000")
# Model priority: qwen2.5-coder:7b > qwen2.5-coder:3b > gemma3:4b > gemma3:1b
_available = {m.model.split(":")[0] for m in ollama.list().models}
_avail_full = {m.model for m in ollama.list().models}

def _pick_model() -> str:
    for candidate in [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:3b",
        "gemma3:4b",
        "gemma3:1b",
    ]:
        if candidate in _avail_full or candidate.split(":")[0] in _available:
            return candidate
    return "gemma3:1b"

DRAFT_MODEL = _pick_model()
RESULTS_TSV = Path(__file__).parent / "results.tsv"

FORBIDDEN_FILES = {
    "castor/api.py", "castor/safety.py", "castor/auth.py",
}

print(f"[autoresearch] Draft model: {DRAFT_MODEL}")

# ── Git / shell helpers ───────────────────────────────────────────────────────

def git(cmd: str, cwd: Path = OPENCASTOR_REPO) -> str:
    result = subprocess.run(["git"] + shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip()


def run_cmd(cmd: str, cwd: Path = OPENCASTOR_REPO, timeout: int = 300) -> tuple[int, str]:
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr

# ── Target pickers ────────────────────────────────────────────────────────────

def pick_target_A() -> Optional[tuple[str, str, str]]:
    """Pick ONE untested function. Returns (file_path, func_name, func_source) or None."""
    _, out = run_cmd("find castor -name '*.py' -not -name '__init__.py' | sort")
    files = [f for f in out.splitlines() if f not in FORBIDDEN_FILES]

    # Find functions without a corresponding test
    _, covered = run_cmd("grep -rh 'def test_' tests/ | grep -oP '(?<=def test_)\\w+'")
    tested_names = set(covered.splitlines())

    for fpath in files:
        full = OPENCASTOR_REPO / fpath
        try:
            tree = ast.parse(full.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_") or name in tested_names:
                    continue
                # Get source lines
                try:
                    src_lines = full.read_text().splitlines()
                    end = node.end_lineno or (node.lineno + 20)
                    func_src = "\n".join(src_lines[node.lineno - 1: min(end, node.lineno + 40)])
                    if len(func_src) < 20:
                        continue
                    return fpath, name, func_src
                except Exception:
                    continue
    return None


def pick_target_B() -> Optional[tuple[str, str, str]]:
    """Pick ONE function/class missing a docstring. Returns (file_path, name, func_source)."""
    _, out = run_cmd("find castor -name '*.py' -not -name '__init__.py' | sort")
    files = [f for f in out.splitlines() if f not in FORBIDDEN_FILES]

    for fpath in files:
        full = OPENCASTOR_REPO / fpath
        try:
            src = full.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node) and not node.name.startswith("__"):
                    try:
                        src_lines = src.splitlines()
                        end = getattr(node, "end_lineno", node.lineno + 20)
                        func_src = "\n".join(src_lines[node.lineno - 1: min(end, node.lineno + 30)])
                        if len(func_src) < 10:
                            continue
                        return fpath, node.name, func_src
                    except Exception:
                        continue
    return None


def pick_target_C() -> Optional[tuple[str, str, str]]:
    """Pick a hardware combination not yet in presets. Returns (hint, '', '')."""
    _, existing = run_cmd("ls config/presets/*.rcan.yaml 2>/dev/null | xargs -I{} basename {} .rcan.yaml")
    existing_names = set(existing.splitlines())
    candidates = [
        "raspberry-pi-5-webcam",
        "jetson-nano-realsense",
        "raspberry-pi-zero-ultrasonic",
        "arduino-mega-sonar",
        "beaglebone-black-camera",
        "orange-pi-5-depth-camera",
        "raspberry-pi-4-oak-d-lite",
        "nvidia-jetson-orin-stereo",
        "rockchip-rk3588-lidar",
    ]
    for c in candidates:
        if c not in existing_names:
            return c, "", ""
    return f"custom-robot-{int(time.time())}", "", ""


def pick_target_D() -> Optional[tuple[str, str, str]]:
    """Pick a failing skill eval case. Returns (skill_name, case_id, case_json) or None."""
    from pathlib import Path as _P
    skills_dir = OPENCASTOR_REPO / "castor" / "skills" / "builtin"
    if not skills_dir.exists():
        return None
    for skill_dir in sorted(skills_dir.iterdir()):
        eval_json = skill_dir / "tests" / "eval.json"
        if eval_json.exists():
            import json
            cases = json.loads(eval_json.read_text())
            # Return first case that needs better documentation
            for case in cases:
                if case.get("should_trigger") and not case.get("expected_checks"):
                    return skill_dir.name, case["id"], json.dumps(case, indent=2)
    # Just return first skill for improvement
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            return skill_dir.name, "instructions", skill_md.read_text()[:2000]
    return None


def pick_target_E() -> Optional[tuple[str, str, str]]:
    """Pick an untested harness/P66 function. Returns (file_path, func_name, func_source)."""
    harness_files = [
        "castor/harness.py",
        "castor/context.py",
        "castor/trajectory.py",
        "castor/dual_model.py",
        "castor/eval_harness.py",
        "castor/skills/loader.py",
        "castor/agent_tools.py",
    ]
    _, covered = run_cmd("grep -rh 'def test_' tests/ | grep -oP '(?<=def test_)\\w+'")
    tested_names = set(covered.splitlines())

    for fpath in harness_files:
        full = OPENCASTOR_REPO / fpath
        if not full.exists():
            continue
        try:
            src = full.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                if name.startswith("_") or name in tested_names:
                    continue
                try:
                    src_lines = src.splitlines()
                    end = getattr(node, "end_lineno", node.lineno + 20)
                    func_src = "\n".join(src_lines[node.lineno - 1: min(end, node.lineno + 40)])
                    if len(func_src) < 20:
                        continue
                    return fpath, name, func_src
                except Exception:
                    continue
    return None


def pick_target_F() -> Optional[tuple[str, str, str]]:
    """Mine trajectory DB for patterns. Read-only. Returns (pattern_type, summary, detail)."""
    import sqlite3, json
    db_path = Path.home() / ".config" / "opencastor" / "trajectories.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        # Find most common tool sequences
        rows = conn.execute(
            "SELECT tool_calls_json, scope, skill_triggered FROM trajectories "
            "WHERE error IS NULL ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
        tool_counts: dict = {}
        for row in rows:
            try:
                calls = json.loads(row[0] or "[]")
                seq = tuple(c.get("tool") for c in calls if c.get("tool"))
                if seq:
                    tool_counts[seq] = tool_counts.get(seq, 0) + 1
            except Exception:
                continue
        if not tool_counts:
            return None
        top = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
        summary = "; ".join(f"{'+'.join(seq)}×{count}" for seq, count in top)
        return "tool_sequence", summary, str(top)
    finally:
        conn.close()

# ── Metric ────────────────────────────────────────────────────────────────────

def get_metric() -> int:
    if TODAY_TRACK == "A":
        _, out = run_cmd("python -m pytest --co -q 2>/dev/null | tail -1")
        m = re.search(r"(\d+)\s+test", out)
        return int(m.group(1)) if m else 0
    elif TODAY_TRACK == "B":
        _, out = run_cmd(
            "python3 -c \""
            "import ast,os; n=0\n"
            "[n := n + sum(1 for x in ast.walk(ast.parse(open(os.path.join(r,f)).read()))"
            " if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))"
            " and not ast.get_docstring(x))"
            " for r,d,fs in os.walk('castor') for f in fs if f.endswith('.py')]\n"
            "print(n)\""
        )
        try:
            return int(out.strip())
        except ValueError:
            return 9999
    elif TODAY_TRACK == "C":
        _, out = run_cmd("ls config/presets/*.rcan.yaml 2>/dev/null | wc -l")
        return int(out.strip()) if out.strip().isdigit() else 0
    elif TODAY_TRACK == "D":
        # Proxy: count skill eval tests with expected_checks
        _, out = run_cmd("grep -r 'expected_checks' castor/skills/builtin/ | grep -v '\\[\\]' | wc -l")
        return int(out.strip()) if out.strip().isdigit() else 0
    elif TODAY_TRACK == "E":
        _, out = run_cmd("grep -c 'def test_' tests/test_harness.py tests/test_p66_harness.py tests/test_trajectory.py tests/test_context.py tests/test_dual_model.py 2>/dev/null | awk -F: '{s+=$2}END{print s}'")
        return int(out.strip()) if out.strip().isdigit() else 0
    else:  # F — track F is read-only, no metric
        return 0


def metric_improved(before: int, after: int) -> bool:
    if TODAY_TRACK == "B":
        return after < before          # fewer missing docstrings = better
    elif TODAY_TRACK == "F":
        return False                   # F never writes code
    else:
        return after > before          # more tests/presets = better

# ── Draft ─────────────────────────────────────────────────────────────────────

def draft_improvement(target: str, name: str, content: str, program: str) -> str:
    """Use on-device Ollama to draft a targeted improvement."""
    prompts = {
        "A": f"""Write ONE pytest test function for the Python function shown below.

File: {target}
Function to test: {name}

```python
{content[:600]}
```

STRICT OUTPUT RULES:
1. Return ONLY the test function — no imports, no class, no explanation
2. First line must be: def test_{name}_<behavior>(...)
3. Import inside the function body: from castor.{Path(target).stem} import {name}
4. Use unittest.mock.patch or MagicMock for any I/O or external deps
5. Test ONE specific behavior (happy path or a known edge case)
6. No code fences, no markdown

Return ONLY the function:""",

        "B": f"""Add a Google-style docstring to exactly this one Python function. It currently has NO docstring.

Function (from {target}):
```python
{content[:500]}
```

Google-style example:
    def fn(x: int) -> str:
        \"\"\"One-line summary.

        Args:
            x: Description.

        Returns:
            Description.
        \"\"\"

STRICT OUTPUT RULES:
1. Return ONLY the complete function definition with the docstring added
2. Insert docstring as the FIRST line inside the function body (after def line)
3. Do NOT change any logic, only add the docstring
4. No code fences, no markdown, no explanation
5. If the function has Args/Returns, document them

Return ONLY the function:""",

        "C": f"""Generate a RCAN config YAML preset for a robot with this hardware: {target}

Required fields: rcan_version, metadata.robot_name, agent.provider, agent.model, drivers list.
Return ONLY valid YAML. No explanation.""",

        "D": f"""Improve this OpenCastor robot skill SKILL.md body to be more specific and actionable.

Skill: {target}
Current content:
{content}

Rules:
- Keep the YAML frontmatter unchanged (between --- markers)
- Make the step-by-step instructions more specific
- Add concrete examples with tool calls shown
- Return the complete SKILL.md including frontmatter
- No explanation""",

        "E": f"""Write ONE pytest test for this harness/P66 function from OpenCastor.

File: {target}
Function: {name}

```python
{content[:600]}
```

STRICT OUTPUT RULES:
1. Return ONLY the test function — no imports, no class, no explanation
2. First line must be: def test_{name}_<behavior>(...) or async def test_{name}_<behavior>(...)
3. Import inside the function body: from castor.{Path(target).stem} import {name}
4. If function touches ESTOP/physical tools/P66, add: assert result is not None (safety invariant check)
5. Use @pytest.mark.asyncio decorator for async functions (add it on the line before def)
6. Mock providers: from unittest.mock import MagicMock, patch
7. No code fences, no markdown

Return ONLY the function:""",

        "F": "readonly",
    }

    prompt = prompts.get(TODAY_TRACK, "")
    if not prompt or prompt == "readonly":
        return ""

    response = ollama.chat(
        model=DRAFT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_predict": 600},
    )
    return response["message"]["content"]

# ── Review ────────────────────────────────────────────────────────────────────

def _review_prompt(draft: str, target: str, name: str) -> str:
    """Build the review prompt string."""
    track_rules = {
        "A": "pytest test: must import real castor modules, test real behavior, use correct pytest syntax",
        "B": "docstring: must be Google-style (one-line summary + Args/Returns), accurate, no hallucinations",
        "C": "RCAN YAML: must have rcan_version, metadata.robot_name, agent.provider, agent.model, non-empty drivers",
        "D": "SKILL.md: must keep frontmatter intact, improve step-by-step instructions",
        "E": "harness test: must include P66 assertion if function touches physical tools or ESTOP",
    }
    rules = track_rules.get(TODAY_TRACK, "code quality")
    return f"""Review this proposed change for the OpenCastor robot runtime.

Target: {target} / {name}
Track rule: {rules}

Proposed change:
{draft[:2000]}

Reply with EXACTLY one line:
PASS - <one sentence why it's good>
FAIL - <one sentence what's wrong>"""


def _review_via_rcan(prompt: str) -> tuple[bool, str]:
    """Send review request to Alex via RCAN HTTP API. Returns (approved, reason)."""
    import urllib.error
    import urllib.request

    payload = json.dumps({
        "cmd": "review",
        "message": prompt,
        "scope": "chat",
        "loa": 1,
        "max_iterations": 1,
    }).encode()

    req = urllib.request.Request(
        f"{REVIEWER_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json", "X-RCAN-Scope": "chat"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = (data.get("response") or data.get("message") or "").strip()
            if not text:
                raise ValueError("empty response from Alex")
            passed = text.upper().startswith("PASS")
            print(f"  [review:alex] {text[:80]}")
            return passed, text
    except (urllib.error.URLError, OSError) as e:
        print(f"  [review:alex] unreachable ({e}) — falling back to Gemini ADC")
        raise


def _review_via_gemini(prompt: str) -> tuple[bool, str]:
    """Review via local Gemini ADC."""
    response = _genai_client.models.generate_content(model=REVIEWER_MODEL, contents=prompt)
    text = response.text.strip()
    passed = text.upper().startswith("PASS")
    return passed, text


def review_draft(draft: str, target: str, name: str) -> tuple[bool, str]:
    """Review the draft via Alex (RCAN) or local Gemini ADC.

    Routes based on REVIEWER env var:
      REVIEWER=rcan   → Alex via RCAN HTTP API, fallback to Gemini on error
      REVIEWER=gemini → local Gemini ADC (default)
    """
    prompt = _review_prompt(draft, target, name)

    if REVIEWER == "rcan":
        try:
            return _review_via_rcan(prompt)
        except Exception:
            print("  [review] RCAN review failed — using Gemini ADC fallback")

    return _review_via_gemini(prompt)

# ── Apply / revert ────────────────────────────────────────────────────────────

def apply_change(target: str, name: str, content: str) -> Optional[Path]:
    """Write the change. Returns path written or None."""
    content = _strip_code_fences(content)

    if TODAY_TRACK == "A":
        stem = Path(target).stem
        dest = OPENCASTOR_REPO / "tests" / f"test_auto_{stem}_{name}.py"
        dest.write_text(content, encoding="utf-8")
        return dest
    elif TODAY_TRACK == "B":
        # Patch just the function into the file (replace old definition)
        dest = OPENCASTOR_REPO / target
        orig = dest.read_text()
        patched = _patch_function(orig, name, content)
        if patched == orig:
            dest.write_text(content, encoding="utf-8")  # fallback: write full
        else:
            dest.write_text(patched, encoding="utf-8")
        return dest
    elif TODAY_TRACK == "C":
        name_match = re.search(r"robot_name:\s*['\"]?([^'\"\n]+)", content)
        preset_name = name_match.group(1).replace(" ", "_") if name_match else target
        dest = OPENCASTOR_REPO / "config" / "presets" / f"{preset_name}.rcan.yaml"
        dest.write_text(content, encoding="utf-8")
        return dest
    elif TODAY_TRACK == "D":
        dest = OPENCASTOR_REPO / "castor" / "skills" / "builtin" / target / "SKILL.md"
        if dest.exists():
            dest.write_text(content, encoding="utf-8")
            return dest
        return None
    elif TODAY_TRACK == "E":
        dest = OPENCASTOR_REPO / "tests" / "test_auto_harness.py"
        # Append to existing file or create
        if dest.exists():
            existing = dest.read_text()
            dest.write_text(existing + "\n\n" + content)
        else:
            dest.write_text(
                "\"\"\"Auto-generated harness tests.\"\"\"\nimport pytest\nfrom unittest.mock import MagicMock\n\n"
                + content
            )
        return dest
    return None


def revert_change(target: str, written_path: Optional[Path]) -> None:
    if written_path and written_path.exists():
        if TODAY_TRACK in ("A", "C"):
            written_path.unlink()
        elif TODAY_TRACK == "B":
            git(f"checkout -- {target}")
        elif TODAY_TRACK in ("D", "E"):
            if TODAY_TRACK == "D":
                git(f"checkout -- castor/skills/builtin/{target}/SKILL.md")
            else:
                written_path.unlink(missing_ok=True)


def _strip_code_fences(text: str) -> str:
    """Remove ```python / ``` / ```yaml fences from model output."""
    text = re.sub(r"^```[\w]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _patch_function(source: str, func_name: str, new_func: str) -> str:
    """Replace an existing function definition in source with new_func."""
    pattern = rf"(def {re.escape(func_name)}\b.*?)(?=\ndef |\nclass |\Z)"
    m = re.search(pattern, source, re.DOTALL)
    if m:
        return source[: m.start()] + new_func.strip() + source[m.end():]
    return source

# ── Metric / verify ───────────────────────────────────────────────────────────

def run_verification() -> tuple[int, str]:
    code, out = run_cmd("python -m pytest tests/ -x -q --tb=no", timeout=300)
    last = "\n".join(out.splitlines()[-5:])
    return code, last

# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(commit: str, before: int, after: int, status: str, desc: str) -> None:
    row = f"{commit}\t{before}\t{after}\t{after - before}\t{status}\t{desc[:80]}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    print(row.strip())


def ensure_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tmetric_before\tmetric_after\tdelta\tstatus\tdescription\n")

# ── Track F: trajectory mining (read-only) ────────────────────────────────────

def run_track_f() -> None:
    """Mine trajectory DB and append patterns to docs/trajectory-patterns.md."""
    result = pick_target_F()
    if not result:
        print("  [F] No trajectory data found yet.")
        return
    pattern_type, summary, detail = result
    patterns_file = OPENCASTOR_REPO / "docs" / "trajectory-patterns.md"
    patterns_file.parent.mkdir(exist_ok=True)
    if not patterns_file.exists():
        patterns_file.write_text("# Trajectory Patterns\n\nAuto-generated from harness run logs.\n\n")
    timestamp = datetime.now().strftime("%Y-%m-%d")
    entry = f"\n## {timestamp} — {pattern_type}\n\nTop tool sequences: {summary}\n\nDetail: {detail[:300]}\n"
    patterns_file.write_text(patterns_file.read_text() + entry)
    git("add docs/trajectory-patterns.md")
    git(f'commit -m "auto(f): trajectory pattern update {timestamp}"')
    print(f"  [F] Pattern logged: {summary[:80]}")

# ── Main loop ─────────────────────────────────────────────────────────────────

TARGET_PICKERS = {
    "A": pick_target_A,
    "B": pick_target_B,
    "C": pick_target_C,
    "D": pick_target_D,
    "E": pick_target_E,
    "F": pick_target_F,
}


def main() -> None:
    program = (Path(__file__).parent / "program.md").read_text()
    ensure_results_tsv()
    print(f"[autoresearch] Starting. Track={TODAY_TRACK} Model={DRAFT_MODEL} Repo={OPENCASTOR_REPO}")

    if TODAY_TRACK == "F":
        run_track_f()
        return

    picker = TARGET_PICKERS[TODAY_TRACK]
    exp = 0

    while True:
        exp += 1
        print(f"\n[exp {exp}] {datetime.now().strftime('%H:%M:%S')} track={TODAY_TRACK}")

        target_info = picker()
        if target_info is None:
            print("  No targets found, sleeping 60s")
            time.sleep(60)
            continue

        target, name, content = target_info
        print(f"  Target: {target} / {name or '-'}")

        # Draft
        try:
            draft = draft_improvement(target, name, content, program)
        except Exception as e:
            print(f"  Draft failed: {e}")
            continue
        if not draft.strip():
            print("  Empty draft, skipping")
            continue

        # Review
        try:
            approved, reason = review_draft(draft, target, name)
        except Exception as e:
            print(f"  Review failed: {e}")
            continue

        if not approved:
            print(f"  REJECTED: {reason}")
            log_result("none", 0, 0, "rejected", f"{target}/{name}: {reason[:60]}")
            continue

        print(f"  APPROVED: {reason[:80]}")

        # Apply and measure
        before = get_metric()
        written_path = apply_change(target, name, draft)
        if written_path is None:
            print("  Apply failed (no path returned)")
            continue

        # Verify tests pass
        exit_code, verify_out = run_verification()
        if exit_code != 0:
            print(f"  Tests FAILED — reverting\n  {verify_out}")
            revert_change(target, written_path)
            log_result("none", before, before, "crash", f"{target}/{name}: tests failed")
            continue

        after = get_metric()

        if metric_improved(before, after):
            git("add -A")
            short = f"{TODAY_TRACK}/{Path(target).name}/{name or 'item'}"
            git(f'commit -m "auto({TODAY_TRACK.lower()}): {short} [{before}->{after}]"')
            commit = git("rev-parse --short HEAD")
            log_result(commit, before, after, "keep", f"{target}/{name}")
            print(f"  KEPT delta={after - before}")
        else:
            revert_change(target, written_path)
            log_result("none", before, after, "discard", f"{target}/{name}: no improvement")
            print(f"  DISCARDED metric unchanged")


if __name__ == "__main__":
    main()
