#!/usr/bin/env python3
"""
OpenCastor AutoResearcher orchestrator.

Loop:
  1. Ask on-device model (gemma3:1b via Ollama) to draft an improvement
  2. Ask Claude Haiku to review the draft
  3. If approved: apply, run metric, keep or revert
  4. Log to results.tsv
  5. Repeat
"""

import os
import re
import shlex
import subprocess
import time
import textwrap
from datetime import datetime
from pathlib import Path

import google.auth
import google.auth.transport.requests
from google import genai
import ollama

# Authenticate using Application Default Credentials (gcloud auth application-default login)
_creds, _project = google.auth.default()
_auth_req = google.auth.transport.requests.Request()
_creds.refresh(_auth_req)
_genai_client = genai.Client(
    vertexai=True,
    project=_project,
    location="us-central1",
    credentials=_creds,
)

OPENCASTOR_REPO = Path(os.environ["OPENCASTOR_REPO_PATH"])
TODAY_TRACK = os.environ.get("TODAY_TRACK", "A")
REVIEWER_MODEL = "gemini-2.0-flash"
DRAFT_MODEL = "gemma3:1b"
RESULTS_TSV = Path(__file__).parent / "results.tsv"

FORBIDDEN_FILES = {
    "castor/api.py",
    "castor/safety.py",
    "castor/auth.py",
}

def git(cmd: str, cwd: Path = OPENCASTOR_REPO) -> str:
    """Run a git command in cwd and return stdout."""
    result = subprocess.run(
        ["git"] + shlex.split(cmd), cwd=cwd, capture_output=True, text=True
    )
    return result.stdout.strip()


def run_cmd(cmd: str, cwd: Path = OPENCASTOR_REPO, timeout: int = 300) -> tuple[int, str]:
    """Run a shell command; return (exit_code, combined_output)."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


def get_metric() -> int:
    """Return the current improvement metric for the active track."""
    if TODAY_TRACK == "A":
        _, out = run_cmd("python -m pytest --co -q 2>/dev/null | grep -E '^[0-9]+ test'")
        try:
            return int(out.split()[0])
        except (IndexError, ValueError):
            return 0
    elif TODAY_TRACK == "B":
        _, out = run_cmd(
            'python3 -c "'
            "import ast,os; missing=[];"
            "[missing.extend([n.name for n in ast.walk(ast.parse(open(os.path.join(r,f)).read()))"
            " if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))"
            " and not ast.get_docstring(n)])"
            " for r,d,files in os.walk('castor') for f in files if f.endswith('.py')];"
            'print(len(missing))"'
        )
        try:
            return int(out.strip())
        except ValueError:
            return 9999
    else:  # Track C
        _, out = run_cmd("ls config/presets/*.rcan.yaml 2>/dev/null | wc -l")
        try:
            return int(out.strip())
        except ValueError:
            return 0


def metric_improved(before: int, after: int) -> bool:
    """Return True if the metric moved in the right direction."""
    if TODAY_TRACK == "A":
        return after > before
    elif TODAY_TRACK == "B":
        return after < before
    else:
        return after > before


def list_candidate_files() -> list[str]:
    """Return source files to target for the active track."""
    if TODAY_TRACK == "A":
        _, out = run_cmd("find castor -name '*.py' -not -name '__init__.py' | sort")
        return [f for f in out.splitlines() if f not in FORBIDDEN_FILES][:20]
    elif TODAY_TRACK == "B":
        _, out = run_cmd(
            'python3 -c "'
            "import ast,os; hits={};"
            "[hits.update({os.path.join(r,f): sum(1 for n in ast.walk(ast.parse(open(os.path.join(r,f)).read()))"
            " if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))"
            " and not ast.get_docstring(n))})"
            " for r,d,files in os.walk('castor') for f in files if f.endswith('.py')];"
            '[print(k,v) for k,v in sorted(hits.items(),key=lambda x:-x[1])[:10]]"'
        )
        return [line.split()[0] for line in out.splitlines() if line.strip()]
    else:
        _, out = run_cmd(
            "ls config/presets/*.rcan.yaml | xargs -I{} basename {} .rcan.yaml | sort"
        )
        return out.splitlines()


def read_file(path: str) -> str:
    """Read a file from the OpenCastor repo, capped at 6000 chars."""
    full = OPENCASTOR_REPO / path
    try:
        return full.read_text(encoding="utf-8")[:6000]
    except Exception:
        return ""


def draft_improvement(file_path: str, file_content: str, program: str) -> str:
    """Use on-device Ollama model to draft an improvement."""
    track_prompt = {
        "A": (
            f"Write new pytest tests for untested functions in {file_path}. "
            "Use pytest conventions. Import from castor. Return ONLY the Python test code."
        ),
        "B": (
            f"Add Google-style docstrings to all functions/classes missing them in {file_path}. "
            "Return the COMPLETE modified Python file."
        ),
        "C": (
            f"Generate a new RCAN config preset YAML for a robot hardware combination "
            f"NOT already in this list: {file_path}. "
            "Required fields: rcan_version, metadata.robot_name, agent.provider, agent.model, drivers (non-empty list). "
            "Return ONLY valid YAML."
        ),
    }
    prompt = textwrap.dedent(f"""
        {program[:2000]}

        Task: {track_prompt[TODAY_TRACK]}

        File content:
        {file_content}

        Output only code/YAML, no explanation.
    """)
    response = ollama.chat(
        model=DRAFT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def haiku_review(draft: str, file_path: str) -> tuple[bool, str]:
    """Ask Gemini to review the draft via Google ADC. Returns (approved, reason)."""
    track_name = {"A": "pytest tests", "B": "docstrings", "C": "RCAN preset"}[TODAY_TRACK]
    prompt = textwrap.dedent(f"""
        You are reviewing a proposed {track_name} change for the OpenCastor robot runtime repo.
        File: {file_path}

        Proposed change:
        {draft[:3000]}

        Review rules:
        - Must not touch forbidden files (api.py, safety.py, auth.py)
        - Track A (tests): must use pytest, import real modules, test real behavior — not trivial stubs
        - Track B (docstrings): must be Google-style, accurate, not hallucinated
        - Track C (RCAN preset): must have rcan_version, metadata.robot_name, agent.model, non-empty drivers list
        - No hallucinated imports or functions that don't exist in the codebase

        Reply with exactly one of:
        PASS - <one sentence explaining why it's good>
        FAIL - <one sentence explaining the problem>
    """)
    response = _genai_client.models.generate_content(model=REVIEWER_MODEL, contents=prompt)
    text = response.text.strip()
    passed = text.upper().startswith("PASS")
    return passed, text


def apply_change(file_path: str, content: str) -> Path:
    """Write the change to the OpenCastor repo. Returns the path written."""
    if TODAY_TRACK == "A":
        test_name = Path(file_path).stem
        dest = OPENCASTOR_REPO / "tests" / f"test_auto_{test_name}.py"
        dest.write_text(content, encoding="utf-8")
        return dest
    elif TODAY_TRACK == "B":
        dest = OPENCASTOR_REPO / file_path
        dest.write_text(content, encoding="utf-8")
        return dest
    else:
        name_match = re.search(r"robot_name:\s*(\S+)", content)
        preset_name = (
            name_match.group(1).replace(" ", "_") if name_match else f"auto_{int(time.time())}"
        )
        dest = OPENCASTOR_REPO / "config" / "presets" / f"{preset_name}.rcan.yaml"
        dest.write_text(content, encoding="utf-8")
        return dest


def revert_change(file_path: str, written_path: Path | None = None) -> None:
    """Undo the applied change."""
    if TODAY_TRACK == "A" and written_path and written_path.exists():
        written_path.unlink()
    elif TODAY_TRACK == "B":
        git(f"checkout -- {file_path}")
    else:
        if written_path and written_path.exists():
            written_path.unlink()


def run_verification() -> tuple[int, str]:
    """Run pytest; return (exit_code, last 5 lines of output)."""
    code, out = run_cmd("python -m pytest tests/ -x -q --tb=no", timeout=300)
    last_lines = "\n".join(out.splitlines()[-5:])
    return code, last_lines


def log_result(commit: str, before: int, after: int, status: str, desc: str) -> None:
    """Append one row to results.tsv."""
    row = f"{commit}\t{before}\t{after}\t{after - before}\t{status}\t{desc}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    print(row.strip())


def ensure_results_tsv() -> None:
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "commit\tmetric_before\tmetric_after\tdelta\tstatus\tdescription\n"
        )


def main() -> None:
    """Main experiment loop — runs until killed."""
    program = (Path(__file__).parent / "program.md").read_text()
    ensure_results_tsv()

    print(f"[autoresearch] Starting. Track={TODAY_TRACK} Repo={OPENCASTOR_REPO}")
    exp = 0

    while True:
        exp += 1
        print(f"\n[exp {exp}] {datetime.now().strftime('%H:%M:%S')}")

        candidates = list_candidate_files()
        if not candidates:
            print("  No candidates found, sleeping 60s")
            time.sleep(60)
            continue

        # Rotate through candidates so we don't hammer the same file
        target = candidates[exp % len(candidates)]
        content = read_file(target) if TODAY_TRACK != "C" else "\n".join(candidates)

        # Draft
        print(f"  Drafting improvement for {target} ...")
        try:
            draft = draft_improvement(target, content, program)
        except Exception as e:
            print(f"  Draft failed: {e}")
            continue

        # Review
        print("  Haiku reviewing ...")
        try:
            approved, reason = haiku_review(draft, target)
        except Exception as e:
            print(f"  Review failed: {e}")
            continue

        if not approved:
            print(f"  REJECTED: {reason}")
            log_result("none", 0, 0, "rejected", f"{target}: {reason[:60]}")
            continue

        print(f"  APPROVED: {reason}")

        # Apply and measure
        before = get_metric()
        written_path = apply_change(target, draft)

        # Verify tests still pass
        exit_code, verify_out = run_verification()
        if exit_code != 0:
            print(f"  Tests FAILED — reverting\n  {verify_out}")
            revert_change(target, written_path)
            log_result("none", before, before, "crash", f"{target}: tests failed")
            continue

        after = get_metric()

        if metric_improved(before, after):
            git("add -A")
            short = Path(target).name
            git(f'commit -m "auto({TODAY_TRACK.lower()}): improve {short} [{before}->{after}]"')
            commit = git("rev-parse --short HEAD")
            log_result(commit, before, after, "keep", target)
            print(f"  KEPT delta={after - before}")
        else:
            revert_change(target, written_path)
            log_result("none", before, after, "discard", f"{target}: no improvement")
            print(f"  DISCARDED (no improvement)")


if __name__ == "__main__":
    main()
