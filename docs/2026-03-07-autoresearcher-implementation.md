# OpenCastor AutoResearcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a nightly cron job that forks karpathy/autoresearch and adapts it to autonomously improve OpenCastor (tests, docs, presets) using a hybrid on-device + cloud LLM loop, running 12am–6am each night.

**Architecture:** Fork autoresearch into `craigm26/opencastor-autoresearch`. Replace `train.py`/`program.md` with OpenCastor-specific equivalents. An orchestrator (`run_agent.py`) drives the loop: gemma3:1b (Ollama) drafts a change, Claude Haiku reviews it, if approved apply it to OpenCastor repo, run pytest/ruff/validate, keep or revert, log to `results.tsv`, repeat forever until killed at 6am by cron.

**Tech Stack:** Python 3.10+, Ollama (gemma3:1b), Anthropic SDK (Claude Haiku 4.5), subprocess, git CLI, gh CLI, cron

---

## Baseline Metrics (as of 2026-03-07)

- Tests collected: 4323
- RCAN presets: 18
- Missing docstrings: 1197

---

## Task 1: Fork the repo and scaffold

**Files:**
- Create: `/tmp/opencastor-autoresearch/` (cloned from fork)

**Step 1: Fork karpathy/autoresearch via gh CLI**

```bash
gh repo fork karpathy/autoresearch --clone --fork-name opencastor-autoresearch
mv autoresearch /tmp/opencastor-autoresearch
cd /tmp/opencastor-autoresearch
```

Expected: repo cloned at `/tmp/opencastor-autoresearch`

**Step 2: Remove ML-specific files we won't use**

```bash
cd /tmp/opencastor-autoresearch
rm -f train.py prepare.py analysis.ipynb progress.png uv.lock pyproject.toml
```

**Step 3: Create new pyproject.toml**

```toml
[project]
name = "opencastor-autoresearch"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",
    "ollama>=0.4.0",
    "gitpython>=3.1.0",
]
```

**Step 4: Install deps**

```bash
pip install anthropic ollama gitpython
```

**Step 5: Commit scaffold**

```bash
git add -A
git commit -m "chore: scaffold opencastor-autoresearch from karpathy/autoresearch fork"
```

---

## Task 2: Write program.md (agent instructions)

**Files:**
- Modify: `/tmp/opencastor-autoresearch/program.md`

**Step 1: Replace program.md content**

```markdown
# OpenCastor AutoResearcher

This agent autonomously improves the OpenCastor codebase overnight.

## Context

OpenCastor is a universal robot runtime (~3.10+ Python, 4323 tests, ruff 100-char).
Repo path: set in env var OPENCASTOR_REPO_PATH.
Conventions: PEP8, snake_case, type hints, lazy imports (HAS_X pattern), structured logging.
Test runner: `pytest tests/ -x -q`
Linter: `ruff check castor/`
RCAN validator: `castor validate --config <file>`

## Active Track

Determined at runtime by TODAY_TRACK env var:
- A = Tests: write new pytest tests for untested code paths
- B = Docs: add missing docstrings to castor/ source files
- C = Presets: generate new RCAN config presets for uncovered hardware

## Metrics

Track A: `pytest --co -q 2>/dev/null | grep -c "test session"` → actually use:
  `python -m pytest --co -q 2>/dev/null | tail -1` → parse test count; must increase

Track B: count missing docstrings:
  `python3 -c "import ast,os; missing=[]; [missing.extend([n.name for n in ast.walk(ast.parse(open(os.path.join(r,f)).read())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not ast.get_docstring(n)]) for r,d,files in os.walk('castor') for f in files if f.endswith('.py')]; print(len(missing))"`
  Must decrease.

Track C: `castor validate --config config/presets/<name>.rcan.yaml` exits 0.
  Count of presets must increase.

## The Loop

LOOP FOREVER (until killed):

1. Pick a target file to improve based on active track
2. Read the file
3. Draft an improvement (new tests / docstrings / preset)
4. The orchestrator sends your draft to Claude Haiku for review
5. If approved: write the file, run the metric command, check result
6. If metric improved: git commit with "keep"
7. If metric same/worse: git checkout -- <file> with "discard"
8. Log result to results.tsv
9. Repeat

## Constraints

- NEVER modify: castor/api.py, castor/safety.py, castor/auth.py, .env, any *_test.py that already passes
- NEVER install new packages
- All pytest runs must exit 0 (no regressions)
- Keep changes small and focused — one function, one docstring, one preset per experiment

## NEVER STOP

Once the loop begins, do NOT pause. The human is asleep. Run until killed.
```

**Step 2: Commit**

```bash
git add program.md
git commit -m "feat: add OpenCastor-specific program.md"
```

---

## Task 3: Write run_agent.py (the orchestrator)

**Files:**
- Create: `/tmp/opencastor-autoresearch/run_agent.py`

**Step 1: Write the orchestrator**

```python
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
import subprocess
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path

import anthropic
import ollama

OPENCASTOR_REPO = Path(os.environ["OPENCASTOR_REPO_PATH"])
TODAY_TRACK = os.environ.get("TODAY_TRACK", "A")
HAIKU_MODEL = "claude-haiku-4-5-20251001"
DRAFT_MODEL = "gemma3:1b"
RESULTS_TSV = Path("results.tsv")

FORBIDDEN_FILES = {
    "castor/api.py",
    "castor/safety.py",
    "castor/auth.py",
}

client = anthropic.Anthropic()


def git(cmd: str, cwd: Path = OPENCASTOR_REPO) -> str:
    result = subprocess.run(
        ["git"] + cmd.split(), cwd=cwd, capture_output=True, text=True
    )
    return result.stdout.strip()


def run_cmd(cmd: str, cwd: Path = OPENCASTOR_REPO, timeout: int = 300) -> tuple[int, str]:
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


def get_metric_before() -> int:
    if TODAY_TRACK == "A":
        _, out = run_cmd("python -m pytest --co -q 2>/dev/null | grep -E '^[0-9]+ test'")
        try:
            return int(out.split()[0])
        except (IndexError, ValueError):
            return 0
    elif TODAY_TRACK == "B":
        _, out = run_cmd(
            "python3 -c \""
            "import ast,os; missing=[];"
            "[missing.extend([n.name for n in ast.walk(ast.parse(open(os.path.join(r,f)).read()))"
            " if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))"
            " and not ast.get_docstring(n)])"
            " for r,d,files in os.walk('castor') for f in files if f.endswith('.py')];"
            "print(len(missing))\""
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
    if TODAY_TRACK == "A":
        return after > before  # more tests = better
    elif TODAY_TRACK == "B":
        return after < before  # fewer missing docstrings = better
    else:
        return after > before  # more presets = better


def list_candidate_files() -> list[str]:
    if TODAY_TRACK == "A":
        # Find source files with low test coverage hints
        _, out = run_cmd("find castor -name '*.py' -not -name '__init__.py' | sort")
        return [f for f in out.splitlines() if f not in FORBIDDEN_FILES][:20]
    elif TODAY_TRACK == "B":
        _, out = run_cmd(
            "python3 -c \""
            "import ast,os;"
            "hits={};"
            "[hits.update({os.path.join(r,f): sum(1 for n in ast.walk(ast.parse(open(os.path.join(r,f)).read()))"
            " if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))"
            " and not ast.get_docstring(n))})"
            " for r,d,files in os.walk('castor') for f in files if f.endswith('.py')];"
            "[print(k,v) for k,v in sorted(hits.items(),key=lambda x:-x[1])[:10]]\""
        )
        return [line.split()[0] for line in out.splitlines() if line.strip()]
    else:
        _, out = run_cmd("ls config/presets/*.rcan.yaml | xargs -I{} basename {} .rcan.yaml | sort")
        return out.splitlines()


def read_file(path: str) -> str:
    full = OPENCASTOR_REPO / path
    try:
        return full.read_text(encoding="utf-8")[:6000]  # cap context
    except Exception:
        return ""


def draft_improvement(file_path: str, file_content: str, program: str) -> str:
    track_prompt = {
        "A": f"Write new pytest tests for untested functions in {file_path}. Return ONLY the test code.",
        "B": f"Add Google-style docstrings to all functions/classes missing them in {file_path}. Return the COMPLETE modified file.",
        "C": f"Generate a new RCAN config preset for a hardware combination not in this list: {file_path}. Return ONLY valid YAML.",
    }
    prompt = textwrap.dedent(f"""
        {program}

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
    prompt = textwrap.dedent(f"""
        You are reviewing a code change for the OpenCastor robot runtime repo.
        File: {file_path}
        Track: {'tests' if TODAY_TRACK=='A' else 'docstrings' if TODAY_TRACK=='B' else 'RCAN preset'}

        Proposed change:
        {draft[:4000]}

        Rules:
        - Must not modify forbidden files (api.py, safety.py, auth.py)
        - Tests must follow pytest conventions and test real behavior
        - Docstrings must be Google-style and accurate
        - RCAN presets must have rcan_version, metadata.robot_name, agent.model, non-empty drivers
        - No hallucinated imports or functions that don't exist

        Reply with exactly: PASS or FAIL
        Then one sentence explaining why.
    """)
    msg = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    passed = text.upper().startswith("PASS")
    return passed, text


def apply_change(file_path: str, content: str) -> None:
    if TODAY_TRACK == "A":
        # Write new test file
        test_name = Path(file_path).stem
        dest = OPENCASTOR_REPO / "tests" / f"test_auto_{test_name}.py"
        dest.write_text(content, encoding="utf-8")
    elif TODAY_TRACK == "B":
        (OPENCASTOR_REPO / file_path).write_text(content, encoding="utf-8")
    else:
        # New preset — extract a name from content
        import re
        name_match = re.search(r"robot_name:\s*(\S+)", content)
        preset_name = name_match.group(1).replace(" ", "_") if name_match else f"auto_{int(time.time())}"
        dest = OPENCASTOR_REPO / "config" / "presets" / f"{preset_name}.rcan.yaml"
        dest.write_text(content, encoding="utf-8")


def revert_change(file_path: str) -> None:
    if TODAY_TRACK == "A":
        test_name = Path(file_path).stem
        dest = OPENCASTOR_REPO / "tests" / f"test_auto_{test_name}.py"
        if dest.exists():
            dest.unlink()
    elif TODAY_TRACK == "B":
        git(f"checkout -- {file_path}")
    else:
        # Remove newly created preset files
        run_cmd("git checkout -- config/presets/ 2>/dev/null; git clean -fd config/presets/ 2>/dev/null")


def run_verification() -> tuple[int, str]:
    """Run pytest and ruff; return (exit_code, output)."""
    code, out = run_cmd("python -m pytest tests/ -x -q --tb=no 2>&1 | tail -5", timeout=300)
    return code, out


def log_result(commit: str, before: int, after: int, status: str, desc: str) -> None:
    row = f"{commit}\t{before}\t{after}\t{after - before}\t{status}\t{desc}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    print(row.strip())


def ensure_results_tsv() -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tmetric_before\tmetric_after\tdelta\tstatus\tdescription\n")


def main() -> None:
    program = (Path(__file__).parent / "program.md").read_text()
    ensure_results_tsv()

    print(f"[autoresearch] Starting. Track={TODAY_TRACK} Repo={OPENCASTOR_REPO}")
    exp = 0

    while True:
        exp += 1
        print(f"\n[exp {exp}] {datetime.now().strftime('%H:%M:%S')}")

        candidates = list_candidate_files()
        if not candidates:
            print("No candidates found, sleeping 60s")
            time.sleep(60)
            continue

        # Rotate through candidates
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

        # Apply
        before = get_metric_before()
        apply_change(target, draft)

        # Verify
        exit_code, verify_out = run_verification()
        if exit_code != 0:
            print(f"  Tests FAILED — reverting\n{verify_out}")
            revert_change(target)
            log_result("none", before, before, "crash", f"{target}: tests failed")
            continue

        after = get_metric_before()

        if metric_improved(before, after):
            git("add -A")
            git(f'commit -m "auto({TODAY_TRACK.lower()}): improve {Path(target).name} [{before}->{after}]"')
            commit = git("rev-parse --short HEAD")
            log_result(commit, before, after, "keep", f"{target}")
            print(f"  KEPT delta={after - before}")
        else:
            revert_change(target)
            log_result("none", before, after, "discard", f"{target}: no improvement")
            print(f"  DISCARDED (no improvement)")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x /tmp/opencastor-autoresearch/run_agent.py
```

**Step 3: Commit**

```bash
cd /tmp/opencastor-autoresearch
git add run_agent.py
git commit -m "feat: add run_agent.py orchestrator (Ollama draft + Haiku review loop)"
```

---

## Task 4: Write cron.sh wrapper

**Files:**
- Create: `/tmp/opencastor-autoresearch/cron.sh`

**Step 1: Write the shell wrapper**

```bash
#!/usr/bin/env bash
# cron.sh — nightly OpenCastor AutoResearcher wrapper
# Cron: 0 0 * * * /home/craigm26/opencastor-autoresearch/cron.sh >> /home/craigm26/autoresearch.log 2>&1

set -euo pipefail

REPO="/home/craigm26/opencastor-autoresearch"
OC_REPO="/home/craigm26/OpenCastor"
LOG="$REPO/run.log"
BRANCH="autoresearch/$(date +%Y-%m-%d)"

# Load env
source "$REPO/.env"
export OPENCASTOR_REPO_PATH="$OC_REPO"

# Determine track by day of week (0=Sun,1=Mon,...6=Sat)
DOW=$(date +%u)
case $DOW in
  1|4) export TODAY_TRACK="A" ;;  # Mon, Thu = Tests
  2|5) export TODAY_TRACK="B" ;;  # Tue, Fri = Docs
  3|6) export TODAY_TRACK="C" ;;  # Wed, Sat = Presets
  7)   export TODAY_TRACK="A" ;;  # Sun = Tests (default)
esac

echo "[cron] $(date) — Starting track=$TODAY_TRACK branch=$BRANCH"

# Create dated branch in OpenCastor repo
cd "$OC_REPO"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

# Run agent until 6am (kill after 6 hours if cron runs at midnight)
cd "$REPO"
timeout 21600 python3 run_agent.py > "$LOG" 2>&1 || true

echo "[cron] $(date) — Agent stopped. Opening PR..."

# Commit any uncommitted results
cd "$OC_REPO"
git add results.tsv run.log 2>/dev/null || true
git diff --staged --quiet || git commit -m "auto: nightly results $(date +%Y-%m-%d) track=$TODAY_TRACK"

# Count experiments from results.tsv
KEPT=$(grep -c "keep" "$OC_REPO/results.tsv" 2>/dev/null || echo 0)
TOTAL=$(grep -vc "^commit" "$OC_REPO/results.tsv" 2>/dev/null || echo 0)

# Open PR
gh pr create \
  --title "autoresearch $(date +%Y-%m-%d): track=$TODAY_TRACK ($KEPT/$TOTAL kept)" \
  --body "$(cat <<EOF
## Nightly AutoResearch Summary

**Track:** $TODAY_TRACK
**Date:** $(date +%Y-%m-%d)
**Experiments:** $TOTAL total, $KEPT kept

### Results

\`\`\`
$(cat "$OC_REPO/results.tsv" | column -t 2>/dev/null || cat "$OC_REPO/results.tsv")
\`\`\`

Auto-generated by [opencastor-autoresearch](https://github.com/craigm26/opencastor-autoresearch)
EOF
)" \
  --base main \
  --head "$BRANCH" 2>/dev/null || echo "[cron] PR creation skipped (no changes)"

echo "[cron] $(date) — Done."
```

**Step 2: Make executable and commit**

```bash
chmod +x /tmp/opencastor-autoresearch/cron.sh
cd /tmp/opencastor-autoresearch
git add cron.sh
git commit -m "feat: add cron.sh nightly wrapper with PR creation"
```

---

## Task 5: Create .env template and README

**Files:**
- Create: `/tmp/opencastor-autoresearch/.env.example`
- Modify: `/tmp/opencastor-autoresearch/README.md`

**Step 1: Write .env.example**

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENCASTOR_REPO_PATH=/home/craigm26/OpenCastor
TODAY_TRACK=A
```

**Step 2: Ensure .env is in .gitignore**

```bash
echo ".env" >> .gitignore
```

**Step 3: Update README.md**

```markdown
# opencastor-autoresearch

Autonomous overnight improvement agent for [OpenCastor](https://github.com/craigm26/OpenCastor).

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and adapted for
software improvement (tests, docs, presets) instead of ML training.

## How it works

- **Draft model**: gemma3:1b via Ollama (on-device, free)
- **Review model**: Claude Haiku 4.5 (cloud, ~$0.15/night)
- **Loop**: propose → review → apply → test → keep/revert → log
- **Schedule**: 12am–6am nightly via cron
- **Cost**: ~$0.07–0.15/night

## Tracks (rotates by day)

| Day | Track | What it does |
|-----|-------|-------------|
| Mon/Thu | A | Write new pytest tests |
| Tue/Fri | B | Add missing docstrings |
| Wed/Sat | C | Generate RCAN presets |
| Sun | A | Tests (default) |

## Setup

1. `cp .env.example .env && nano .env` — add your Anthropic key
2. `pip install anthropic ollama gitpython`
3. Ensure `ollama run gemma3:1b` works
4. `crontab -e` and add: `0 0 * * * /home/craigm26/opencastor-autoresearch/cron.sh >> /home/craigm26/autoresearch.log 2>&1`

## Manual run

```bash
source .env
export OPENCASTOR_REPO_PATH=/home/craigm26/OpenCastor
export TODAY_TRACK=A
python3 run_agent.py
```

## Results

Each night produces `results.tsv` and a GitHub PR in OpenCastor with the experiment log.
```

**Step 4: Commit**

```bash
cd /tmp/opencastor-autoresearch
git add .env.example .gitignore README.md
git commit -m "docs: add README, .env.example for opencastor-autoresearch"
```

---

## Task 6: Install repo and configure cron

**Step 1: Clone the fork to final location**

```bash
cp -r /tmp/opencastor-autoresearch /home/craigm26/opencastor-autoresearch
```

**Step 2: Create .env**

```bash
cp /home/craigm26/opencastor-autoresearch/.env.example /home/craigm26/opencastor-autoresearch/.env
# Edit: nano /home/craigm26/opencastor-autoresearch/.env
# Add real ANTHROPIC_API_KEY
```

**Step 3: Install dependencies**

```bash
pip install anthropic ollama gitpython
```

**Step 4: Smoke test the loop (5 minutes, Track A)**

```bash
cd /home/craigm26/opencastor-autoresearch
source .env
export OPENCASTOR_REPO_PATH=/home/craigm26/OpenCastor
export TODAY_TRACK=A
timeout 300 python3 run_agent.py
```

Expected: at least 1 experiment completes, `results.tsv` gets a row.

**Step 5: Add cron job**

```bash
crontab -e
```

Add this line:

```
0 0 * * * /home/craigm26/opencastor-autoresearch/cron.sh >> /home/craigm26/autoresearch.log 2>&1
```

**Step 6: Verify cron is registered**

```bash
crontab -l | grep autoresearch
```

Expected: line appears.

**Step 7: Push fork to GitHub**

```bash
cd /home/craigm26/opencastor-autoresearch
git remote set-url origin git@github.com:craigm26/opencastor-autoresearch.git
git push -u origin main
```

---

## Task 7: Smoke test end-to-end

**Step 1: Verify Ollama is running and gemma3:1b responds**

```bash
ollama run gemma3:1b "say hello in one word"
```

Expected: one-word response

**Step 2: Verify Anthropic key works**

```bash
python3 -c "
import anthropic, os
from dotenv import load_dotenv
load_dotenv('/home/craigm26/opencastor-autoresearch/.env')
c = anthropic.Anthropic()
m = c.messages.create(model='claude-haiku-4-5-20251001', max_tokens=10, messages=[{'role':'user','content':'say hi'}])
print(m.content[0].text)
"
```

Expected: short greeting

**Step 3: Run one full experiment cycle**

```bash
cd /home/craigm26/opencastor-autoresearch
source .env
export OPENCASTOR_REPO_PATH=/home/craigm26/OpenCastor
export TODAY_TRACK=A
python3 -c "
import run_agent
# Run just 1 experiment by patching the loop
run_agent.ensure_results_tsv()
candidates = run_agent.list_candidate_files()
print('Candidates:', candidates[:3])
before = run_agent.get_metric_before()
print('Metric before:', before)
"
```

Expected: lists candidate files and prints current test count (~4323)

---

## Summary

After completion you will have:
- `craigm26/opencastor-autoresearch` on GitHub (forked from karpathy/autoresearch)
- Nightly cron at `0 0 * * *` running `cron.sh`
- Agent loop: gemma3:1b draft → Haiku review → apply → pytest → keep/revert
- Results logged to `results.tsv` each night
- Auto PR opened in OpenCastor at 6am with experiment summary
- ~$0.07–0.15/night in API costs
