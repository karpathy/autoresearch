# OpenCastor AutoResearcher

This agent autonomously improves the OpenCastor codebase overnight.

## Context

OpenCastor is a universal robot runtime (~Python 3.10+, 4323 tests, ruff 100-char line limit).
Repo path: set in env var OPENCASTOR_REPO_PATH.
Conventions: PEP8, snake_case, type hints on public signatures, lazy imports (HAS_X pattern), structured logging.
Test runner: `python -m pytest tests/ -x -q`
Linter: `ruff check castor/`
RCAN validator: `castor validate --config <file>`

## Active Track

Determined at runtime by TODAY_TRACK env var:
- A = Tests: write new pytest tests for untested code paths in castor/
- B = Docs: add missing Google-style docstrings to castor/ source files
- C = Presets: generate new RCAN config presets for hardware not yet in config/presets/

## Metrics

Track A — test count must increase:
  `python -m pytest --co -q 2>/dev/null | grep -E "^[0-9]+ test" | awk "{print \$1}"`

Track B — missing docstring count must decrease:
  `python3 -c "import ast,os; missing=[]; [missing.extend([n.name for n in ast.walk(ast.parse(open(os.path.join(r,f)).read())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not ast.get_docstring(n)]) for r,d,files in os.walk('castor') for f in files if f.endswith('.py')]; print(len(missing))"`

Track C — preset count must increase:
  `ls config/presets/*.rcan.yaml 2>/dev/null | wc -l`

## The Loop

LOOP FOREVER (until killed):

1. Pick a target file to improve based on the active track
2. Read the file content
3. Draft an improvement (new tests / docstrings / preset YAML)
4. The orchestrator sends your draft to Claude Haiku for quality review
5. If approved: write the file, run the metric command, check result
6. If metric improved: git commit — status "keep"
7. If metric same or worse: git checkout -- <file> — status "discard"
8. Log result to results.tsv
9. Go to step 1

## Constraints

- NEVER modify: castor/api.py, castor/safety.py, castor/auth.py, .env
- NEVER install new packages
- All pytest runs must exit 0 — zero regressions allowed
- Keep changes small and focused — one function docstring, one test function, or one preset per experiment
- Test file names for Track A: tests/test_auto_<module>.py

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask questions. The human is asleep.
Run until killed externally. If you run out of ideas, rotate to the next candidate file.
