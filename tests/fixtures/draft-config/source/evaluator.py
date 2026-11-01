"""Synthetic adapter. A real project supplies its own data reader and evaluator."""

import json
import sys
from pathlib import Path


candidate, run_dir, config_dir = map(Path, sys.argv[1:4])
factor = int((candidate / "rule.txt").read_text(encoding="utf-8"))
cases = json.loads((config_dir / "cases.json").read_text(encoding="utf-8"))
errors = [abs(case["input"] * factor - case["expected"]) for case in cases]
result = {
    "metrics": {"mean_error": sum(errors) / len(errors), "coverage": len(errors)},
    "usage": {"requests": 0, "cost_usd": 0},
    "details": {"errors_by_case": errors},
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
