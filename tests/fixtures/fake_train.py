from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import time


VAL_RE = re.compile(r"#\s*val_bpb:\s*([-+]?(?:\d+\.?\d*|\.\d+))")
BEHAVIOR_RE = re.compile(r"#\s*behavior:\s*([a-z_]+)")


def main() -> int:
    train_py = Path("train.py").read_text(encoding="utf-8")
    behavior_match = BEHAVIOR_RE.search(train_py)
    behavior = behavior_match.group(1) if behavior_match else "success"
    if behavior == "timeout":
        time.sleep(5)
        return 0
    if behavior == "crash":
        print("simulated crash", file=sys.stderr)
        return 1
    if behavior == "missing_metric":
        print("completed without metric")
        return 0
    match = VAL_RE.search(train_py)
    val_bpb = float(match.group(1)) if match else 1.0
    Path("metrics.json").write_text(json.dumps({"val_bpb": val_bpb}) + "\n", encoding="utf-8")
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
