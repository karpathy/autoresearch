"""Lightweight telemetry for benchmark runs.

Collects per-task metrics (tokens, timing, tool calls, errors) and writes
telemetry.jsonl. CLI: ``python observe.py --summary FILE`` or ``--errors FILE``.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

ERROR_CATEGORIES = frozenset(
    {"constraint_violation", "tool_failure", "timeout", "quality_gate_fail", "unknown"}
)


@dataclass
class TaskTelemetry:
    """Per-task metrics container with provenance tracking."""

    task_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "pass"
    error_category: str | None = None

    # Provenance fields for audit trail
    agent_response: str = ""
    expected_outcome: dict = field(default_factory=dict)
    outcome_match: dict = field(default_factory=dict)
    session_id: str | None = None
    prompt_hash: str = ""
    harness_version: str = ""
    evaluation_mode: str = ""
    model: str = ""
    check_score: float = 0.0

    def record_tokens(self, input: int = 0, output: int = 0) -> None:
        self.input_tokens += input
        self.output_tokens += output

    def record_tool(self, name: str, success: bool = True, duration_ms: float = 0.0) -> None:
        self.tool_calls.append({"name": name, "success": success, "duration_ms": duration_ms})

    def record_timing(self, start: float, end: float) -> None:
        self.start_time, self.end_time = start, end

    def record_evaluation(
        self,
        response: str,
        expected: dict,
        checks: list[dict],
        mode: str = "",
        score: float = 0.0,
        prompt_hash: str = "",
        harness_version: str = "",
        model: str = "",
    ) -> None:
        """Record evaluation provenance for audit trail."""
        self.agent_response = response[:5000]  # cap stored response size
        self.expected_outcome = expected
        self.outcome_match = {"checks": checks, "score": score}
        self.evaluation_mode = mode
        self.check_score = score
        self.prompt_hash = prompt_hash
        self.harness_version = harness_version
        self.model = model

    def to_dict(self) -> dict:
        dur = round(self.end_time - self.start_time, 3) if self.end_time else 0.0
        d = {
            "task_name": self.task_name, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "total_tokens": self.input_tokens + self.output_tokens,
            "tool_calls": self.tool_calls, "tool_call_count": len(self.tool_calls),
            "start_time": self.start_time, "end_time": self.end_time, "duration_s": dur,
            "status": self.status, "error_category": self.error_category,
        }
        # Include provenance fields when populated
        if self.prompt_hash:
            d["prompt_hash"] = self.prompt_hash
        if self.harness_version:
            d["harness_version"] = self.harness_version
        if self.model:
            d["model"] = self.model
        if self.evaluation_mode:
            d["evaluation_mode"] = self.evaluation_mode
        if self.outcome_match:
            d["outcome_match"] = self.outcome_match
        if self.check_score > 0:
            d["check_score"] = self.check_score
        return d

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if exc_type is not None:
            self.status, self.error_category = "error", "unknown"
        return False


class TelemetryCollector:
    """Aggregates TaskTelemetry records and writes them to telemetry.jsonl."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self._records: list[TaskTelemetry] = []

    def task(self, name: str) -> TaskTelemetry:
        t = TaskTelemetry(task_name=name)
        self._records.append(t)
        return t

    def flush(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / "telemetry.jsonl"
        with out.open("a", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        self._records.clear()
        return out

    def summary(self) -> dict:
        return _summarize(self._records)


def _summarize(records) -> dict:
    def _get(r, key, default=0):
        return getattr(r, key, None) if isinstance(r, TaskTelemetry) else r.get(key, default)

    total_in = sum(_get(r, "input_tokens", 0) for r in records)
    total_out = sum(_get(r, "output_tokens", 0) for r in records)
    durations, tool_count, errors = [], 0, {}
    for r in records:
        d = r.to_dict() if isinstance(r, TaskTelemetry) else r
        if d["duration_s"]:
            durations.append(d["duration_s"])
        tool_count += d.get("tool_call_count", len(d.get("tool_calls", [])))
        if cat := d.get("error_category"):
            errors[cat] = errors.get(cat, 0) + 1
    avg_dur = round(sum(durations) / len(durations), 3) if durations else 0.0
    return {"task_count": len(records), "total_input_tokens": total_in,
            "total_output_tokens": total_out, "total_tokens": total_in + total_out,
            "avg_duration_s": avg_dur, "total_tool_calls": tool_count, "error_breakdown": errors}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Telemetry viewer")
    parser.add_argument("--summary", metavar="FILE", help="Print aggregated summary")
    parser.add_argument("--errors", metavar="FILE", help="Print error taxonomy breakdown")
    args = parser.parse_args()

    if args.summary:
        print(json.dumps(_summarize(_load_jsonl(Path(args.summary))), indent=2))
    elif args.errors:
        recs = _load_jsonl(Path(args.errors))
        errs = {}
        for r in recs:
            if cat := r.get("error_category"):
                errs[cat] = errs.get(cat, 0) + 1
        print(json.dumps(errs, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
