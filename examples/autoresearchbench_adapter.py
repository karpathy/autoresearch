"""Dependency-free AutoResearchBench JSONL adapter.

This module loads and normalizes already-available AutoResearchBench example
or decrypted JSONL records. Its download helper only writes bytes from an
explicit user-provided URL; it does not decrypt obfuscated data, call
model/search providers, run official external
evaluators, or claim benchmark scores. The emitted report is an adapter report:
one normalized task row per input row, with expected answer titles and arXiv IDs
prepared for downstream tooling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter
from typing import Any

BENCHMARK_NAME = "AutoResearchBench"
SOURCE_NAME = "external_autoresearchbench"
TASK_TYPE_DEEP = "deep_research"
TASK_TYPE_WIDE = "wide_research"
TASK_TYPES = (TASK_TYPE_DEEP, TASK_TYPE_WIDE)
_ARXIV_PREFIX_RE = re.compile(r"^\s*arxiv\s*:\s*", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
_JSON_DUMP_KWARGS = {"ensure_ascii": False, "sort_keys": True}


def load_autoresearchbench_jsonl(input_file: str | Path) -> list[dict]:
    """Load, validate, and normalize non-empty AutoResearchBench JSONL rows.

    ``input_file`` must point at a local JSONL file whose non-blank lines are
    JSON objects with non-empty ``question``, ``answer``, and ``arxiv_id``
    fields. Malformed JSON and validation failures raise ``ValueError`` with
    the source line number. Returned records are normalized task dictionaries
    suitable for ``build_autoresearchbench_report``.
    """

    path = Path(input_file)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: malformed JSON: {exc.msg}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            records.append(normalize_autoresearchbench_record(parsed, line_number=line_number))
    return records


def normalize_autoresearchbench_record(record: dict, *, line_number: int | None = None) -> dict:
    """Return the normalized JSON-safe task shape for one benchmark record."""

    if not isinstance(record, dict):
        raise TypeError("record must be a dict")

    question = _required_non_empty_string(record, "question", line_number=line_number)
    answer_titles = _normalize_answer_titles(record.get("answer"), line_number=line_number)
    expected_arxiv_ids = _normalize_arxiv_ids(record.get("arxiv_id"), line_number=line_number)
    task_type = _normalize_task_type(record, answer_titles, expected_arxiv_ids, line_number=line_number)
    task_id = _normalize_task_id(record.get("id"), question, line_number)

    metadata: dict[str, Any] = {}
    if line_number is not None:
        metadata["line_number"] = line_number

    return {
        "benchmark": BENCHMARK_NAME,
        "source": SOURCE_NAME,
        "task_id": task_id,
        "task_type": task_type,
        "question": question,
        "answer_titles": answer_titles,
        "expected_arxiv_ids": expected_arxiv_ids,
        "metadata": metadata,
    }


def build_autoresearchbench_report(records: list[dict] | str | Path) -> dict:
    """Build a JSON-safe adapter report from raw, normalized, or JSONL records."""

    if isinstance(records, (str, Path)):
        tasks = load_autoresearchbench_jsonl(records)
    else:
        tasks = [_ensure_normalized_task(record) for record in records]
    counts = Counter(task["task_type"] for task in tasks)
    task_type_counts = {task_type: counts.get(task_type, 0) for task_type in TASK_TYPES}
    return {
        "benchmark": BENCHMARK_NAME,
        "mode": "adapter",
        "task_count": len(tasks),
        "task_type_counts": task_type_counts,
        "tasks": tasks,
    }

def download_autoresearchbench_file(
    url: str,
    output_path: str | Path,
    *,
    token_env: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Explicitly download a user-provided AutoResearchBench file without parsing it."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError("output_path must be a non-empty path")

    path = Path(output_path)
    output_existed = path.exists()
    if output_existed:
        if path.is_dir():
            raise IsADirectoryError(f"output_path is a directory: {path}")
        if not overwrite:
            raise FileExistsError(f"output_path already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url.strip())
    token = _token_from_env(token_env)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")

    byte_count = 0
    with urllib.request.urlopen(request) as response, path.open("wb") as target:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            target.write(chunk)

    return {
        "benchmark": BENCHMARK_NAME,
        "mode": "download",
        "source_url": url.strip(),
        "output_path": str(path),
        "output_file": str(path),
        "byte_count": byte_count,
        "downloaded": True,
        "metadata": {
            "url_scheme": urllib.parse.urlparse(url.strip()).scheme,
            "output_exists": path.exists(),
            "overwritten": output_existed,
        },
    }


def _token_from_env(token_env: str | None) -> str | None:
    if token_env is None:
        return None
    if not isinstance(token_env, str) or not token_env.strip():
        raise ValueError("token_env must name a non-empty environment variable")
    token = os.environ.get(token_env.strip(), "")
    if not token.strip():
        raise ValueError(f"environment variable {token_env.strip()} is empty or unset")
    return token.strip()



def export_autoresearchbench_jsonl(report: dict) -> str:
    """Serialize normalized report tasks as one JSON object per JSONL line."""

    tasks = report.get("tasks") if isinstance(report, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("report must contain a tasks list")
    lines = [json.dumps(_ensure_normalized_task(task), **_JSON_DUMP_KWARGS) for task in tasks]
    return "\n".join(lines) + ("\n" if lines else "")


def researchfs_task_rows(report: dict) -> list[dict]:
    """Convert normalized adapter tasks to ResearchFS-friendly external task rows."""

    rows: list[dict] = []
    for task in _report_tasks(report):
        expected_arxiv_ids = _ordered_normalized_arxiv_ids(task["expected_arxiv_ids"])
        expected_answer_titles = [str(value) for value in task["answer_titles"]]
        rows.append(
            {
                "benchmark": task["benchmark"],
                "source": task["source"],
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "question": task["question"],
                "expected_answer_titles": expected_answer_titles,
                "expected_arxiv_ids": expected_arxiv_ids,
                "gold_paper_ids": list(expected_arxiv_ids),
                "gold_arxiv_ids": list(expected_arxiv_ids),
                "required_citations": _required_citation_rows(expected_answer_titles, expected_arxiv_ids),
                "candidate_papers": [],
                "evaluation_contract": {
                    "name": "deterministic_arxiv_id_overlap",
                    "scope": "adapter_side_sanity_evaluation",
                    "expected_field": "expected_arxiv_ids",
                    "prediction_fields": ["predicted_arxiv_ids", "final_candidates[].arxiv_id"],
                    "candidate_papers": "not_included_by_adapter",
                },
                "metadata": _json_safe_metadata(task.get("metadata", {})),
            }
        )
    return rows


def export_researchfs_tasks_jsonl(report: dict) -> str:
    """Serialize ResearchFS-friendly external task rows as JSONL."""
    return export_researchfs_jsonl(report)

def export_researchfs_jsonl(report_or_rows: dict | list[dict]) -> str:
    """Serialize ResearchFS-friendly external task rows as JSONL."""

    if isinstance(report_or_rows, list):
        rows = report_or_rows
    else:
        rows = researchfs_task_rows(report_or_rows)
    lines = [json.dumps(row, **_JSON_DUMP_KWARGS) for row in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def evaluate_autoresearchbench_predictions(report: dict, predictions: list[dict] | dict) -> dict:
    """Evaluate predictions with deterministic arXiv-ID overlap for adapter sanity checks."""

    tasks = _report_tasks(report)
    prediction_map = _prediction_map(predictions)
    task_reports: list[dict] = []
    task_ids = set()

    for task in tasks:
        task_id = task["task_id"]
        task_ids.add(task_id)
        expected_ids = _ordered_normalized_arxiv_ids(task["expected_arxiv_ids"])
        prediction = prediction_map.get(task_id)
        evaluated = prediction is not None
        predicted_ids = _extract_prediction_arxiv_ids(prediction) if evaluated else []
        task_reports.append(
            _prediction_overlap_report(
                task=task,
                expected_ids=expected_ids,
                predicted_ids=predicted_ids,
                evaluated=evaluated,
            )
        )

    evaluated_reports = [task_report for task_report in task_reports if task_report["evaluated"]]
    evaluated_deep_reports = [task_report for task_report in evaluated_reports if task_report["task_type"] == TASK_TYPE_DEEP]
    evaluated_wide_reports = [task_report for task_report in evaluated_reports if task_report["task_type"] == TASK_TYPE_WIDE]
    unexpected_prediction_task_ids = sorted(task_id for task_id in prediction_map if task_id not in task_ids)
    aggregate = {
        "task_count": len(tasks),
        "evaluated_count": len(evaluated_reports),
        "deep_accuracy": _average_metric(evaluated_deep_reports, "correct"),
        "wide_iou": _average_metric(evaluated_wide_reports, "iou"),
        "wide_recall": _average_metric(evaluated_wide_reports, "recall"),
        "wide_precision": _average_metric(evaluated_wide_reports, "precision"),
        "overall_recall": _average_metric(evaluated_reports, "recall"),
        "mean_iou": _average_metric(task_reports, "iou"),
        "mean_recall": _average_metric(task_reports, "recall"),
        "mean_precision": _average_metric(task_reports, "precision"),
        "missing_prediction_count": len(tasks) - len(evaluated_reports),
    }
    return {
        "benchmark": BENCHMARK_NAME,
        "source": SOURCE_NAME,
        "mode": "deterministic_adapter_evaluation",
        "kind": "deterministic_arxiv_id_overlap",
        "evaluation_scope": "deterministic_arxiv_id_overlap",
        "scope": "adapter_side_sanity_evaluation",
        "task_count": len(tasks),
        "prediction_count": len(prediction_map),
        "aggregate": aggregate,
        "results": task_reports,
        "tasks": task_reports,
        "unexpected_prediction_task_ids": unexpected_prediction_task_ids,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the AutoResearchBench adapter CLI."""

    parser = argparse.ArgumentParser(description="Normalize local AutoResearchBench JSONL records.")
    parser.add_argument("--input-file", help="Local decrypted/example AutoResearchBench JSONL file.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON without indentation.")
    parser.add_argument("--output", help="Optional path to write the report JSON.")
    parser.add_argument("--jsonl-output", help="Optional path to write normalized task JSONL rows.")
    parser.add_argument("--researchfs-output", help="Optional path to write ResearchFS task JSONL rows.")
    parser.add_argument("--predictions-file", help="Optional JSON or JSONL predictions file to evaluate.")
    parser.add_argument("--evaluation-output", help="Optional path to write only the deterministic evaluation JSON.")
    parser.add_argument("--download-url", help="Explicit URL to download before any adapter loading.")
    parser.add_argument("--download-output", help="Local path where --download-url bytes are written.")
    parser.add_argument("--download-token-env", help="Environment variable containing a bearer token for --download-url.")
    parser.add_argument("--overwrite-download", action="store_true", help="Allow --download-output to replace an existing file.")
    parser.add_argument("--overwrite", action="store_true", dest="overwrite_download", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    has_download_url = args.download_url is not None
    has_download_output = args.download_output is not None
    if has_download_url != has_download_output:
        parser.error("--download-url and --download-output must be supplied together")
    if args.download_token_env is not None and not args.download_token_env.strip():
        parser.error("--download-token-env must name a non-empty environment variable")
    if args.input_file is None and not (has_download_url and has_download_output):
        parser.error("--input-file is required unless --download-url and --download-output are supplied")

    if has_download_url and has_download_output:
        download_report = download_autoresearchbench_file(
            args.download_url,
            args.download_output,
            token_env=args.download_token_env,
            overwrite=args.overwrite_download,
        )
        if args.input_file is None:
            print(_dumps_report(download_report, compact=args.compact))
            return 0

    report = build_autoresearchbench_report(load_autoresearchbench_jsonl(args.input_file))
    evaluation = None
    if args.predictions_file:
        predictions = _load_predictions_file(args.predictions_file)
        evaluation = evaluate_autoresearchbench_predictions(report, predictions)
        report["deterministic_adapter_evaluation"] = evaluation

    report_json = _dumps_report(report, compact=args.compact)
    if args.output:
        Path(args.output).write_text(report_json + "\n", encoding="utf-8")
    if args.jsonl_output:
        Path(args.jsonl_output).write_text(export_autoresearchbench_jsonl(report), encoding="utf-8")
    if args.researchfs_output:
        Path(args.researchfs_output).write_text(export_researchfs_tasks_jsonl(report), encoding="utf-8")
    if args.evaluation_output and evaluation is not None:
        Path(args.evaluation_output).write_text(_dumps_report(evaluation, compact=args.compact) + "\n", encoding="utf-8")
    print(report_json)
    return 0


def _ensure_normalized_task(record: dict) -> dict:
    if _is_normalized_task(record):
        return {
            "benchmark": BENCHMARK_NAME,
            "source": SOURCE_NAME,
            "task_id": str(record["task_id"]),
            "task_type": str(record["task_type"]),
            "question": str(record["question"]),
            "answer_titles": [str(value) for value in record["answer_titles"]],
            "expected_arxiv_ids": [str(value) for value in record["expected_arxiv_ids"]],
            "metadata": _json_safe_metadata(record.get("metadata", {})),
        }
    return normalize_autoresearchbench_record(record)


def _is_normalized_task(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    return (
        record.get("benchmark") == BENCHMARK_NAME
        and record.get("source") == SOURCE_NAME
        and record.get("task_type") in TASK_TYPES
        and _non_empty(record.get("task_id"))
        and _non_empty(record.get("question"))
        and _has_non_empty_items(record.get("answer_titles"))
        and _has_non_empty_items(record.get("expected_arxiv_ids"))
        and isinstance(record.get("metadata"), dict)
    )


def _report_tasks(report: dict | list[dict]) -> list[dict]:
    if isinstance(report, list):
        return [_ensure_normalized_task(task) for task in report]
    tasks = report.get("tasks") if isinstance(report, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("report must contain a tasks list")
    return [_ensure_normalized_task(task) for task in tasks]


def _required_citation_rows(expected_answer_titles: list[str], expected_arxiv_ids: list[str]) -> list[dict]:
    row_count = max(len(expected_answer_titles), len(expected_arxiv_ids))
    rows: list[dict] = []
    for index in range(row_count):
        row: dict[str, str] = {}
        if index < len(expected_answer_titles):
            row["title"] = expected_answer_titles[index]
        if index < len(expected_arxiv_ids):
            arxiv_id = expected_arxiv_ids[index]
            row["arxiv_id"] = arxiv_id
            row["paper_id"] = arxiv_id
        rows.append(row)
    return rows


def _prediction_map(predictions: list[dict] | dict) -> dict[str, object]:
    if isinstance(predictions, list):
        return _prediction_list_map(predictions)
    if not isinstance(predictions, dict):
        raise TypeError("predictions must be a list or dict")
    if _is_prediction_item(predictions):
        return {str(predictions["task_id"]): predictions}
    nested_predictions = predictions.get("predictions")
    if isinstance(nested_predictions, list):
        return _prediction_list_map(nested_predictions)

    mapped: dict[str, object] = {}
    for task_id, prediction in predictions.items():
        normalized_task_id = str(task_id)
        if isinstance(prediction, dict):
            prediction_with_id = dict(prediction)
            prediction_with_id.setdefault("task_id", normalized_task_id)
            mapped[normalized_task_id] = prediction_with_id
        else:
            mapped[normalized_task_id] = {
                "task_id": normalized_task_id,
                "predicted_arxiv_ids": prediction,
            }
    return mapped


def _prediction_list_map(predictions: list[dict]) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for index, prediction in enumerate(predictions, start=1):
        if not isinstance(prediction, dict):
            raise ValueError(f"prediction item {index} must be a JSON object")
        task_id = prediction.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"prediction item {index} must contain non-empty task_id")
        mapped[task_id.strip()] = prediction
    return mapped


def _is_prediction_item(value: dict) -> bool:
    return _non_empty(value.get("task_id")) and (
        "predicted_arxiv_ids" in value or "final_candidates" in value
    )


def _extract_prediction_arxiv_ids(prediction: object) -> list[str]:
    values: list[object] = []
    if isinstance(prediction, dict):
        if "predicted_arxiv_ids" in prediction:
            values.extend(_list_or_scalar_values(prediction["predicted_arxiv_ids"]))
        if "final_candidates" in prediction:
            final_candidates = prediction["final_candidates"]
            if isinstance(final_candidates, list):
                for candidate in final_candidates:
                    if isinstance(candidate, dict):
                        values.append(candidate.get("arxiv_id"))
                    else:
                        values.append(candidate)
            else:
                values.append(final_candidates)
    else:
        values.extend(_list_or_scalar_values(prediction))
    return _ordered_normalized_arxiv_ids(values)


def _list_or_scalar_values(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    return [value]


def _ordered_normalized_arxiv_ids(values: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in _list_or_scalar_values(values):
        arxiv_id = _normalize_arxiv_id(value)
        if arxiv_id and arxiv_id not in seen:
            normalized.append(arxiv_id)
            seen.add(arxiv_id)
    return normalized


def _prediction_overlap_report(
    *,
    task: dict,
    expected_ids: list[str],
    predicted_ids: list[str],
    evaluated: bool,
) -> dict:
    expected_set = set(expected_ids)
    predicted_set = set(predicted_ids)
    hit_set = expected_set & predicted_set
    missed_set = expected_set - predicted_set
    extra_set = predicted_set - expected_set
    union_set = expected_set | predicted_set
    precision = len(hit_set) / len(predicted_set) if predicted_set else (1.0 if not expected_set else 0.0)
    recall = len(hit_set) / len(expected_set) if expected_set else 1.0
    iou = len(hit_set) / len(union_set) if union_set else 1.0
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "evaluated": evaluated,
        "expected_arxiv_ids": expected_ids,
        "gold_arxiv_ids": expected_ids,
        "predicted_arxiv_ids": predicted_ids,
        "hit_arxiv_ids": [arxiv_id for arxiv_id in expected_ids if arxiv_id in hit_set],
        "matched_arxiv_ids": [arxiv_id for arxiv_id in expected_ids if arxiv_id in hit_set],
        "missed_arxiv_ids": [arxiv_id for arxiv_id in expected_ids if arxiv_id in missed_set],
        "extra_arxiv_ids": [arxiv_id for arxiv_id in predicted_ids if arxiv_id in extra_set],
        "has_prediction": evaluated,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "supported": bool(expected_set) and not missed_set,
        "correct": bool(expected_set) and not missed_set and not extra_set,
    }


def _average_metric(task_reports: list[dict], key: str) -> float:
    if not task_reports:
        return 0.0
    return sum(float(task_report[key]) for task_report in task_reports) / len(task_reports)


def _load_predictions_file(path: str | Path) -> list[dict] | dict:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError("predictions file is empty")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        predictions: list[dict] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: malformed prediction JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: prediction JSONL rows must be objects")
            predictions.append(item)
        return predictions
    if not isinstance(parsed, (dict, list)):
        raise ValueError("predictions file must contain a JSON object, JSON array, or JSONL objects")
    return parsed


def _required_non_empty_string(record: dict, key: str, *, line_number: int | None) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(_line_prefix(line_number) + f"missing non-empty {key}")
    return value.strip()


def _normalize_answer_titles(value: object, *, line_number: int | None) -> list[str]:
    titles: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            titles.append(stripped)
    elif isinstance(value, list):
        for item in value:
            title = _answer_title_from_item(item)
            if title:
                titles.append(title)
    else:
        title = _answer_title_from_item(value)
        if title:
            titles.append(title)

    if not titles:
        raise ValueError(_line_prefix(line_number) + "missing non-empty answer")
    return titles


def _answer_title_from_item(item: object) -> str | None:
    if isinstance(item, str):
        stripped = item.strip()
        return stripped or None
    if isinstance(item, dict):
        for key in ("title", "paper_title", "answer"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalize_arxiv_ids(value: object, *, line_number: int | None) -> list[str]:
    normalized: list[str] = []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]

    for raw_value in raw_values:
        arxiv_id = _normalize_arxiv_id(raw_value)
        if arxiv_id:
            normalized.append(arxiv_id)

    if not normalized:
        raise ValueError(_line_prefix(line_number) + "missing non-empty arxiv_id")
    return normalized


def _normalize_arxiv_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    without_prefix = _ARXIV_PREFIX_RE.sub("", text).strip()
    match = _ARXIV_ID_RE.search(without_prefix)
    if match:
        return match.group(1)
    return without_prefix


def _normalize_task_type(
    record: dict,
    answer_titles: list[str],
    expected_arxiv_ids: list[str],
    *,
    line_number: int | None,
) -> str:
    explicit = record.get("task_type")
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "deep": TASK_TYPE_DEEP,
            "deep_research": TASK_TYPE_DEEP,
            "wide": TASK_TYPE_WIDE,
            "wide_research": TASK_TYPE_WIDE,
        }
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(_line_prefix(line_number) + f"unknown task_type: {explicit}")

    original_arxiv_id = record.get("arxiv_id")
    if isinstance(original_arxiv_id, list) or len(answer_titles) > 1 or len(expected_arxiv_ids) > 1:
        return TASK_TYPE_WIDE
    return TASK_TYPE_DEEP


def _normalize_task_id(value: object, question: str, line_number: int | None) -> str:
    if value is not None:
        task_id = str(value).strip()
        if task_id:
            return task_id
    if line_number is not None:
        return f"line:{line_number}"
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"question:{digest}"


def _json_safe_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return {}
    return json.loads(json.dumps(metadata, ensure_ascii=False))


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_non_empty_items(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value)


def _line_prefix(line_number: int | None) -> str:
    return f"line {line_number}: " if line_number is not None else ""


def _dumps_report(report: dict, *, compact: bool) -> str:
    if compact:
        return json.dumps(report, separators=(",", ":"), **_JSON_DUMP_KWARGS)
    return json.dumps(report, indent=2, **_JSON_DUMP_KWARGS)


if __name__ == "__main__":
    raise SystemExit(main())
