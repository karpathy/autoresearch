"""Compare HTTP response bodies between two repeat directories."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sha256_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _case_records(cases_path: Path) -> list[dict[str, Any]]:
    if not cases_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        case_id = row.get("id")
        if isinstance(case_id, str):
            records.append(row)
    return records


def _ok_case_ids(cases_path: Path) -> list[str]:
    return [str(row["id"]) for row in _case_records(cases_path) if row.get("status") == "ok"]


def _usable_response_hashes(sent_path: Path) -> list[str]:
    if not sent_path.is_file():
        return []
    hashes: list[str] = []
    for line in sent_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        status = row.get("status")
        if status in (429, 503):
            continue
        digest = row.get("response_body_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            hashes.append(digest)
            continue
        body = row.get("response_body")
        if body is None:
            raise ValueError(
                "Response body was not captured for case comparison; enable AUTORESEARCH_WIRE_LOG_BODY=1 and rerun"
            )
        if not isinstance(body, str):
            body = str(body)
        hashes.append(_sha256_body(body))
    return hashes


def _case_hash_map(repeat_dir: Path) -> dict[str, str]:
    records = _case_records(repeat_dir / "cases.jsonl")
    sent_path = repeat_dir / "sent-requests.jsonl"
    hashes = _usable_response_hashes(sent_path)
    if any(row.get("status") == "ok" for row in records) and not sent_path.is_file():
        raise ValueError(
            "Response body log is missing for case comparison; enable AUTORESEARCH_WIRE_LOG_BODY=1 and rerun"
        )
    mapped: dict[str, str] = {}
    # Каждый записанный кейс, и успешный и нет, занимает одну строку ответа.
    # Иначе следующий ok-кейс получает чужое тело.
    for index, row in enumerate(records):
        if row.get("status") != "ok":
            continue
        case_id = str(row["id"])
        if index >= len(hashes):
            raise ValueError(f"Response body hash is missing for successful case {case_id}")
        mapped[case_id] = hashes[index]
    return mapped


def _ordered_case_ids(left: Path, right: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for repeat_dir in (left, right):
        for case_id in _ok_case_ids(repeat_dir / "cases.jsonl"):
            if case_id in seen:
                continue
            seen.add(case_id)
            ordered.append(case_id)
    return ordered


def compare_repeat_dirs(left: Path, right: Path) -> dict[str, Any]:
    left_map = _case_hash_map(left)
    right_map = _case_hash_map(right)
    rows: list[dict[str, Any]] = []
    mismatched = 0
    for case_id in _ordered_case_ids(left, right):
        left_sha256 = left_map.get(case_id, "")
        right_sha256 = right_map.get(case_id, "")
        same = left_sha256 == right_sha256
        if not same:
            mismatched += 1
        rows.append(
            {
                "case_id": case_id,
                "same": same,
                "left_sha256": left_sha256,
                "right_sha256": right_sha256,
            }
        )
    total = len(rows)
    return {"mismatched": mismatched, "total": total, "rows": rows}
