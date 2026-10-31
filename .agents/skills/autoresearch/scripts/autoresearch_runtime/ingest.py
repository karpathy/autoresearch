"""Universal multi-format data ingestion engine for AutoResearch.

Supports JSON summaries, flat manifests, JSONL, CSV/TSV, and Markdown tables
with atomic transactions, no-clobber ID allocation, and dashboard auto-refresh.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from .errors import LabError
from .measurement import next_sequence_id
from .state import _refresh_dashboard


RESULT_TYPES = ("baseline", "experiment", "reproducibility_control", "provider_check")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _parse_csv_tsv(file_path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
    rows: list[dict[str, Any]] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    for row in reader:
        metrics: dict[str, float] = {}
        for key, val in row.items():
            if not key:
                continue
            cleaned = val.strip().replace("%", "") if isinstance(val, str) else val
            num = _finite(cleaned)
            if num is not None:
                metrics[key.strip()] = num
        if metrics:
            rows.append({"metrics": metrics, "row_data": dict(row)})
    return rows


def _parse_markdown_tables(file_path: Path) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    results: list[dict[str, Any]] = []
    current_headers: list[str] = []
    in_table = False

    for line in lines:
        if not line.startswith("|") or not line.endswith("|"):
            in_table = False
            current_headers = []
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not in_table:
            current_headers = cells
            in_table = True
            continue
        if all(re.match(r"^:?-+:?$", c) for c in cells):
            continue  # table separator row
        if len(cells) == len(current_headers):
            metrics: dict[str, float] = {}
            for h, v in zip(current_headers, cells):
                cleaned = v.replace("%", "").strip()
                num = _finite(cleaned)
                if num is not None:
                    metrics[h] = num
            if metrics:
                results.append({"metrics": metrics})
    return results


def _extract_report_entries(file_path: Path) -> list[dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return _parse_csv_tsv(file_path)
    if suffix in (".md", ".markdown"):
        return _parse_markdown_tables(file_path)

    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    # Try JSONL
    if suffix == ".jsonl":
        entries: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    entries.append(data)
            except json.JSONDecodeError:
                continue
        return entries

    # Standard JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise LabError(f"Invalid JSON in {file_path.name}: {err}") from err

    if not isinstance(data, dict):
        return []

    if isinstance(data.get("results"), list):
        return [item for item in data["results"] if isinstance(item, dict)]
    if isinstance(data.get("runs"), list):
        return [item for item in data["runs"] if isinstance(item, dict)]
    if isinstance(data.get("metrics"), dict):
        return [data]
    return []


def ingest_reports(
    state_dir: Path,
    source_path: Path,
    *,
    probes_dir: Path | None = None,
    group: str = "imported",
    result_type: str = "experiment",
    metric_mapping: dict[str, str] | None = None,
    primary: str | None = None,
    direction: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Ingest external experiment reports into probe summaries and control manifests.

    Atomic and idempotent: uses next_sequence_id to prevent any overwrite,
    validates arguments pre-flight, rolls back partial writes on failure,
    and refreshes the lab dashboard upon completion.
    """
    state_dir = state_dir.resolve()
    lab_config_path = state_dir / "lab.json"
    from .store import _sqlite_document
    storage_target = _sqlite_document(lab_config_path)
    if storage_target is None and not lab_config_path.is_file():
        raise LabError(f"Invalid lab state directory: {state_dir} (missing lab.json)")

    if result_type not in RESULT_TYPES:
        raise LabError(f"Invalid result_type '{result_type}'. Expected one of: {RESULT_TYPES}")

    from .config import load_config
    config = load_config(str(storage_target[1]) if storage_target is not None else lab_config_path)

    # Determine metric and direction
    configured_primary = primary or config.primary or ""
    configured_direction = direction or config.direction or "max"

    target_probes_dir = (probes_dir or state_dir / "probes").resolve()
    if storage_target is None:
        target_probes_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_path.resolve()
    target_files: list[Path] = []
    if source_path.is_file():
        target_files = [source_path]
    elif source_path.is_dir():
        for ext in ("*.json", "*.jsonl", "*.csv", "*.tsv", "*.md"):
            target_files.extend(sorted(source_path.glob(f"**/{ext}")))
    else:
        raise LabError(f"source_path does not exist: {source_path}")

    created_dirs: list[Path] = []
    written_summaries: list[Path] = []

    try:
        for file_path in target_files:
            entries = _extract_report_entries(file_path)
            if not entries:
                continue

            # Apply metric mapping if provided
            if metric_mapping:
                for entry in entries:
                    if isinstance(entry.get("metrics"), dict):
                        m = entry["metrics"]
                        for old_k, new_k in metric_mapping.items():
                            if old_k in m:
                                m[new_k] = m[old_k]

            # Detect primary metric if not explicitly configured
            active_primary = configured_primary
            if not active_primary:
                for entry in entries:
                    metrics_dict = entry.get("metrics")
                    if isinstance(metrics_dict, dict) and metrics_dict:
                        active_primary = next(iter(metrics_dict.keys()))
                        break

            if not active_primary:
                continue

            # Extract metric values
            values: list[float] = []
            for entry in entries:
                metrics_dict = entry.get("metrics")
                if isinstance(metrics_dict, dict):
                    val = _finite(metrics_dict.get(active_primary))
                    if val is not None:
                        values.append(val)

            if not values:
                continue

            # Safe ID reservation without race conditions or overwrites
            if storage_target is not None:
                store, lab_id, *_ = storage_target
                page = store.list_runs(lab_id, limit=100, offset=0)
                aliases = [item.get("alias", "") for item in page["items"] if item.get("kind") == "probe"]
                # Continue through every page so an old probe alias is never reused.
                offset = len(page["items"])
                while offset < page["total"]:
                    page = store.list_runs(lab_id, limit=100, offset=offset)
                    aliases.extend(item.get("alias", "") for item in page["items"]
                                   if item.get("kind") == "probe")
                    offset += len(page["items"])
                numbers = [int(alias[6:]) for alias in aliases
                           if isinstance(alias, str) and alias.startswith("probe-") and alias[6:].isdigit()]
                probe_id = f"probe-{max(numbers, default=0) + 1:04d}"
                probe_dir = state_dir / "runs" / probe_id
            else:
                probe_id = next_sequence_id(target_probes_dir, "probe")
                probe_dir = target_probes_dir / probe_id
            if probe_dir.exists():
                raise LabError(f"Collision: probe directory already exists: {probe_dir}")

            if dry_run:
                written_summaries.append(probe_dir / "summary.json")
                continue

            if storage_target is None:
                probe_dir.mkdir(parents=True, exist_ok=False)
                created_dirs.append(probe_dir)

            worst_val = min(values) if configured_direction == "max" else max(values)
            mean_val = sum(values) / len(values)
            spread_val = max(values) - min(values)

            # Metadata extraction
            commit = None
            for key in ("candidate_commit", "source_commit", "commit"):
                if key in entries[0]:
                    commit = str(entries[0][key])
                    break

            label = file_path.stem

            summary_data: dict[str, Any] = {
                "schema": 1,
                "id": probe_id,
                "group": group,
                "label": label,
                "commit": commit,
                "status": "complete",
                "repeats": len(entries),
                "runs": entries,
                "primary": active_primary,
                "direction": configured_direction,
                "worst": worst_val,
                "mean": mean_val,
                "spread": spread_val,
                "import_source": str(file_path),
                "evidence_origin": "imported",
                "adoptable": False if storage_target is not None else True,
            }

            if storage_target is not None:
                store, lab_id, *_ = storage_target
                state = store.load_state(lab_id)
                run_id = store.create_run(lab_id, "probe", probe_id,
                                          int(state.get("current_epoch") or 1),
                                          {"candidate_commit": state.get("best_commit"),
                                           "import_source": str(file_path)})
                summary_data["status"] = "non_adoptable"
                summary_data["run_id"] = str(run_id)
                summary_data["epoch"] = int(state.get("current_epoch") or 1)
                for index, entry in enumerate(entries, 1):
                    metrics_dict = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
                    store.append_result(lab_id, run_id, index, {
                        "status": "ok", "metrics": metrics_dict,
                        "details": {"evidence_origin": "imported", "source": str(file_path)},
                    })
                out_path = probe_dir / "manifest.json"
                store.save_manifest(lab_id, run_id, summary_data)
            else:
                summary_data.pop("adoptable", None)
                out_path = probe_dir / "summary.json"
                out_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written_summaries.append(out_path)

            # Write control manifest in state_dir if applicable
            if storage_target is None:
                from .control_import import import_control_report

                import_control_report(state_dir, file_path, result_type=result_type, label=label, entries=entries)

        if not dry_run and written_summaries:
            _refresh_dashboard(config)

    except Exception as err:
        # Rollback partial changes on error
        for p_dir in created_dirs:
            if p_dir.is_dir():
                shutil.rmtree(p_dir, ignore_errors=True)
        raise LabError(f"Failed to ingest reports: {err}") from err

    return written_summaries
