from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from build_heavy_equipment_dataset import DATASET_ROOT, PHASES


CATEGORY_PHASE_MAP = {
    range(1, 8): "1",
    range(8, 13): "2",
    range(13, 16): "3",
    range(16, 21): "4",
    range(21, 25): "5",
}


def phase_for_category(category: str) -> tuple[str, str]:
    match = re.match(r"^(\d+)\.\s+", category)
    if not match:
        return "", ""
    category_number = int(match.group(1))
    for group, phase_number in CATEGORY_PHASE_MAP.items():
        if category_number in group:
            return phase_number, PHASES[phase_number]
    return "", ""


def parse_report(report_path: Path) -> list[dict[str, str]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    current_category = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("### "):
            current_category = line[4:].strip()
            i += 1
            continue
        if line.startswith("| # | Equipment Name |"):
            i += 2
            while i < len(lines):
                row_line = lines[i].strip()
                if not row_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                if len(cells) >= 6:
                    phase_number, phase_label = phase_for_category(current_category)
                    rows.append(
                        {
                            "phase_number": phase_number,
                            "phase": phase_label,
                            "category": current_category,
                            "row_number": cells[0],
                            "equipment_name": cells[1],
                            "also_known_as": cells[2],
                            "description": cells[3],
                            "typical_use_phase": cells[4],
                            "image_url": cells[5],
                        }
                    )
                i += 1
            continue
        i += 1
    return rows


def write_dataset(report_path: Path, output_dir: Path, rows: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "heavy_construction_equipment_dataset.csv"
    json_path = output_dir / "heavy_construction_equipment_dataset.json"
    manifest_path = output_dir / "manifest.json"
    report_copy_path = output_dir / "combined_report.md"

    fieldnames = [
        "phase_number",
        "phase",
        "category",
        "row_number",
        "equipment_name",
        "also_known_as",
        "description",
        "typical_use_phase",
        "image_url",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_copy_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_report": str(report_path),
        "row_count": len(rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a combined heavy-equipment markdown report to CSV/JSON.")
    parser.add_argument("--report", type=Path, required=True, help="Path to the combined markdown report.")
    parser.add_argument("--output-dir", type=Path, help="Directory for exported dataset artifacts.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = DATASET_ROOT / f"{stamp}-exported-heavy-construction-report"

    rows = parse_report(args.report.resolve())
    write_dataset(args.report.resolve(), output_dir.resolve(), rows)
    print(f"dataset output: {output_dir.resolve()}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
