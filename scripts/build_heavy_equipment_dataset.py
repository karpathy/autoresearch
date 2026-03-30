from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from run_research import DEFAULT_OUTPUT_ROOT, REPO_ROOT, RunConfig, run_codex_job


SPEC_PATH = REPO_ROOT / "specs" / "heavy_construction_industrial_equipment.md"
PROGRAM_PATH = REPO_ROOT / "program.md"
DATASET_ROOT = DEFAULT_OUTPUT_ROOT / "datasets"
PHASES: dict[str, str] = {
    "1": "1: Heavy Equipment",
    "2": "2: Vehicles & Transport",
    "3": "3: Specialty & Land Clearing",
    "4": "4: Trade Tools",
    "5": "5: General Tools, Survey & Site Support",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a merged heavy-construction equipment dataset.")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=sorted(PHASES),
        default=sorted(PHASES),
        help="Phase numbers to research and merge.",
    )
    parser.add_argument("--depth", default="seed", help="Research depth injected into ${depth}.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="Codex model used for each phase run.")
    parser.add_argument("--tag", help="Optional dataset tag. Defaults to heavy-construction-equipment-<depth>.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going if one phase fails.")
    parser.add_argument(
        "--image-followup",
        action="store_true",
        help="Run targeted recovery passes for rows whose image_url is still Not confirmed after the first pass.",
    )
    parser.add_argument("--image-followup-model", default="gpt-5.4", help="Model used for missing-image recovery jobs.")
    parser.add_argument(
        "--image-followup-batch-size",
        type=int,
        default=20,
        help="Number of unresolved rows to send in each targeted image-recovery batch.",
    )
    parser.add_argument(
        "--image-followup-rounds",
        type=int,
        default=1,
        help="Maximum recovery rounds per phase when --image-followup is enabled.",
    )
    parser.add_argument(
        "--max-unconfirmed-images",
        type=int,
        help="Exit non-zero if the final unresolved image_url count exceeds this threshold.",
    )
    parser.add_argument("--no-search", action="store_false", dest="search", help="Disable Codex web search.")
    parser.set_defaults(search=True)
    return parser.parse_args()


def dataset_dir(depth: str, tag: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = tag or f"heavy-construction-equipment-{depth}"
    path = DATASET_ROOT / f"{timestamp}-{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def parse_markdown_report(report_path: Path, phase_number: str, phase_label: str) -> list[dict[str, str]]:
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


def count_unconfirmed_images(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row["image_url"].strip() == "Not confirmed")


def chunk_rows(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    if size <= 0:
        raise ValueError("Batch size must be greater than zero.")
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def markdown_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "/").strip()


def build_missing_image_spec(phase_label: str, batch_rows: list[dict[str, str]]) -> str:
    lines = [
        "# Heavy Construction Equipment Missing Image Recovery",
        "",
        "## Objective",
        f"Confirm direct image URLs for the listed rows in {phase_label}.",
        "",
        "## Rules",
        "1. Do not add, remove, merge, or rename rows.",
        "2. Revisit every listed row once before stopping.",
        "3. Prefer manufacturer, rental, trade, or Wikimedia Commons image URLs.",
        "4. Return `Not confirmed` only if you still cannot verify a direct image asset.",
        "5. Return only the final markdown table.",
        "",
        "## Output Format",
        "| # | Category | Equipment Name | Also Known As | Description | Image URL |",
        "|---|----------|----------------|---------------|-------------|-----------|",
        "",
        "## Rows To Resolve",
        "| # | Category | Equipment Name | Also Known As | Description | Current Image URL |",
        "|---|----------|----------------|---------------|-------------|-------------------|",
    ]
    for index, row in enumerate(batch_rows, start=1):
        lines.append(
            "| {index} | {category} | {equipment_name} | {also_known_as} | {description} | {image_url} |".format(
                index=index,
                category=markdown_cell(row["category"]),
                equipment_name=markdown_cell(row["equipment_name"]),
                also_known_as=markdown_cell(row["also_known_as"]),
                description=markdown_cell(row["description"]),
                image_url=markdown_cell(row["image_url"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_missing_image_report(report_path: Path) -> list[dict[str, str]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("| # | Category | Equipment Name |"):
            i += 2
            while i < len(lines):
                row_line = lines[i].strip()
                if not row_line.startswith("|"):
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                if len(cells) >= 6:
                    rows.append(
                        {
                            "batch_row_number": cells[0],
                            "category": cells[1],
                            "equipment_name": cells[2],
                            "also_known_as": cells[3],
                            "description": cells[4],
                            "image_url": cells[5],
                        }
                    )
                i += 1
            break
        i += 1
    return rows


def run_missing_image_followup(
    dataset_output_dir: Path,
    phase_number: str,
    phase_label: str,
    rows: list[dict[str, str]],
    *,
    model: str,
    search: bool,
    batch_size: int,
    max_rounds: int,
) -> list[dict[str, object]]:
    specs_dir = dataset_output_dir / "_followup_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    followup_runs: list[dict[str, object]] = []

    for round_number in range(1, max_rounds + 1):
        unresolved = [row for row in rows if row["image_url"].strip() == "Not confirmed"]
        if not unresolved:
            break

        recovered_this_round = 0
        for batch_number, batch_rows in enumerate(chunk_rows(unresolved, batch_size), start=1):
            spec_path = specs_dir / f"phase{phase_number}-missing-images-r{round_number}-b{batch_number}.md"
            spec_path.write_text(build_missing_image_spec(phase_label, batch_rows), encoding="utf-8")
            config = RunConfig(
                name=f"heavy-equipment-phase-{phase_number}-image-followup",
                spec_path=spec_path,
                vars={},
                program_path=PROGRAM_PATH,
                output_root=DEFAULT_OUTPUT_ROOT,
                model=model,
                search=search,
                tag=f"dataset-phase{phase_number}-images-r{round_number}-b{batch_number}",
                extra_instructions=(
                    "This is a targeted image recovery pass. Do not add new equipment rows, "
                    "and do not stop early after finding a few URLs. Revisit every listed row once."
                ),
            )

            run_record: dict[str, object] = {
                "phase_number": phase_number,
                "phase": phase_label,
                "round": round_number,
                "batch": batch_number,
                "requested_rows": len(batch_rows),
            }
            try:
                output_dir = run_codex_job(config)
                parsed_rows = parse_missing_image_report(output_dir / "final_report.md")
                recovered_in_batch = 0
                for parsed_row in parsed_rows:
                    try:
                        batch_index = int(parsed_row["batch_row_number"])
                    except ValueError:
                        continue
                    if batch_index < 1 or batch_index > len(batch_rows):
                        continue
                    new_url = parsed_row["image_url"].strip()
                    target_row = batch_rows[batch_index - 1]
                    if target_row["image_url"].strip() == "Not confirmed" and new_url and new_url != "Not confirmed":
                        target_row["image_url"] = new_url
                        recovered_in_batch += 1
                run_record["output_dir"] = str(output_dir)
                run_record["report_path"] = str(output_dir / "final_report.md")
                run_record["recovered_rows"] = recovered_in_batch
                run_record["remaining_unconfirmed_after_batch"] = count_unconfirmed_images(rows)
                recovered_this_round += recovered_in_batch
            except Exception as exc:  # noqa: BLE001
                run_record["error"] = str(exc)
            followup_runs.append(run_record)

        if recovered_this_round == 0:
            break

    return followup_runs


def render_combined_markdown(dataset_rows: list[dict[str, str]]) -> str:
    lines = ["# Heavy Construction Equipment Dataset Reports", ""]
    current_phase_number = None
    current_phase = None
    current_category = None

    for row in dataset_rows:
        phase_number = row["phase_number"]
        phase = row["phase"]
        category = row["category"]
        if phase_number != current_phase_number or phase != current_phase:
            if lines[-1] != "":
                lines.append("")
            lines.append(f"## Phase {phase_number} - {phase}")
            lines.append("")
            current_phase_number = phase_number
            current_phase = phase
            current_category = None
        if category != current_category:
            if lines[-1] != "":
                lines.append("")
            lines.append(f"### {category}")
            lines.append("")
            lines.append("| # | Equipment Name | Also Known As | Description | Typical Use Phase | Image URL |")
            lines.append("|---|----------------|---------------|-------------|-------------------|-----------|")
            current_category = category
        lines.append(
            "| {row_number} | {equipment_name} | {also_known_as} | {description} | {typical_use_phase} | {image_url} |".format(
                row_number=markdown_cell(row["row_number"]),
                equipment_name=markdown_cell(row["equipment_name"]),
                also_known_as=markdown_cell(row["also_known_as"]),
                description=markdown_cell(row["description"]),
                typical_use_phase=markdown_cell(row["typical_use_phase"]),
                image_url=markdown_cell(row["image_url"]),
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    dataset_output_dir: Path,
    dataset_rows: list[dict[str, str]],
    phase_runs: list[dict[str, object]],
    *,
    extra_manifest: dict[str, object] | None = None,
) -> None:
    csv_path = dataset_output_dir / "heavy_construction_equipment_dataset.csv"
    json_path = dataset_output_dir / "heavy_construction_equipment_dataset.json"
    manifest_path = dataset_output_dir / "manifest.json"
    combined_markdown_path = dataset_output_dir / "combined_reports.md"

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
        writer.writerows(dataset_rows)

    json_path.write_text(json.dumps(dataset_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "row_count": len(dataset_rows),
        "unconfirmed_image_count": count_unconfirmed_images(dataset_rows),
        "phase_runs": phase_runs,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    combined_markdown_path.write_text(render_combined_markdown(dataset_rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_output_dir = dataset_dir(args.depth, args.tag)
    dataset_rows: list[dict[str, str]] = []
    phase_runs: list[dict[str, object]] = []
    image_followup_runs: list[dict[str, object]] = []

    for phase_number in args.phases:
        phase_label = PHASES[phase_number]
        run_tag = f"dataset-phase{phase_number}-{args.depth}"
        instructions = (
            "This is part of a multi-phase dataset build. Research only the requested phase, "
            "include every listed category inside that phase, return only the final markdown tables, "
            "and do not inspect or mine old results folders for content."
        )
        config = RunConfig(
            name=f"heavy-equipment-phase-{phase_number}",
            spec_path=SPEC_PATH,
            vars={"phase": phase_label, "depth": args.depth},
            program_path=PROGRAM_PATH,
            output_root=DEFAULT_OUTPUT_ROOT,
            model=args.model,
            search=args.search,
            tag=run_tag,
            extra_instructions=instructions,
        )

        try:
            output_dir = run_codex_job(config)
            report_path = output_dir / "final_report.md"
            rows = parse_markdown_report(report_path, phase_number, phase_label)
            dataset_rows.extend(rows)
            phase_runs.append(
                {
                    "phase_number": phase_number,
                    "phase": phase_label,
                    "output_dir": str(output_dir),
                    "report_path": str(report_path),
                    "row_count": len(rows),
                    "unconfirmed_image_count": count_unconfirmed_images(rows),
                }
            )
        except Exception as exc:  # noqa: BLE001
            phase_runs.append(
                {
                    "phase_number": phase_number,
                    "phase": phase_label,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                write_outputs(dataset_output_dir, dataset_rows, phase_runs)
                raise

    initial_unconfirmed = count_unconfirmed_images(dataset_rows)
    if args.image_followup:
        for phase_number in args.phases:
            phase_label = PHASES[phase_number]
            phase_rows = [row for row in dataset_rows if row["phase_number"] == phase_number]
            if not phase_rows or count_unconfirmed_images(phase_rows) == 0:
                continue
            image_followup_runs.extend(
                run_missing_image_followup(
                    dataset_output_dir,
                    phase_number,
                    phase_label,
                    phase_rows,
                    model=args.image_followup_model,
                    search=args.search,
                    batch_size=args.image_followup_batch_size,
                    max_rounds=args.image_followup_rounds,
                )
            )

    final_unconfirmed = count_unconfirmed_images(dataset_rows)
    write_outputs(
        dataset_output_dir,
        dataset_rows,
        phase_runs,
        extra_manifest={
            "initial_unconfirmed_image_count": initial_unconfirmed,
            "final_unconfirmed_image_count": final_unconfirmed,
            "image_followup_enabled": args.image_followup,
            "image_followup_model": args.image_followup_model if args.image_followup else None,
            "image_followup_runs": image_followup_runs,
            "max_unconfirmed_images": args.max_unconfirmed_images,
        },
    )
    print(f"dataset output: {dataset_output_dir}")
    print(f"rows: {len(dataset_rows)}")
    print(f"unconfirmed_images: {final_unconfirmed}")
    if final_unconfirmed and not args.image_followup:
        print("hint: rerun with --image-followup for targeted recovery of Not confirmed image rows.")
    if args.max_unconfirmed_images is not None and final_unconfirmed > args.max_unconfirmed_images:
        print(
            (
                f"error: unresolved image_url rows ({final_unconfirmed}) exceed "
                f"--max-unconfirmed-images {args.max_unconfirmed_images}"
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
