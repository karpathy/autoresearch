import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "autoresearchbench"
EXAMPLE_JSONL = FIXTURE_DIR / "examples.jsonl"


def test_download_autoresearchbench_file_file_url_writes_bytes_and_metadata(tmp_path):
    from examples import autoresearchbench_adapter as adapter

    source_bytes = b'{"question":"download fixture","answer":"Paper","arxiv_id":"2401.12345"}\n'
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "downloaded.jsonl"
    source_path.write_bytes(source_bytes)

    metadata = adapter.download_autoresearchbench_file(source_path.as_uri(), output_path)

    assert output_path.read_bytes() == source_bytes
    assert metadata["benchmark"] == "AutoResearchBench"
    assert metadata["mode"] == "download"
    assert metadata["source_url"] == source_path.as_uri()
    assert metadata["output_path"] == str(output_path)
    assert metadata["byte_count"] == len(source_bytes)
    assert metadata["metadata"]["url_scheme"] == "file"
    assert metadata["metadata"]["output_exists"] is True
    assert "official_score" not in metadata
    assert "tasks" not in metadata


def test_download_autoresearchbench_file_refuses_overwrite_unless_requested(tmp_path):
    from examples import autoresearchbench_adapter as adapter

    first_source = tmp_path / "first.jsonl"
    second_source = tmp_path / "second.jsonl"
    output_path = tmp_path / "downloaded.jsonl"
    first_bytes = b'{"question":"first","answer":"Paper","arxiv_id":"2401.11111"}\n'
    second_bytes = b'{"question":"second","answer":"Paper","arxiv_id":"2401.22222"}\n'
    first_source.write_bytes(first_bytes)
    second_source.write_bytes(second_bytes)

    adapter.download_autoresearchbench_file(first_source.as_uri(), output_path)

    with pytest.raises(FileExistsError):
        adapter.download_autoresearchbench_file(second_source.as_uri(), output_path)
    assert output_path.read_bytes() == first_bytes

    metadata = adapter.download_autoresearchbench_file(
        second_source.as_uri(),
        output_path,
        overwrite=True,
    )

    assert output_path.read_bytes() == second_bytes
    assert metadata["byte_count"] == len(second_bytes)
    assert metadata["metadata"]["overwritten"] is True



def test_download_autoresearchbench_file_missing_token_env_raises(tmp_path, monkeypatch):
    from examples import autoresearchbench_adapter as adapter

    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "downloaded.jsonl"
    source_path.write_text(
        '{"question":"token fixture","answer":"Paper","arxiv_id":"2401.33333"}\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("AUTORESEARCHBENCH_TEST_TOKEN", raising=False)

    with pytest.raises(ValueError, match="AUTORESEARCHBENCH_TEST_TOKEN"):
        adapter.download_autoresearchbench_file(
            source_path.as_uri(),
            output_path,
            token_env="AUTORESEARCHBENCH_TEST_TOKEN",
        )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_autoresearchbench_loads_and_normalizes_deep_and_wide_records():
    from examples import autoresearchbench_adapter as adapter

    records = adapter.load_autoresearchbench_jsonl(EXAMPLE_JSONL)
    report = adapter.build_autoresearchbench_report(records)

    assert report["benchmark"] == "AutoResearchBench"
    assert report["task_count"] == 3
    assert report["task_type_counts"] == {"deep_research": 1, "wide_research": 2}

    records = report["tasks"]
    deep = records[0]
    assert deep["task_type"] == "deep_research"
    assert deep["question"] == "Which paper introduced Example Deep Method?"
    assert deep["answer_titles"] == ["Example Deep Method"]
    assert deep["expected_arxiv_ids"] == ["2401.00001"]

    wide = records[1]
    assert wide["task_type"] == "wide_research"
    assert wide["answer_titles"] == ["Wide Paper One", "Wide Paper Two"]
    assert wide["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]

    explicit = records[2]
    assert explicit["task_type"] == "wide_research"
    assert explicit["answer_titles"] == ["Explicit Wide Paper"]
    assert explicit["expected_arxiv_ids"] == ["2401.00004"]

    for record in records:
        assert "candidate_papers" not in record
        assert "score" not in record


def test_autoresearchbench_load_jsonl_returns_normalized_records():
    from examples import autoresearchbench_adapter as adapter

    records = adapter.load_autoresearchbench_jsonl(EXAMPLE_JSONL)

    assert [record["task_type"] for record in records] == [
        "deep_research",
        "wide_research",
        "wide_research",
    ]
    assert records[0]["answer_titles"] == ["Example Deep Method"]
    assert records[1]["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]




def test_autoresearchbench_researchfs_task_rows_export_gold_ids_and_empty_candidates():
    from examples import autoresearchbench_adapter as adapter

    records = adapter.load_autoresearchbench_jsonl(EXAMPLE_JSONL)
    report = adapter.build_autoresearchbench_report(records)
    rows = adapter.researchfs_task_rows(report)

    assert len(rows) == 3
    assert [row["task_type"] for row in rows] == [
        "deep_research",
        "wide_research",
        "wide_research",
    ]
    assert rows[0]["expected_arxiv_ids"] == ["2401.00001"]
    assert rows[0]["gold_paper_ids"] == ["2401.00001"]
    assert rows[0]["gold_arxiv_ids"] == ["2401.00001"]
    assert rows[1]["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert rows[1]["gold_paper_ids"] == ["2401.00002", "2401.00003"]
    assert rows[1]["gold_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert rows[2]["expected_arxiv_ids"] == ["2401.00004"]
    assert rows[2]["gold_paper_ids"] == ["2401.00004"]
    assert rows[2]["gold_arxiv_ids"] == ["2401.00004"]
    assert all(row["candidate_papers"] == [] for row in rows)

    exported = adapter.export_researchfs_tasks_jsonl(report)
    exported_rows = [json.loads(line) for line in exported.splitlines()]
    assert exported_rows == rows
    assert all(row["candidate_papers"] == [] for row in exported_rows)


def test_autoresearchbench_evaluate_predictions_reports_deep_hit_wide_partial_and_missing():
    from examples import autoresearchbench_adapter as adapter

    records = adapter.load_autoresearchbench_jsonl(EXAMPLE_JSONL)
    predictions = [
        {"task_id": records[0]["task_id"], "predicted_arxiv_ids": ["2401.00001"]},
        {"task_id": records[1]["task_id"], "predicted_arxiv_ids": ["2401.00002", "2401.99999"]},
    ]

    report = adapter.evaluate_autoresearchbench_predictions(
        adapter.build_autoresearchbench_report(records),
        predictions,
    )

    assert report["benchmark"] == "AutoResearchBench"
    assert report["kind"] == "deterministic_arxiv_id_overlap"
    assert report["scope"] == "adapter_side_sanity_evaluation"
    assert report["mode"] == "deterministic_adapter_evaluation"
    assert report["evaluation_scope"] == "deterministic_arxiv_id_overlap"
    assert "official_score" not in report
    assert report["aggregate"]["task_count"] == 3
    assert report["aggregate"]["evaluated_count"] == 2
    assert report["task_count"] == 3
    assert report["prediction_count"] == 2
    assert report["results"] == report["tasks"]

    by_task_id = {result["task_id"]: result for result in report["tasks"]}
    deep = by_task_id[records[0]["task_id"]]
    assert deep["task_type"] == "deep_research"
    assert deep["expected_arxiv_ids"] == ["2401.00001"]
    assert deep["gold_arxiv_ids"] == ["2401.00001"]
    assert deep["predicted_arxiv_ids"] == ["2401.00001"]
    assert deep["hit_arxiv_ids"] == ["2401.00001"]
    assert deep["matched_arxiv_ids"] == ["2401.00001"]
    assert deep["iou"] == 1.0
    assert deep["recall"] == 1.0
    assert deep["precision"] == 1.0
    assert deep["evaluated"] is True
    assert deep["correct"] is True

    wide = by_task_id[records[1]["task_id"]]
    assert wide["task_type"] == "wide_research"
    assert wide["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert wide["gold_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert wide["predicted_arxiv_ids"] == ["2401.00002", "2401.99999"]
    assert wide["hit_arxiv_ids"] == ["2401.00002"]
    assert wide["matched_arxiv_ids"] == ["2401.00002"]
    assert wide["missed_arxiv_ids"] == ["2401.00003"]
    assert wide["extra_arxiv_ids"] == ["2401.99999"]
    assert wide["iou"] == pytest.approx(1 / 3)
    assert wide["recall"] == pytest.approx(1 / 2)
    assert wide["precision"] == pytest.approx(1 / 2)
    assert wide["evaluated"] is True
    assert wide["correct"] is False

    missing = by_task_id[records[2]["task_id"]]
    assert missing["task_type"] == "wide_research"
    assert missing["expected_arxiv_ids"] == ["2401.00004"]
    assert missing["gold_arxiv_ids"] == ["2401.00004"]
    assert missing["predicted_arxiv_ids"] == []
    assert missing["hit_arxiv_ids"] == []
    assert missing["matched_arxiv_ids"] == []
    assert missing["iou"] == 0.0
    assert missing["recall"] == 0.0
    assert missing["precision"] == 0.0
    assert missing["evaluated"] is False

    aggregate = report["aggregate"]
    assert aggregate["deep_accuracy"] == 1.0
    assert aggregate["wide_iou"] == pytest.approx(1 / 3)
    assert aggregate["wide_recall"] == pytest.approx(1 / 2)
    assert aggregate["wide_precision"] == pytest.approx(1 / 2)
    assert aggregate["overall_recall"] == pytest.approx((1 + (1 / 2)) / 2)


def test_autoresearchbench_malformed_json_line_raises_value_error_with_line_number(tmp_path):
    from examples import autoresearchbench_adapter as adapter

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text(
        '{"question":"ok","answer":"A","arxiv_id":"2401.1"}\n'
        '{"question":"broken","answer":\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 2"):
        adapter.load_autoresearchbench_jsonl(bad_jsonl)


def test_autoresearchbench_missing_required_field_raises_value_error(tmp_path):
    from examples import autoresearchbench_adapter as adapter

    missing = tmp_path / "missing.jsonl"
    missing.write_text(
        json.dumps({"question": "Missing answer", "arxiv_id": "2401.00005"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"answer"):
        adapter.load_autoresearchbench_jsonl(missing)


def test_autoresearchbench_cli_compact_json_parses():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--input-file",
            str(EXAMPLE_JSONL),
            "--compact",
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["benchmark"] == "AutoResearchBench"
    assert report["task_count"] == 3
    assert report["task_type_counts"] == {"deep_research": 1, "wide_research": 2}
    assert report["tasks"][0]["answer_titles"] == ["Example Deep Method"]


def test_autoresearchbench_cli_download_only_compact_json(tmp_path):
    source_bytes = b'{"question":"cli download","answer":"Paper","arxiv_id":"2401.33333"}\n'
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "downloaded.jsonl"
    source_path.write_bytes(source_bytes)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--download-url",
            source_path.as_uri(),
            "--download-output",
            str(output_path),
            "--compact",
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert output_path.read_bytes() == source_bytes
    assert report["benchmark"] == "AutoResearchBench"
    assert report["mode"] == "download"
    assert report["source_url"] == source_path.as_uri()
    assert report["output_path"] == str(output_path)
    assert report["byte_count"] == len(source_bytes)
    assert report["metadata"]["url_scheme"] == "file"
    assert report["metadata"]["output_exists"] is True
    assert "tasks" not in report
    assert "official_score" not in report


def test_autoresearchbench_cli_requires_input_file_or_download_url():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--compact",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert "input-file" in completed.stderr
    assert "download-url" in completed.stderr


def test_autoresearchbench_cli_output_and_jsonl_output_write_expected_files(tmp_path):
    report_path = tmp_path / "report.json"
    jsonl_path = tmp_path / "normalized.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--input-file",
            str(EXAMPLE_JSONL),
            "--compact",
            "--output",
            str(report_path),
            "--jsonl-output",
            str(jsonl_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_report = json.loads(completed.stdout)
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert written_report == stdout_report
    assert written_report["task_type_counts"] == {"deep_research": 1, "wide_research": 2}

    normalized_lines = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert normalized_lines == written_report["tasks"]
    assert normalized_lines[0]["task_type"] == "deep_research"
    assert normalized_lines[1]["task_type"] == "wide_research"
    assert normalized_lines[1]["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]


def test_autoresearchbench_cli_researchfs_output_writes_rows_without_candidates(tmp_path):
    researchfs_path = tmp_path / "researchfs.jsonl"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--input-file",
            str(EXAMPLE_JSONL),
            "--compact",
            "--researchfs-output",
            str(researchfs_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rows = [json.loads(line) for line in researchfs_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["task_type"] == "deep_research"
    assert rows[0]["expected_arxiv_ids"] == ["2401.00001"]
    assert rows[0]["gold_paper_ids"] == ["2401.00001"]
    assert rows[0]["gold_arxiv_ids"] == ["2401.00001"]
    assert rows[1]["task_type"] == "wide_research"
    assert rows[1]["expected_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert rows[1]["gold_paper_ids"] == ["2401.00002", "2401.00003"]
    assert rows[1]["gold_arxiv_ids"] == ["2401.00002", "2401.00003"]
    assert all(row["candidate_papers"] == [] for row in rows)


def test_autoresearchbench_cli_predictions_file_writes_deterministic_evaluation(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    evaluation_path = tmp_path / "evaluation.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"task_id": "line:1", "predicted_arxiv_ids": ["2401.00001"]},
                {"task_id": "line:2", "predicted_arxiv_ids": ["2401.00002", "2401.99999"]},
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.autoresearchbench_adapter",
            "--input-file",
            str(EXAMPLE_JSONL),
            "--compact",
            "--predictions-file",
            str(predictions_path),
            "--evaluation-output",
            str(evaluation_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["kind"] == "deterministic_arxiv_id_overlap"
    assert evaluation["scope"] == "adapter_side_sanity_evaluation"
    assert evaluation["mode"] == "deterministic_adapter_evaluation"
    assert evaluation["evaluation_scope"] == "deterministic_arxiv_id_overlap"
    assert "official_score" not in evaluation
    assert evaluation["aggregate"]["task_count"] == 3
    assert evaluation["aggregate"]["evaluated_count"] == 2

    by_task_id = {result["task_id"]: result for result in evaluation["tasks"]}
    assert by_task_id["line:1"]["iou"] == 1.0
    assert by_task_id["line:1"]["recall"] == 1.0
    assert by_task_id["line:1"]["precision"] == 1.0
    assert by_task_id["line:2"]["iou"] == pytest.approx(1 / 3)
    assert by_task_id["line:2"]["recall"] == pytest.approx(1 / 2)
    assert by_task_id["line:2"]["precision"] == pytest.approx(1 / 2)
    assert by_task_id["line:3"]["evaluated"] is False
    assert by_task_id["line:3"]["predicted_arxiv_ids"] == []
