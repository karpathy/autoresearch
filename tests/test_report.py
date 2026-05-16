from pathlib import Path

import report


def test_parse_results_and_trend():
    results_tsv = (
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a1\t1.500\t5.0\tkeep\tbaseline\n"
        "b2\t1.450\t5.1\tkeep\timproved\n"
        "c3\t1.700\t5.2\tdiscard\tworse\n"
    )

    rows = report._parse_results(results_tsv)

    assert len(rows) == 3
    assert rows[0]["val_bpb"] == 1.5

    trend = report._trend_analysis(rows)
    assert "Trend:" in trend


def test_generate_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path)

    rg = report.ReportGenerator()
    path = rg.generate(
        results_tsv=(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\na1\t1.500\t5.0\tkeep\tbaseline\n"
        ),
        directives=[
            {
                "category": "ml",
                "hypothesis": "h",
                "priority": "high",
                "risk": "low",
                "rationale": "r",
            }
        ],
        round_num=1,
        tag="apr21",
    )

    written = Path(path)
    assert written.exists()
    text = written.read_text()
    assert "Autoresearch Report" in text
    assert "apr21" in text
