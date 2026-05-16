from pathlib import Path

import api


def test_read_results_json_and_best_bpb(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = (
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "a1\t1.500\t5.0\tkeep\tbaseline\n"
        "b2\t1.700\t5.1\tdiscard\tworse\n"
        "c3\t1.400\t5.2\tkeep\tbetter\n"
    )
    Path(api.RESULTS_FILE).write_text(sample)

    rows = api._read_results_json()

    assert len(rows) == 3
    assert rows[0]["val_bpb"] == 1.5
    assert rows[2]["memory_gb"] == 5.2
    assert api._get_best_bpb() == 1.4


def test_best_bpb_ignores_non_keep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\na1\t1.200\t5.0\tdiscard\tnot kept\n"
    Path(api.RESULTS_FILE).write_text(sample)

    assert api._get_best_bpb() is None
