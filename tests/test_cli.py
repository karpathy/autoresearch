import io
import json
from contextlib import redirect_stdout

import pytest

from autoresearch.cli import main, build_parser


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as ex:
        main(["--help"])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "autoresearch" in out
    assert "toy-train" in out
    assert "info" in out


def test_info_command():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["info"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["autoresearch_version"]
    assert data["tribunal_implemented"] is False
    assert "cuda_available" in data


def test_config_command_defaults():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["n_embd"] > 0
    assert data["device"] == "cpu"


def test_parser_has_expected_subcommands():
    p = build_parser()
    # Introspect via parse — unknown subcommand should fail:
    with pytest.raises(SystemExit):
        p.parse_args(["bogus"])


def test_tribunal_stub_raises():
    from autoresearch import tribunal
    assert tribunal.IMPLEMENTED is False
    with pytest.raises(tribunal.TribunalNotImplementedError):
        tribunal.evaluate()
