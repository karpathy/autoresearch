"""Documented lab.py commands must parse with the runtime parser and carry --confirmed when they run the evaluator."""

import contextlib
import io
import re
import shlex
import unittest
from pathlib import Path

try:
    import support  # noqa: F401  discover -s tests
except ImportError:
    from tests import support  # noqa: F401  package import

from autoresearch_runtime.cli import build_parser
from autoresearch_runtime.runtimeops import _OPERATION_KINDS

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / ".agents" / "skills" / "autoresearch"
DOCS = [REPO / "README.md", SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
MIN_COMMANDS = 10
FENCE = re.compile(r"^\s*(```|~~~)")
PLACEHOLDER = re.compile(r"<[^<>\s]+>")
PLACEHOLDER_TOKEN = "PLACEHOLDER"


def documented_commands():
    """Yield (where, command) for each lab.py line in a fenced block, joining backslash continuations."""
    for path in DOCS:
        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        index = 0
        while index < len(lines):
            line = lines[index]
            if FENCE.match(line):
                in_fence = not in_fence
            elif in_fence and "lab.py" in line:
                where = f"{path.relative_to(REPO).as_posix()}:{index + 1}"
                logical = line.rstrip()
                while logical.endswith("\\") and index + 1 < len(lines):
                    index += 1
                    logical = logical[:-1] + " " + lines[index].strip()
                yield where, logical
            index += 1


def command_argv(where, command):
    """Return the argv after lab.py. Placeholders and bare ellipses become a dummy value; optional brackets are dropped."""
    tokens = shlex.split(PLACEHOLDER.sub(PLACEHOLDER_TOKEN, command), posix=True)
    starts = [i for i, token in enumerate(tokens) if token.endswith("lab.py")]
    if not starts:
        raise ValueError(f"{where}: no lab.py token found")
    argv = []
    for token in tokens[starts[0] + 1:]:
        if token == "...":
            argv.append(PLACEHOLDER_TOKEN)
            continue
        if "{" not in token:
            token = token.removeprefix("[").removesuffix("]")
        if token:
            argv.append(token)
    return argv


def parse_documented(parser, where, argv):
    """Parse with the real parser. argparse output is captured, and a rejection fails with where."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        try:
            return parser.parse_args(argv)
        except SystemExit:
            pass
    detail = stderr.getvalue().strip().splitlines()
    raise AssertionError(f"{where}: runtime parser rejected the documented command: "
                         f"{detail[-1] if detail else 'no argparse message'}")


def runs_evaluator(args):
    """Mirror the runtime: baseline/trial/resume/recover/epoch/probe/replay, plus step --run and trial start."""
    if args.command == "step":
        return bool(getattr(args, "run", False))
    if args.command == "trial":
        return getattr(args, "trial_command", None) in (None, "start")
    return args.command in _OPERATION_KINDS


class DocCommandsTest(unittest.TestCase):
    def test_documented_lab_commands_parse_and_confirm_evaluator_runs(self):
        commands = list(documented_commands())
        self.assertGreaterEqual(len(commands), MIN_COMMANDS, "extraction found too few lab.py commands")
        parser = build_parser()
        evaluator_runs = 0
        for where, command in commands:
            with self.subTest(where=where, command=command):
                args = parse_documented(parser, where, command_argv(where, command))
                if runs_evaluator(args):
                    evaluator_runs += 1
                    self.assertIs(args.confirmed, True,
                                  f"{where}: evaluator-running command needs --confirmed before the subcommand")
        self.assertGreater(evaluator_runs, 0, "no evaluator-running commands were recognised")


if __name__ == "__main__":
    unittest.main()
