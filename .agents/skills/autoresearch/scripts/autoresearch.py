"""One entry point for harness agents. prepare is the only special command."""

from __future__ import annotations

import sys

from autoresearch_runtime.bootstrap import main as prepare_main
from autoresearch_runtime.lab import main as lab_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "prepare":
        return prepare_main(args[1:])
    return lab_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
