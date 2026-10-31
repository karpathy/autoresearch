"""Validate a prepare contract and a saved result without starting evaluation."""
import json
import tempfile
from pathlib import Path
from typing import Any

from ..config import load_config
from ..errors import LabError
from ..evaluator import _validate_result


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("preflight")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--program", type=Path, help="Existing confirmed brief, required to validate quoted limits")
    parser.add_argument("--result", type=Path, help="Saved fixture result; no evaluator is executed")


def run(args: Any, labs_root: Path | None) -> dict[str, Any]:
    # Import lazily: prepare itself imports the shared lab CLI.
    from ..bootstrap import prepare_lab
    try:
        output = prepare_lab(args.program, args.spec, dry_run=True)
        preview = output.pop("config_preview")
        output["result_valid"] = None
        if args.result is not None:
            brief_bytes = args.program.resolve().read_bytes() if args.program is not None else b""
            with tempfile.TemporaryDirectory(prefix="autoresearch-result-check-") as directory:
                root = Path(directory)
                # Generated-evaluator staging is gone: validate the original pinned script.
                spec = json.loads(args.spec.read_text(encoding="utf-8"))
                if spec.get("evaluator_generated"):
                    uses_python = preview["evaluator"][0] == "{python}" or Path(preview["evaluator"][0]).name.lower().startswith("python")
                    index = 1 if uses_python else 0
                    base = args.program.parent if args.program else args.spec.parent
                    preview["evaluator"][index] = str((base / spec["evaluator"][index]).resolve())
                config_path = root / "lab.json"
                config_path.write_text(json.dumps(preview), encoding="utf-8")
                (root / "program.snapshot.md").write_bytes(brief_bytes)
                (root / "result.json").write_bytes(args.result.read_bytes())
                _validate_result(load_config(config_path, allow_pending=True, read_only=True), root)
                output["result_valid"] = True
        return output
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise LabError(str(error), code="E_CONTRACT_PREFLIGHT") from error
