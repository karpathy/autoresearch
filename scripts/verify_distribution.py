"""Verify that ResearchFS release archives contain only the lightweight product."""

from __future__ import annotations

import argparse
import email
import re
import tarfile
import zipfile
from pathlib import Path


EXPECTED_NAME = "researchfs-core"
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
FORBIDDEN_DEPENDENCIES = {
    "kernels",
    "matplotlib",
    "modal",
    "numpy",
    "pandas",
    "pyarrow",
    "rustbpe",
    "tiktoken",
    "torch",
}
FORBIDDEN_PATH_PARTS = (
    "/benchmarks/",
    "/nanochat/",
    "/prepare.py",
    "/train.py",
)
FORBIDDEN_SUFFIXES = (".db", ".key", ".pem", ".png")


def _archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise AssertionError(
                f"{path.name}: expected one wheel METADATA file, found {len(metadata_paths)}"
            )
        return archive.read(metadata_paths[0]).decode("utf-8")


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\s(<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def verify_distribution(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise AssertionError(
            f"expected one wheel and one sdist in {directory}, found "
            f"{len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    for archive_path in (*wheels, *sdists):
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise AssertionError(
                f"{archive_path.name}: archive exceeds {MAX_ARCHIVE_BYTES} bytes"
            )
        names = _archive_names(archive_path)
        normalized_names = [f"/{name.lower().lstrip('/')}" for name in names]
        leaked_paths = [
            name
            for name, normalized in zip(names, normalized_names)
            if any(part in normalized for part in FORBIDDEN_PATH_PARTS)
            or normalized.endswith(FORBIDDEN_SUFFIXES)
        ]
        if leaked_paths:
            raise AssertionError(
                f"{archive_path.name}: forbidden release paths: {', '.join(leaked_paths)}"
            )
        if not any(name.endswith("LICENSE") for name in names):
            raise AssertionError(f"{archive_path.name}: LICENSE is missing")

    metadata = email.message_from_string(_wheel_metadata(wheels[0]))
    if metadata.get("Name") != EXPECTED_NAME:
        raise AssertionError(
            f"wheel name is {metadata.get('Name')!r}, expected {EXPECTED_NAME!r}"
        )
    requirements = metadata.get_all("Requires-Dist", [])
    forbidden = sorted(
        {_dependency_name(requirement) for requirement in requirements}
        & FORBIDDEN_DEPENDENCIES
    )
    if forbidden:
        raise AssertionError(f"forbidden runtime dependencies: {', '.join(forbidden)}")

    wheel_names = _archive_names(wheels[0])
    if "researchfs/migrations.py" not in wheel_names:
        raise AssertionError("wheel does not contain researchfs/migrations.py")

    print(
        f"verified {wheels[0].name} and {sdists[0].name}: "
        "lightweight ResearchFS product boundary intact"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="dist", type=Path)
    args = parser.parse_args()
    verify_distribution(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
