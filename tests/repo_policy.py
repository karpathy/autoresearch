"""Repository-wide checks shared by the test suite.

Generic rules always apply: personal profile paths, credentials, and the tracked-path length limit.
Project-specific rules are not kept in the repository. They are read from `.private/policy-rules.txt`
(ignored by git) or from the file given with --private-rules; each line is `name<TAB>regex`.

`python tests/repo_policy.py` checks the staged blobs, which are what the next commit would contain.
`python tests/repo_policy.py --history` checks every blob reachable from any ref and requires private
rules, because history is the publication gate. Output names the path and the rule only; matched text
is redacted, so the report never repeats the data it found.
"""

import argparse
import io
import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

# Windows clones live under a root of up to ~139 characters, so tracked paths stay below 120.
MAX_TRACKED_PATH = 120

PRIVATE_RULES_RELATIVE = Path(".private") / "policy-rules.txt"
GITLINK_MODE = "160000"

# Placeholder account names that docs and examples may use in profile paths.
STUB_NAMES = frozenset({"user", "example", "test", "runner", "me", "you", "name"})

UNREADABLE_ARCHIVE = "unreadable archive"
REDACTED = "[redacted]"
_PROFILE_NAME = r"([^\\/\s\"'`<>|*?()\[\],;:]+)"

# (rule name, pattern, whether group 1 is an account name that stub names exempt)
_GENERIC_RULES = (
    ("windows user profile", re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+" + _PROFILE_NAME, re.IGNORECASE), True),
    ("unix home directory", re.compile(r"/(?:Users|home)/" + _PROFILE_NAME, re.IGNORECASE), True),
    ("openai-style API key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"), False),
    ("GitHub token", re.compile(r"(?<![A-Za-z0-9])(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), False),
    ("Slack token", re.compile(r"(?<![A-Za-z0-9])xox[abposr]-[A-Za-z0-9-]{10,}"), False),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}"), False),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}"), False),
    ("private key", re.compile(r"PRIVATE[ ]KEY"), False),
)


def load_private_rules(path: Path) -> list[tuple[str, re.Pattern[str]]]:
    """Read `name<TAB>regex` lines. Blank lines and lines starting with # are ignored.

    Errors name the file and line only, never the pattern, so a bad rule cannot leak through a traceback.
    """
    rules = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, separator, pattern = line.partition("\t")
        if not separator or not name.strip() or not pattern:
            raise ValueError(f"{path.name}: line {number} must read 'name<TAB>regex'")
        try:
            rules.append((name.strip(), re.compile(pattern, re.IGNORECASE)))
        except re.error:
            raise ValueError(f"{path.name}: line {number} has an invalid regex") from None
    return rules


def default_private_rules(repo: Path) -> list[tuple[str, re.Pattern[str]]]:
    path = repo / PRIVATE_RULES_RELATIVE
    return load_private_rules(path) if path.is_file() else []


def tracked_paths(repo: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo, check=True, capture_output=True,
    ).stdout
    return [path for path in output.decode("utf-8").split("\0") if path]


def long_paths(paths: list[str], limit: int = MAX_TRACKED_PATH) -> list[str]:
    return [path for path in paths if len(path) > limit]


def _active_rules(extra_rules):
    return [*_GENERIC_RULES, *((name, pattern, False) for name, pattern in extra_rules)]


def _matches(text: str, extra_rules=()):
    """Yield (rule, start, end) for every match that breaks a rule."""
    for rule, pattern, stub_aware in _active_rules(extra_rules):
        for match in pattern.finditer(text):
            if stub_aware and match.group(1).casefold() in STUB_NAMES:
                continue
            yield rule, match.start(), match.end()


def scan_bytes(data: bytes, extra_rules=()) -> list[str]:
    """Names of the rules that the content breaks. Matched text is never returned.

    Configured rules are text patterns, so they skip binary content, where short names match by chance.
    """
    if b"\0" in data:
        extra_rules = ()
    text = data.decode("utf-8", errors="replace")
    return sorted({rule for rule, _, _ in _matches(text, extra_rules)})


def scan_member(name: str, data: bytes, extra_rules=()) -> list[str]:
    """Rules broken by a file's path and content; zip archives are scanned member by member."""
    found = set(scan_bytes(name.encode("utf-8"), extra_rules))
    if name.casefold().endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    found.update(scan_member(info.filename, archive.read(info), extra_rules))
        except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, OSError):
            found.add(UNREADABLE_ARCHIVE)
    else:
        found.update(scan_bytes(data, extra_rules))
    return sorted(found)


def redact_matches(text: str, extra_rules=()) -> str:
    """Replace every rule match in text, so a path can be printed without repeating private data."""
    spans = sorted((start, end) for _, start, end in _matches(text, extra_rules))
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start >= cursor:
            parts.append(text[cursor:start])
            parts.append(REDACTED)
            cursor = end
        elif end > cursor:
            cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, input=stdin,
    ).stdout


def _read_blobs(repo: Path, oids: list[str]) -> dict[str, bytes]:
    """Contents of the blob objects among oids, read with one `git cat-file --batch` session."""
    if not oids:
        return {}
    stream = io.BytesIO(_git(repo, "cat-file", "--batch", stdin="\n".join(oids).encode("utf-8") + b"\n"))
    blobs: dict[str, bytes] = {}
    for _ in oids:
        header = stream.readline().split()
        if len(header) != 3:
            raise RuntimeError("git cat-file returned an unexpected header")
        oid, kind, size = header[0].decode("ascii"), header[1], int(header[2])
        content = stream.read(size)
        stream.read(1)
        if kind == b"blob":
            blobs[oid] = content
    return blobs


def _scan_entries(repo: Path, entries: list[tuple[str, str]], extra_rules) -> list[tuple[str, str]]:
    """Scan (object id, path) pairs. Blobs are checked by path and content; other objects by path only."""
    blobs = _read_blobs(repo, list(dict.fromkeys(oid for oid, _ in entries)))
    violations = set()
    for oid, path in entries:
        label = path or oid
        if oid in blobs:
            rules = scan_member(label, blobs[oid], extra_rules)
        else:
            rules = scan_bytes(label.encode("utf-8"), extra_rules)
        violations.update((redact_matches(label, extra_rules), rule) for rule in rules)
    return sorted(violations)


def tracked_violations(repo: Path, extra_rules=()) -> list[tuple[str, str]]:
    """(redacted path, rule) for each staged blob that breaks a rule.

    The index is scanned rather than the working tree, so a clean working copy cannot hide staged
    content and a file deleted from disk is still checked as it would be committed.
    """
    entries = []
    for record in _git(repo, "ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        meta, _, path = record.decode("utf-8").partition("\t")
        mode, oid, _stage = meta.split()
        if mode != GITLINK_MODE:
            entries.append((oid, path))
    return _scan_entries(repo, entries, extra_rules)


def history_violations(repo: Path, extra_rules=()) -> list[tuple[str, str]]:
    """(redacted path, rule) for each blob reachable from any ref, including deleted files."""
    listing = _git(repo, "rev-list", "--objects", "--all").decode("utf-8")
    entries = []
    for line in listing.splitlines():
        oid, _, path = line.partition(" ")
        entries.append((oid, path))
    return _scan_entries(repo, entries, extra_rules)


def _printable(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main(argv: list[str] | None = None, repo: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when the repository carries private or personal data.")
    parser.add_argument("--history", action="store_true",
                        help="scan every blob reachable from any ref; requires private rules")
    parser.add_argument("--private-rules", type=Path, metavar="PATH",
                        help="file of name<TAB>regex rules (default: .private/policy-rules.txt if present)")
    args = parser.parse_args(argv)
    repo = repo or Path(__file__).resolve().parents[1]
    if args.private_rules is not None and not args.private_rules.is_file():
        print(f"policy error: rules file not found: {args.private_rules.name}", file=sys.stderr)
        return 2
    try:
        rules = (load_private_rules(args.private_rules) if args.private_rules is not None
                 else default_private_rules(repo))
    except ValueError as error:
        print(f"policy error: {error}", file=sys.stderr)
        return 2
    print(f"private rules: {len(rules)} loaded" if rules else "private rules: none", file=sys.stderr)
    if args.history and not rules:
        print("policy error: --history needs private rules (--private-rules PATH)", file=sys.stderr)
        return 2
    violations = history_violations(repo, rules) if args.history else tracked_violations(repo, rules)
    for path, rule in violations:
        print(_printable(f"{path}: {rule}"))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
