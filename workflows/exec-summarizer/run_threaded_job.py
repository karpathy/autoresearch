#!/usr/bin/env python3
"""Run an executive-summary job inside an explicit thread subtree."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import evaluate

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


WORKFLOW_ROOT = Path(__file__).parent
THREADS_ROOT = WORKFLOW_ROOT / "threads"
THREAD_INDEX_PATH = THREADS_ROOT / "index.json"
THREAD_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# ---------------------------------------------------------------------------
# Gate configuration (loaded from workflow.yaml)
# ---------------------------------------------------------------------------

def load_gate_config() -> dict[str, Any]:
    """Load the gates section from workflow.yaml. Returns {} if absent or unreadable."""
    manifest_path = WORKFLOW_ROOT / "workflow.yaml"
    if not manifest_path.exists():
        return {}
    if yaml is None:
        print("Warning: PyYAML not installed — quality gates disabled", file=sys.stderr)
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return manifest.get("gates", {})


def _gate_section(gates: dict[str, Any], section: str) -> dict[str, Any]:
    """Extract a gate sub-section with defaults."""
    raw = gates.get(section, {})
    if not isinstance(raw, dict):
        return {}
    return raw


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_thread_slug(value: str) -> str:
    if not THREAD_SLUG_RE.fullmatch(value):
        raise ValueError(
            "Thread slug must use lowercase letters, numbers, and hyphens only: "
            f"{value!r}"
        )
    return value


def load_prompt() -> str:
    return (WORKFLOW_ROOT / "prompt.txt").read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_single_article(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(f"Expected exactly one article in {path}, found {len(payload)}")
        article = payload[0]
    elif isinstance(payload, dict):
        article = payload
    else:
        raise ValueError(f"Unsupported article payload in {path}")

    required_keys = ["title", "url", "text"]
    missing = [key for key in required_keys if not article.get(key)]
    if missing:
        raise ValueError(f"Article is missing required fields: {', '.join(missing)}")
    return article


def parse_upload_date(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    if re.fullmatch(r"\d{8}", raw_value):
        return f"{raw_value[:4]}-{raw_value[4:6]}-{raw_value[6:8]}"
    return raw_value


def clean_vtt(vtt_path: Path, max_chars: int) -> str:
    transcript_lines: list[str] = []
    last_line = ""

    for raw_line in vtt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in line or re.fullmatch(r"\d+", line):
            continue

        clean_line = re.sub(r"<[^>]+>", "", line).replace("&amp;", "&").strip()
        if not clean_line or clean_line == last_line:
            continue

        transcript_lines.append(clean_line)
        last_line = clean_line

    transcript = " ".join(transcript_lines)
    if len(transcript) <= max_chars:
        return transcript
    return transcript[:max_chars].rsplit(" ", 1)[0]


def run_command(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(command)
            + "\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout


def fetch_youtube_article(url: str, scratch_dir: Path, max_chars: int) -> dict[str, Any]:
    metadata_output = run_command(
        [sys.executable, "-m", "yt_dlp", "--dump-single-json", "--skip-download", url],
        WORKFLOW_ROOT,
    )
    metadata = json.loads(metadata_output)

    run_command(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--write-auto-subs",
            "--sub-lang",
            "en",
            "--skip-download",
            "--output",
            str(scratch_dir / "source"),
            url,
        ],
        WORKFLOW_ROOT,
    )

    vtt_files = sorted(scratch_dir.glob("source*.en.vtt"))
    if not vtt_files:
        raise FileNotFoundError("yt-dlp completed without writing an English subtitle file")

    transcript = clean_vtt(vtt_files[0], max_chars=max_chars)
    (scratch_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    return {
        "id": metadata.get("id") or "youtube-source",
        "title": metadata.get("title") or url,
        "source": metadata.get("uploader") or "YouTube",
        "url": metadata.get("webpage_url") or url,
        "date": parse_upload_date(metadata.get("upload_date"))
        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "text": transcript,
    }


async def generate_summary(article: dict[str, Any]) -> str:
    result = await evaluate.generate_summary(load_prompt(), article)
    if result["error"]:
        raise RuntimeError(result["error"])
    return result["summary"]


# ---------------------------------------------------------------------------
# Quality-gate pipeline
# ---------------------------------------------------------------------------

async def run_gate_pipeline(
    *,
    summary_text: str,
    article: dict[str, Any],
    gates: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[str, dict[str, Any]]:
    """Run self-eval and optional judge gates after summary generation.

    Returns (final_summary_text, gate_results_dict).
    May regenerate the summary on retry.
    """
    gate_results: dict[str, Any] = {"self_eval": None, "judge": None, "attempts": 1, "status": "passed"}

    self_eval_cfg = _gate_section(gates, "self_eval")
    judge_cfg = _gate_section(gates, "judge")

    if not self_eval_cfg.get("enabled", False):
        return summary_text, gate_results

    composite_min = float(self_eval_cfg.get("composite_min", 0))
    dimension_mins = self_eval_cfg.get("dimension_mins") or {}
    on_fail = self_eval_cfg.get("on_fail", "retry")
    max_retries = int(self_eval_cfg.get("max_retries", 2))

    current_summary = summary_text

    for attempt in range(1, max_retries + 2):  # max_retries + 1 total attempts
        gate_results["attempts"] = attempt

        # --- Self-evaluation ---
        print(f"  Self-eval (attempt {attempt})...", file=sys.stderr)
        scores = await evaluate.score_single_summary(current_summary, article)
        gate_results["self_eval"] = scores

        passed, failures = evaluate.check_gate(scores, composite_min, dimension_mins)
        if passed:
            gate_results["status"] = "passed"
            break

        failure_msg = "; ".join(failures)
        print(f"  Gate failed: {failure_msg}", file=sys.stderr)

        if on_fail == "retry" and attempt <= max_retries:
            # Regenerate with feedback
            print("  Retrying with feedback...", file=sys.stderr)
            feedback_prompt = (
                f"Your previous summary scored below quality thresholds:\n"
                f"  Failures: {failure_msg}\n"
                f"  Scores: conciseness={scores.get('conciseness', 0)}, "
                f"relevance={scores.get('relevance', 0)}, "
                f"provenance={scores.get('provenance', 0)}, "
                f"ecq={scores.get('ecq', 0)}, "
                f"composite={scores.get('composite', 0):.2f}\n\n"
                f"Previous summary:\n{current_summary}\n\n"
                f"Please regenerate the summary, addressing the specific failures above."
            )
            system_prompt = load_prompt()
            retry_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n---\n\n"
                f"QUALITY FEEDBACK:\n{feedback_prompt}\n\n---\n\n"
                f"Article Title: {article.get('title', 'Untitled')}\n"
                f"Source: {article.get('source', 'Unknown')}\n"
                f"URL: {article.get('url', 'No URL')}\n\n"
                f"{article.get('text', '')}\n\n---\n\n"
                f"Provide an improved executive summary addressing the quality failures."
            )
            current_summary = (await evaluate.call_llm(retry_prompt)).strip()
            continue

        # Retries exhausted or on_fail is not retry
        if on_fail == "stop":
            gate_results["status"] = "failed"
            write_json(paths["artifacts_dir"] / "gate-results.json", gate_results)
            raise RuntimeError(
                f"Quality gate failed after {attempt} attempt(s): {failure_msg}. "
                f"on_fail=stop — halting."
            )
        # on_fail == "flag" or retries exhausted
        gate_results["status"] = "flagged"
        print(f"  Quality flag set (below threshold after {attempt} attempt(s))", file=sys.stderr)
        break

    # --- Judge evaluation (optional) ---
    if judge_cfg.get("enabled", False):
        judge_model = judge_cfg.get("model")
        judge_persona = judge_cfg.get("persona")
        judge_composite_min = float(judge_cfg.get("composite_min", composite_min))
        judge_dimension_mins = judge_cfg.get("dimension_mins") or dimension_mins
        judge_on_fail = judge_cfg.get("on_fail", "flag")

        print("  Judge evaluation...", file=sys.stderr)
        judge_scores = await evaluate.judge_summary(
            current_summary, article, model=judge_model, persona=judge_persona,
        )
        gate_results["judge"] = judge_scores

        j_passed, j_failures = evaluate.check_gate(judge_scores, judge_composite_min, judge_dimension_mins)
        if not j_passed:
            j_failure_msg = "; ".join(j_failures)
            print(f"  Judge gate failed: {j_failure_msg}", file=sys.stderr)
            if judge_on_fail == "stop":
                gate_results["status"] = "failed"
                write_json(paths["artifacts_dir"] / "gate-results.json", gate_results)
                raise RuntimeError(f"Judge gate failed: {j_failure_msg}. on_fail=stop — halting.")
            if gate_results["status"] != "flagged":
                gate_results["status"] = "flagged"
        else:
            print(f"  Judge passed (composite={judge_scores.get('composite', 0):.2f})", file=sys.stderr)

    # Write gate results to job artifacts
    write_json(paths["artifacts_dir"] / "gate-results.json", gate_results)
    return current_summary, gate_results


def load_thread_index() -> dict[str, Any]:
    if THREAD_INDEX_PATH.exists():
        return read_json(THREAD_INDEX_PATH)
    return {"version": 1, "threads": []}


def update_thread_index(thread_summary: dict[str, Any]) -> None:
    index = load_thread_index()
    threads = index.setdefault("threads", [])
    for position, item in enumerate(threads):
        if item.get("slug") == thread_summary["slug"]:
            threads[position] = thread_summary
            break
    else:
        threads.append(thread_summary)

    threads.sort(key=lambda item: item.get("slug", ""))
    write_json(THREAD_INDEX_PATH, index)


def build_thread_paths(thread_slug: str, job_id: str) -> dict[str, Path]:
    thread_dir = THREADS_ROOT / thread_slug
    job_dir = thread_dir / "jobs" / job_id
    return {
        "thread_dir": thread_dir,
        "job_dir": job_dir,
        "inputs_dir": job_dir / "inputs",
        "artifacts_dir": job_dir / "artifacts",
        "logs_dir": job_dir / "logs",
        "scratch_dir": job_dir / "scratch",
    }


def ensure_directories(paths: dict[str, Path]) -> None:
    THREADS_ROOT.mkdir(exist_ok=True)
    for key in ["thread_dir", "job_dir", "inputs_dir", "artifacts_dir", "logs_dir", "scratch_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def write_job_artifacts(
    *,
    thread_slug: str,
    job_id: str,
    article: dict[str, Any],
    summary_text: str,
    source_type: str,
    article_origin: str,
    paths: dict[str, Path],
    gate_results: dict[str, Any] | None = None,
    quality_flag: str | None = None,
) -> dict[str, Any]:
    generated_at = utc_timestamp()
    article_copy = deepcopy(article)

    write_json(paths["inputs_dir"] / "article.json", article_copy)
    write_json(paths["inputs_dir"] / "articles.json", [article_copy])

    job_manifest: dict[str, Any] = {
        "thread": thread_slug,
        "job_id": job_id,
        "generated_at": generated_at,
        "source_type": source_type,
        "article_origin": article_origin,
        "title": article.get("title"),
        "url": article.get("url"),
        "source": article.get("source"),
        "date": article.get("date"),
        "artifacts": {
            "article": "inputs/article.json",
            "summary": "artifacts/executive-summary.json",
            "gate_results": "artifacts/gate-results.json",
        },
    }
    if quality_flag:
        job_manifest["quality_flag"] = quality_flag
    if gate_results:
        job_manifest["gate_status"] = gate_results.get("status", "skipped")
        job_manifest["gate_attempts"] = gate_results.get("attempts", 1)

    write_json(paths["job_dir"] / "job.json", job_manifest)

    summary_payload: dict[str, Any] = {
        "generated": generated_at,
        "thread": thread_slug,
        "job_id": job_id,
        "source_type": source_type,
        "title": article.get("title"),
        "source": article.get("source"),
        "url": article.get("url"),
        "date": article.get("date"),
        "executive_summary": summary_text,
    }
    if quality_flag:
        summary_payload["quality_flag"] = quality_flag

    write_json(paths["artifacts_dir"] / "executive-summary.json", summary_payload)
    return summary_payload


def promote_thread_summary(
    thread_slug: str,
    summary_payload: dict[str, Any],
    article: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    thread_dir = paths["thread_dir"]
    write_json(thread_dir / "summary.json", summary_payload)
    write_json(
        thread_dir / "thread.json",
        {
            "slug": thread_slug,
            "updated": summary_payload["generated"],
            "title": article.get("title"),
            "source": article.get("source"),
            "url": article.get("url"),
            "latest_job": summary_payload["job_id"],
            "latest_summary": "summary.json",
            "executive_summary": summary_payload["executive_summary"],
        },
    )
    update_thread_index(
        {
            "slug": thread_slug,
            "title": article.get("title"),
            "source": article.get("source"),
            "url": article.get("url"),
            "latest_job": summary_payload["job_id"],
            "updated": summary_payload["generated"],
            "path": f"threads/{thread_slug}",
        }
    )


async def run_thread_job(
    *,
    thread_slug: str,
    job_id: str,
    article: dict[str, Any],
    source_type: str,
    article_origin: str,
    prebuilt_paths: dict[str, Path] | None = None,
    skip_gates: bool = False,
) -> None:
    paths = prebuilt_paths or build_thread_paths(thread_slug, job_id)
    ensure_directories(paths)

    summary_text = await generate_summary(article)

    # --- Quality gates ---
    gate_results: dict[str, Any] = {"self_eval": None, "judge": None, "attempts": 1, "status": "skipped"}
    quality_flag: str | None = None

    if not skip_gates:
        gates = load_gate_config()
        if gates:
            summary_text, gate_results = await run_gate_pipeline(
                summary_text=summary_text,
                article=article,
                gates=gates,
                paths=paths,
            )
            if gate_results["status"] == "flagged":
                quality_flag = "below_threshold"

    summary_payload = write_job_artifacts(
        thread_slug=thread_slug,
        job_id=job_id,
        article=article,
        summary_text=summary_text,
        source_type=source_type,
        article_origin=article_origin,
        paths=paths,
        gate_results=gate_results,
        quality_flag=quality_flag,
    )
    promote_thread_summary(thread_slug, summary_payload, article, paths)

    print(f"Thread: {thread_slug}")
    print(f"Job: {job_id}")
    print(f"Gate: {gate_results['status']} (attempts: {gate_results['attempts']})")
    if gate_results.get("self_eval"):
        se = gate_results["self_eval"]
        print(f"  Self-eval: composite={se.get('composite', 0):.2f} "
              f"(conciseness={se.get('conciseness', 0)} "
              f"relevance={se.get('relevance', 0)} "
              f"provenance={se.get('provenance', 0)} "
              f"ecq={se.get('ecq', 0)})")
    if gate_results.get("judge"):
        j = gate_results["judge"]
        print(f"  Judge: composite={j.get('composite', 0):.2f} "
              f"(conciseness={j.get('conciseness', 0)} "
              f"relevance={j.get('relevance', 0)} "
              f"provenance={j.get('provenance', 0)} "
              f"ecq={j.get('ecq', 0)})")
    print(f"Summary: {paths['thread_dir'] / 'summary.json'}")
    print(f"Job artifacts: {paths['artifacts_dir'] / 'executive-summary.json'}")


async def run_from_json(args: argparse.Namespace) -> None:
    article_path = Path(args.article_json).resolve()
    article = load_single_article(article_path)
    await run_thread_job(
        thread_slug=args.thread,
        job_id=args.job_id,
        article=article,
        source_type="article-json",
        article_origin=str(article_path),
        skip_gates=args.skip_gates,
    )


async def run_from_youtube(args: argparse.Namespace) -> None:
    paths = build_thread_paths(args.thread, args.job_id)
    ensure_directories(paths)
    article = fetch_youtube_article(args.url, paths["scratch_dir"], max_chars=args.max_chars)
    await run_thread_job(
        thread_slug=args.thread,
        job_id=args.job_id,
        article=article,
        source_type="youtube",
        article_origin=args.url,
        prebuilt_paths=paths,
        skip_gates=args.skip_gates,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an executive-summary job in an explicit thread subtree."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_json_parser = subparsers.add_parser(
        "from-json",
        help="Summarize a prepared one-article JSON payload inside a thread subtree.",
    )
    from_json_parser.add_argument("--thread", required=True, type=ensure_thread_slug)
    from_json_parser.add_argument("--article-json", required=True)
    from_json_parser.add_argument("--job-id", default=default_job_id())
    from_json_parser.add_argument("--skip-gates", action="store_true",
                                  help="Bypass quality gates (for debugging)")

    from_youtube_parser = subparsers.add_parser(
        "from-youtube",
        help="Fetch a YouTube transcript into a job subtree and summarize it.",
    )
    from_youtube_parser.add_argument("--thread", required=True, type=ensure_thread_slug)
    from_youtube_parser.add_argument("--url", required=True)
    from_youtube_parser.add_argument("--job-id", default=default_job_id())
    from_youtube_parser.add_argument("--max-chars", type=int, default=12000)
    from_youtube_parser.add_argument("--skip-gates", action="store_true",
                                     help="Bypass quality gates (for debugging)")

    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "from-json":
        await run_from_json(args)
        return
    if args.command == "from-youtube":
        await run_from_youtube(args)
        return
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())