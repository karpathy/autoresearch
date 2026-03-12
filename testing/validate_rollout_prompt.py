from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_autoresearch.config import TTTAutoResearchConfig
from ttt_autoresearch.discover_compat import (
    patch_transformers_kimi_trust_remote_code,
    patch_ttt_discover_kimi_renderer,
    patch_ttt_discover_kimi_tokenizer,
)
from ttt_autoresearch.env import AutoResearchState
from ttt_autoresearch.prompt_builder import build_prompt_for_state
from ttt_autoresearch.runner import AutoResearchRunner, parse_patch_candidate_for_state


patch_ttt_discover_kimi_tokenizer()
patch_ttt_discover_kimi_renderer()
patch_transformers_kimi_trust_remote_code()

from ttt_discover.tinker_utils import renderers


DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing" / "output"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    renderer_name: str
    label: str
    use_gpt_oss_sections: bool = False


MODEL_SPECS = {
    "gpt_oss": ModelSpec(
        key="gpt_oss",
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_high_reasoning",
        label="GPT-OSS-120B high reasoning via Tinker",
        use_gpt_oss_sections=True,
    ),
    "kimi": ModelSpec(
        key="kimi",
        model_name="moonshotai/Kimi-K2.5",
        renderer_name="kimi_k25",
        label="Kimi K2.5 via Tinker",
        use_gpt_oss_sections=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate rollout-style prompt outputs against the current train.py."
    )
    parser.add_argument(
        "--api-key",
        help="Tinker API key. Defaults to TINKER_API_KEY from the environment.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default="gpt_oss",
        help="Model to test. Defaults to the repo's current rollout default.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of rollout-style prompt samples to generate.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "train.py",
        help="train.py to embed in the prompt and validate against.",
    )
    parser.add_argument(
        "--current-val-bpb",
        type=float,
        default=1.0,
        help="Current val_bpb used to build the prompt.",
    )
    parser.add_argument(
        "--target-val-bpb",
        type=float,
        default=0.97,
        help="Target val_bpb used to build the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=26000,
        help="Total prompt-plus-output token budget. Defaults to the repo's GPT-OSS rollout setting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts are saved.",
    )
    return parser.parse_args()


def build_prompt(train_file: Path, current_val_bpb: float, target_val_bpb: float) -> tuple[str, str]:
    train_path = train_file.resolve()
    train_text = train_path.read_text(encoding="utf-8")
    state = AutoResearchState(
        timestep=-1,
        construction=[],
        code=train_text,
        value=-current_val_bpb,
    )
    prompt = build_prompt_for_state(state, target_val_bpb)
    return prompt, train_text


def extract_kimi_sections(text: str) -> dict[str, str | None]:
    think_open = "<think>"
    think_close = "</think>"
    if think_open in text and think_close in text:
        thinking_start = text.index(think_open) + len(think_open)
        thinking_end = text.index(think_close, thinking_start)
        thinking = text[thinking_start:thinking_end].strip()
        final = text[thinking_end + len(think_close) :].strip()
        return {"thinking": thinking or None, "final": final or None}
    return {"thinking": None, "final": text.strip() or None}


def extract_gpt_oss_sections(text: str) -> dict[str, str | None]:
    analysis_marker = "<|channel|>analysis<|message|>"
    final_marker = "<|channel|>final<|message|>"
    handoff_marker = "<|end|><|start|>assistant"
    thinking: str | None = None
    final: str | None = None

    final_index = text.find(final_marker)
    if final_index != -1:
        final = text[final_index + len(final_marker) :].strip() or None

    analysis_index = text.find(analysis_marker)
    if analysis_index != -1:
        analysis_start = analysis_index + len(analysis_marker)
        analysis_end = text.find(handoff_marker, analysis_start)
        if analysis_end == -1:
            analysis_end = final_index if final_index != -1 else len(text)
        thinking = text[analysis_start:analysis_end].strip() or None

    if final is None and final_index == -1:
        final = text.strip() or None

    return {"thinking": thinking, "final": final}


def extract_sections(spec: ModelSpec, text: str) -> dict[str, str | None]:
    if spec.use_gpt_oss_sections:
        return extract_gpt_oss_sections(text)
    return extract_kimi_sections(text)


def classify_boldness(lines_changed: int, final_text: str, updated_train_py: str, current_train_py: str) -> dict[str, Any]:
    categories: list[str] = []
    if "def " in final_text and ">>>>>>> REPLACE" in final_text:
        categories.append("code_logic")

    architecture_markers = (
        "DEPTH",
        "ASPECT_RATIO",
        "HEAD_DIM",
        "WINDOW_PATTERN",
        "build_model_config",
        "class GPT",
        "class Block",
        "class CausalSelfAttention",
    )
    optimizer_markers = (
        "EMBEDDING_LR",
        "UNEMBEDDING_LR",
        "MATRIX_LR",
        "SCALAR_LR",
        "WEIGHT_DECAY",
        "ADAM_BETAS",
        "WARMUP_RATIO",
        "WARMDOWN_RATIO",
        "FINAL_LR_FRAC",
        "setup_optimizer",
        "get_lr_multiplier",
    )
    throughput_markers = (
        "TOTAL_BATCH_SIZE",
        "DEVICE_BATCH_SIZE",
        "grad_accum_steps",
    )

    if any(marker in final_text for marker in architecture_markers):
        categories.append("architecture")
    if any(marker in final_text for marker in optimizer_markers):
        categories.append("optimizer_or_schedule")
    if any(marker in final_text for marker in throughput_markers):
        categories.append("throughput")

    if not categories:
        categories.append("unclear")

    if lines_changed >= 80 or "code_logic" in categories:
        rating = "bold"
    elif lines_changed >= 25 or "architecture" in categories:
        rating = "moderate"
    else:
        rating = "conservative"

    return {
        "rating": rating,
        "categories": categories,
        "lines_changed": lines_changed,
        "net_char_delta": len(updated_train_py) - len(current_train_py),
    }


def evaluate_sample(
    sample_index: int,
    spec: ModelSpec,
    renderer: Any,
    sequence: Any,
    current_train_py: str,
) -> dict[str, Any]:
    parsed_message, parsed_ok = renderer.parse_response(sequence.tokens)
    content = parsed_message.get("content", "")
    if not isinstance(content, str):
        raise ValueError(f"Expected string content from renderer, got: {type(content)!r}")

    sections = extract_sections(spec, content)
    final_text = sections["final"] or ""

    validation: dict[str, Any] = {
        "sample_index": sample_index,
        "parsed_ok": parsed_ok,
        "token_count": len(sequence.tokens),
        "thinking": sections["thinking"],
        "final": sections["final"],
        "raw_content": content,
    }

    try:
        candidate = parse_patch_candidate_for_state(final_text, current_train_py)
    except ValueError as exc:
        validation.update(
            {
                "format_pass": False,
                "format_reason": str(exc),
                "candidate_format": None,
                "patch_block_count": 0,
                "lines_changed": 0,
                "preflight_ok": False,
                "preflight_stage": None,
                "preflight_reason": None,
                "boldness": {
                    "rating": "unknown",
                    "categories": ["invalid_format"],
                    "lines_changed": 0,
                    "net_char_delta": 0,
                },
            }
        )
        return validation

    with tempfile.TemporaryDirectory(prefix="rollout-validate-") as tmpdir:
        root = Path(tmpdir)
        config = TTTAutoResearchConfig(execution_backend="local").normalized(REPO_ROOT)
        runner = AutoResearchRunner(REPO_ROOT, config, root / "run")
        workspace = runner.prepare_candidate_workspace(candidate, step=sample_index, prefix="validate")
        preflight = runner.preflight_candidate(workspace, candidate)

    boldness = classify_boldness(
        candidate.lines_changed,
        final_text,
        candidate.train_py,
        current_train_py,
    )
    validation.update(
        {
            "format_pass": True,
            "format_reason": "Patch parsed successfully.",
            "candidate_format": candidate.candidate_format,
            "patch_block_count": candidate.patch_block_count,
            "lines_changed": candidate.lines_changed,
            "preflight_ok": preflight.ok,
            "preflight_stage": preflight.stage,
            "preflight_reason": preflight.reason,
            "preflight_details": preflight.details,
            "boldness": boldness,
        }
    )
    return validation


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    format_passes = sum(1 for item in results if item["format_pass"])
    preflight_passes = sum(1 for item in results if item["format_pass"] and item["preflight_ok"])

    bold_counts: dict[str, int] = {}
    for item in results:
        rating = item["boldness"]["rating"]
        bold_counts[rating] = bold_counts.get(rating, 0) + 1

    impressive = [
        {
            "sample_index": item["sample_index"],
            "lines_changed": item["lines_changed"],
            "categories": item["boldness"]["categories"],
            "preflight_ok": item["preflight_ok"],
            "preview": (item["final"] or "")[:400],
        }
        for item in results
        if item["format_pass"] and (
            item["boldness"]["rating"] == "bold"
            or "architecture" in item["boldness"]["categories"]
            or "code_logic" in item["boldness"]["categories"]
        )
    ]

    return {
        "num_samples": len(results),
        "format_passes": format_passes,
        "format_failures": len(results) - format_passes,
        "preflight_passes": preflight_passes,
        "preflight_failures": len(results) - preflight_passes,
        "format_pass_rate": format_passes / len(results) if results else 0.0,
        "preflight_pass_rate": preflight_passes / len(results) if results else 0.0,
        "boldness_counts": bold_counts,
        "impressive_candidates": impressive,
    }


def write_artifact(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"rollout_prompt_validation_{timestamp}.json"
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise SystemExit("Missing TINKER_API_KEY. Pass --api-key or set the environment variable.")

    spec = MODEL_SPECS[args.model]
    prompt, current_train_py = build_prompt(
        args.train_file,
        args.current_val_bpb,
        args.target_val_bpb,
    )

    print(f"Model: {spec.label}", flush=True)
    print(f"Samples: {args.num_samples}", flush=True)
    print(f"Prompt source: {args.train_file.resolve()}", flush=True)

    tokenizer = renderers.get_tokenizer(spec.model_name)
    renderer = renderers.get_renderer(spec.renderer_name, tokenizer)
    prompt_input = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    available_output_tokens = args.max_tokens - prompt_input.length
    if available_output_tokens <= 0:
        raise SystemExit(
            f"Prompt length {prompt_input.length} exceeds max token budget {args.max_tokens}."
        )

    service_client = tinker.ServiceClient(api_key=api_key)
    sampling_client = service_client.create_sampling_client(base_model=spec.model_name)

    print("Submitting Tinker sample request...", flush=True)
    sample_response = sampling_client.sample(
        prompt=prompt_input,
        num_samples=args.num_samples,
        sampling_params=tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=available_output_tokens,
            temperature=args.temperature,
        ),
    ).result()
    print("Tinker response received. Validating candidates...", flush=True)

    results: list[dict[str, Any]] = []
    for index, sequence in enumerate(sample_response.sequences, start=1):
        result = evaluate_sample(
            sample_index=index,
            spec=spec,
            renderer=renderer,
            sequence=sequence,
            current_train_py=current_train_py,
        )
        results.append(result)
        status = "PASS" if result["format_pass"] and result["preflight_ok"] else "FAIL"
        print(
            f"Sample {index}: {status} | format={result['format_pass']} | "
            f"preflight={result['preflight_ok']} | boldness={result['boldness']['rating']}",
            flush=True,
        )

    summary = summarize(results)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": asdict(spec),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_tokens_budget": args.max_tokens,
        "available_output_tokens": available_output_tokens,
        "train_file": str(args.train_file.resolve()),
        "summary": summary,
        "results": results,
    }
    artifact_path = write_artifact(args.output_dir, payload)

    print(
        f"Summary: format {summary['format_passes']}/{summary['num_samples']}, "
        f"preflight {summary['preflight_passes']}/{summary['num_samples']}",
        flush=True,
    )
    print(f"Saved artifact: {artifact_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
