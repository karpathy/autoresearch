from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_autoresearch.discover_compat import (
    patch_transformers_kimi_trust_remote_code,
    patch_ttt_discover_kimi_renderer,
    patch_ttt_discover_kimi_tokenizer,
)
from ttt_autoresearch.env import AutoResearchState
from ttt_autoresearch.prompt_builder import build_prompt_for_state


patch_ttt_discover_kimi_tokenizer()
patch_ttt_discover_kimi_renderer()
patch_transformers_kimi_trust_remote_code()

from ttt_discover.tinker_utils import renderers


DEFAULT_PROMPT_FILE = REPO_ROOT / "prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "testing" / "output"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    renderer_name: str
    label: str


MODEL_SPECS = (
    ModelSpec(
        key="kimi",
        model_name="moonshotai/Kimi-K2.5",
        renderer_name="kimi_k25",
        label="Kimi K2.5 via Tinker",
    ),
    ModelSpec(
        key="gpt_oss",
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_high_reasoning",
        label="GPT-OSS-120B high reasoning via Tinker",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporary smoke test for Tinker reasoning models."
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text to send. If omitted, --prompt-file is used, or prompt.txt if present.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing the prompt text.",
    )
    parser.add_argument(
        "--api-key",
        help="Tinker API key. Defaults to TINKER_API_KEY from the environment.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["raw", "normal_full"],
        default="raw",
        help="Use a raw prompt or build the normal full AutoResearch prompt from train.py.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO_ROOT / "train.py",
        help="train.py to embed when --prompt-mode normal_full is used.",
    )
    parser.add_argument(
        "--current-val-bpb",
        type=float,
        default=1.0,
        help="Current val_bpb used when building the normal full prompt.",
    )
    parser.add_argument(
        "--target-val-bpb",
        type=float,
        default=0.97,
        help="Target val_bpb used when building the normal full prompt.",
    )
    parser.add_argument(
        "--base-url",
        help="Override Tinker base URL. Defaults to TINKER_BASE_URL or SDK default.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional maximum tokens to sample per model. Omit for no generation cap.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts are saved.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
        default=[spec.key for spec in MODEL_SPECS],
        help="Subset of models to run.",
    )
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    if args.prompt_mode == "normal_full":
        if args.prompt or args.prompt_file:
            raise SystemExit("--prompt/--prompt-file cannot be combined with --prompt-mode normal_full.")
        train_path = args.train_file.resolve()
        train_text = train_path.read_text(encoding="utf-8")
        state = AutoResearchState(
            timestep=-1,
            construction=[],
            code=train_text,
            value=-args.current_val_bpb,
        )
        prompt = build_prompt_for_state(state, args.target_val_bpb)
        return prompt, f"normal_full:{train_path}"
    if args.prompt and args.prompt_file:
        raise SystemExit("Pass either --prompt or --prompt-file, not both.")
    if args.prompt:
        return args.prompt, "inline"
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8"), str(args.prompt_file.resolve())
    if DEFAULT_PROMPT_FILE.exists():
        return DEFAULT_PROMPT_FILE.read_text(encoding="utf-8"), str(DEFAULT_PROMPT_FILE)
    raise SystemExit("No prompt provided. Pass --prompt or --prompt-file.")


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
    if spec.key == "kimi":
        return extract_kimi_sections(text)
    if spec.key == "gpt_oss":
        return extract_gpt_oss_sections(text)
    return {"thinking": None, "final": text.strip() or None}


def run_single_model(
    service_client: tinker.ServiceClient,
    spec: ModelSpec,
    prompt: str,
    max_tokens: int | None,
    temperature: float,
) -> dict[str, Any]:
    print(f"Starting model: {spec.label}", flush=True)
    tokenizer = renderers.get_tokenizer(spec.model_name)
    renderer = renderers.get_renderer(spec.renderer_name, tokenizer)
    prompt_input = renderer.build_generation_prompt(
        [{"role": "user", "content": prompt}],
    )
    sampling_client = service_client.create_sampling_client(base_model=spec.model_name)
    sample_response = sampling_client.sample(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    ).result()
    sequence = sample_response.sequences[0]
    parsed_message, parsed_ok = renderer.parse_response(sequence.tokens)
    content = parsed_message.get("content", "")
    if not isinstance(content, str):
        raise ValueError(f"Expected string content from renderer, got: {type(content)!r}")
    sections = extract_sections(spec, content)
    return {
        "spec": asdict(spec),
        "parsed_ok": parsed_ok,
        "token_count": len(sequence.tokens),
        "raw_content": content,
        "thinking": sections["thinking"],
        "final": sections["final"],
    }


def format_console_output(result: dict[str, Any]) -> str:
    title = result["spec"]["label"]
    parts = [f"===== {title} ====="]
    thinking = result.get("thinking")
    final = result.get("final")
    if thinking:
        parts.append("THINKING:")
        parts.append(thinking)
    if final:
        parts.append("FINAL:")
        parts.append(final)
    if not thinking and not final:
        parts.append("RAW:")
        parts.append(result.get("raw_content", ""))
    return "\n".join(parts)


def save_artifact(
    output_dir: Path,
    prompt: str,
    prompt_source: str,
    results: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"tinker_reasoning_smoke_{timestamp}.json"
    payload = {
        "created_at_utc": timestamp,
        "prompt_source": prompt_source,
        "prompt": prompt,
        "results": results,
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def main() -> int:
    args = parse_args()
    prompt, prompt_source = resolve_prompt(args)
    print(f"Prompt source: {prompt_source}", flush=True)
    print(f"Selected models: {', '.join(args.models)}", flush=True)
    api_key = args.api_key or os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise SystemExit("Missing TINKER_API_KEY. Pass --api-key or set the environment variable.")

    service_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        service_kwargs["base_url"] = args.base_url

    requested = set(args.models)
    selected_specs = [spec for spec in MODEL_SPECS if spec.key in requested]
    if not selected_specs:
        raise SystemExit("No models selected.")

    service_client = tinker.ServiceClient(**service_kwargs)
    results: list[dict[str, Any]] = []
    for spec in selected_specs:
        result = run_single_model(
            service_client=service_client,
            spec=spec,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        results.append(result)
        print(format_console_output(result), flush=True)
        print(flush=True)

    artifact_path = save_artifact(args.output_dir, prompt, prompt_source, results)
    print(f"Saved artifact: {artifact_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
