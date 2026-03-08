"""Model registry and dataset helpers for autoresearch_mlx.

Usage:
    python models.py                          # List recommended models for your machine
    python models.py --list-datasets coding   # Search for coding datasets
    python models.py --prep-data alpaca       # Download and format a dataset for fine-tuning
"""
import subprocess, json, os, argparse

def get_memory_gb():
    """Detect unified memory on Apple Silicon via sysctl."""
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(r.stdout.strip()) / (1024**3)
    except Exception:
        return 16

# Model registry organized by memory tier
# (hf_repo, size_gb_approx, description, best_for)
# mlx-community/ prefix = pre-converted MLX weights, fastest to start
MODELS = {
    "8gb": [
        ("mlx-community/Qwen2.5-0.5B-Instruct-4bit", 0.4, "Qwen 0.5B 4-bit", "Fast iteration, tiny model experiments"),
        ("mlx-community/gemma-3-1b-it-4bit", 0.8, "Gemma 3 1B 4-bit", "Google's latest small model"),
        ("mlx-community/SmolLM2-1.7B-Instruct-4bit", 1.2, "SmolLM2 1.7B 4-bit", "Compact, surprisingly capable"),
    ],
    "16gb": [
        ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", 1.2, "Qwen 1.5B 4-bit", "Best bang-for-buck on 16GB"),
        ("mlx-community/Llama-3.2-3B-Instruct-4bit", 2.0, "Llama 3.2 3B 4-bit", "Meta's small Llama, Ollama-native"),
        ("mlx-community/Phi-4-mini-instruct-4bit", 2.5, "Phi-4 Mini 4-bit", "Microsoft reasoning model"),
        ("mlx-community/gemma-3-4b-it-4bit", 2.8, "Gemma 3 4B 4-bit", "Strong small model from Google"),
    ],
    "32gb": [
        ("mlx-community/Qwen2.5-7B-Instruct-4bit", 4.5, "Qwen 7B 4-bit", "Excellent general purpose"),
        ("mlx-community/Llama-3.2-3B-Instruct", 6.0, "Llama 3.2 3B fp16", "Full precision -- memory advantage!"),
        ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", 4.5, "Mistral 7B 4-bit", "Strong general model"),
        ("mlx-community/gemma-3-12b-it-4bit", 7.5, "Gemma 3 12B 4-bit", "Best quality at this tier"),
    ],
    "64gb": [
        ("mlx-community/Qwen2.5-14B-Instruct-4bit", 8.5, "Qwen 14B 4-bit", "Strong mid-size model"),
        ("mlx-community/Qwen2.5-7B-Instruct", 14.0, "Qwen 7B fp16", "Full precision -- no quantization needed"),
        ("mlx-community/Llama-3.1-8B-Instruct-4bit", 5.0, "Llama 3.1 8B 4-bit", "Battle-tested"),
        ("mlx-community/Mistral-Small-24B-Instruct-2501-4bit", 14.0, "Mistral Small 24B 4-bit", "Multimodal capable"),
    ],
    "96gb": [
        ("mlx-community/Qwen2.5-32B-Instruct-4bit", 18.0, "Qwen 32B 4-bit", "Frontier quality, fits easily"),
        ("mlx-community/Qwen2.5-14B-Instruct", 28.0, "Qwen 14B fp16", "Full precision large model"),
        ("mlx-community/QwQ-32B-4bit", 18.0, "QwQ 32B 4-bit", "Reasoning model"),
        ("mlx-community/Llama-3.1-8B-Instruct", 16.0, "Llama 3.1 8B fp16", "Full precision -- memory advantage!"),
    ],
    "128gb": [
        ("mlx-community/Qwen2.5-72B-Instruct-4bit", 40.0, "Qwen 72B 4-bit", "Near-frontier, single machine!"),
        ("mlx-community/Llama-3.3-70B-Instruct-4bit", 40.0, "Llama 70B 4-bit", "Meta's flagship"),
        ("mlx-community/Qwen2.5-32B-Instruct", 64.0, "Qwen 32B fp16", "Full precision at scale"),
        ("mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit", 25.0, "Mixtral 8x7B 4-bit MoE", "MoE -- memory advantage!"),
    ],
    "192gb": [
        ("mlx-community/Qwen2.5-72B-Instruct", 144.0, "Qwen 72B fp16", "Full precision 72B -- impossible on NVIDIA!"),
        ("mlx-community/Llama-3.3-70B-Instruct-8bit", 70.0, "Llama 70B 8-bit", "Near-full precision 70B"),
        ("mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit", 80.0, "Mixtral 8x22B 4-bit MoE", "141B MoE -- unified memory showcase"),
    ],
}

# Dataset presets for fine-tuning
DATASET_PRESETS = {
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "description": "52k instruction-following examples (Stanford Alpaca)",
        "format": "completions",
        "convert": lambda ex: {"prompt": f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex.get('input','')}\n\n### Response:\n", "completion": ex['output']},
    },
    "code-feedback": {
        "hf_name": "m-a-p/CodeFeedback-Filtered-Instruction",
        "description": "157k code instruction-response pairs",
        "format": "completions",
        "convert": lambda ex: {"prompt": ex["query"], "completion": ex["answer"]},
    },
    "openassistant": {
        "hf_name": "OpenAssistant/oasst1",
        "description": "Multi-turn human+assistant conversations",
        "format": "chat",
        "convert": None,
    },
    "medical-qa": {
        "hf_name": "medalpaca/medical_meadow_medical_flashcards",
        "description": "Medical Q&A flashcards for domain fine-tuning",
        "format": "completions",
        "convert": lambda ex: {"prompt": ex["input"], "completion": ex["output"]},
    },
    "sql": {
        "hf_name": "mlx-community/wikisql",
        "description": "Natural language to SQL (MLX-native, ready to use)",
        "format": "completions",
        "convert": None,
    },
    "fineweb-edu": {
        "hf_name": "karpathy/fineweb-edu-100b-shuffle",
        "description": "Karpathy's curated web text for pretraining (used in train.py mode)",
        "format": "text",
        "convert": lambda ex: {"text": ex["text"]},
    },
    "custom": {
        "hf_name": None,
        "description": "Any HuggingFace dataset -- specify with --hf-dataset <name>",
        "format": "auto",
        "convert": None,
    },
}

# Tier thresholds in ascending order, derived from MODELS keys
_TIER_THRESHOLDS = [(int(k.rstrip("gb")), k) for k in MODELS]

def recommend_models(memory_gb=None):
    """Print model recommendations based on available unified memory.

    Detects memory automatically if not provided. Shows all models for the
    matching tier and below, and highlights full-precision / MoE options as
    "MEMORY ADVANTAGE MODELS" for machines with 64 GB or more.
    """
    if memory_gb is None:
        memory_gb = get_memory_gb()

    print(f"Detected {memory_gb:.0f} GB unified memory\n")

    # Find the best matching tier
    matched_tier = "8gb"
    for threshold, tier_name in _TIER_THRESHOLDS:
        if memory_gb >= threshold:
            matched_tier = tier_name

    # Collect tiers up to and including the matched one
    tier_order = [t for _, t in _TIER_THRESHOLDS]
    active_tiers = tier_order[:tier_order.index(matched_tier) + 1]

    # Print models per tier
    for tier in active_tiers:
        models = MODELS[tier]
        print(f"--- {tier.upper()} tier ---")
        for repo, size, desc, best_for in models:
            print(f"  {desc:<30s} ~{size:>5.1f} GB  {best_for}")
            print(f"    {repo}")
        print()

    # Highlight memory-advantage models for 64 GB+ machines
    if memory_gb >= 64:
        advantage = []
        for tier in active_tiers:
            for repo, size, desc, best_for in MODELS[tier]:
                if "fp16" in desc or "MoE" in desc or "memory advantage" in best_for.lower():
                    advantage.append((repo, size, desc, best_for))
        if advantage:
            print("=== MEMORY ADVANTAGE MODELS ===")
            print("These leverage Apple Silicon unified memory in ways NVIDIA GPUs cannot:\n")
            for repo, size, desc, best_for in advantage:
                print(f"  {desc:<30s} ~{size:>5.1f} GB  {best_for}")
                print(f"    {repo}")
            print()


def prep_dataset(preset_name, output_dir="data", hf_dataset=None, max_samples=None):
    """Download and format a dataset for MLX fine-tuning.

    Writes train.jsonl and valid.jsonl into output_dir, formatted for
    mlx-lm's LoRA / QLoRA fine-tuning (completions or chat format).

    Args:
        preset_name: Key from DATASET_PRESETS, or "custom".
        output_dir: Directory to write output files.
        hf_dataset: HuggingFace dataset name (required when preset is "custom").
        max_samples: Cap the number of training examples.
    """
    from datasets import load_dataset  # lazy import -- only needed here

    if preset_name not in DATASET_PRESETS:
        print(f"Unknown preset '{preset_name}'. Available: {', '.join(DATASET_PRESETS)}")
        return

    preset = DATASET_PRESETS[preset_name]
    ds_name = hf_dataset if preset_name == "custom" else preset["hf_name"]
    if ds_name is None:
        print("Specify --hf-dataset <name> when using the 'custom' preset.")
        return

    print(f"Loading dataset: {ds_name} ...")
    ds = load_dataset(ds_name, split="train")

    if max_samples and len(ds) > max_samples:
        total = len(ds)
        ds = ds.shuffle(seed=42).select(range(max_samples))
        print(f"  Sampled {max_samples} examples from {total} total")

    convert_fn = preset["convert"]
    if convert_fn is not None:
        ds = ds.map(convert_fn, remove_columns=ds.column_names)

    os.makedirs(output_dir, exist_ok=True)

    # 90/10 train/valid split
    split = ds.train_test_split(test_size=0.1, seed=42)
    for split_name, split_key in [("train", "train"), ("valid", "test")]:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for row in split[split_key]:
                f.write(json.dumps(row) + "\n")
        print(f"  Wrote {len(split[split_key])} examples to {path}")

    print(f"\nDataset ready in {output_dir}/")
    print(f"Format: {preset['format']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Model registry and dataset helpers for autoresearch_mlx"
    )
    p.add_argument("--list-datasets", nargs="?", const="", default=None, metavar="QUERY",
                    help="List dataset presets, optionally filtered by keyword")
    p.add_argument("--prep-data", metavar="PRESET",
                    help="Download and format a dataset preset for fine-tuning")
    p.add_argument("--hf-dataset", metavar="NAME",
                    help="HuggingFace dataset name (for 'custom' preset)")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Cap training examples")
    p.add_argument("--output-dir", default="data",
                    help="Output directory for prepared data (default: data)")
    args = p.parse_args()

    if args.list_datasets is not None:
        query = args.list_datasets.lower()
        print("Available dataset presets:\n")
        for name, info in DATASET_PRESETS.items():
            text = f"{name}: {info['description']}"
            if query and query not in text.lower():
                continue
            hf = info["hf_name"] or "(user-specified)"
            print(f"  {name:<16s} {info['description']}")
            print(f"    {'HF:':<5s} {hf}   Format: {info['format']}")
        print(f"\nUse --prep-data <preset> to download and format a dataset.")
    elif args.prep_data:
        prep_dataset(args.prep_data, output_dir=args.output_dir,
                     hf_dataset=args.hf_dataset, max_samples=args.max_samples)
    else:
        recommend_models()
