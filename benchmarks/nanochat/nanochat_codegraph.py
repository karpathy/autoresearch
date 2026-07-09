"""NanoChat-specific method-to-code linking over generic ResearchFS indexing."""

from __future__ import annotations

from pathlib import Path

from researchfs.codegraph import refresh_code_map, replace_method_code_targets
from researchfs.db import get_connection


BENCHMARK_ROOT = Path(__file__).resolve().parent
NANOCHAT_CODE_PATHS = (BENCHMARK_ROOT / "train.py", BENCHMARK_ROOT / "prepare.py")

TARGET_RULES = (
    (
        ("mlp", "swiglu", "feed-forward", "feed forward"),
        ("MLP",),
        "nanochat: mlp keywords map to the MLP block",
    ),
    (
        ("attention", "gqa", "grouped-query", "window", "sliding", "token shifting"),
        ("CausalSelfAttention", "Block", "GPT._compute_window_sizes", "WINDOW_PATTERN"),
        "nanochat: attention keywords map to attention and block symbols",
    ),
    (
        ("optimizer", "muon", "lr", "learning rate", "warmup", "warmdown", "gradient", "clip"),
        (
            "MuonAdamW",
            "GPT.setup_optimizer",
            "optimizer",
            "get_lr_multiplier",
            "get_muon_momentum",
            "get_weight_decay",
            "MATRIX_LR",
            "EMBEDDING_LR",
            "UNEMBEDDING_LR",
            "SCALAR_LR",
            "WEIGHT_DECAY",
            "ADAM_BETAS",
            "WARMUP_RATIO",
            "WARMDOWN_RATIO",
            "FINAL_LR_FRAC",
        ),
        "nanochat: optimizer keywords map to optimizer setup and schedules",
    ),
    (
        ("normalization", "rmsnorm", "layernorm", "norm"),
        ("RMSNorm", "GPT.setup_optimizer"),
        "nanochat: normalization keywords map to norm symbols",
    ),
    (
        ("training", "batch", "dataloader", "data", "tokens", "precision", "bf16", "fp8"),
        (
            "infinite_loop",
            "TOTAL_BATCH_SIZE",
            "DEVICE_BATCH_SIZE",
            "grad_accum_steps",
            "tokens_per_fwdbwd",
            "train_loader",
            "MAX_SEQ_LEN",
            "TIME_BUDGET",
            "make_dataloader",
            "evaluate_bpb",
        ),
        "nanochat: training keywords map to loop, dataloader, and batch symbols",
    ),
)


def _symbols_by_name(symbols: list[dict]) -> dict[str, list[dict]]:
    indexed: dict[str, list[dict]] = {}
    for symbol in symbols:
        name = str(symbol["symbol_name"]).lower()
        indexed.setdefault(name, []).append(symbol)
        if "." in name:
            indexed.setdefault(name.rsplit(".", 1)[-1], []).append(symbol)
    return indexed


def link_nanochat_methods_to_code(conn=None) -> int:
    """Refresh NanoChat sources and replace only their explicit method links."""
    close = conn is None
    if conn is None:
        conn = get_connection()
    try:
        symbols = refresh_code_map(NANOCHAT_CODE_PATHS, conn=conn)
        indexed = _symbols_by_name(symbols)
        file_paths = tuple(dict.fromkeys(str(symbol["file_path"]) for symbol in symbols))
        targets = []
        for method in conn.execute(
            "SELECT method_id, name, category FROM methods ORDER BY method_id"
        ).fetchall():
            method_text = f"{method['category'] or ''} {method['name'] or ''}".lower()
            seen_symbols = set()
            for keywords, symbol_names, rationale in TARGET_RULES:
                if not any(keyword in method_text for keyword in keywords):
                    continue
                for symbol_name in symbol_names:
                    for symbol in indexed.get(symbol_name.lower(), ()):
                        key = (
                            symbol["file_path"],
                            symbol["symbol_name"],
                            symbol["symbol_kind"],
                        )
                        if key in seen_symbols:
                            continue
                        seen_symbols.add(key)
                        targets.append(
                            {
                                "method_id": int(method["method_id"]),
                                "file_path": symbol["file_path"],
                                "symbol_name": symbol["symbol_name"],
                                "symbol_kind": symbol["symbol_kind"],
                                "rationale": rationale,
                            }
                        )
        return replace_method_code_targets(file_paths, targets, conn=conn)
    finally:
        if close:
            conn.close()
