"""Seed the ResearchFS database and ingest papers/methods into ResearchFS."""

import argparse
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from typing import Callable, Optional

from researchfs import db as db_module
from researchfs import extractor
from researchfs.code_repos import enrich_paper_code_repositories
from researchfs.db import get_connection, initialize_db, migrate_db
from researchfs.fetcher import fetch_arxiv_source, fetch_arxiv_query, fetch_github_readme
from researchfs.pdf import extract_pdf_text


def _load_markdown(filepath: Path) -> str:
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""

SUPPORTED_LOCAL_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".tex", ".pdf"}

_SOURCE_TYPE_EXPLICIT_META_KEY = "_researchfs_source_type_explicit"

def _read_only_sqlite_uri(db_uri: str) -> str | None:
    """Return a read-only SQLite URI for a configured file: target."""
    parsed = urlsplit(db_uri)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if any(key == "mode" and value == "memory" for key, value in query):
        return None
    query = [(key, value) for key, value in query if key != "mode"]
    query.append(("mode", "ro"))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def _open_dry_run_target_connection():
    """Open an existing target DB read-only for dry-run duplicate checks."""
    try:
        target = db_module._resolve_db_path(None)
        target_str = str(target)
        if target_str in ("", ":memory:"):
            return None
        if target_str.startswith("file:"):
            uri = _read_only_sqlite_uri(target_str)
            if uri is None:
                return None
        else:
            target_path = Path(target_str).expanduser()
            if not target_path.exists():
                return None
            uri = f"file:{quote(str(target_path.resolve()), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except (OSError, sqlite3.Error):
        return None



def _mark_source_type_explicit(meta: dict) -> dict:
    """Mark source_type as a caller-supplied label rather than an inferred default."""
    meta[_SOURCE_TYPE_EXPLICIT_META_KEY] = True
    return meta


def _explicit_source_type(meta: dict) -> Optional[str]:
    """Return the caller-supplied source_type, excluding known inferred defaults."""
    if meta.get(_SOURCE_TYPE_EXPLICIT_META_KEY) is False:
        return None
    source_type = meta.get("source_type")
    if source_type is None:
        return None
    return _source_type_label(str(source_type))


def _title_from_file_path(path: Path) -> str:
    title = re.sub(r"[-_]+", " ", path.stem).strip()
    return title.title() if title else path.name


class LocalFileExtractionError(ValueError):
    """Raised when a supported local file cannot be converted to text."""


def _source_type_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise argparse.ArgumentTypeError("--source-type must not be empty")
    return label


def _apply_source_type_override(meta: dict, source_type: Optional[str]) -> dict:
    if source_type is not None:
        meta["source_type"] = _source_type_label(source_type)
        _mark_source_type_explicit(meta)
    return meta


def load_local_file_metadata(path: str | Path, overrides: Optional[dict] = None) -> dict:
    """Load a local text/markdown/TeX/PDF paper into ResearchFS metadata."""
    file_path = Path(path).expanduser()
    suffix = file_path.suffix.lower()
    supported = ", ".join(sorted(SUPPORTED_LOCAL_FILE_EXTENSIONS))
    if suffix not in SUPPORTED_LOCAL_FILE_EXTENSIONS:
        display_suffix = suffix or "<none>"
        raise ValueError(
            f"Unsupported local file type {display_suffix!r}; supported extensions are {supported}."
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"Local paper file not found: {file_path}")

    resolved_path = file_path.resolve()
    if suffix == ".pdf":
        try:
            content = extract_pdf_text(resolved_path)
        except ValueError as exc:
            raise LocalFileExtractionError(str(exc)) from exc
    else:
        content = resolved_path.read_text(encoding="utf-8")
    meta = dict(overrides or {})
    title = str(meta.get("title") or "").strip() or _title_from_file_path(resolved_path)
    source_type_is_explicit = bool(meta.get("source_type"))
    source_type = _source_type_label(str(meta["source_type"])) if source_type_is_explicit else "file"
    meta.update(
        {
            "title": title,
            "url": resolved_path.as_uri(),
            "source_type": source_type,
            "content": content,
        }
    )
    if source_type_is_explicit:
        _mark_source_type_explicit(meta)
    else:
        meta[_SOURCE_TYPE_EXPLICIT_META_KEY] = False
    meta.setdefault("authors", None)
    meta.setdefault("abstract", None)
    meta.setdefault("publish_date", None)
    return meta


def seed_initial_data(conn=None) -> None:
    """Populate the database with the v0 seed corpus."""
    close = False
    if conn is None:
        initialize_db()
        conn = get_connection()
        close = True
    migrate_db(conn)

    repo_root = Path(__file__).resolve().parent.parent

    # -----------------------------------------------------------------------
    # Papers
    # -----------------------------------------------------------------------
    papers = [
        {
            "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
            "authors": "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai",
            "abstract": (
                "Multi-query attention (MQA) speeds up decoder inference but quality degrades. "
                "Grouped-query attention (GQA) divides query heads into groups that share key/value heads, "
                "interpolating between MHA and MQA to balance quality and speed."
            ),
            "url": "https://arxiv.org/abs/2305.13245",
            "publish_date": "2023-05-22",
        },
        {
            "title": "GLU Variants Improve Transformer",
            "authors": "Noam Shazeer",
            "abstract": (
                "Replaces the Transformer feed-forward ReLU activation with gated linear units (GLUs) "
                "such as ReGLU, GEGLU, SwiGLU, and approximates parameter count by scaling the hidden "
                "dimension. SwiGLU is shown to improve language modeling quality."
            ),
            "url": "https://arxiv.org/abs/2002.05202",
            "publish_date": "2020-02-12",
        },
        {
            "title": "Muon: An Optimizer for Hidden Layers in Neural Networks",
            "authors": "Keller Jordan, Jeremy Bernstein, Brendan Rappazzo, ...",
            "abstract": (
                "Muon applies orthogonal updates to 2D matrix parameters via Newton-Schulz iterations, "
                "combined with AdamW for non-matrix parameters. It is especially effective for small-scale "
                "transformer training."
            ),
            "url": "https://github.com/KellerJordan/Muon",
            "publish_date": "2024",
        },
        {
            "title": "Longformer: The Long-Document Transformer",
            "authors": "Iz Beltagy, Matthew E. Peters, Arman Cohan",
            "abstract": (
                "Introduces attention patterns that combine local sliding-window attention with global "
                "attention on pre-selected tokens, reducing the quadratic cost of self-attention and "
                "enabling longer contexts."
            ),
            "url": "https://arxiv.org/abs/2004.05150",
            "publish_date": "2020-04-10",
        },
        {
            "title": "Layer Normalization",
            "authors": "Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton",
            "abstract": (
                "LayerNorm normalizes across the feature dimension. RMSNorm is a reparameterized variant "
                "that removes the mean-centering step and often trains with comparable stability at lower cost."
            ),
            "url": "https://arxiv.org/abs/1607.06450",
            "publish_date": "2016-07-21",
        },
        {
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "authors": "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov",
            "abstract": (
                "Dropout randomly zeroes a subset of activations during training, acting as a strong "
                "regularizer that reduces co-adaptation and improves generalization."
            ),
            "url": "https://jmlr.org/papers/v15/srivastava14a.html",
            "publish_date": "2014",
        },
        {
            "title": "Hash Embeddings for Efficient Word Representations",
            "authors": "Daniele Sgrò",
            "abstract": (
                "Hash embeddings compress a large vocabulary by mapping tokens through multiple hash "
                "functions into a shared embedding space, trading a small collision risk for large "
                "memory savings."
            ),
            "url": "https://arxiv.org/abs/1709.03993",
            "publish_date": "2017-09-12",
        },
        {
            "title": "FP8-LM: Training FP8 Large Language Models",
            "authors": "Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Fei Chao, Dongsheng Li, Yuqing Yang, Lili Qiu",
            "abstract": (
                "Demonstrates training of large language models in FP8 while maintaining accuracy, using "
                "per-tensor scaling, delayed scaling, and mixed precision where needed for numerical stability."
            ),
            "url": "https://arxiv.org/abs/2310.18313",
            "publish_date": "2023-10-27",
        },
        {
            "title": "RWKV: Reinventing RNNs for the Transformer Era",
            "authors": "Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzhan He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanslav Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu",
            "abstract": (
                "RWKV combines transformer-like parallel training with efficient linear-time inference. "
                "A key ingredient is a small time-shift (token shift) that mixes the current token with "
                "the previous token before gating, improving local statistics without adding sequence cost."
            ),
            "url": "https://arxiv.org/abs/2305.13048",
            "publish_date": "2023-05-22",
        },
        {
            "title": "Decoupled Weight Decay Regularization",
            "authors": "Ilya Loshchilov, Frank Hutter",
            "abstract": (
                "AdamW decouples weight decay from the gradient update, often improving generalization "
                "over Adam with L2 regularization. The weight decay is typically annealed over training."
            ),
            "url": "https://arxiv.org/abs/1711.05101",
            "publish_date": "2017-11-14",
        },
        {
            "title": "On the Difficulty of Training Recurrent Neural Networks",
            "authors": "Razvan Pascanu, Tomas Mikolov, Yoshua Bengio",
            "abstract": (
                "Introduces gradient clipping as a simple and effective stabilization technique for deep "
                "networks, bounding the norm of gradients to prevent parameter spikes."
            ),
            "url": "https://arxiv.org/abs/1211.5063",
            "publish_date": "2012-11-21",
        },
    ]

    cur = conn.cursor()
    paper_ids = {}
    for paper in papers:
        cur.execute(
            "SELECT paper_id FROM papers WHERE title = ?",
            (paper["title"],),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                INSERT INTO papers (title, authors, abstract, url, publish_date)
                VALUES (:title, :authors, :abstract, :url, :publish_date)
                """,
                paper,
            )
            cur.execute(
                "SELECT paper_id FROM papers WHERE title = ?",
                (paper["title"],),
            )
            row = cur.fetchone()
        paper_ids[paper["title"]] = row["paper_id"]

    # -----------------------------------------------------------------------
    # Chunks
    # -----------------------------------------------------------------------
    chunks = [
        # GQA
        (
            "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
            "Architecture",
            (
                "Grouped-Query Attention (GQA) sets n_kv_head to a value strictly between 1 and n_head, "
                "with n_head divisible by n_kv_head. The fallback scaled_dot_product_attention and "
                "FlashAttention implementations support GQA natively, and the rotary-embedding broadcasting "
                "does not require changes."
            ),
        ),
        (
            "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
            "Configuration",
            (
                "To configure GQA, change GPTConfig.n_kv_head to a divisor of n_head, e.g. n_head=6, n_kv_head=2. "
                "This yields 3 query heads per key/value head, reducing KV-cache and parameter footprint while "
                "preserving most representational capacity."
            ),
        ),
        # SwiGLU
        (
            "GLU Variants Improve Transformer",
            "Architecture",
            (
                "SwiGLU computes (SiLU(x W_gate) * (x W_val)) W_proj. The gate and value projections are "
                "frequently fused into a single linear layer outputting 2 * hidden_dim channels and chunked "
                "inside the forward pass."
            ),
        ),
        (
            "GLU Variants Improve Transformer",
            "Parameter matching",
            (
                "To keep parameter count comparable to the baseline ReLU^2 block with intermediate dimension "
                "4 * d_model, a SwiGLU block should use d_ff = int(8/3 * d_model) and round to a multiple of 64. "
                "The fused gate+value projection is then nn.Linear(d_model, 2 * d_ff, bias=False)."
            ),
        ),
        # Muon
        (
            "Muon: An Optimizer for Hidden Layers in Neural Networks",
            "Optimizers",
            (
                "MuonAdamW applies Muon orthogonal updates to 2D matrix parameters and AdamW to all other "
                "parameters (embeddings, LM head, scalars). The Newton-Schulz iteration default is ns_steps=5."
            ),
        ),
        (
            "Muon: An Optimizer for Hidden Layers in Neural Networks",
            "Hyperparameters",
            (
                "MATRIX_LR controls the learning rate for 2D matrix parameters. The default of 0.04 is a "
                "reasonable starting point, but optimal values often fall between 0.02 and 0.08 for small "
                "transformers and scale with sqrt(d_model)."
            ),
        ),
        # Longformer / sliding window
        (
            "Longformer: The Long-Document Transformer",
            "Sliding window",
            (
                "Sliding-window attention restricts each token to attend only to nearby tokens within a "
                "fixed window size. In small-transformer training, alternating short-window (S) and full-context "
                "(L) layers trades context modeling for speed under a fixed time budget."
            ),
        ),
        (
            "Longformer: The Long-Document Transformer",
            "Patterns",
            (
                "Common window patterns are strings of S (short, half sequence length) and L (long, full "
                "sequence length), e.g. 'SSSL' or 'SLSL'. The last layer is typically kept as L to preserve "
                "final-context aggregation."
            ),
        ),
        # LayerNorm / RMSNorm
        (
            "Layer Normalization",
            "Forms",
            (
                "Functional RMSNorm can be swapped for functional LayerNorm with no extra parameters. "
                "RMSNorm uses F.rms_norm(x, (x.size(-1),)); LayerNorm uses F.layer_norm(x, (x.size(-1),))."
            ),
        ),
        (
            "Layer Normalization",
            "Learnable scale",
            (
                "A learnable RMSNorm registers a weight parameter and multiplies after normalization. "
                "When used, it must be added as a module so MuonAdamW groups it correctly as a scalar/vector parameter."
            ),
        ),
        # Dropout
        (
            "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "Residual dropout",
            (
                "Adding nn.Dropout after the attention and MLP residuals before the addition can reduce "
                "overfitting on small datasets or longer training runs. Typical rates are 0.0 to 0.1."
            ),
        ),
        # Hash embeddings
        (
            "Hash Embeddings for Efficient Word Representations",
            "Hashed n-grams",
            (
                "Hashed bigram/trigram value embeddings enrich token representations with local statistics "
                "by combining the current token embedding with hashed representations of nearby tokens. "
                "They are gated and added per-layer where value embeddings are enabled."
            ),
        ),
        (
            "Hash Embeddings for Efficient Word Representations",
            "Tradeoffs",
            (
                "Hash collisions add noise but allow a large effective vocabulary with a small embedding "
                "matrix. Ablate bigram-only, trigram-only, and gated mixtures to find the best speed/quality point."
            ),
        ),
        # FP8
        (
            "FP8-LM: Training FP8 Large Language Models",
            "Mixed precision",
            (
                "FP8 matmul can accelerate training on supported hardware, but optimizer states and sensitive "
                "layers are often kept in bf16 or fp32 for stability. On unsupported hardware, bf16 autocast "
                "remains the safe default."
            ),
        ),
        # RWKV / token shift
        (
            "RWKV: Reinventing RNNs for the Transformer Era",
            "Token shift",
            (
                "Token shift mixes the current token representation with the previous token representation "
                "before the gating path. In transformers this can be implemented as a small shift-and-mix "
                "operation on the input to each layer, improving local context at negligible cost."
            ),
        ),
        (
            "RWKV: Reinventing RNNs for the Transformer Era",
            "Stability",
            (
                "Token shifting can destabilize training if the shift coefficient is too large or if it "
                "interacts badly with learning-rate schedules. Start with a small coefficient and gate from zero."
            ),
        ),
        # AdamW / weight decay
        (
            "Decoupled Weight Decay Regularization",
            "Scheduling",
            (
                "Weight decay is commonly annealed with training progress. In the MuonAdamW setup used here, "
                "WEIGHT_DECAY starts at a configured value and is multiplied by (1 - progress) over the run."
            ),
        ),
        # Gradient clipping
        (
            "On the Difficulty of Training Recurrent Neural Networks",
            "Gradient clipping",
            (
                "Clipping gradients by global norm prevents rare large updates from destabilizing training. "
                "A typical threshold is 1.0; it is most useful when using large learning rates or new architectures."
            ),
        ),
    ]

    paper_chunk_ids_by_title = {paper["title"]: [] for paper in papers}
    paper_chunk_content_by_title = {paper["title"]: [] for paper in papers}
    chunk_ids = []
    for title, section, content in chunks:
        paper_id = paper_ids[title]
        cur.execute(
            """
            SELECT chunk_id FROM chunks
            WHERE paper_id = ? AND section = ? AND content = ?
            LIMIT 1
            """,
            (paper_id, section, content),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
                (paper_id, section, content),
            )
            chunk_id = cur.lastrowid
        else:
            chunk_id = row["chunk_id"]
        chunk_ids.append(chunk_id)
        paper_chunk_ids_by_title[title].append(chunk_id)
        paper_chunk_content_by_title[title].append(content)
    chunk_content_by_id = {
        chunk_id: content for chunk_id, (_, _, content) in zip(chunk_ids, chunks)
    }
    chunk_title_by_id = {
        chunk_id: title for chunk_id, (title, _, _) in zip(chunk_ids, chunks)
    }

    # -----------------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------------
    methods = [
        {
            "name": "Grouped-Query Attention (GQA)",
            "description": (
                "Reduce KV heads to a divisor of n_head to lower parameter count and sequence "
                "processing overhead while retaining model quality."
            ),
            "implementation_hint": (
                "Set GPTConfig.n_kv_head to a divisor of n_head (e.g. n_head=6, n_kv_head=2). "
                "No attention code changes are required because scaled_dot_product_attention supports GQA."
            ),
            "risks": (
                "Too few KV heads may degrade long-context modeling; n_head must remain divisible by n_kv_head."
            ),
            "category": "attention",
        },
        {
            "name": "SwiGLU MLP",
            "description": (
                "Replace ReLU^2 with a gated SiLU activation to improve representational capacity "
                "in the feed-forward block."
            ),
            "implementation_hint": (
                "Implement MLP with hidden_dim = int(8/3 * n_embd) rounded to 64, a single c_fc layer "
                "of size 2*hidden_dim, chunk into gate/value, apply F.silu(gate) * value, then c_proj."
            ),
            "risks": (
                "Memory and FLOP increase if hidden_dim is not parameter-matched; c_proj must initialize to zero."
            ),
            "category": "mlp",
        },
        {
            "name": "Muon LR Tuning",
            "description": (
                "Tune MATRIX_LR for 2D matrix parameters to accelerate convergence of the Muon optimizer."
            ),
            "implementation_hint": (
                "Adjust the MATRIX_LR constant in train.py between 0.02 and 0.08, keeping other LRs fixed."
            ),
            "risks": (
                "Values too high cause instability/NaN; values too low slow convergence."
            ),
            "category": "optimizer",
        },
        {
            "name": "Sliding Window Attention Patterns",
            "description": (
                "Alternate short-context (S) and full-context (L) layers to reduce per-step cost and "
                "fit more training steps into the fixed time budget."
            ),
            "implementation_hint": (
                "Change WINDOW_PATTERN in train.py to strings like 'SSSL', 'SLSL', or 'SSSS'. "
                "The last layer is always treated as full-context internally."
            ),
            "risks": (
                "Too many short layers hurts long-range dependency modeling; throughput gains are hardware-dependent."
            ),
            "category": "attention",
        },
        {
            "name": "LayerNorm / RMSNorm Swap",
            "description": (
                "Replace functional RMSNorm with functional LayerNorm or add learnable scale parameters "
                "to change normalization behavior."
            ),
            "implementation_hint": (
                "Swap F.rms_norm for F.layer_norm in the norm() function, or implement a LearnableRMSNorm "
                "module and use it in Block."
            ),
            "risks": (
                "Learnable norms must be registered as modules for correct parameter grouping and checkpointing."
            ),
            "category": "normalization",
        },
        {
            "name": "Dropout Regularization",
            "description": (
                "Add dropout to attention and MLP residuals to reduce overfitting on small data or longer runs."
            ),
            "implementation_hint": (
                "Add nn.Dropout(p) layers in Block and apply them to the attention and MLP outputs "
                "before the residual addition."
            ),
            "risks": (
                "Too much dropout slows convergence; too little has no effect. Rates above 0.1 are rarely useful here."
            ),
            "category": "regularization",
        },
        {
            "name": "Hashed n-gram Value Embeddings",
            "description": (
                "Augment token representations with hashed bigram/trigram value embeddings to capture "
                "local token statistics."
            ),
            "implementation_hint": (
                "Where value embeddings are enabled (has_ve), combine the current token embedding with "
                "hashed embeddings of previous/next tokens, gated before addition."
            ),
            "risks": (
                "Hash collisions add noise; memory overhead increases with n-gram order; ablate carefully."
            ),
            "category": "embeddings",
        },
        {
            "name": "FP8 / bf16 Training",
            "description": (
                "Use lower-precision matmul to accelerate training on supported hardware while keeping "
                "sensitive states in bf16/fp32 for stability."
            ),
            "implementation_hint": (
                "On CUDA with FP8 support, enable FP8 autocast for matmuls. Otherwise keep bf16 autocast "
                "and ensure scaler/optimizer states remain stable."
            ),
            "risks": (
                "FP8 requires hardware support and careful scaling; bf16 is the safe fallback on most GPUs."
            ),
            "category": "precision",
        },
        {
            "name": "Token Shifting",
            "description": (
                "Mix each token with the previous token before the layer's main computation to improve "
                "local context modeling at negligible cost."
            ),
            "implementation_hint": (
                "In the Block forward, shift the input by one position and combine with a learned or fixed "
                "coefficient before attention/MLP. Gate the contribution from zero."
            ),
            "risks": (
                "Large shift coefficients can destabilize training, especially with aggressive schedules."
            ),
            "category": "attention",
        },
        {
            "name": "Learning Rate Warmup / Warmdown",
            "description": (
                "Schedule learning rate with an optional warmup phase and a warmdown/cooldown phase to "
                "stabilize early training and anneal smoothly."
            ),
            "implementation_hint": (
                "Adjust WARMUP_RATIO, WARMDOWN_RATIO, and FINAL_LR_FRAC in train.py. Schedules are "
                "computed from training_time / TIME_BUDGET."
            ),
            "risks": (
                "Long warmup wastes budget on small updates; abrupt warmdown can truncate useful training."
            ),
            "category": "optimizer",
        },
        {
            "name": "Gradient Clipping",
            "description": (
                "Clip gradients by global norm to prevent rare large updates from destabilizing training."
            ),
            "implementation_hint": (
                "Call torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step()."
            ),
            "risks": (
                "Too-low clipping slows convergence; it is usually unnecessary with MuonAdamW at modest LRs."
            ),
            "category": "optimizer",
        },
    ]

    method_ids = {}
    for method in methods:
        cur.execute(
            """
            INSERT OR IGNORE INTO methods
            (name, description, implementation_hint, risks, category)
            VALUES (:name, :description, :implementation_hint, :risks, :category)
            """,
            method,
        )
        cur.execute(
            "SELECT method_id FROM methods WHERE name = ?",
            (method["name"],),
        )
        method_ids[method["name"]] = cur.fetchone()["method_id"]

    # Map methods to source chunks
    method_chunk_map = [
        ("Grouped-Query Attention (GQA)", chunk_ids[0]),
        ("Grouped-Query Attention (GQA)", chunk_ids[1]),
        ("SwiGLU MLP", chunk_ids[2]),
        ("SwiGLU MLP", chunk_ids[3]),
        ("Muon LR Tuning", chunk_ids[4]),
        ("Muon LR Tuning", chunk_ids[5]),
        ("Sliding Window Attention Patterns", chunk_ids[6]),
        ("Sliding Window Attention Patterns", chunk_ids[7]),
        ("LayerNorm / RMSNorm Swap", chunk_ids[8]),
        ("LayerNorm / RMSNorm Swap", chunk_ids[9]),
        ("Dropout Regularization", chunk_ids[10]),
        ("Hashed n-gram Value Embeddings", chunk_ids[11]),
        ("Hashed n-gram Value Embeddings", chunk_ids[12]),
        ("FP8 / bf16 Training", chunk_ids[13]),
        ("Token Shifting", chunk_ids[14]),
        ("Token Shifting", chunk_ids[15]),
        ("Learning Rate Warmup / Warmdown", chunk_ids[16]),
        ("Gradient Clipping", chunk_ids[17]),
    ]
    method_by_name = {method["name"]: method for method in methods}
    seed_methods_by_title = {paper["title"]: {} for paper in papers}
    for method_name, chunk_id in method_chunk_map:
        evidence = _build_method_source_evidence(
            chunk_content_by_id[chunk_id], method_by_name[method_name]
        )
        cur.execute(
            """
            INSERT INTO method_sources (
                method_id, chunk_id, evidence_text, evidence_start, evidence_end, evidence_strategy
            )
            VALUES (
                :method_id, :chunk_id, :evidence_text, :evidence_start, :evidence_end, :evidence_strategy
            )
            ON CONFLICT(method_id, chunk_id) DO UPDATE SET
                evidence_text = COALESCE(method_sources.evidence_text, excluded.evidence_text),
                evidence_start = COALESCE(method_sources.evidence_start, excluded.evidence_start),
                evidence_end = COALESCE(method_sources.evidence_end, excluded.evidence_end),
                evidence_strategy = COALESCE(method_sources.evidence_strategy, excluded.evidence_strategy)
            """,
            {"method_id": method_ids[method_name], "chunk_id": chunk_id, **evidence},
        )
        paper_title = chunk_title_by_id[chunk_id]
        method_for_status = seed_methods_by_title[paper_title].setdefault(
            method_name,
            {**method_by_name[method_name], "source_chunk_indices": []},
        )
        method_for_status["source_chunk_indices"].append(
            paper_chunk_ids_by_title[paper_title].index(chunk_id)
        )

    for paper in papers:
        paper_title = paper["title"]
        paper_chunks = paper_chunk_content_by_title[paper_title]
        status_methods = list(seed_methods_by_title[paper_title].values())
        status_meta = {
            **paper,
            "content": "\n\n".join(paper_chunks),
        }
        _upsert_extraction_status(
            conn,
            paper_ids[paper_title],
            _build_extraction_status(
                status_meta,
                "extracted" if status_methods else "skipped",
                paper_chunks,
                status_methods,
            ),
        )

    # -----------------------------------------------------------------------
    # Experiment briefs
    # -----------------------------------------------------------------------
    brief_path = repo_root / "experiment_brief.md"
    brief_content = _load_markdown(brief_path)
    cur.execute(
        """
        INSERT OR IGNORE INTO experiment_briefs
        (name, objective, constraints, hypotheses, content)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "nanochat_autoresearch",
            "Minimize validation bits per byte (val_bpb) within a fixed training time budget.",
            "Single device; no new dependencies; do not edit prepare.py; time budget enforced after step 10.",
            "SwiGLU improves over ReLU^2; GQA improves speed/BPB; Muon LR tuning accelerates convergence; "
            "sliding-window patterns trade context for speed; normalization/regularization prevents overfitting; "
            "hashed value embeddings improve local statistics; token shifting adds cheap local context.",
            brief_content,
        ),
    )

    conn.commit()
    if close:
        conn.close()


def ensure_seed(conn=None) -> None:
    """Seed the database only if it contains no methods."""
    close = False
    if conn is None:
        from researchfs.db import get_connection

        conn = get_connection()
        close = True
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS count FROM methods")
        if cur.fetchone()["count"] == 0:
            seed_initial_data(conn)
    finally:
        if close:
            conn.close()


def _build_llm_fn() -> Optional[Callable[[str], str]]:
    """Return a callable LLM client if one can be configured locally.

    Supports OpenAI-compatible endpoints via the ``OPENAI_API_KEY`` environment
    variable.  Returns ``None`` if no local LLM can be built so callers can fall
    back to ingestion without extraction.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:  # pragma: no cover - optional dependency
        return None

    client = openai.OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    return _llm


def _iter_extracted_methods(result: dict) -> list:
    """Return the extracted methods list only when the LLM schema is usable."""
    methods = result.get("methods", [])
    return methods if isinstance(methods, list) else []


def _normalize_extracted_method(method: object) -> Optional[dict]:
    """Validate and normalize one LLM-extracted method before ingestion."""
    if not isinstance(method, dict):
        return None
    name = str(method.get("name") or "").strip()
    if not name:
        return None
    source_indices = method.get("source_chunk_indices") or []
    if not isinstance(source_indices, (list, tuple)):
        source_indices = []
    return {
        "name": name,
        "description": str(method.get("description") or "").strip(),
        "implementation_hint": str(method.get("implementation_hint") or "").strip(),
        "risks": str(method.get("risks") or "").strip(),
        "category": _normalize_extracted_category(method.get("category")),
        "source_chunk_indices": [
            idx
            for idx in source_indices
            if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0
        ],
    }


def _normalize_extracted_category(value: object) -> str:
    """Return a canonical extractor category token for one LLM item."""
    category = str(value or "").strip().lower()
    if category in extractor.METHOD_CATEGORY_TOKENS:
        return category
    return "method"


_EVIDENCE_EXCERPT_CHARS = 360
_EVIDENCE_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "can",
    "for",
    "from",
    "into",
    "method",
    "none",
    "the",
    "this",
    "use",
    "uses",
    "using",
    "with",
}


def _method_evidence_terms(method: dict) -> list[str]:
    """Return deterministic candidate terms from normalized method metadata."""
    terms = []
    seen = set()
    for field in ("name", "description", "implementation_hint"):
        for term in re.findall(r"[A-Za-z0-9]+", method.get(field) or ""):
            normalized = term.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized in _EVIDENCE_STOPWORDS:
                continue
            if len(normalized) < 3 and not any(ch.isdigit() for ch in normalized):
                continue
            terms.append(term)
    return terms


def _bounded_evidence_window(content: str, anchor_start: int, anchor_end: int) -> tuple[str, int, int]:
    """Return a stable excerpt window containing the anchor span."""
    if not content:
        return "", 0, 0
    excerpt_len = min(_EVIDENCE_EXCERPT_CHARS, len(content))
    anchor_mid = (anchor_start + anchor_end) // 2
    start = max(0, anchor_mid - excerpt_len // 2)
    end = min(len(content), start + excerpt_len)
    start = max(0, end - excerpt_len)
    return content[start:end], start, end


def _build_method_source_evidence(content: str, method: dict) -> dict:
    """Build deterministic provenance evidence for a method-source link."""
    content = content or ""
    for term in _method_evidence_terms(method):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        match = pattern.search(content)
        if match:
            evidence_text, start, end = _bounded_evidence_window(
                content, match.start(), match.end()
            )
            return {
                "evidence_text": evidence_text,
                "evidence_start": start,
                "evidence_end": end,
                "evidence_strategy": f"matched_term:{term.lower()}",
            }

    end = min(_EVIDENCE_EXCERPT_CHARS, len(content))
    return {
        "evidence_text": content[:end],
        "evidence_start": 0,
        "evidence_end": end,
        "evidence_strategy": "leading_excerpt" if content else "empty_chunk",
    }


def _build_method_fts_query(name: str) -> str:
    """Build an FTS5-safe OR query from a method name."""
    terms = re.findall(r"[A-Za-z0-9]+", name)
    return " OR ".join(f'"{term}"' for term in terms)


def _check_duplicate_method(
    method: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    conn=None,
) -> Optional[int]:
    """Check if the extracted method is a duplicate of an existing method.

    Checks:
    1. Case-insensitive name match.
    2. FTS5 query on method name keywords for similar candidates.
    3. LLM decision on the candidates if llm_fn is provided.
    """
    name = method.get("name")
    if not name:
        return None

    close = False
    if conn is None:
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT method_id FROM methods WHERE LOWER(name) = LOWER(?)",
                (name,),
            )
            row = cur.fetchone()
        except sqlite3.OperationalError:
            return None
        if row:
            return row[0]

        # 2. FTS5 search using keywords from the method's name
        fts_query = _build_method_fts_query(name)
        if not fts_query:
            return None
        try:
            cur.execute(
                "SELECT rowid, name, description FROM methods_fts WHERE methods_fts MATCH ? ORDER BY rank LIMIT 3",
                (fts_query,),
            )
            candidates = cur.fetchall()
        except sqlite3.OperationalError:
            candidates = []

        # 3. LLM comparison
        if candidates and llm_fn is not None:
            extracted_name = name
            extracted_desc = method.get("description", "")
            for cand in candidates:
                existing_name = cand["name"]
                existing_desc = cand["description"] or ""
                
                prompt = (
                    f"You are an expert machine learning database curator.\n"
                    f"We are ingesting a new method:\n"
                    f"Name: {extracted_name}\n"
                    f"Description: {extracted_desc}\n\n"
                    f"We found an existing method in the database:\n"
                    f"Name: {existing_name}\n"
                    f"Description: {existing_desc}\n\n"
                    f"Are these the exact same optimization method?\n"
                    f"Reply ONLY with 'YES' or 'NO'. Do not include explanation or markdown."
                )
                response = llm_fn(prompt)
                if response and response.strip().upper() == "YES":
                    return cand["rowid"]

        return None
    finally:
        if close:
            conn.close()


def _infer_source_type(meta: dict) -> Optional[str]:
    """Return source type metadata when it can be inferred cheaply."""
    source_type = meta.get("source_type")
    if source_type:
        return str(source_type)
    url = meta.get("url") or ""
    title = meta.get("title") or ""
    if "arxiv.org" in url:
        return "arxiv"
    if "github.com" in url or title.startswith("GitHub:"):
        return "github"
    return None


def _extractor_metadata(meta: dict, llm_fn: Optional[Callable[[str], str]]) -> tuple[Optional[str], Optional[str]]:
    """Return extractor model and prompt version when the caller exposes them."""
    model = meta.get("extractor_model")
    prompt_version = meta.get("extractor_prompt_version")
    if llm_fn is not None:
        model = model or getattr(llm_fn, "model", None) or getattr(llm_fn, "model_name", None)
        prompt_version = prompt_version or getattr(llm_fn, "prompt_version", None)
    prompt_version = prompt_version or getattr(extractor, "PROMPT_VERSION", None)
    return (str(model) if model else None, str(prompt_version) if prompt_version else None)


def _compressed_method_char_count(methods: list[dict]) -> int:
    """Count characters in the compressed method memory payload."""
    fields = ("name", "description", "implementation_hint", "risks", "category")
    return sum(len(str(method.get(field) or "")) for method in methods for field in fields)

def _compression_confidence_component(compression_ratio: Optional[float]) -> tuple[float, str]:
    """Return a bounded confidence component for compression sanity."""
    if compression_ratio is None:
        return 0.0, "missing"
    if compression_ratio <= 0:
        return 0.0, "nonpositive"
    if compression_ratio <= 1.0:
        return 1.0, "sane"
    if compression_ratio <= 2.0:
        return 0.5, "large"
    return 0.1, "oversized"


def _method_source_confidence_stats(methods: list[dict], chunks: list[str]) -> dict:
    """Measure observable source-link and evidence coverage for normalized methods."""
    method_count = len(methods)
    total_source_links = 0
    valid_source_links = 0
    methods_with_valid_source = 0
    evidence_links = 0
    methods_with_evidence = 0

    for method in methods:
        method_has_valid_source = False
        method_has_evidence = False
        for idx in method.get("source_chunk_indices") or []:
            total_source_links += 1
            if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < len(chunks)):
                continue
            valid_source_links += 1
            method_has_valid_source = True
            evidence = _build_method_source_evidence(chunks[idx], method)
            if (
                evidence.get("evidence_text")
                and isinstance(evidence.get("evidence_start"), int)
                and isinstance(evidence.get("evidence_end"), int)
                and evidence["evidence_end"] > evidence["evidence_start"]
            ):
                evidence_links += 1
                method_has_evidence = True
        if method_has_valid_source:
            methods_with_valid_source += 1
        if method_has_evidence:
            methods_with_evidence += 1

    return {
        "method_count": method_count,
        "total_source_links": total_source_links,
        "valid_source_links": valid_source_links,
        "methods_with_valid_source": methods_with_valid_source,
        "evidence_links": evidence_links,
        "methods_with_evidence": methods_with_evidence,
    }


def _extraction_confidence(
    status: str,
    methods: list[dict],
    chunks: list[str],
    compression_ratio: Optional[float],
    error_text: Optional[str] = None,
) -> tuple[float, str]:
    """Score deterministic extraction confidence from observable extraction quality."""
    if status == "skipped":
        return 0.0, "skipped"
    if status == "failed":
        reason = "failed"
        if error_text:
            reason = f"{reason}: {error_text}"
        return 0.0, reason

    stats = _method_source_confidence_stats(methods, chunks)
    method_count = stats["method_count"]
    total_source_links = stats["total_source_links"]
    valid_source_links = stats["valid_source_links"]
    methods_with_valid_source = stats["methods_with_valid_source"]
    evidence_links = stats["evidence_links"]
    methods_with_evidence = stats["methods_with_evidence"]

    method_score = min(method_count / 3.0, 1.0) if method_count else 0.0
    source_link_score = (
        valid_source_links / total_source_links if total_source_links else 0.0
    )
    source_coverage_score = (
        methods_with_valid_source / method_count if method_count else 0.0
    )
    evidence_coverage_score = (
        methods_with_evidence / method_count if method_count else 0.0
    )
    compression_score, compression_label = _compression_confidence_component(
        compression_ratio
    )
    score = (
        0.15
        + 0.20 * method_score
        + 0.20 * source_link_score
        + 0.20 * source_coverage_score
        + 0.15 * evidence_coverage_score
        + 0.10 * compression_score
    )
    bounded_score = round(max(0.0, min(1.0, score)), 3)
    ratio_text = "missing" if compression_ratio is None else f"{compression_ratio:.3f}"
    reason = (
        f"extracted: methods={method_count}; "
        f"valid_source_links={valid_source_links}/{total_source_links}; "
        f"method_source_coverage={methods_with_valid_source}/{method_count}; "
        f"evidence_coverage={methods_with_evidence}/{method_count}; "
        f"evidence_links={evidence_links}/{valid_source_links}; "
        f"compression_ratio={ratio_text}; compression={compression_label}"
    )
    return bounded_score, reason


def _build_extraction_status(
    meta: dict,
    status: str,
    chunks: list[str],
    methods: Optional[list[dict]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    error_text: Optional[str] = None,
) -> dict:
    """Build the durable extraction status payload returned and stored for a paper."""
    methods = methods or []
    source_text = meta.get("content", meta.get("abstract", "")) or ""
    source_char_count = len(source_text)
    compressed_char_count = _compressed_method_char_count(methods)
    compression_ratio = (
        compressed_char_count / source_char_count if source_char_count else None
    )
    extractor_model, extractor_prompt_version = _extractor_metadata(meta, llm_fn)
    confidence_score, confidence_reason = _extraction_confidence(
        status, methods, chunks, compression_ratio, error_text
    )
    return {
        "status": status,
        "source_type": _infer_source_type(meta),
        "chunk_count": len(chunks),
        "method_count": len(methods),
        "source_char_count": source_char_count,
        "compressed_char_count": compressed_char_count,
        "compression_ratio": compression_ratio,
        "confidence_score": confidence_score,
        "confidence_reason": confidence_reason,
        "external_id": meta.get("external_id"),
        "extractor_model": extractor_model,
        "extractor_prompt_version": extractor_prompt_version,
        "error_text": error_text,
    }


def _row_to_extraction_status(row) -> Optional[dict]:
    if row is None:
        return None
    return {
        "paper_id": row["paper_id"],
        "status": row["status"],
        "source_type": row["source_type"],
        "chunk_count": row["chunk_count"],
        "method_count": row["method_count"],
        "source_char_count": row["source_char_count"],
        "compressed_char_count": row["compressed_char_count"],
        "compression_ratio": row["compression_ratio"],
        "confidence_score": row["confidence_score"],
        "confidence_reason": row["confidence_reason"],
        "external_id": row["external_id"],
        "extractor_model": row["extractor_model"],
        "extractor_prompt_version": row["extractor_prompt_version"],
        "error_text": row["error_text"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _get_extraction_status(conn, paper_id: int) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM paper_extraction_status WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    return _row_to_extraction_status(row)

def _paper_confidence_inputs(conn, paper_id: int) -> tuple[list[str], list[dict]]:
    """Reconstruct observable paper chunks and method links for legacy status confidence."""
    chunk_rows = conn.execute(
        """
        SELECT chunk_id, content
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        """,
        (paper_id,),
    ).fetchall()
    chunks = [(row["content"] or "") for row in chunk_rows]
    chunk_index_by_id = {
        row["chunk_id"]: idx for idx, row in enumerate(chunk_rows)
    }
    method_rows = conn.execute(
        """
        SELECT m.method_id, m.name, m.description, m.implementation_hint, m.risks, m.category, ms.chunk_id
        FROM method_sources ms
        JOIN methods m ON m.method_id = ms.method_id
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        ORDER BY m.method_id, ms.chunk_id
        """,
        (paper_id,),
    ).fetchall()
    methods_by_id = {}
    for row in method_rows:
        method = methods_by_id.setdefault(
            row["method_id"],
            {
                "name": row["name"] or "",
                "description": row["description"] or "",
                "implementation_hint": row["implementation_hint"] or "",
                "risks": row["risks"] or "",
                "category": row["category"] or "",
                "source_chunk_indices": [],
            },
        )
        chunk_index = chunk_index_by_id.get(row["chunk_id"])
        if chunk_index is not None:
            method["source_chunk_indices"].append(chunk_index)
    return chunks, list(methods_by_id.values())


def _status_payload_from_row(status: dict) -> dict:
    """Return an upsert payload from a stored extraction status dictionary."""
    return {
        "status": status["status"],
        "source_type": status["source_type"],
        "chunk_count": status["chunk_count"],
        "method_count": status["method_count"],
        "source_char_count": status["source_char_count"],
        "compressed_char_count": status["compressed_char_count"],
        "compression_ratio": status["compression_ratio"],
        "confidence_score": status["confidence_score"],
        "confidence_reason": status["confidence_reason"],
        "external_id": status["external_id"],
        "extractor_model": status["extractor_model"],
        "extractor_prompt_version": status["extractor_prompt_version"],
        "error_text": status["error_text"],
    }



def _upsert_extraction_status(conn, paper_id: int, status_payload: dict) -> dict:
    conn.execute(
        """
        INSERT INTO paper_extraction_status (
            paper_id, status, source_type, external_id, chunk_count, method_count,
            source_char_count, compressed_char_count, compression_ratio,
            confidence_score, confidence_reason,
            extractor_model, extractor_prompt_version, error_text
        )
        VALUES (
            :paper_id, :status, :source_type, :external_id, :chunk_count, :method_count,
            :source_char_count, :compressed_char_count, :compression_ratio,
            :confidence_score, :confidence_reason,
            :extractor_model, :extractor_prompt_version, :error_text
        )
        ON CONFLICT(paper_id) DO UPDATE SET
            status = excluded.status,
            source_type = excluded.source_type,
            external_id = excluded.external_id,
            chunk_count = excluded.chunk_count,
            method_count = excluded.method_count,
            source_char_count = excluded.source_char_count,
            compressed_char_count = excluded.compressed_char_count,
            compression_ratio = excluded.compression_ratio,
            confidence_score = excluded.confidence_score,
            confidence_reason = excluded.confidence_reason,
            extractor_model = excluded.extractor_model,
            extractor_prompt_version = excluded.extractor_prompt_version,
            error_text = excluded.error_text,
            updated_at = CURRENT_TIMESTAMP
        """,
        {"paper_id": paper_id, **status_payload},
    )
    return _get_extraction_status(conn, paper_id) or {"paper_id": paper_id, **status_payload}


def _update_existing_source_type_if_explicit(
    conn,
    paper_id: int,
    extraction_status: dict,
    meta: dict,
) -> tuple[dict, bool]:
    """Apply an explicit duplicate-ingest source_type correction without touching status metadata."""
    explicit_source_type = _explicit_source_type(meta)
    if (
        explicit_source_type is None
        or extraction_status["source_type"] == explicit_source_type
    ):
        return extraction_status, False
    conn.execute(
        "UPDATE paper_extraction_status SET source_type = ? WHERE paper_id = ?",
        (explicit_source_type, paper_id),
    )
    updated_status = _get_extraction_status(conn, paper_id)
    return (updated_status or {**extraction_status, "source_type": explicit_source_type}), True

def _extract_methods_from_existing_chunks(
    meta: dict,
    chunks: list[str],
    llm_fn: Callable[[str], str],
) -> dict:
    """Extract methods from already-stored paper chunks without rechunking them."""
    return extractor.extract_methods_from_chunks(meta, chunks, llm_fn)


def _stored_paper_inputs(conn, paper_id: int, fallback_meta: dict) -> tuple[dict, list[str], list[int]]:
    """Return stored paper metadata, chunk text, and chunk ids for duplicate extraction."""
    paper = conn.execute(
        """
        SELECT title, authors, abstract, url, publish_date
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    chunk_rows = conn.execute(
        """
        SELECT chunk_id, content
        FROM chunks
        WHERE paper_id = ?
        ORDER BY chunk_id
        """,
        (paper_id,),
    ).fetchall()
    chunks = [(row["content"] or "") for row in chunk_rows]
    content = "\n\n".join(chunks) if chunks else (fallback_meta.get("content") or fallback_meta.get("abstract") or "")
    stored_meta = dict(fallback_meta)
    if paper is not None:
        stored_meta.update(
            {
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "url": paper["url"],
                "publish_date": paper["publish_date"],
            }
        )
    stored_meta["content"] = content
    return stored_meta, chunks, [row["chunk_id"] for row in chunk_rows]


def _paper_has_method_sources(conn, paper_id: int) -> bool:
    """Return whether a paper already has method provenance links."""
    row = conn.execute(
        """
        SELECT 1
        FROM method_sources ms
        JOIN chunks c ON c.chunk_id = ms.chunk_id
        WHERE c.paper_id = ?
        LIMIT 1
        """,
        (paper_id,),
    ).fetchone()
    return row is not None


def _insert_or_link_extracted_methods(
    conn,
    chunks: list[str],
    chunk_ids: list[int],
    raw_methods: list,
    llm_fn: Callable[[str], str],
) -> tuple[list[int], list[dict]]:
    """Insert or reuse extracted methods and link them to the supplied chunks."""
    cur = conn.cursor()
    method_ids = []
    normalized_methods = []
    for raw_method in raw_methods:
        method = _normalize_extracted_method(raw_method)
        if method is None:
            continue
        normalized_methods.append(method)
        resolved_id = _check_duplicate_method(method, llm_fn=llm_fn, conn=conn)
        if resolved_id is not None:
            method_id = resolved_id
        else:
            cur.execute(
                """
                INSERT OR IGNORE INTO methods
                (name, description, implementation_hint, risks, category)
                VALUES (:name, :description, :implementation_hint, :risks, :category)
                """,
                {
                    "name": method["name"],
                    "description": method["description"],
                    "implementation_hint": method["implementation_hint"],
                    "risks": method["risks"],
                    "category": method["category"],
                },
            )
            cur.execute(
                "SELECT method_id FROM methods WHERE name = ?",
                (method["name"],),
            )
            method_id = cur.fetchone()["method_id"]

        method_ids.append(method_id)

        for idx in method["source_chunk_indices"]:
            if 0 <= idx < len(chunk_ids):
                evidence = _build_method_source_evidence(chunks[idx], method)
                cur.execute(
                    """
                    INSERT INTO method_sources (
                        method_id, chunk_id, evidence_text, evidence_start, evidence_end, evidence_strategy
                    )
                    VALUES (
                        :method_id, :chunk_id, :evidence_text, :evidence_start, :evidence_end, :evidence_strategy
                    )
                    ON CONFLICT(method_id, chunk_id) DO UPDATE SET
                        evidence_text = COALESCE(method_sources.evidence_text, excluded.evidence_text),
                        evidence_start = COALESCE(method_sources.evidence_start, excluded.evidence_start),
                        evidence_end = COALESCE(method_sources.evidence_end, excluded.evidence_end),
                        evidence_strategy = COALESCE(method_sources.evidence_strategy, excluded.evidence_strategy)
                    """,
                    {"method_id": method_id, "chunk_id": chunk_ids[idx], **evidence},
                )

    return method_ids, normalized_methods


def _reextract_existing_paper(
    conn,
    paper_id: int,
    meta: dict,
    llm_fn: Callable[[str], str],
) -> dict:
    """Run extraction for an already-stored paper that has no usable compressed memory."""
    stored_meta, stored_chunks, stored_chunk_ids = _stored_paper_inputs(conn, paper_id, meta)
    extraction_chunks = stored_chunks or extractor.chunk_text(stored_meta.get("content", "") or "")
    try:
        result = _extract_methods_from_existing_chunks(stored_meta, extraction_chunks, llm_fn)
        method_ids, normalized_methods = _insert_or_link_extracted_methods(
            conn,
            extraction_chunks,
            stored_chunk_ids,
            _iter_extracted_methods(result),
            llm_fn,
        )
        extraction_status = _upsert_extraction_status(
            conn,
            paper_id,
            _build_extraction_status(
                stored_meta, "extracted", extraction_chunks, normalized_methods, llm_fn=llm_fn
            ),
        )
    except Exception as exc:
        method_ids = []
        extraction_status = _upsert_extraction_status(
            conn,
            paper_id,
            _build_extraction_status(
                stored_meta, "failed", extraction_chunks, llm_fn=llm_fn, error_text=str(exc)
            ),
        )
    return {
        "paper_id": paper_id,
        "chunk_ids": stored_chunk_ids,
        "method_ids": method_ids,
        "extraction_status": extraction_status,
        "duplicate": True,
    }

def _visual_index_failed_payload(error_text: object) -> dict:
    """Return a stable failed PixelRAG payload without raising into ingestion."""
    return {
        "provider": "pixelrag",
        "status": "failed",
        "source": None,
        "tiles_path": None,
        "embeddings_path": None,
        "index_path": None,
        "error_text": str(error_text),
    }


def _visual_plan_identifier(meta: dict) -> str:
    """Return a stable dry-run identifier for planned visual-index paths."""
    raw = meta.get("external_id") or meta.get("title") or meta.get("url") or "planned"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw)).strip("-")
    return f"planned-{(label or 'paper')[:80]}"


def _plan_visual_index_payload(meta: dict, output_root: str | Path | None) -> dict:
    """Plan an optional visual index without importing PixelRAG or running commands."""
    try:
        from researchfs import visual

        return visual.plan_pixelrag_visual_index(_visual_plan_identifier(meta), meta, output_root=output_root)
    except Exception as exc:
        return _visual_index_failed_payload(exc)


def _attach_planned_visual_index(
    result: dict,
    meta: dict,
    output_root: str | Path | None,
) -> dict:
    visual_index_payload = _plan_visual_index_payload(meta, output_root)
    result["visual_index"] = visual_index_payload
    planned = result.get("planned")
    if isinstance(planned, dict):
        planned["visual_index"] = visual_index_payload
    return result


def _attach_stored_visual_index(
    conn,
    result: dict,
    meta: dict,
    output_root: str | Path | None,
    runner=None,
) -> dict:
    paper_id = result.get("paper_id")
    if paper_id is None:
        return result
    try:
        from researchfs import visual

        result["visual_index"] = visual.build_and_store_pixelrag_visual_index(
            conn,
            paper_id,
            meta,
            output_root=output_root,
            runner=runner,
        )
        conn.commit()
    except Exception as exc:
        result["visual_index"] = _visual_index_failed_payload(exc)
    return result


def _resolve_code_repo_search_fn(code_repo_search_fn=None, code_repo_searcher=None):
    """Return the configured code repository search callable, accepting legacy aliases."""
    if code_repo_search_fn is not None and code_repo_searcher is not None:
        if code_repo_search_fn is not code_repo_searcher:
            raise ValueError("Pass either code_repo_search_fn or code_repo_searcher, not both")
    return code_repo_search_fn if code_repo_search_fn is not None else code_repo_searcher


def _code_repository_rows_from_enrichment(enrichment: object) -> list[dict]:
    """Return selected/stored repository rows from an enrichment result."""
    if isinstance(enrichment, list):
        return [row for row in enrichment if isinstance(row, dict)]
    if not isinstance(enrichment, dict):
        return []
    selected = enrichment.get("selected")
    if isinstance(selected, dict):
        return [selected]
    if isinstance(selected, list):
        return [row for row in selected if isinstance(row, dict)]
    stored = enrichment.get("stored")
    if isinstance(stored, list):
        return [row for row in stored if isinstance(row, dict)]
    return []


def _attach_code_repository_enrichment(
    conn,
    result: dict,
    *,
    enrich_code: bool,
    code_repo_search_fn=None,
) -> dict:
    """Attach automatic code repository enrichment results to an ingestion result."""
    if not enrich_code:
        return result
    paper_id = result.get("paper_id")
    if paper_id is None:
        return result
    try:
        enrichment = enrich_paper_code_repositories(
            conn,
            int(paper_id),
            search_fn=code_repo_search_fn,
        )
    except Exception as exc:
        result["code_repository_enrichment"] = {
            "paper_id": int(paper_id),
            "selected": None,
            "stored": [],
            "candidates": [],
            "error_text": str(exc),
        }
        result["code_repositories"] = []
        return result
    result["code_repository_enrichment"] = enrichment
    result["code_repositories"] = _code_repository_rows_from_enrichment(enrichment)
    return result


def ingest_paper(
    meta: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    conn=None,
    dry_run: bool = False,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> dict:
    """Insert (or plan) a paper, its chunks, methods, and extraction status."""
    code_repo_search_fn = _resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher)
    if dry_run:
        close_conn = False
        if conn is None:
            conn = _open_dry_run_target_connection()
            if conn is None:
                conn = sqlite3.connect(":memory:")
                conn.row_factory = sqlite3.Row
                initialize_db(conn)
            close_conn = True
        try:
            content = meta.get("content", meta.get("abstract", "")) or ""
            chunks = extractor.chunk_text(content)
            planned = {
                "paper": {
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "abstract": meta.get("abstract"),
                    "url": meta.get("url"),
                    "publish_date": meta.get("publish_date"),
                    "external_id": meta.get("external_id"),
                },
                "chunks": [
                    {"section": "extracted", "content": chunk} for chunk in chunks
                ],
                "methods": [],
                "merged_methods": [],
                "method_sources": [],
            }
            normalized_methods = []
            if llm_fn is None:
                extraction_status = _build_extraction_status(
                    meta, "skipped", chunks, llm_fn=llm_fn
                )
            else:
                try:
                    result = extractor.extract_methods_from_paper(meta, llm_fn)
                    for raw_method in _iter_extracted_methods(result):
                        method = _normalize_extracted_method(raw_method)
                        if method is None:
                            continue
                        normalized_methods.append(method)
                        resolved_id = _check_duplicate_method(method, llm_fn=llm_fn, conn=conn)
                        if resolved_id is not None:
                            planned["merged_methods"].append(
                                {
                                    "name": method["name"],
                                    "merged_into_id": resolved_id,
                                }
                            )
                        else:
                            planned["methods"].append(
                                {
                                    "name": method["name"],
                                    "description": method["description"],
                                    "implementation_hint": method["implementation_hint"],
                                    "risks": method["risks"],
                                    "category": method["category"],
                                }
                            )
                        for idx in method["source_chunk_indices"]:
                            if 0 <= idx < len(chunks):
                                evidence = _build_method_source_evidence(chunks[idx], method)
                                planned["method_sources"].append(
                                    {
                                        "method_name": method["name"],
                                        "chunk_index": idx,
                                        **evidence,
                                    }
                                )
                    extraction_status = _build_extraction_status(
                        meta, "extracted", chunks, normalized_methods, llm_fn=llm_fn
                    )
                except Exception as exc:  # pragma: no cover
                    planned["extraction_error"] = str(exc)
                    extraction_status = _build_extraction_status(
                        meta, "failed", chunks, llm_fn=llm_fn, error_text=str(exc)
                    )
            planned["extraction_status"] = extraction_status
            result = {"planned": planned, "extraction_status": extraction_status}
            if visual_index:
                result = _attach_planned_visual_index(result, meta, visual_index_root)
            return result
        finally:
            if close_conn:
                conn.close()

    close = False
    if conn is None:
        initialize_db()
        conn = get_connection()
        close = True

    try:
        migrate_db(conn)
        cur = conn.cursor()
        content = meta.get("content", meta.get("abstract", "")) or ""
        chunks = extractor.chunk_text(content)
        cur.execute(
            "SELECT paper_id FROM papers WHERE title = ?",
            (meta["title"],),
        )
        existing = cur.fetchone()
        if existing:
            paper_id = existing["paper_id"]
            extraction_status = _get_extraction_status(conn, paper_id)
            has_method_sources = _paper_has_method_sources(conn, paper_id)
            should_reextract = llm_fn is not None and (
                extraction_status is None
                or extraction_status["status"] in {"skipped", "failed"}
                or not has_method_sources
            )
            if should_reextract:
                result = _reextract_existing_paper(conn, paper_id, meta, llm_fn)
                conn.commit()
                if visual_index:
                    result = _attach_stored_visual_index(
                        conn,
                        result,
                        meta,
                        visual_index_root,
                        runner=visual_index_runner,
                    )
                result = _attach_code_repository_enrichment(
                    conn,
                    result,
                    enrich_code=enrich_code,
                    code_repo_search_fn=code_repo_search_fn,
                )
                return result
            if extraction_status is None:
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(meta, "skipped", chunks, llm_fn=llm_fn),
                )
                conn.commit()
            elif (
                extraction_status["confidence_score"] is None
                or extraction_status["confidence_reason"] is None
            ):
                stored_chunks, stored_methods = _paper_confidence_inputs(conn, paper_id)
                if not stored_chunks:
                    stored_chunks = chunks
                payload = _status_payload_from_row(extraction_status)
                confidence_score, confidence_reason = _extraction_confidence(
                    extraction_status["status"],
                    stored_methods,
                    stored_chunks,
                    extraction_status["compression_ratio"],
                    extraction_status["error_text"],
                )
                payload["confidence_score"] = confidence_score
                payload["confidence_reason"] = confidence_reason
                extraction_status = _upsert_extraction_status(conn, paper_id, payload)
                conn.commit()
            extraction_status, source_type_changed = _update_existing_source_type_if_explicit(
                conn,
                paper_id,
                extraction_status,
                meta,
            )
            if source_type_changed:
                conn.commit()
            result = {
                "paper_id": paper_id,
                "chunk_ids": [],
                "method_ids": [],
                "extraction_status": extraction_status,
                "duplicate": True,
            }
            if visual_index:
                result = _attach_stored_visual_index(
                    conn,
                    result,
                    meta,
                    visual_index_root,
                    runner=visual_index_runner,
                )
            result = _attach_code_repository_enrichment(
                conn,
                result,
                enrich_code=enrich_code,
                code_repo_search_fn=code_repo_search_fn,
            )
            return result

        cur.execute(
            """
            INSERT INTO papers (title, authors, abstract, url, publish_date)
            VALUES (:title, :authors, :abstract, :url, :publish_date)
            """,
            {
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "abstract": meta.get("abstract"),
                "url": meta.get("url"),
                "publish_date": meta.get("publish_date"),
            },
        )
        paper_id = cur.lastrowid

        chunk_ids = []
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
                (paper_id, "extracted", chunk),
            )
            chunk_ids.append(cur.lastrowid)

        method_ids = []
        normalized_methods = []
        if llm_fn is None:
            extraction_status = _upsert_extraction_status(
                conn,
                paper_id,
                _build_extraction_status(meta, "skipped", chunks, llm_fn=llm_fn),
            )
        else:
            try:
                result = extractor.extract_methods_from_paper(meta, llm_fn)
                method_ids, normalized_methods = _insert_or_link_extracted_methods(
                    conn,
                    chunks,
                    chunk_ids,
                    _iter_extracted_methods(result),
                    llm_fn,
                )
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(
                        meta, "extracted", chunks, normalized_methods, llm_fn=llm_fn
                    ),
                )
            except Exception as exc:
                extraction_status = _upsert_extraction_status(
                    conn,
                    paper_id,
                    _build_extraction_status(
                        meta, "failed", chunks, llm_fn=llm_fn, error_text=str(exc)
                    ),
                )
        conn.commit()
        result = {
            "paper_id": paper_id,
            "chunk_ids": chunk_ids,
            "method_ids": method_ids,
            "extraction_status": extraction_status,
        }
        if visual_index:
            result = _attach_stored_visual_index(
                conn,
                result,
                meta,
                visual_index_root,
                runner=visual_index_runner,
            )
        result = _attach_code_repository_enrichment(
            conn,
            result,
            enrich_code=enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        return result
    finally:
        if close:
            conn.close()


def _default_github_code_searcher(queries: list[str]) -> list[dict]:
    """Search GitHub repositories for automatic CLI code enrichment."""
    import requests

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    seen: set[str] = set()
    results: list[dict] = []
    for query_index, query in enumerate(queries):
        query = str(query or "").strip()
        if not query:
            continue
        response = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        for rank, item in enumerate(items):
            html_url = item.get("html_url")
            if not html_url or html_url in seen:
                continue
            seen.add(html_url)
            results.append(
                {
                    "url": html_url,
                    "full_name": item.get("full_name"),
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "evidence": f"GitHub search query: {query}; rank={rank + 1}",
                    "confidence": max(0.35, 0.74 - 0.04 * query_index - 0.02 * rank),
                }
            )
    return results


def ingest_arxiv_id(
    arxiv_id: str,
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> dict:
    """Fetch and ingest a single arXiv paper.

    Wraps :func:`fetch_arxiv_source` and :func:`ingest_paper`, optionally
    overriding the ``source_type`` metadata label. Code repository enrichment
    runs automatically unless disabled. Returns both metadata and ingestion.
    """
    meta = fetch_arxiv_source(arxiv_id)
    _apply_source_type_override(meta, source_type)
    return {
        "meta": meta,
        "result": ingest_paper(
            meta,
            llm_fn=llm_fn,
            conn=conn,
            dry_run=dry_run,
            visual_index=visual_index,
            visual_index_root=visual_index_root,
            visual_index_runner=visual_index_runner,
            enrich_code=enrich_code,
            code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
        ),
    }


def ingest_arxiv_ids(
    arxiv_ids: list[str],
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> list[dict]:
    """Fetch and ingest arXiv papers with automatic code repository enrichment."""
    results: list[dict] = []
    for raw_id in arxiv_ids:
        arxiv_id = raw_id.strip()
        if not arxiv_id:
            continue
        try:
            outcome = ingest_arxiv_id(
                arxiv_id,
                llm_fn=llm_fn,
                conn=conn,
                dry_run=dry_run,
                source_type=source_type,
                visual_index=visual_index,
                visual_index_root=visual_index_root,
                visual_index_runner=visual_index_runner,
                enrich_code=enrich_code,
                code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
            )
            meta = outcome.get("meta", {})
            result = outcome.get("result", {})
            title = meta.get("title", "")
            status = "unknown"
            paper_id = "planned" if dry_run else None
            if dry_run:
                planned = result.get("planned", {})
                status = (planned.get("extraction_status") or {}).get("status", "unknown")
            else:
                paper_id = result.get("paper_id")
                status = (result.get("extraction_status") or {}).get("status", "unknown")
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "paper_id": paper_id,
                    "status": status,
                    "error": None,
                    "result": result,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": "",
                    "paper_id": None,
                    "status": "error",
                    "error": str(exc),
                    "result": None,
                }
            )
    return results


def ingest_arxiv_query(
    query: str,
    llm_fn=None,
    conn=None,
    dry_run: bool = False,
    source_type: Optional[str] = None,
    max_results: int = 10,
    visual_index: bool = False,
    visual_index_root: str | Path | None = None,
    visual_index_runner=None,
    enrich_code: bool = True,
    code_repo_search_fn=None,
    code_repo_searcher=None,
) -> list[dict]:
    """Search arXiv and ingest matches with automatic code repository enrichment."""
    arxiv_ids = fetch_arxiv_query(query, max_results=max_results)
    return ingest_arxiv_ids(
        arxiv_ids,
        llm_fn=llm_fn,
        conn=conn,
        dry_run=dry_run,
        source_type=source_type,
        visual_index=visual_index,
        visual_index_root=visual_index_root,
        visual_index_runner=visual_index_runner,
        enrich_code=enrich_code,
        code_repo_search_fn=_resolve_code_repo_search_fn(code_repo_search_fn, code_repo_searcher),
    )


def _print_arxiv_batch_summary(results: list[dict], dry_run: bool = False) -> None:
    """Print a concise summary line for each arXiv batch result."""
    if dry_run:
        print("DRY RUN - planned batch ingestion")
    else:
        print("Ingested batch:")
    has_visual_status = any((item.get("result") or {}).get("visual_index") for item in results)
    rule_width = 82 if has_visual_status else 70
    print("-" * rule_width)
    if has_visual_status:
        print(f"{'ID':<20}{'Paper ID':<14}{'Status':<14}{'Visual':<12}Title")
    else:
        print(f"{'ID':<20}{'Paper ID':<14}{'Status':<14}Title")
    print("-" * rule_width)
    for item in results:
        arxiv_id = item.get("arxiv_id", "")
        title = (item.get("title") or "").strip()
        paper_id = item.get("paper_id")
        if paper_id is None:
            paper_id_label = "error"
        elif paper_id == "planned":
            paper_id_label = "planned"
        else:
            paper_id_label = str(paper_id)
        status = item.get("status", "unknown")
        visual_index_payload = (item.get("result") or {}).get("visual_index") or {}
        visual_status = visual_index_payload.get("status", "")
        error = item.get("error")
        if has_visual_status:
            line = f"{arxiv_id:<20}{paper_id_label:<14}{status:<14}{visual_status:<12}{title}"
        else:
            line = f"{arxiv_id:<20}{paper_id_label:<14}{status:<14}{title}"
        print(line)
        if error:
            print(f"  -> error: {error}")


def _print_single_ingestion_result(result: dict, dry_run: bool = False) -> None:
    """Print the single-paper ingestion result, preserving the legacy format."""
    if dry_run:
        planned = result.get("planned", {})
        paper = planned.get("paper", {})
        print("DRY RUN - planned ingestion")
        print("=" * 40)
        print(f"Paper title : {paper.get('title')}")
        print(f"Authors     : {paper.get('authors')}")
        print(f"URL         : {paper.get('url')}")
        print(f"Publish date: {paper.get('publish_date')}")
        print(f"Chunks      : {len(planned.get('chunks', []))}")
        methods = planned.get("methods", [])
        merged_methods = planned.get("merged_methods", [])
        print(f"Methods     : {len(methods) + len(merged_methods)}")
        extraction_status = result.get("extraction_status") or planned.get("extraction_status", {})
        print(f"Extraction : {extraction_status.get('status')}")
        print(f"External ID : {extraction_status.get('external_id')}")
        print(f"Compression: {extraction_status.get('compressed_char_count')} / {extraction_status.get('source_char_count')} chars")
        print(f"Confidence : {extraction_status.get('confidence_score')} ({extraction_status.get('confidence_reason')})")
        visual_index_payload = result.get("visual_index") or planned.get("visual_index")
        if visual_index_payload:
            print(f"Visual index: {visual_index_payload.get('status')}")
            if visual_index_payload.get("error_text"):
                print(f"Visual error: {visual_index_payload.get('error_text')}")
        for method in methods:
            print(f"  - {method.get('name')} ({method.get('category')})")
        for method in merged_methods:
            print(f"  - {method.get('name')} (merged into existing)")
        if planned.get("extraction_error"):
            print(f"Extraction error: {planned['extraction_error']}")
    else:
        print("Ingested:")
        print(f"  paper_id : {result['paper_id']}")
        print(f"  chunk_ids: {result['chunk_ids']}")
        print(f"  method_ids: {result['method_ids']}")
        status = result.get("extraction_status", {})
        print(f"  extraction_status: {status.get('status')}")
        visual_index_payload = result.get("visual_index")
        if visual_index_payload:
            print(f"  visual_index: {visual_index_payload.get('status')}")
            if visual_index_payload.get("error_text"):
                print(f"  visual_error: {visual_index_payload.get('error_text')}")
        code_repositories = result.get("code_repositories") or []
        if code_repositories:
            print(f"  code_repositories: {len(code_repositories)} linked")
            for repo in code_repositories:
                url = repo.get("repository_url") or repo.get("url") or repo.get("html_url") or repo.get("repo_url")
                if url:
                    print(f"    selected: {url}")
        enrichment = result.get("code_repository_enrichment") or {}
        if enrichment.get("error_text"):
            print(f"  code_repository_error: {enrichment.get('error_text')}")


def main() -> None:
    """CLI entry point for ResearchFS ingestion."""
    parser = argparse.ArgumentParser(
        prog="researchfs.ingest",
        description="Ingest papers and methods into ResearchFS.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--arxiv", metavar="ID", help="Ingest an arXiv paper by ID")
    input_group.add_argument(
        "--arxiv-list",
        metavar="IDS",
        help="Ingest a comma-separated list of arXiv paper IDs",
    )
    input_group.add_argument(
        "--arxiv-query", metavar="QUERY", help="Ingest papers matching an arXiv search query"
    )
    input_group.add_argument("--repo", metavar="PATH", help="Ingest a GitHub README as a paper")
    input_group.add_argument(
        "--file", metavar="PATH", help="Ingest a local text/markdown/TeX/PDF paper note"
    )
    parser.add_argument(
        "--source-type",
        metavar="LABEL",
        type=_source_type_label,
        help="Set an explicit source_type metadata label for the ingested artifact",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract methods with a local LLM (only if one can be configured)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the records that would be created without writing to the DB",
    )
    parser.add_argument(
        "--enrich-code",
        action="store_true",
        help="Code repository enrichment is automatic; accepted for compatibility",
    )
    parser.add_argument(
        "--no-enrich-code",
        action="store_true",
        help="Disable automatic code repository enrichment and GitHub repository search",
    )
    parser.add_argument(
        "--visual-index",
        action="store_true",
        help="Build a PixelRAG visual index for the ingested paper",
    )
    parser.add_argument(
        "--visual-index-root",
        metavar="PATH",
        help="Directory for generated visual index artifacts",
    )
    parser.add_argument(
        "--max-results",
        metavar="N",
        type=int,
        default=10,
        help="Maximum arXiv results to ingest with --arxiv-query (default: 10)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the database with the v0 corpus",
    )

    args = parser.parse_args()

    if args.seed:
        seed_initial_data()
        print("Seeded researchfs.db")
        return

    if not args.arxiv and not args.arxiv_list and not args.arxiv_query and not args.repo and not args.file:
        parser.print_help()
        return

    llm_fn: Optional[Callable[[str], str]] = None
    if args.extract:
        llm_fn = _build_llm_fn()
        if llm_fn is None:
            print(
                "Warning: --extract was requested but no local LLM could be built. "
                "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL / OPENAI_MODEL) "
                "or implement _build_llm_fn(). Ingesting without extraction."
            )

    effective_enrich_code = not args.no_enrich_code
    code_repo_search_fn = _default_github_code_searcher if effective_enrich_code else None

    if args.arxiv:
        meta = fetch_arxiv_source(args.arxiv)
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)
    elif args.arxiv_list:
        arxiv_ids = [id_.strip() for id_ in args.arxiv_list.split(",")]
        arxiv_ids = [id_ for id_ in arxiv_ids if id_]
        results = ingest_arxiv_ids(
            arxiv_ids,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            source_type=args.source_type,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_arxiv_batch_summary(results, dry_run=args.dry_run)
    elif args.arxiv_query:
        results = ingest_arxiv_query(
            args.arxiv_query,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            source_type=args.source_type,
            max_results=args.max_results,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_arxiv_batch_summary(results, dry_run=args.dry_run)
    elif args.repo:
        meta = fetch_github_readme(args.repo)
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)
    else:
        try:
            meta = load_local_file_metadata(args.file)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        _apply_source_type_override(meta, args.source_type)
        result = ingest_paper(
            meta,
            llm_fn=llm_fn,
            dry_run=args.dry_run,
            visual_index=args.visual_index,
            visual_index_root=args.visual_index_root,
            enrich_code=effective_enrich_code,
            code_repo_search_fn=code_repo_search_fn,
        )
        _print_single_ingestion_result(result, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
