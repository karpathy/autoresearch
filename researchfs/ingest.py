"""Seed the ResearchFS database and ingest papers/methods into ResearchFS."""

import argparse
import os
from pathlib import Path
from typing import Callable, Optional

from researchfs import extractor
from researchfs.db import get_connection, initialize_db
from researchfs.fetcher import fetch_arxiv_source, fetch_github_readme


def _load_markdown(filepath: Path) -> str:
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def seed_initial_data(conn=None) -> None:
    """Populate the database with the v0 seed corpus."""
    close = False
    if conn is None:
        initialize_db()
        conn = get_connection()
        close = True

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

    chunk_ids = []
    for title, section, content in chunks:
        paper_id = paper_ids[title]
        cur.execute(
            "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
            (paper_id, section, content),
        )
        chunk_ids.append(cur.lastrowid)

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
    for method_name, chunk_id in method_chunk_map:
        cur.execute(
            "INSERT OR IGNORE INTO method_sources (method_id, chunk_id) VALUES (?, ?)",
            (method_ids[method_name], chunk_id),
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
        # 1. Exact case-insensitive name match
        cur.execute(
            "SELECT method_id FROM methods WHERE LOWER(name) = LOWER(?)",
            (name,),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        # 2. FTS5 search using keywords from the method's name
        import re
        keywords = [k.strip() for k in re.split(r"[^a-zA-Z0-9]+", name) if k.strip()]
        if not keywords:
            return None

        fts_query = " OR ".join(keywords)
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


def ingest_paper(
    meta: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
    conn=None,
    dry_run: bool = False,
) -> dict:
    """Insert (or plan) a paper, its chunks, and extracted methods into ResearchFS.

    Parameters
    ----------
    meta:
        Paper metadata dict. Must contain ``title``; ``content`` or
        ``abstract`` is used for chunking.
    llm_fn:
        Optional callable ``llm_fn(prompt) -> response_text`` used to extract
        methods.  If ``None``, only the paper and chunks are ingested.
    conn:
        Optional SQLite connection.  A new connection is opened if none is
        provided (and ``dry_run`` is ``False``).
    dry_run:
        If ``True``, no database writes are performed.  A planned summary is
        returned instead.

    Returns
    -------
    dict
        ``{paper_id, chunk_ids, method_ids}`` on a real run, or
        ``{planned: {...}}`` on a dry run.
    """
    if dry_run:
        close_conn = False
        if conn is None:
            initialize_db()
            conn = get_connection()
            close_conn = True
        try:
            content = meta.get("content", meta.get("abstract", ""))
            chunks = extractor.chunk_text(content)
            planned = {
                "paper": {
                    "title": meta.get("title"),
                    "authors": meta.get("authors"),
                    "abstract": meta.get("abstract"),
                    "url": meta.get("url"),
                    "publish_date": meta.get("publish_date"),
                },
                "chunks": [
                    {"section": "extracted", "content": chunk} for chunk in chunks
                ],
                "methods": [],
                "merged_methods": [],
                "method_sources": [],
            }
            if llm_fn is not None:
                try:
                    result = extractor.extract_methods_from_paper(meta, llm_fn)
                    for method in result.get("methods", []):
                        resolved_id = _check_duplicate_method(method, llm_fn=llm_fn, conn=conn)
                        if resolved_id is not None:
                            planned["merged_methods"].append(
                                {
                                    "name": method.get("name"),
                                    "merged_into_id": resolved_id,
                                }
                            )
                        else:
                            planned["methods"].append(
                                {
                                    "name": method.get("name"),
                                    "description": method.get("description", ""),
                                    "implementation_hint": method.get("implementation_hint", ""),
                                    "risks": method.get("risks", ""),
                                    "category": method.get("category", ""),
                                }
                            )
                        for idx in method.get("source_chunk_indices") or []:
                            if isinstance(idx, int) and 0 <= idx < len(chunks):
                                planned["method_sources"].append(
                                    {
                                        "method_name": method.get("name"),
                                        "chunk_index": idx,
                                    }
                                )
                except Exception as exc:  # pragma: no cover
                    planned["extraction_error"] = str(exc)
            return {"planned": planned}
        finally:
            if close_conn:
                conn.close()

    close = False
    if conn is None:
        initialize_db()
        conn = get_connection()
        close = True

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT paper_id FROM papers WHERE title = ?",
            (meta["title"],),
        )
        existing = cur.fetchone()
        if existing:
            return {"paper_id": existing["paper_id"], "chunk_ids": [], "method_ids": []}

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

        content = meta.get("content", meta.get("abstract", ""))
        chunks = extractor.chunk_text(content)
        chunk_ids = []
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (paper_id, section, content) VALUES (?, ?, ?)",
                (paper_id, "extracted", chunk),
            )
            chunk_ids.append(cur.lastrowid)

        method_ids = []
        if llm_fn is not None:
            result = extractor.extract_methods_from_paper(meta, llm_fn)
            for method in result.get("methods", []):
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
                            "name": method.get("name"),
                            "description": method.get("description", ""),
                            "implementation_hint": method.get("implementation_hint", ""),
                            "risks": method.get("risks", ""),
                            "category": method.get("category", ""),
                        },
                    )
                    cur.execute(
                        "SELECT method_id FROM methods WHERE name = ?",
                        (method["name"],),
                    )
                    method_id = cur.fetchone()["method_id"]
                
                method_ids.append(method_id)

                for idx in method.get("source_chunk_indices") or []:
                    if isinstance(idx, int) and 0 <= idx < len(chunk_ids):
                        cur.execute(
                            "INSERT OR IGNORE INTO method_sources (method_id, chunk_id) VALUES (?, ?)",
                            (method_id, chunk_ids[idx]),
                        )
        conn.commit()
        return {"paper_id": paper_id, "chunk_ids": chunk_ids, "method_ids": method_ids}
    finally:
        if close:
            conn.close()


def main() -> None:
    """CLI entry point for ResearchFS ingestion."""
    parser = argparse.ArgumentParser(
        prog="researchfs.ingest",
        description="Ingest papers and methods into ResearchFS.",
    )
    parser.add_argument("--arxiv", metavar="ID", help="Ingest an arXiv paper by ID")
    parser.add_argument("--repo", metavar="PATH", help="Ingest a GitHub README as a paper")
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
        "--seed",
        action="store_true",
        help="Seed the database with the v0 corpus",
    )

    args = parser.parse_args()

    if args.seed:
        seed_initial_data()
        print("Seeded researchfs.db")
        return

    if not args.arxiv and not args.repo:
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

    if args.arxiv:
        meta = fetch_arxiv_source(args.arxiv)
    else:
        meta = fetch_github_readme(args.repo)

    result = ingest_paper(meta, llm_fn=llm_fn, dry_run=args.dry_run)

    if args.dry_run:
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


if __name__ == "__main__":
    main()
