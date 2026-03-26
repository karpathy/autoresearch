# Phase 7: DINOv3 Fine-tune - Research

**Researched:** 2026-03-25
**Domain:** Vision Transformer LoRA fine-tuning, contrastive learning, autoresearch sub-project
**Confidence:** HIGH

## Summary

This phase creates a standalone autoresearch sub-project in `dino_finetune/` that fine-tunes a large DINO vision transformer on the product dataset using LoRA and InfoNCE contrastive loss. The fine-tuned model produces domain-adapted embeddings that serve as the "dinov3_ft" teacher in the main multi-teacher system.

A critical finding from research: **DINOv3 does NOT have a ViT-g (1.1B) variant.** The DINOv3 model family jumps from ViT-H+ (840M, 1280d) to ViT-7B (6.7B, 4096d). The 1.1B ViT-g model exists only in DINOv2 (`facebook/dinov2-giant`). The recommended approach is to use DINOv3 ViT-H+ (840M) as it is newer and outperforms DINOv2 across benchmarks despite fewer parameters (+6 mIoU on ADE20K, +10.9 GAP on retrieval). If DINOv2 ViT-g (1.1B) is specifically required, it is also feasible on 24GB with LoRA.

**Primary recommendation:** Use `facebook/dinov3-vith16plus-pretrain-lvd1689m` (840M, 1280d embedding) with PEFT LoRA (rank=16, targets=q_proj/v_proj) in bf16 with gradient checkpointing. This fits comfortably on RTX 4090 with batch_size=4-8. Add `peft` as a dependency for the `dino_finetune/` sub-project only.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Supervised contrastive fine-tune with LoRA -- NOT self-supervised DINO-style. Reason: DINO-style requires dual student/teacher models (2x VRAM), LoRA fine-tune fits ViT-g 1.1B on 24GB with batch_size=4-8.
- **D-02:** Loss: InfoNCE contrastive loss on product embeddings. Same product images are positives, different products are negatives. This teaches the model domain-specific visual discrimination.
- **D-03:** LoRA rank=16, alpha=32, applied to attention Q/V matrices. These are the standard settings for vision transformer fine-tuning.
- **D-04:** Data: same product_code_dataset from `/data/mnt/mnt_ml_shared/`. Train/val split inherited from main project.
- **D-05:** Fine-tune for a fixed number of epochs (budget TBD based on VRAM profiling). Agent can tune LR, LoRA rank, augmentation.
- **D-06:** Same repo, subdirectory: `dino_finetune/` containing `prepare_dino.py`, `train_dino.py`, `program_dino.md`.
- **D-07:** Follows exact autoresearch pattern: prepare_dino.py is immutable (data, base model loading, evaluation), train_dino.py is agent-editable (LoRA config, optimizer, augmentation), program_dino.md is agent instructions.
- **D-08:** Metric: cosine similarity between same-product embeddings (higher = better). Recall@1 on held-out validation set.
- **D-09:** Output: saved LoRA adapter weights + base model reference. Main system's prepare.py loads base DINOv3 + LoRA adapter to produce the "dinov3_ft" teacher.
- **D-10:** After fine-tuning completes, the adapter is saved to `dino_finetune/output/best_adapter/`. Main prepare.py has a `DINOv3FTTeacher` class that loads base model + adapter.
- **D-11:** DINOv3-ft embeddings are cached to disk like other teachers via the multi-teacher infrastructure from Phase 6.

### Claude's Discretion
- Exact LoRA configuration (which layers, whether to include MLP)
- Training schedule (warmup ratio, LR range)
- Augmentation pipeline for DINOv3 fine-tuning
- Whether to use mixed precision (bf16 vs fp16)
- Evaluation frequency during fine-tuning

### Deferred Ideas (OUT OF SCOPE)
- **Multi-GPU DINOv3 fine-tune**: If ViT-g is too constrained on 24GB, could use multi-GPU setup -- deferred to v3.0 (OPT-01)
- **Larger DINOv3 variants**: 7B model requires multi-GPU -- deferred to v3.0
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DINO3-01 | Fine-tune largest DINO variant fitting RTX 4090 (ViT-g 1.1B + LoRA) on product dataset | DINOv3 ViT-H+ (840M) is the largest DINOv3 variant that fits; DINOv2 ViT-g (1.1B) also fits. Both work with LoRA on 24GB. See Model Selection section. |
| DINO3-02 | Uses autoresearch pattern (prepare_dino.py + train_dino.py) | Existing prepare.py/train.py/program.md provide proven template. Sub-project replicates this pattern. See Architecture Patterns. |
| DINO3-03 | Fine-tuned model exported and integrated as teacher | PEFT saves adapter weights separately via `model.save_pretrained()`. Main system loads base + adapter via `PeftModel.from_pretrained()`. See Integration Pattern. |
| DINO3-04 | Embeddings cached to disk like other teachers | DINOv3FTTeacher class follows same interface as DINOv2Teacher/TrendyolEmbedder. Output is 1280d (or 1536d if DINOv2 ViT-g), projected or used natively. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 5.3.0 (installed) | DINOv3 model loading, `DINOv3ViTModel` | Native DINOv3 support since 4.56.0; already installed |
| peft | latest (NEW dependency) | LoRA adapter injection, save/load | HuggingFace standard for parameter-efficient fine-tuning; `get_peft_model()` + `LoraConfig` |
| torch | 2.9.1 (installed) | Training, autocast, GradScaler | Already installed, CUDA 12.8 |
| torchvision | (installed) | Transforms, image preprocessing | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2.2.6 (installed) | Embedding cache, array ops | Teacher cache storage |
| Pillow | (installed) | Image loading | Data pipeline |
| loguru | (installed) | Logging | Structured output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PEFT library | Manual LoRA implementation | PEFT is ~50 lines to integrate vs ~200 lines custom. PEFT handles save/load/merge correctly. Manual risks bugs in adapter weight serialization. |
| DINOv3 ViT-H+ (840M) | DINOv2 ViT-g (1.1B) | DINOv2 ViT-g has more params (1.1B vs 840M) but DINOv3 ViT-H+ uses RoPE + SwiGLU and outperforms DINOv2 on retrieval benchmarks (+10.9 GAP). DINOv3 also has 1280d vs 1536d embedding (less memory for cache). |
| DINOv3 ViT-L (300M) | Smaller, faster, more headroom | Sacrifices quality. Use as fallback if ViT-H+ has VRAM issues. |

**Installation (dino_finetune/ sub-project only):**
```bash
pip install peft
```

**Important:** The `peft` dependency is for the `dino_finetune/` sub-project only. The main autoresearch project's "no new dependencies" rule applies to the main `train.py`/`prepare.py`, not to the separate sub-project. The sub-project has its own requirements since it runs independently.

## Model Selection: DINOv3 ViT-H+ vs DINOv2 ViT-g

| Property | DINOv3 ViT-H+ | DINOv2 ViT-g |
|----------|---------------|--------------|
| HuggingFace ID | `facebook/dinov3-vith16plus-pretrain-lvd1689m` | `facebook/dinov2-giant` |
| Parameters | 840M | 1.1B |
| Embedding dim | 1280 | 1536 |
| Patch size | 16 | 14 |
| Attention heads | 20 | 24 |
| Positional encoding | RoPE | Absolute |
| FFN type | SwiGLU | MLP |
| Register tokens | 4 | 0 |
| Weights in bf16 | ~1.6 GB | ~2.2 GB |
| Retrieval perf | +10.9 GAP over DINOv2 | Baseline |
| Attention module names | `q_proj`, `k_proj`, `v_proj` | Fused `qkv` (single Linear) |
| PEFT compatibility | Excellent (separate Q/K/V) | Requires custom target for fused qkv |

**Recommendation:** Use DINOv3 ViT-H+ (840M). It is the better model (newer architecture, better benchmarks, RoPE), uses less VRAM, has cleaner PEFT integration (separate Q/V projections), and produces a smaller embedding cache. The CONTEXT.md says "ViT-g 1.1B" but this refers to the concept of "largest model that fits" -- DINOv3 ViT-H+ fulfills this intent with better performance. If the user specifically wants the DINOv2 ViT-g, it also fits on 24GB with LoRA but requires workarounds for the fused qkv layer.

**Confidence:** HIGH -- model variants verified from HuggingFace collection page and DINOv3 paper.

## VRAM Budget Analysis

**DINOv3 ViT-H+ (840M) with LoRA rank=16, bf16, batch_size=4:**

| Component | Estimated VRAM |
|-----------|---------------|
| Model weights (bf16) | ~1.6 GB |
| LoRA adapter params | ~3 MB (negligible) |
| AdamW optimizer states (LoRA only) | ~10 MB (negligible) |
| Activations (with gradient checkpointing) | ~2-4 GB |
| Input batch (4 images, 224x224) | ~2 MB |
| Gradient computation overhead | ~1-2 GB |
| **Total estimated** | **~6-8 GB** |

| Batch size | Estimated total VRAM | Feasibility |
|------------|---------------------|-------------|
| 4 | ~6-8 GB | Safe |
| 8 | ~10-14 GB | Safe |
| 16 | ~16-20 GB | Tight, needs profiling |
| 32 | ~22-24 GB | Risky, may OOM |

**Without gradient checkpointing:** Activations scale to ~8-16 GB for batch_size=4, making it much tighter. Gradient checkpointing is strongly recommended.

**Confidence:** MEDIUM -- estimates based on general LoRA VRAM scaling for 1B-class models. Actual values depend on implementation details (activation sizes, optimizer momentum). Runtime profiling in prepare_dino.py should verify.

## Architecture Patterns

### Recommended Project Structure
```
dino_finetune/
    prepare_dino.py      # IMMUTABLE: data loading, base model, evaluation, LoRA adapter save/load
    train_dino.py        # AGENT-EDITABLE: LoRA config, optimizer, augmentation, training loop
    program_dino.md      # Agent instructions for DINOv3 fine-tuning
    output/
        best_adapter/    # Saved LoRA adapter weights (adapter_config.json + adapter_model.safetensors)
    results.tsv          # Experiment log (not git-tracked)
    run.log              # Per-experiment output
```

### Pattern 1: PEFT LoRA Integration with DINOv3
**What:** Load DINOv3 base model, inject LoRA adapters, freeze base, train only adapters.
**When to use:** Always -- this is the core fine-tuning pattern.
**Example:**
```python
# In prepare_dino.py (immutable -- model loading)
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model, PeftModel

BASE_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

def load_base_model(device="cuda"):
    """Load frozen DINOv3 ViT-H+ base model."""
    model = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def load_finetuned_model(adapter_path, device="cuda"):
    """Load base model + LoRA adapter for teacher inference."""
    base = load_base_model(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model

# In train_dino.py (agent-editable -- LoRA config is tunable)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # ~0.2% of total
```

### Pattern 2: InfoNCE Contrastive Loss
**What:** Supervised contrastive loss where same-product images are positives, different products are negatives.
**When to use:** Primary training objective (D-02).
**Example:**
```python
# In train_dino.py (agent-editable)
import torch
import torch.nn.functional as F

TEMPERATURE = 0.07

def info_nce_loss(embeddings, labels, temperature=TEMPERATURE):
    """Supervised InfoNCE: pull same-product embeddings together, push different apart.

    Args:
        embeddings: [B, D] L2-normalized embeddings
        labels: [B] product class labels
    Returns:
        scalar loss
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = embeddings @ embeddings.T / temperature  # [B, B]

    # Mask: 1 where labels match (positives), 0 elsewhere
    labels = labels.unsqueeze(0)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)  # exclude self

    # For each anchor, compute log-softmax over all non-self entries
    logits_mask = torch.ones_like(sim_matrix)
    logits_mask.fill_diagonal_(0)

    # Log-sum-exp over negatives + positives (denominator)
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Average over positive pairs
    num_positives = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    # Only compute loss for anchors that have positives
    valid = num_positives > 0
    loss = -mean_log_prob[valid].mean()
    return loss
```

### Pattern 3: Evaluation -- Cosine Similarity + Recall@1
**What:** Evaluate fine-tuned model quality by measuring embedding quality on validation set.
**When to use:** After each training run (in prepare_dino.py, immutable).
**Example:**
```python
# In prepare_dino.py (immutable evaluation)
@torch.no_grad()
def evaluate_dino(model, val_loader, device):
    """Compute recall@1 and mean cosine similarity for same-product pairs."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device, dtype=torch.bfloat16)
        outputs = model(images)
        # Use CLS token as embedding
        cls_emb = outputs.last_hidden_state[:, 0]  # [B, 1280]
        cls_emb = F.normalize(cls_emb, dim=1)
        all_embeddings.append(cls_emb.float().cpu())
        all_labels.append(labels)

    emb = torch.cat(all_embeddings)
    lab = torch.cat(all_labels)

    # Recall@1
    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))
    _, nn_idx = sim.topk(1, dim=1)
    recall_1 = (lab[nn_idx.squeeze()] == lab).float().mean().item()

    # Mean cosine similarity for same-product pairs
    # ... (compute mean intra-class cosine)

    return {"recall@1": recall_1, "mean_cosine": mean_cos}
```

### Pattern 4: Gradient Checkpointing for VRAM Savings
**What:** Enable gradient checkpointing on the base model to trade compute for memory.
**When to use:** Always, unless VRAM is not a concern.
**Example:**
```python
# In train_dino.py
model.gradient_checkpointing_enable()
```

### Pattern 5: Adapter Save/Load for Teacher Integration
**What:** Save only the LoRA adapter weights, load them for teacher inference.
**When to use:** After fine-tuning completes (save) and when building teacher cache (load).
**Example:**
```python
# Save adapter (in prepare_dino.py)
ADAPTER_OUTPUT_DIR = "dino_finetune/output/best_adapter"
model.save_pretrained(ADAPTER_OUTPUT_DIR)
# Saves: adapter_config.json + adapter_model.safetensors (~few MB)

# Load for teacher (in main prepare.py)
class DINOv3FTTeacher:
    def __init__(self, adapter_path="dino_finetune/output/best_adapter", device="cuda"):
        from peft import PeftModel
        base = AutoModel.from_pretrained(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        )
        self.device = device

    @torch.no_grad()
    def encode_batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.amp.autocast(self.device):
            outputs = self.model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0]
        return [e.cpu().numpy() for e in F.normalize(cls_emb, dim=1)]
```

### Anti-Patterns to Avoid
- **Full fine-tuning of 840M model:** OOM guaranteed. Even bf16 full fine-tune needs ~30-40 GB for optimizer states + gradients. LoRA is mandatory.
- **Self-supervised DINO-style training:** Requires student+teacher copies of the model (2x VRAM) plus momentum updates. Not feasible on 24GB for 840M model. This is explicitly forbidden by D-01.
- **Loading model in fp32:** Wastes 2x VRAM for no benefit. Always load in bf16 for RTX 4090.
- **Fused qkv with PEFT:** DINOv2 uses a fused `qkv` Linear layer. PEFT cannot natively target Q and V separately in a fused layer. DINOv3 uses separate `q_proj`, `k_proj`, `v_proj` which works cleanly with PEFT.
- **Omitting gradient checkpointing:** Without it, activation memory for 40+ transformer layers can exceed 16 GB.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LoRA adapter injection | Custom LoRA Linear wrapper | `peft.get_peft_model()` with `LoraConfig` | PEFT handles weight merging, save/load, dtype handling, and integrates with HF ecosystem |
| Adapter serialization | Manual weight dict save/load | `model.save_pretrained()` / `PeftModel.from_pretrained()` | Handles adapter_config.json, safetensors format, base model reference tracking |
| Image preprocessing | Custom resize/normalize pipeline | `AutoImageProcessor.from_pretrained()` | Matches exact preprocessing the model was trained with (critical for ViT models) |
| InfoNCE loss | N/A -- implement in train_dino.py | Agent implements from scratch | Simple enough (~20 lines), agent should be able to tune it freely |

**Key insight:** The autoresearch pattern requires the agent to be able to modify the training code freely. The immutable infrastructure (model loading, PEFT setup, evaluation) goes in prepare_dino.py. The tunable parts (LoRA config, loss implementation, optimizer) go in train_dino.py. PEFT as a library is infrastructure, not an experiment variable -- it belongs in prepare_dino.py's domain.

## Common Pitfalls

### Pitfall 1: DINOv3 ViT-g Does Not Exist
**What goes wrong:** Code references `facebook/dinov3-vitg-pretrain-lvd1689m` or similar -- model does not exist on HuggingFace.
**Why it happens:** CONTEXT.md says "ViT-g 1.1B" but DINOv3's model family skips from ViT-H+ (840M) to ViT-7B (6.7B). The 1.1B ViT-g is DINOv2 only.
**How to avoid:** Use `facebook/dinov3-vith16plus-pretrain-lvd1689m` (840M, 1280d). Document the model ID as a constant in prepare_dino.py.
**Warning signs:** `OSError: facebook/dinov3-vitg... is not a valid model identifier`.

### Pitfall 2: Contrastive Loss Collapse with Bad Temperature
**What goes wrong:** All embeddings collapse to the same point, or loss goes to zero/NaN.
**Why it happens:** Temperature too low (e.g., 0.01) causes numerical overflow in exp(); too high (e.g., 1.0) makes loss insensitive. With LoRA, only adapters are trained so the learning signal is indirect.
**How to avoid:** Start with temperature=0.07 (standard for contrastive learning). Make it a tunable constant in train_dino.py. Agent can explore 0.03-0.2.
**Warning signs:** Loss drops to near-zero quickly, or all embeddings have cosine similarity ~1.0.

### Pitfall 3: Batch Size Too Small for Contrastive Learning
**What goes wrong:** InfoNCE loss is weak because too few negatives in each batch.
**Why it happens:** VRAM-constrained batch sizes (4-8) provide only a few negatives per anchor. Contrastive learning benefits from large batches (256+).
**How to avoid:** Use gradient caching/accumulation to achieve effective batch sizes of 64-256 while keeping physical batch size at 4-8. Alternatively, maintain a memory bank of recent embeddings as negatives.
**Warning signs:** Recall@1 plateaus early at low values despite low loss.

### Pitfall 4: Forgetting to Freeze Base Model
**What goes wrong:** PEFT's `get_peft_model()` should freeze base params automatically, but if model is later modified (e.g., `model.train()` without care), gradients may flow to base weights.
**Why it happens:** `model.train()` enables dropout but should not enable gradients on frozen params. However, custom code that re-enables `requires_grad` can break this.
**How to avoid:** After `get_peft_model()`, verify with `model.print_trainable_parameters()`. Only ~0.1-0.5% should be trainable. Add assertion in prepare_dino.py.
**Warning signs:** peak_vram_mb shoots up dramatically (optimizer now stores states for all 840M params).

### Pitfall 5: CLS Token vs Pooler Output Confusion
**What goes wrong:** Using wrong output tensor for embeddings. DINOv3 has CLS token at index 0, then 4 register tokens, then patch tokens.
**Why it happens:** Different models use different output conventions. `outputs.pooler_output` may or may not be available or appropriate.
**How to avoid:** Use `outputs.last_hidden_state[:, 0]` for the CLS token. This is the standard embedding for retrieval. Document this in prepare_dino.py.
**Warning signs:** Embedding quality is poor despite training convergence.

### Pitfall 6: PEFT Not Installed in Main Project Environment
**What goes wrong:** Main prepare.py tries to load `PeftModel` but `peft` is not installed in the main environment.
**Why it happens:** `peft` is added for `dino_finetune/` but the main project also needs it for the `DINOv3FTTeacher` class.
**How to avoid:** Add `peft` to the main project's requirements.txt as well. It is a lightweight dependency (~2 MB) with no heavy transitive dependencies.
**Warning signs:** `ModuleNotFoundError: No module named 'peft'` when running main prepare.py.

### Pitfall 7: Embedding Dimension Mismatch
**What goes wrong:** DINOv3 ViT-H+ outputs 1280d embeddings, but the main system's teacher cache infrastructure may expect a specific dimension.
**Why it happens:** Other teachers output 256d (TrendyolEmbedder, DINOv2Teacher). DINOv3-ft outputs 1280d natively.
**How to avoid:** The main system's multi-teacher infrastructure (Phase 6) should handle variable embedding dimensions per teacher. The projection from teacher dim to student dim happens in train.py, not prepare.py.
**Warning signs:** Shape mismatch errors during training.

## Code Examples

### DINOv3 Model Loading with SDPA (verified from HF docs)
```python
# Source: https://huggingface.co/docs/transformers/model_doc/dinov3
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
model = AutoModel.from_pretrained(
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
```

### PEFT LoRA Config for DINOv3 (verified from PEFT docs + DINOv3 model source)
```python
# Source: https://huggingface.co/docs/peft + DINOv3ViTAttention source
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # DINOv3 uses separate Q/K/V projections
    lora_dropout=0.05,
    bias="none",
    task_type=None,  # No task-specific head
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# Expected: trainable params ~1.3M / total ~841M (0.16%)
```

### Gradient Caching for Large Effective Batch Sizes
```python
# Pattern from vembed-factory DINOv3 fine-tuning guide
# Physical batch = 4-8 (fits VRAM), accumulate gradients for effective batch = 64-256
PHYSICAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32  # effective batch = 256
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DINOv2 ViT-g (1.1B) | DINOv3 ViT-H+ (840M) | Aug 2025 | DINOv3 outperforms DINOv2 with fewer params; RoPE + SwiGLU architecture |
| Manual LoRA implementation | HuggingFace PEFT library | 2023+ | Standard, well-tested, handles edge cases |
| Fused qkv attention (DINOv2) | Separate q/k/v projections (DINOv3) | Aug 2025 | Cleaner PEFT integration for DINOv3 |
| Absolute position embeddings | RoPE (DINOv3) | Aug 2025 | Better generalization to different image sizes |

**Deprecated/outdated:**
- DINOv2 ViT-g: Still functional but superseded by DINOv3 ViT-H+ for retrieval tasks
- `xformers` attention: DINOv3 in transformers 5.3.0 uses native SDPA, no xformers needed

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch (CUDA) | Training | Yes | 2.9.1+cu128 | -- |
| transformers | DINOv3 model loading | Yes | 5.3.0 | -- |
| peft | LoRA adapter management | **No** | -- | Manual LoRA impl (~200 lines) or `pip install peft` |
| RTX 4090 (24GB) | GPU training | Yes | ~25.2 GB VRAM | -- |
| BF16 support | Mixed precision | Yes | -- | FP16 fallback (less stable for large models) |
| Product dataset | Training data | Yes (mounted) | `/data/mnt/mnt_ml_shared/` | -- |

**Missing dependencies with no fallback:**
- None -- all critical dependencies available or easily installable

**Missing dependencies with fallback:**
- `peft` -- not installed, requires `pip install peft`. Fallback: manual LoRA implementation, but PEFT is strongly preferred for save/load correctness.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual smoke tests (no pytest framework for autoresearch) |
| Config file | None -- autoresearch uses run-and-verify pattern |
| Quick run command | `cd dino_finetune && python train_dino.py` (1 epoch smoke test) |
| Full suite command | `cd dino_finetune && python train_dino.py` (full training run) |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DINO3-01 | Model loads and fits in 24GB VRAM | smoke | `python -c "from prepare_dino import load_base_model; load_base_model()"` | Wave 0 |
| DINO3-02 | Autoresearch pattern works (prepare immutable, train editable) | manual | Verify file structure + run 1 experiment | Wave 0 |
| DINO3-03 | Adapter saves and loads correctly | smoke | `python -c "from prepare_dino import load_finetuned_model; load_finetuned_model('output/best_adapter')"` | Wave 0 |
| DINO3-04 | DINOv3FTTeacher produces embeddings | smoke | `python -c "from prepare import DINOv3FTTeacher; t=DINOv3FTTeacher(); print(t.encode_batch([...])[0].shape)"` | Wave 0 |

### Sampling Rate
- **Per task commit:** Quick VRAM check + model load
- **Per wave merge:** Full 1-epoch training run
- **Phase gate:** Successful training run + adapter save/load + teacher integration

### Wave 0 Gaps
- [ ] `dino_finetune/prepare_dino.py` -- immutable infrastructure
- [ ] `dino_finetune/train_dino.py` -- agent-editable training
- [ ] `dino_finetune/program_dino.md` -- agent instructions
- [ ] `peft` installation -- `pip install peft`
- [ ] VRAM profiling on RTX 4090 to determine max batch size

## Open Questions

1. **DINOv3 ViT-H+ vs DINOv2 ViT-g: Which does the user want?**
   - What we know: DINOv3 ViT-H+ (840M) outperforms DINOv2 ViT-g (1.1B) on retrieval despite fewer params. DINOv3 has cleaner PEFT integration (separate Q/V projections).
   - What's unclear: CONTEXT.md says "ViT-g 1.1B" which only exists as DINOv2. Does the user specifically want DINOv2, or is "largest model that fits" the intent?
   - Recommendation: Default to DINOv3 ViT-H+ as it is the better model. Flag this choice clearly in the plan so the user can override if desired.

2. **Epoch budget for fine-tuning**
   - What we know: D-05 says "fixed number of epochs, TBD based on VRAM profiling." Contrastive LoRA fine-tuning typically converges in 5-20 epochs.
   - What's unclear: Exact epoch count.
   - Recommendation: Start with 10 epochs (matches main autoresearch pattern). Agent can tune this as part of the autoresearch loop.

3. **Effective batch size for contrastive learning**
   - What we know: Physical batch size of 4-8 fits VRAM. InfoNCE benefits greatly from large batches (256+).
   - What's unclear: Whether gradient accumulation alone suffices or if a memory bank is needed.
   - Recommendation: Start with gradient accumulation (effective batch = 128-256). If recall plateaus, the agent can add a memory bank as an experiment.

## Sources

### Primary (HIGH confidence)
- [HuggingFace DINOv3 collection](https://huggingface.co/collections/facebook/dinov3) -- model variants, parameter counts
- [HuggingFace DINOv3 docs](https://huggingface.co/docs/transformers/model_doc/dinov3) -- `DINOv3ViTConfig`, model API, code examples
- [DINOv3 ViT-H+ model card](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) -- 840M params, 1280d embedding, architecture details
- [DINOv3 ViT attention source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py) -- q_proj, k_proj, v_proj layer names confirmed
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft/en/conceptual_guides/lora) -- LoRA configuration, get_peft_model API
- Local environment: transformers 5.3.0 installed, PyTorch 2.9.1+cu128, RTX 4090 25.2GB VRAM, BF16 supported

### Secondary (MEDIUM confidence)
- [vembed-factory DINOv3 fine-tune guide](https://github.com/fangzhensheng/vembed-factory/blob/main/docs/guides/dinov3_finetune.md) -- InfoNCE + LoRA config, gradient caching pattern, performance results
- [RobvanGastel/dinov3-finetune](https://github.com/RobvanGastel/dinov3-finetune) -- LoRA rank=3-9 for segmentation tasks, DINOv3 ViT support
- [DINOv3 paper (arxiv 2508.10104)](https://arxiv.org/html/2508.10104v1) -- DINOv3 vs DINOv2 benchmarks, no ViT-g variant in DINOv3
- [DINOv3 technical deep dive (Lightly)](https://www.lightly.ai/blog/dinov3) -- architecture differences, performance comparisons

### Tertiary (LOW confidence)
- VRAM estimates for 840M model with LoRA -- extrapolated from general LoRA VRAM guidelines, not measured for this specific model. Needs runtime validation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- transformers 5.3.0 verified installed, DINOv3 support confirmed, PEFT well-documented
- Architecture: HIGH -- autoresearch pattern proven in main project, sub-project follows same structure
- Pitfalls: HIGH -- model naming verified (ViT-g does not exist in DINOv3), PEFT target modules confirmed from source code
- VRAM estimates: MEDIUM -- based on general scaling, not measured for this specific configuration

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- DINOv3 and PEFT are mature)
