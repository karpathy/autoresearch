"""Agent-editable training script for DINOv3 LoRA fine-tuning.

The AI agent modifies this file to experiment with LoRA configuration,
optimizer settings, loss function, and augmentation. All tunable parameters
are module-level constants below.

Usage: cd dino_finetune && python train_dino.py
"""

from prepare_dino import (
    load_base_model, get_image_processor, build_dataset,
    extract_cls_embedding, evaluate_dino, save_adapter,
    EPOCHS, EMBEDDING_DIM, ADAPTER_OUTPUT_DIR,
    TRAIN_DIR, VAL_DIR,
    set_seed, collate_fn,
)

import math
import time
import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

# ============================================================
# EXPERIMENT VARIABLES (agent edits these)
# ============================================================
LORA_R = 16                          # LoRA rank (per D-03)
LORA_ALPHA = 32                      # LoRA alpha (per D-03)
LORA_DROPOUT = 0.05                  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # DINOv3 attention projections (per D-03)

BATCH_SIZE = 8                       # Physical batch size (VRAM safe for 24GB)
GRADIENT_ACCUMULATION_STEPS = 16     # Effective batch = 128
LR = 1e-3                           # Learning rate for AdamW
WEIGHT_DECAY = 0.01                  # AdamW weight decay
WARMUP_RATIO = 0.2                   # Fraction of total steps for LR warmup
TEMPERATURE = 0.15                   # InfoNCE temperature

# ArcFace metric learning
ARCFACE_WEIGHT = 0.0                 # Weight of ArcFace loss (0.0 = disabled)
ARCFACE_SCALE = 30.0                 # ArcFace scale parameter
ARCFACE_MARGIN = 0.3                 # ArcFace angular margin (radians)

SEED = 42
USE_GRADIENT_CHECKPOINTING = True
EVAL_EVERY_N_EPOCHS = 1              # Evaluate after every N epochs
MAX_STEPS_PER_EPOCH = 0              # Cap steps per epoch (0 = no cap)
MAX_TRAINING_SECONDS = 1800          # Hard time limit: 30 minutes per experiment
NUM_WORKERS = 4                      # DataLoader workers
DEVICE = "cuda"


# ============================================================
# Loss function
# ============================================================

def info_nce_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                  temperature: float = TEMPERATURE) -> torch.Tensor:
    """Supervised InfoNCE contrastive loss (per D-02).

    Same-product images are positives, different products are negatives.
    Teaches domain-specific visual discrimination.

    Args:
        embeddings: [B, D] L2-normalized embeddings.
        labels: [B] integer product class labels.
        temperature: Scaling factor for similarity logits.

    Returns:
        Scalar loss. Returns 0 if no valid positive pairs in batch.
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = embeddings @ embeddings.T / temperature  # [B, B]

    # Positive mask: 1 where labels match (excluding self)
    labels_col = labels.unsqueeze(0)  # [1, B]
    labels_row = labels.unsqueeze(1)  # [B, 1]
    pos_mask = (labels_row == labels_col).float()
    pos_mask.fill_diagonal_(0)  # exclude self-pairs

    # Logits mask: all pairs except self
    logits_mask = torch.ones_like(sim_matrix)
    logits_mask.fill_diagonal_(0)

    # Log-softmax denominator: sum of exp over all non-self entries
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Average log-prob over positive pairs for each anchor
    num_positives = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    # Only compute loss for anchors that have at least one positive
    valid = num_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    loss = -mean_log_prob[valid].mean()
    return loss


class ArcFaceHead(torch.nn.Module):
    """ArcFace angular margin classification head for metric learning.

    Forces embeddings to have clear angular separation between classes.
    Complements InfoNCE which only learns relative distances.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = torch.nn.Parameter(torch.randn(num_classes, embedding_dim))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        target_logits = torch.cos(theta + self.margin * one_hot)
        logits = self.scale * target_logits
        return F.cross_entropy(logits, labels)


# ============================================================
# Optimizer and scheduler
# ============================================================

def build_optimizer(model, arcface_head=None) -> torch.optim.Optimizer:
    """Create AdamW optimizer for trainable (LoRA + ArcFace) parameters."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if arcface_head is not None:
        trainable_params += list(arcface_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


def build_scheduler(optimizer, num_training_steps: int):
    """Cosine LR schedule with linear warmup.

    Uses torch.optim.lr_scheduler (no transformers dependency).
    """
    warmup_steps = int(num_training_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / max(1.0, float(warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / max(
            1.0, float(num_training_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, train_loader: DataLoader, optimizer, scheduler,
                    device: str, epoch: int,
                    arcface_head: "ArcFaceHead | None" = None) -> float:
    """Train one epoch with gradient accumulation and bf16 autocast.

    Args:
        model: PEFT model with LoRA adapters.
        train_loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler (stepped per optimizer step).
        device: CUDA device string.
        epoch: Current epoch number (for logging).
        arcface_head: Optional ArcFace classification head.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    effective_steps = len(train_loader)
    if MAX_STEPS_PER_EPOCH > 0:
        effective_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)

    for step, (images, labels) in enumerate(train_loader):
        if MAX_STEPS_PER_EPOCH > 0 and step >= MAX_STEPS_PER_EPOCH:
            break

        images = images.to(device, dtype=torch.bfloat16)
        labels = labels.to(device)

        # Forward pass with bf16 autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            cls_emb = extract_cls_embedding(outputs)
            loss_nce = info_nce_loss(cls_emb, labels)
            loss = loss_nce
            if arcface_head is not None and ARCFACE_WEIGHT > 0:
                loss_arc = arcface_head(cls_emb.float(), labels)
                loss = loss_nce + ARCFACE_WEIGHT * loss_arc
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale for accumulation

        # Backward (no GradScaler needed for bfloat16)
        loss.backward()

        # Optimizer step after accumulation
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        num_batches += 1

        # Log every 50 steps
        if (step + 1) % 50 == 0:
            avg = total_loss / num_batches
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} step {step + 1}/{effective_steps}: "
                f"loss={avg:.4f}  lr={lr_now:.2e}"
            )

    # Handle remaining accumulated gradients
    if num_batches % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)

    # Log peak VRAM at end of first epoch for profiling
    if epoch == 1:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Peak VRAM after epoch 1: {peak_vram_mb:.0f} MB")

    return avg_loss


# ============================================================
# Main
# ============================================================

def main():
    """Full DINOv3 LoRA fine-tuning pipeline."""
    set_seed(SEED)
    logger.info("=" * 60)
    logger.info("DINOv3 ViT-H+ LoRA Fine-tuning")
    logger.info("=" * 60)
    logger.info(
        f"Config: LoRA r={LORA_R} alpha={LORA_ALPHA} targets={LORA_TARGET_MODULES} "
        f"BS={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS} LR={LR} T={TEMPERATURE} "
        f"ArcFace={ARCFACE_WEIGHT} margin={ARCFACE_MARGIN} scale={ARCFACE_SCALE}"
    )

    # -- Load base model --
    base_model = load_base_model(DEVICE)

    # -- Inject LoRA adapters --
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Expected: ~0.2% of total

    # -- Enable gradient checkpointing for VRAM savings --
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # -- Build datasets and DataLoaders --
    processor = get_image_processor()
    train_dataset, num_train_classes = build_dataset(TRAIN_DIR, processor, split="train")
    val_dataset, num_val_classes = build_dataset(VAL_DIR, processor, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -- ArcFace head (optional) --
    arcface_head = None
    if ARCFACE_WEIGHT > 0:
        arcface_head = ArcFaceHead(EMBEDDING_DIM, num_train_classes,
                                    ARCFACE_SCALE, ARCFACE_MARGIN).to(DEVICE)
        logger.info(f"ArcFace enabled: weight={ARCFACE_WEIGHT} scale={ARCFACE_SCALE} margin={ARCFACE_MARGIN} classes={num_train_classes}")

    # -- Build optimizer and scheduler --
    optimizer = build_optimizer(model, arcface_head=arcface_head)
    steps_per_epoch = min(len(train_loader), MAX_STEPS_PER_EPOCH) if MAX_STEPS_PER_EPOCH > 0 else len(train_loader)
    num_training_steps = (steps_per_epoch // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    scheduler = build_scheduler(optimizer, num_training_steps)

    # -- Training loop --
    best_combined = -1.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        elapsed = time.time() - start_time
        if MAX_TRAINING_SECONDS > 0 and elapsed >= MAX_TRAINING_SECONDS:
            logger.info(f"Time limit reached ({elapsed:.0f}s >= {MAX_TRAINING_SECONDS}s) after {epoch - 1} epochs")
            break

        epoch_start = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch,
            arcface_head=arcface_head,
        )
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f}  time={epoch_time:.1f}s  total={time.time() - start_time:.0f}s")

        # -- Evaluation --
        if epoch % EVAL_EVERY_N_EPOCHS == 0:
            metrics = evaluate_dino(model, val_loader, DEVICE)

            if metrics["combined"] > best_combined:
                best_combined = metrics["combined"]
                save_adapter(model)
                logger.info(
                    f"New best! combined={best_combined:.4f} -- adapter saved"
                )

    # -- Final results --
    total_time = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Evaluate best adapter
    logger.info("Loading best adapter for final evaluation ...")
    from prepare_dino import load_finetuned_model
    best_model = load_finetuned_model(ADAPTER_OUTPUT_DIR, DEVICE)
    final_metrics = evaluate_dino(best_model, val_loader, DEVICE)

    logger.info("=" * 60)
    logger.info(
        f"RESULT: recall@1={final_metrics['recall@1']:.4f} "
        f"mean_cosine={final_metrics['mean_cosine']:.4f} "
        f"combined={final_metrics['combined']:.4f} "
        f"peak_vram_mb={int(peak_vram_mb)}"
    )
    print(f"METRIC: {final_metrics['combined']:.6f}")
    logger.info(f"Total training time: {total_time:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        logger.error("CRASH: OOM")
        print("CRASH: OOM")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CRASH: OOM")
            print("CRASH: OOM")
        else:
            raise
