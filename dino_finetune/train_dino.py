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
    TRAIN_DIR, VAL_DIR, SKIP_CLASSES,
    set_seed, collate_fn,
)

# -- Production overrides (do NOT edit prepare_dino.py) --
EPOCHS = 20                          # Override imported EPOCHS for production run

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# EXPERIMENT VARIABLES (agent edits these)
# ============================================================
LORA_R = 16                          # LoRA rank (per D-03)
LORA_ALPHA = 32                      # LoRA alpha (per D-03)
LORA_DROPOUT = 0.05                  # LoRA dropout
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # DINOv3 attention projections (per D-03)

BATCH_SIZE = 8                       # Physical batch size (VRAM safe for 24GB)
GRADIENT_ACCUMULATION_STEPS = 16     # Effective batch = 128
LR = 5e-4                           # Learning rate for AdamW (reduced to prevent collapse)
WEIGHT_DECAY = 0.01                  # AdamW weight decay
WARMUP_RATIO = 0.2                   # Fraction of total steps for LR warmup
TEMPERATURE = 0.20                   # InfoNCE temperature (softer for combined dataset)

# ArcFace metric learning
ARCFACE_WEIGHT = 0.0                 # Weight of ArcFace loss (0.0 = disabled)
ARCFACE_SCALE = 30.0                 # ArcFace scale parameter
ARCFACE_MARGIN = 0.3                 # ArcFace angular margin (radians)

SEED = 42
USE_GRADIENT_CHECKPOINTING = True
EVAL_EVERY_N_EPOCHS = 1              # Evaluate after every N epochs
MAX_STEPS_PER_EPOCH = 0              # Cap steps per epoch (0 = no cap)
MAX_TRAINING_SECONDS = 0             # No time limit for production run (0 = disabled)

# Early stopping
EARLY_STOP_COSINE_THRESHOLD = 0.95   # Stop if mean_cosine exceeds this (cosine collapse)
EARLY_STOP_PATIENCE = 10             # Stop if combined metric hasn't improved for N epochs
EARLY_STOP_RECALL_DROP = 0.15        # Stop if recall@1 drops more than this from best
EARLY_STOP_COLLAPSE_CONSECUTIVE = 3  # Stop only after N consecutive collapsed epochs

NUM_WORKERS = 4                      # DataLoader workers
DEVICE = "cuda"
LAST_ADAPTER_DIR = "dino_finetune/output/last_adapter"   # Saved every epoch
CHECKPOINT_PATH = "dino_finetune/output/last_adapter/checkpoint.pt"  # For resume

# Combined dataset paths (richer training data)
USE_COMBINED_DATASET = True                  # Use combined data from multiple sources
REID_PRODUCTS_DIR = "/data/mnt/mnt_ml_shared/joesu/reid/data/reid_train/train/products"
RETAIL_DIR = "/data/mnt/mnt_ml_shared/Vic/retail_product_checkout_crop"
RETAIL_MAX_PER_CLASS = 100                   # Cap retail samples per class


# ============================================================
# Combined dataset (multi-source training data)
# ============================================================

class CombinedDinoDataset(Dataset):
    """Combine product_code + REID products + retail data for DINOv3 training.

    Merges multiple data sources with unified class indexing.
    Excludes val barcodes to prevent data leakage.
    Returns (image_tensor, label) matching prepare_dino's collate_fn.
    """

    def __init__(self, primary_roots: list[str], retail_root: str,
                 transform, retail_max_per_class: int = 100,
                 skip_classes: set[str] | None = None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        # Collect all class IDs from primary roots (union)
        ref_class_ids: set[str] = set()
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for d in root_path.iterdir():
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                    ref_class_ids.add(d.name)

        # Build class list: primary classes first, then retail
        all_classes: list[str] = []
        primary_class_dirs: list[Path] = []

        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                logger.warning(f"Primary root not found: {root}")
                continue
            for d in sorted(root_path.iterdir()):
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name in ref_class_ids:
                    primary_class_dirs.append(d)

        seen_class_names: set[str] = set()
        for d in primary_class_dirs:
            class_name = f"primary_{d.name}"
            if class_name not in seen_class_names:
                all_classes.append(class_name)
                seen_class_names.add(class_name)

        # Retail classes
        retail_path = Path(retail_root)
        retail_class_dirs: list[Path] = []
        if retail_path.exists():
            retail_class_dirs = sorted(
                [d for d in retail_path.iterdir()
                 if d.is_dir() and not d.name.startswith((".", "@", "__"))]
            )
            for d in retail_class_dirs:
                all_classes.append(f"retail_{d.name}")

        self.classes = all_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

        # Load primary samples
        primary_count = 0
        for class_dir in primary_class_dirs:
            class_name = f"primary_{class_dir.name}"
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))
                    primary_count += 1

        # Load retail samples (capped per class)
        retail_count = 0
        for class_dir in retail_class_dirs:
            class_name = f"retail_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = []
            for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPEG"):
                image_files.extend(class_dir.glob(ext))
            if len(image_files) > retail_max_per_class:
                image_files = random.sample(image_files, retail_max_per_class)
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                retail_count += 1

        logger.info(
            f"CombinedDinoDataset: {len(self.classes)} classes "
            f"(primary={len(seen_class_names)}, retail={len(retail_class_dirs)}), "
            f"{len(self.samples)} samples (primary={primary_count}, retail={retail_count})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_combined_train_dataset(processor) -> tuple[Dataset, int]:
    """Build combined training dataset from multiple sources.

    Excludes val barcodes to prevent data leakage.
    """
    val_dir = Path(VAL_DIR)
    val_barcodes = {
        d.name for d in val_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    } if val_dir.exists() else set()
    skip = SKIP_CLASSES | val_barcodes
    logger.info(f"Combined dataset: skipping {len(skip)} classes ({len(SKIP_CLASSES)} skip + {len(val_barcodes)} val barcodes)")

    # Build DINOv3-compatible train transform
    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", 518)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = CombinedDinoDataset(
        primary_roots=[TRAIN_DIR, REID_PRODUCTS_DIR],
        retail_root=RETAIL_DIR,
        transform=train_transform,
        retail_max_per_class=RETAIL_MAX_PER_CLASS,
        skip_classes=skip,
    )
    return dataset, len(dataset.classes)


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
# Checkpoint save / load (for crash resume)
# ============================================================

def save_last_checkpoint(model, optimizer, scheduler, epoch: int,
                         best_combined: float, best_recall: float,
                         patience_counter: int, collapse_counter: int):
    """Save last adapter + training state for crash resume."""
    import os
    os.makedirs(LAST_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(LAST_ADAPTER_DIR)
    torch.save({
        "epoch": epoch,
        "best_combined": best_combined,
        "best_recall": best_recall,
        "patience_counter": patience_counter,
        "collapse_counter": collapse_counter,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, CHECKPOINT_PATH)
    logger.info(f"Last adapter + checkpoint saved (epoch {epoch})")



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
    logger.info(
        f"Early stopping: cosine>{EARLY_STOP_COSINE_THRESHOLD} "
        f"patience={EARLY_STOP_PATIENCE} recall_drop>{EARLY_STOP_RECALL_DROP} "
        f"collapse_consecutive={EARLY_STOP_COLLAPSE_CONSECUTIVE}"
    )

    # -- Load base model --
    base_model = load_base_model(DEVICE)

    # -- Check if resuming or warm-starting from saved adapter --
    import os
    has_adapter = os.path.exists(os.path.join(LAST_ADAPTER_DIR, "adapter_model.safetensors"))
    has_checkpoint = os.path.exists(CHECKPOINT_PATH)

    if has_adapter:
        # Load saved adapter onto base model (resume or warm start)
        from peft import PeftModel
        logger.info(f"Loading saved adapter from {LAST_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, LAST_ADAPTER_DIR, is_trainable=True)
    else:
        # Fresh run: inject new LoRA adapters
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # -- Enable gradient checkpointing for VRAM savings --
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # -- Build datasets and DataLoaders --
    processor = get_image_processor()
    if USE_COMBINED_DATASET:
        train_dataset, num_train_classes = build_combined_train_dataset(processor)
    else:
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

    # -- Resume or warm start --
    best_combined = -1.0
    best_recall = -1.0
    patience_counter = 0
    collapse_counter = 0
    start_epoch = 1

    if has_adapter and has_checkpoint:
        # Full resume: restore optimizer/scheduler/tracking state
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        start_epoch = ckpt["epoch"] + 1
        best_combined = ckpt["best_combined"]
        best_recall = ckpt["best_recall"]
        patience_counter = ckpt["patience_counter"]
        collapse_counter = ckpt["collapse_counter"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info(
            f"Full resume: epoch={start_epoch} best_combined={best_combined:.4f} "
            f"best_recall={best_recall:.4f} patience={patience_counter} collapse={collapse_counter}"
        )
    elif has_adapter:
        # Warm start: adapter weights loaded, fresh optimizer/scheduler
        logger.info("Warm start: adapter weights loaded, fresh optimizer + scheduler")

    # -- Training loop --
    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
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

            current_recall = metrics["recall@1"]
            current_cosine = metrics["mean_cosine"]

            # -- 1. Cosine collapse check (BEFORE any save) --
            is_collapsed = current_cosine > EARLY_STOP_COSINE_THRESHOLD

            if is_collapsed:
                collapse_counter += 1
                logger.warning(
                    f"Cosine collapse detected "
                    f"(mean_cosine={current_cosine:.4f} > {EARLY_STOP_COSINE_THRESHOLD}), "
                    f"consecutive={collapse_counter}/{EARLY_STOP_COLLAPSE_CONSECUTIVE} -- skipping save"
                )
                if collapse_counter >= EARLY_STOP_COLLAPSE_CONSECUTIVE:
                    logger.warning(
                        f"EARLY STOP: {EARLY_STOP_COLLAPSE_CONSECUTIVE} consecutive collapsed epochs"
                    )
                    break
            else:
                collapse_counter = 0  # Reset on healthy epoch

                # -- 2. Check improvement and save (only non-collapsed) --
                improved = metrics["combined"] > best_combined
                if improved:
                    best_combined = metrics["combined"]
                    save_adapter(model)
                    logger.info(f"New best! combined={best_combined:.4f} -- adapter saved")

                # -- 3. Patience: no improvement in combined metric --
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        logger.warning(
                            f"EARLY STOP: no improvement for {EARLY_STOP_PATIENCE} epochs "
                            f"(best_combined={best_combined:.4f})"
                        )
                        break

            # -- 4. Track best recall and check recall drop (always, regardless of collapse) --
            if current_recall > best_recall:
                best_recall = current_recall

            if best_recall > 0 and (best_recall - current_recall) > EARLY_STOP_RECALL_DROP:
                logger.warning(
                    f"EARLY STOP: recall@1 dropped {best_recall - current_recall:.4f} "
                    f"from best={best_recall:.4f} (threshold={EARLY_STOP_RECALL_DROP})"
                )
                break

            # -- 5. Save last adapter + checkpoint (every epoch, for crash resume) --
            save_last_checkpoint(
                model, optimizer, scheduler, epoch,
                best_combined, best_recall, patience_counter, collapse_counter,
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
