"""Agent-editable training script for ReID autoresearch.

The AI agent modifies this file to experiment with model architecture,
loss functions, augmentations, and hyperparameters. All tunable parameters
are module-level constants below.

Usage: python train.py
"""

from prepare import (
    # Data
    CombinedDistillDataset, CombinedArcFaceDataset,
    collate_distill, collate_arcface,
    PadToSquare,
    # Teacher
    TrendyolEmbedder, init_teacher, load_teacher_embeddings,
    # Evaluation
    run_retrieval_eval, compute_combined_metric,
    # Transforms
    build_val_transform,
    # Dataset builders
    build_distill_dataset, build_arcface_dataset, build_val_dataset,
    # Constants
    EPOCHS, EMBEDDING_DIM, IMAGE_SIZE, DEFAULT_TEACHER_CACHE_DIR,
    SKIP_CLASSES, VAL_DIR,
    # Utility
    set_seed,
)

import sys
import time
import timm
import numpy as np
import random
import torch
import torch.nn.functional as functional
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# ============================================================
# EXPERIMENT VARIABLES (agent edits these)
# ============================================================
MODEL_NAME = "hf-hub:timm/lcnet_050.ra2_in1k"
BATCH_SIZE = 256
ARCFACE_BATCH_SIZE = 128
LR = 2e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 16
SEED = 42
DEVICE = "cuda"
QUALITY_DEGRADATION_PROB = 0.5
DROP_HARD_RATIO = 0.2
USE_ARCFACE = True
ARCFACE_S = 32.0
ARCFACE_M = 0.50
ARCFACE_LOSS_WEIGHT = 0.03
ARCFACE_PHASEOUT_EPOCH = 0     # 0 = disabled
ARCFACE_MAX_PER_CLASS = 100
VAT_WEIGHT = 0.0
VAT_EPSILON = 8.0
SEP_WEIGHT = 1.0
UNFREEZE_EPOCH = 0             # 0 = unfreeze from start (per D-02)
BACKBONE_LR_MULT = 0.1        # Backbone LR = LR * this
TEACHER_CACHE_DIR = DEFAULT_TEACHER_CACHE_DIR
OUTPUT_DIR = "workspace/output/distill_trendyol_lcnet050_retail"
RETRIEVAL_MAX_SAMPLES = 5000
RETRIEVAL_TOPK = 5


# ============================================================
# MODEL (agent edits architecture)
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = functional.linear(functional.normalize(embeddings), functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * np.cos(self.m) - sine * np.sin(self.m)
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FrozenBackboneWithHead(nn.Module):
    """Student model with frozen backbone + trainable projection head."""

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 256,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            in_features = out.shape[-1]
        logger.info(f"Backbone {model_name} output dim: {in_features}")

        self.proj = ProjectionHead(in_features, embedding_dim)

    def train(self, mode: bool = True) -> "FrozenBackboneWithHead":
        super().train(mode)
        self.backbone.eval()
        return self

    def unfreeze_last_stage(self) -> None:
        """Unfreeze the last stage of the backbone."""
        for i in [-1, -2, -3, -4]:
            if hasattr(self.backbone, "stages"):
                for p in self.backbone.stages[i].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "features"):
                for p in self.backbone.features[i].parameters():
                    p.requires_grad = True
            elif hasattr(self.backbone, "blocks"):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True
        self.backbone.train()

    def forward_embeddings_train(self, images: torch.Tensor) -> torch.Tensor:
        has_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if has_trainable:
            features = self.backbone(images)
        else:
            with torch.no_grad():
                features = self.backbone(images)
        emb = self.proj(features)
        emb = functional.normalize(emb, p=2, dim=1)
        return emb

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            features = self.backbone(images)
            emb = self.proj(features)
            emb = functional.normalize(emb, p=2, dim=1)
        return emb


# ============================================================
# AUGMENTATIONS (agent edits transforms)
# ============================================================

class RandomQualityDegradation:
    """Randomly degrade image quality by downsampling and JPEG compression."""

    def __init__(
        self,
        prob: float = 0.5,
        downsample_ratio: tuple[float, float] = (0.3, 0.6),
        quality_range: tuple[int, int] = (50, 80),
    ) -> None:
        self.prob = prob
        self.downsample_ratio = downsample_ratio
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        ratio = random.uniform(*self.downsample_ratio)
        new_w, new_h = max(1, int(img.width * ratio)), max(1, int(img.height * ratio))
        orig_size = (img.width, img.height)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        img = img.resize(orig_size, Image.Resampling.BILINEAR)
        return img


def build_train_transform(model_name: str, image_size: int) -> transforms.Compose:
    """Build training transform (agent-editable augmentations)."""
    tmp_model = timm.create_model(model_name, pretrained=True)
    from timm.data import resolve_data_config
    data_config = resolve_data_config(tmp_model.pretrained_cfg)
    mean = data_config["mean"]
    std = data_config["std"]
    del tmp_model
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)], p=0.4),
        PadToSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ============================================================
# DEBUGGING (agent may modify)
# ============================================================

def save_batch_visualization(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    title: str = "Training Batch",
    max_images: int = 16,
    denormalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    denormalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> None:
    """Save a grid visualization of batch images.

    Args:
        images: Tensor of shape (B, C, H, W)
        labels: Tensor of shape (B,)
        output_path: Path to save the visualization
        title: Title for the plot
        max_images: Maximum number of images to show
        denormalize_mean: Mean used for normalization
        denormalize_std: Std used for normalization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(images), max_images)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    mean = torch.tensor(denormalize_mean).view(3, 1, 1)
    std = torch.tensor(denormalize_std).view(3, 1, 1)

    for i in range(n):
        row, col = i // ncols, i % ncols
        ax = axes[row][col]

        # Denormalize
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}", fontsize=8)
        ax.axis("off")

    # Hide empty subplots
    for i in range(n, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row][col].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved batch visualization to {output_path}")


# ============================================================
# LOSSES (agent edits loss functions)
# ============================================================

def vat_embedding_loss(
    model: FrozenBackboneWithHead,
    x: torch.Tensor,
    epsilon: float = 2.0,
    xi: float = 0.1,
    num_power_iter: int = 1,
) -> torch.Tensor:
    """Feature-level VAT loss (Miyato et al., 2018 -- arXiv 1704.03976).

    Perturbs backbone features (not raw pixels) to find adversarial
    direction, then penalises cosine distance between clean and perturbed
    embeddings.  Operating in ~512-dim feature space instead of 150K-dim
    pixel space is faster and finds better adversarial directions.
    """
    # Get clean backbone features and embedding
    with torch.no_grad():
        feat_clean = model.backbone(x)
        emb_clean = functional.normalize(model.proj(feat_clean), p=2, dim=1)

    # Random unit vector in feature space
    d = torch.randn_like(feat_clean)
    d = d / (d.norm(dim=1, keepdim=True) + 1e-12)

    for _ in range(num_power_iter):
        d = d.detach().requires_grad_(True)
        emb_perturbed = functional.normalize(model.proj(feat_clean.detach() + xi * d), p=2, dim=1)
        dist = (1.0 - functional.cosine_similarity(emb_clean, emb_perturbed, dim=1)).mean()
        (grad_d,) = torch.autograd.grad(dist, d)
        d = grad_d.detach()
        d = d / (d.norm(dim=1, keepdim=True) + 1e-12)

    # Final VAT loss with adversarial perturbation
    r_adv = epsilon * d.detach()
    emb_adv = functional.normalize(model.proj(feat_clean.detach() + r_adv), p=2, dim=1)
    return (1.0 - functional.cosine_similarity(emb_clean, emb_adv, dim=1)).mean()


# ============================================================
# TRAINING LOOP (agent edits everything here)
# ============================================================

@dataclass
class EpochStats:
    loss: float
    distill_loss: float
    arc_loss: float
    vat_loss: float
    sep_loss: float
    mean_cosine: float


def run_train_epoch(
    model: FrozenBackboneWithHead,
    distill_loader: DataLoader,
    arcface_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler,
    teacher: TrendyolEmbedder,
    device: torch.device,
    amp: bool,
    arc_margin: ArcMarginProduct | None,
    arc_loss_weight: float,
    cache_dir: str | None,
    drop_hard_ratio: float = 0.0,
    vat_weight: float = 0.0,
    vat_epsilon: float = 0.1,
    sep_weight: float = 0.0,
    blacklist_class_indices: set[int] | None = None,
    wl_centroid_ema: dict | None = None,
    backbone_unfrozen: bool = False,
    save_first_batch_path: Path | None = None,
) -> EpochStats:
    """Run one training epoch with separate distillation and ArcFace data."""
    model.train()

    total_loss = 0.0
    total_distill = 0.0
    total_arc = 0.0
    total_vat = 0.0
    total_sep = 0.0
    total_align = 0.0
    _bl_idx = blacklist_class_indices or set()
    n = 0
    first_batch_saved = False

    # Create iterators
    arcface_iter = iter(arcface_loader) if arcface_loader else None

    for images, labels, paths in distill_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Save first batch visualization
        if save_first_batch_path and not first_batch_saved:
            save_batch_visualization(
                images, labels, save_first_batch_path / "distill_batch.png", title="Distillation Batch (First)"
            )
            first_batch_saved = True

        # Load teacher embeddings for distillation
        teacher_emb = load_teacher_embeddings(paths, teacher, device, cache_dir)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            student_emb = model.forward_embeddings_train(images)

            # --- Distillation Loss ---
            teacher_emb = teacher_emb.to(device=device, dtype=student_emb.dtype)
            cosine = functional.cosine_similarity(student_emb, teacher_emb, dim=1)
            distill_loss = (1.0 - cosine).mean()
            batch_align = float(cosine.mean().item())

            # --- ArcFace Loss (from retail dataset) ---
            arc_loss = torch.tensor(0.0, device=device)
            if arc_margin is not None and arcface_iter is not None:
                try:
                    arc_images, arc_labels, _arc_paths = next(arcface_iter)
                except StopIteration:
                    arcface_iter = iter(arcface_loader)
                    arc_images, arc_labels, _arc_paths = next(arcface_iter)

                arc_images = arc_images.to(device, non_blocking=True)
                arc_labels = arc_labels.to(device, non_blocking=True)

                # Save first ArcFace batch visualization
                if save_first_batch_path and n == 0:
                    save_batch_visualization(
                        arc_images,
                        arc_labels,
                        save_first_batch_path / "arcface_batch.png",
                        title="ArcFace Batch (First)",
                    )

                arc_emb = model.forward_embeddings_train(arc_images)

                # ArcFace classification loss
                arc_logits = arc_margin(arc_emb, arc_labels)
                per_sample_arc = functional.cross_entropy(arc_logits, arc_labels, reduction="none")

                if drop_hard_ratio > 0.0:
                    keep = max(int(len(per_sample_arc) * (1 - drop_hard_ratio)), 1)
                    trimmed, _ = torch.topk(per_sample_arc, k=keep, largest=False)
                    arc_loss = trimmed.mean()
                else:
                    arc_loss = per_sample_arc.mean()

            # --- Separation Loss: push blacklist away from whitelist ---
            l_sep = torch.tensor(0.0, device=device)
            if sep_weight > 0 and _bl_idx:
                bl_mask = torch.tensor([int(lab) in _bl_idx for lab in labels], device=device)
                wl_mask = ~bl_mask

                # Update EMA whitelist centroid
                if wl_mask.any() and wl_centroid_ema is not None:
                    wl_mean = student_emb[wl_mask].detach().mean(dim=0)
                    if wl_centroid_ema.get("centroid") is None:
                        wl_centroid_ema["centroid"] = wl_mean
                    else:
                        wl_centroid_ema["centroid"] = 0.9 * wl_centroid_ema["centroid"] + 0.1 * wl_mean

                # Compute separation: blacklist vs EMA centroid (always available)
                if bl_mask.any() and wl_centroid_ema is not None and wl_centroid_ema.get("centroid") is not None:
                    bl_emb = student_emb[bl_mask]
                    centroid = functional.normalize(wl_centroid_ema["centroid"].unsqueeze(0), p=2, dim=1)
                    l_sep = (bl_emb @ centroid.T).clamp(min=0).mean()

            loss = distill_loss + arc_loss_weight * arc_loss + sep_weight * l_sep

        # --- VAT Loss (fp32, outside autocast to avoid precision issues) ---
        # Skip VAT while backbone is frozen -- perturbations have no effect.
        l_vat = torch.tensor(0.0, device=device)
        if vat_weight > 0 and backbone_unfrozen:
            l_vat = vat_embedding_loss(model, images, epsilon=vat_epsilon)
            loss = loss + vat_weight * l_vat

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_distill += float(distill_loss.item())
        total_arc += float(arc_loss.item())
        total_vat += float(l_vat.item())
        total_sep += float(l_sep.item())
        total_align += batch_align
        n += 1

        # ML-style progress bar
        total_steps = len(distill_loader)
        if total_steps > 0:
            bar_len = 30
            filled = bar_len * n // total_steps
            bar = "=" * filled + ">" + "." * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len
            avg_loss = total_loss / n
            avg_cos = total_align / n
            print(f"\r  {n}/{total_steps} [{bar}] - loss: {avg_loss:.4f} - cos: {avg_cos:.4f}", end="", flush=True)
    print()  # newline after epoch

    return EpochStats(
        loss=total_loss / max(n, 1),
        distill_loss=total_distill / max(n, 1),
        arc_loss=total_arc / max(n, 1),
        vat_loss=total_vat / max(n, 1),
        sep_loss=total_sep / max(n, 1),
        mean_cosine=total_align / max(n, 1),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Build training augmentations (agent controls these)
    train_transform = build_train_transform(MODEL_NAME, IMAGE_SIZE)
    quality_degradation = RandomQualityDegradation(prob=QUALITY_DEGRADATION_PROB)

    # Build datasets (prepare.py controls data, train.py controls transforms)
    distill_dataset = build_distill_dataset(train_transform, quality_degradation)

    arcface_dataset: CombinedArcFaceDataset | None = None
    num_arcface_classes = 0
    if USE_ARCFACE:
        arcface_dataset, num_arcface_classes = build_arcface_dataset(
            train_transform, quality_degradation, max_per_class=ARCFACE_MAX_PER_CLASS
        )

    val_dataset = build_val_dataset(MODEL_NAME, IMAGE_SIZE)

    # DataLoaders
    distill_loader = DataLoader(
        distill_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_distill,
    )

    arcface_loader: DataLoader | None = None
    if arcface_dataset is not None:
        arcface_loader = DataLoader(
            arcface_dataset,
            batch_size=ARCFACE_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            collate_fn=collate_arcface,
        )

    # Model
    model = FrozenBackboneWithHead(
        model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        device=str(device),
    ).to(device)

    # ArcFace head
    arc_margin: ArcMarginProduct | None = None
    if USE_ARCFACE and arcface_dataset is not None:
        arc_margin = ArcMarginProduct(
            in_features=EMBEDDING_DIM,
            out_features=num_arcface_classes,
            s=ARCFACE_S,
            m=ARCFACE_M,
        ).to(device)
        logger.info(f"ArcFace enabled: {num_arcface_classes} classes, s={ARCFACE_S}, m={ARCFACE_M}")

    # Teacher model
    teacher = init_teacher(str(device))

    # Unfreeze backbone (per D-02, UNFREEZE_EPOCH=0 means unfreeze from start)
    backbone_unfrozen = False
    if UNFREEZE_EPOCH == 0:
        model.unfreeze_last_stage()
        backbone_unfrozen = True

    # Optimizer with differential LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.proj.parameters())
    if arc_margin is not None:
        head_params += list(arc_margin.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": LR},
            {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Output directory
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    wl_centroid_ema: dict = {"centroid": None}

    # Training loop (fixed epoch budget)
    t_start = time.time()
    recall_at_1 = 0.0
    mean_cos = 0.0
    combined = 0.0

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Save first batch visualization only on epoch 0
        first_batch_path = out_dir if epoch == 0 else None

        # Unfreeze backbone at UNFREEZE_EPOCH (if not already unfrozen)
        if not backbone_unfrozen and epoch >= UNFREEZE_EPOCH:
            model.unfreeze_last_stage()
            backbone_unfrozen = True
            # Rebuild optimizer with backbone params
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = list(model.proj.parameters())
            if arc_margin is not None:
                head_params += list(arc_margin.parameters())
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": LR},
                    {"params": backbone_params, "lr": LR * BACKBONE_LR_MULT},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)
            logger.info(f"Unfroze backbone at epoch {epoch}")

        # ArcFace phase-out: linearly decay weight to 0 after phaseout epoch
        if ARCFACE_PHASEOUT_EPOCH > 0 and epoch >= ARCFACE_PHASEOUT_EPOCH:
            remaining = EPOCHS - ARCFACE_PHASEOUT_EPOCH
            progress = (epoch - ARCFACE_PHASEOUT_EPOCH) / max(remaining, 1)
            effective_arc_weight = ARCFACE_LOSS_WEIGHT * (1.0 - progress)
        else:
            effective_arc_weight = ARCFACE_LOSS_WEIGHT

        stats = run_train_epoch(
            model=model,
            distill_loader=distill_loader,
            arcface_loader=arcface_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            teacher=teacher,
            device=device,
            amp=(device.type == "cuda"),
            arc_margin=arc_margin,
            arc_loss_weight=effective_arc_weight,
            cache_dir=TEACHER_CACHE_DIR,
            drop_hard_ratio=DROP_HARD_RATIO,
            vat_weight=VAT_WEIGHT,
            vat_epsilon=VAT_EPSILON,
            sep_weight=SEP_WEIGHT,
            blacklist_class_indices=distill_dataset.blacklist_class_indices,
            wl_centroid_ema=wl_centroid_ema,
            backbone_unfrozen=backbone_unfrozen,
            save_first_batch_path=first_batch_path,
        )
        elapsed_epoch = time.time() - t0

        arc_w_str = f" arc_w={effective_arc_weight:.4f}" if effective_arc_weight != ARCFACE_LOSS_WEIGHT else ""
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={stats.loss:.4f} distill={stats.distill_loss:.4f} "
            f"arc={stats.arc_loss:.4f} vat={stats.vat_loss:.4f} sep={stats.sep_loss:.4f} cosine={stats.mean_cosine:.4f}{arc_w_str} | "
            f"{elapsed_epoch:.1f}s"
        )

        # Retrieval evaluation
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        mean_cos = stats.mean_cosine
        if val_dataset is not None:
            retrieval_metrics = run_retrieval_eval(
                model=model,
                dataset=val_dataset,
                device=device,
                amp=(device.type == "cuda"),
                max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK,
                seed=SEED,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = retrieval_metrics["recall@1"]
            recall_at_5 = retrieval_metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            logger.info(
                f"  Retrieval: recall@1={recall_at_1:.4f} "
                f"recall@{RETRIEVAL_TOPK}={recall_at_5:.4f}"
            )

        combined = compute_combined_metric(recall_at_1, mean_cos)
        logger.info(f"  Combined metric: {combined:.6f}")

    # Compute final metrics
    elapsed = time.time() - t_start
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    final_distill_loss = stats.distill_loss
    final_arc_loss = stats.arc_loss
    final_vat_loss = stats.vat_loss
    final_sep_loss = stats.sep_loss

    # Write metrics.json (per D-02, INFRA-02, INFRA-05, INFRA-06)
    import json

    metrics = {
        "status": "success",
        "combined_metric": combined,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mean_cosine": mean_cos,
        "distill_loss": final_distill_loss,
        "arc_loss": final_arc_loss,
        "vat_loss": final_vat_loss,
        "sep_loss": final_sep_loss,
        "peak_vram_mb": peak_vram_mb,
        "epochs": EPOCHS,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Greppable summary block (per INFRA-06)
    print("---")
    print(f"status:           success")
    print(f"combined_metric:  {combined:.6f}")
    print(f"recall@1:         {recall_at_1:.6f}")
    print(f"recall@5:         {recall_at_5:.6f}")
    print(f"mean_cosine:      {mean_cos:.6f}")
    print(f"distill_loss:     {final_distill_loss:.6f}")
    print(f"arc_loss:         {final_arc_loss:.6f}")
    print(f"vat_loss:         {final_vat_loss:.6f}")
    print(f"sep_loss:         {final_sep_loss:.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"epochs:           {EPOCHS}")
    print(f"elapsed_seconds:  {elapsed:.1f}")


if __name__ == "__main__":
    import json
    import traceback

    try:
        main()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics = {
            "status": "oom",
            "peak_vram_mb": round(peak, 1),
            "error": "CUDA out of memory",
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("---")
        print("status: OOM")
        print(f"peak_vram_mb: {peak:.1f}")
        sys.exit(1)
    except Exception as e:
        peak = 0.0
        try:
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            pass
        metrics = {
            "status": "crash",
            "peak_vram_mb": round(peak, 1),
            "error": str(e),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        traceback.print_exc()
        sys.exit(1)
