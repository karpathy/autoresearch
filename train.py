"""
Medical image classification training script. Single-GPU, single-file.
Fixed 5-minute wall-clock training budget.

Usage:
    python train.py
    DATASET=pathmnist IMAGE_SIZE=224 python train.py

Environment variables (passed through to prepare.py):
    DATASET     MedMNIST dataset name (default: chestmnist)
    IMAGE_SIZE  Image resolution 28 or 224 (default: 28)
"""

import logging
import time

import torch
import torch.nn as nn
from torchvision import models

from prepare import (
    DATASET,
    IMAGE_SIZE,
    TRAIN_TIME_MINUTES,
    evaluate,
    get_dataloaders,
    get_num_classes,
    is_multilabel,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# === HYPERPARAMETERS (agent can modify) ===
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
PRETRAINED: bool = True  # use ImageNet pretrained weights

# ---------------------------------------------------------------------------
# === MODEL ARCHITECTURE (agent can modify) ===
# ---------------------------------------------------------------------------


def build_model(num_classes: int, pretrained: bool = PRETRAINED) -> nn.Module:
    """
    Build ResNet-18 with a modified final layer for *num_classes* outputs.

    Args:
        num_classes: Number of output logits (labels for multi-label, classes for multi-class).
        pretrained:  If True, load ImageNet pretrained weights.

    Returns:
        A ``torchvision`` ResNet-18 model with adapted ``fc`` layer.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Adapt for 1-channel (greyscale) MedMNIST images when not using 224px pretrained
    # Note: only swap conv1 when image is 28x28 to preserve pretrained weights otherwise
    if IMAGE_SIZE == 28:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# === DATA AUGMENTATION (agent can modify) ===
# ---------------------------------------------------------------------------
# Augmentation is currently handled in prepare.py get_dataloaders().
# To add augmentation: import transforms here and pass custom_train_transform
# to get_dataloaders() — or modify the transforms inside prepare.py.
#
# Example (uncomment and extend as desired):
#
# from torchvision import transforms
# TRAIN_TRANSFORM = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5]),
# ])


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: %s", device)
logger.info("Dataset: %s  image_size: %d", DATASET, IMAGE_SIZE)

num_classes = get_num_classes(DATASET)
multilabel = is_multilabel(DATASET)
logger.info("num_classes=%d  multilabel=%s", num_classes, multilabel)

train_loader, val_loader, _ = get_dataloaders(
    dataset_name=DATASET,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# === OPTIMIZER & SCHEDULER (agent can modify) ===
# ---------------------------------------------------------------------------

model = build_model(num_classes=num_classes, pretrained=PRETRAINED).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# T_max is a placeholder; reset each epoch so scheduler tracks relative progress
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=len(train_loader),
    eta_min=LEARNING_RATE * 0.01,
)

criterion: nn.Module = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
logger.info("Loss: %s", criterion.__class__.__name__)

# ---------------------------------------------------------------------------
# === TRAINING LOOP (agent can modify) ===
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = TRAIN_TIME_MINUTES * 60

logger.info("Time budget: %ds (%d min)", TIME_BUDGET_SECONDS, TRAIN_TIME_MINUTES)

t_train_start = time.time()
step = 0
epoch = 0

while True:
    model.train()
    epoch += 1
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)

        if multilabel:
            labels = labels.float().to(device, non_blocking=True)
        else:
            labels = labels.squeeze(1).long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        step += 1
        elapsed = time.time() - t_train_start

        if step % 50 == 0:
            pct = 100 * elapsed / TIME_BUDGET_SECONDS
            logger.info(
                "epoch=%d step=%d loss=%.4f lr=%.2e elapsed=%.0fs (%.1f%%)",
                epoch,
                step,
                loss.item(),
                scheduler.get_last_lr()[0],
                elapsed,
                pct,
            )

        if elapsed >= TIME_BUDGET_SECONDS:
            break

    # Reset scheduler T_max for the next epoch so LR cosine restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=LEARNING_RATE * 0.01,
    )

    if time.time() - t_train_start >= TIME_BUDGET_SECONDS:
        break

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

logger.info("Training complete. Running final evaluation...")

val_metrics = evaluate(model, val_loader, num_classes=num_classes, device=device)

total_time = time.time() - t_start
train_time = time.time() - t_train_start

logger.info("---")
logger.info("val_auc:          %.6f", val_metrics["auc"])
logger.info("val_acc:          %.6f", val_metrics["accuracy"])
logger.info("training_seconds: %.1f", train_time)
logger.info("total_seconds:    %.1f", total_time)
logger.info("num_steps:        %d", step)
logger.info("num_epochs:       %d", epoch)
if torch.cuda.is_available():
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    logger.info("peak_vram_mb:     %.1f", peak_vram_mb)
