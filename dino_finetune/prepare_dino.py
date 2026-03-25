"""Immutable infrastructure for DINOv3 ViT-H+ fine-tuning.

This file owns: model loading, data pipeline, evaluation, adapter save/load.
The AI agent does NOT modify this file -- all tunable parameters are in train_dino.py.

Usage: python prepare_dino.py  (runs VRAM smoke test)
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Constants (immutable)
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
EMBEDDING_DIM = 1280          # DINOv3 ViT-H+ native dimension
IMAGE_SIZE = 518              # DINOv3 default resolution from AutoImageProcessor
TRAIN_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/train"
VAL_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/val"
SKIP_CLASSES = {"0000000000"}
EPOCHS = 10                   # Fixed budget per experiment
ADAPTER_OUTPUT_DIR = "dino_finetune/output/best_adapter"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility (torch, numpy, random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(device: str = "cuda"):
    """Load frozen DINOv3 ViT-H+ (840M) in bf16 with SDPA attention.

    All parameters are frozen (requires_grad=False). This is the base model
    before LoRA injection -- train_dino.py adds LoRA adapters on top.
    """
    model = AutoModel.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Verify all params are frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 0, f"Expected 0 trainable params after freeze, got {trainable}"
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"DINOv3 ViT-H+ loaded: {total / 1e6:.1f}M params, all frozen, bf16+SDPA")

    return model


def get_image_processor() -> AutoImageProcessor:
    """Return the HuggingFace image processor for DINOv3 ViT-H+.

    Handles correct normalization and resize for DINOv3.
    """
    return AutoImageProcessor.from_pretrained(BASE_MODEL_ID)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str, processor, split: str = "train"):
    """Build an ImageFolder dataset with DINOv3-compatible transforms.

    Args:
        data_dir: Path to image directory (ImageFolder structure).
        processor: HuggingFace AutoImageProcessor for DINOv3.
        split: "train" (with augmentations) or "val" (deterministic).

    Returns:
        (dataset, num_classes) tuple.
    """
    # Extract normalization params from processor
    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", IMAGE_SIZE)

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # Build ImageFolder, filtering SKIP_CLASSES
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Filter out skip classes
    filtered_indices = []
    for idx, (path, label) in enumerate(full_dataset.samples):
        class_name = full_dataset.classes[label]
        if class_name not in SKIP_CLASSES:
            filtered_indices.append(idx)

    if len(filtered_indices) < len(full_dataset):
        from torch.utils.data import Subset
        dataset = Subset(full_dataset, filtered_indices)
        num_classes = len([c for c in full_dataset.classes if c not in SKIP_CLASSES])
        logger.info(
            f"Dataset {split}: {len(dataset)} images, {num_classes} classes "
            f"(filtered {len(full_dataset) - len(dataset)} from SKIP_CLASSES)"
        )
    else:
        dataset = full_dataset
        num_classes = len(full_dataset.classes)
        logger.info(f"Dataset {split}: {len(dataset)} images, {num_classes} classes")

    return dataset, num_classes


def collate_fn(batch):
    """Collate function for DataLoader that stacks images and labels."""
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_cls_embedding(model_output) -> torch.Tensor:
    """Extract CLS token embedding from DINOv3 model output.

    DINOv3 output layout for last_hidden_state:
        Index 0: CLS token  <-- we want this
        Index 1-4: Register tokens (4 register tokens in DINOv3)
        Index 5+: Patch tokens

    The CLS token is the global image representation used for retrieval.

    Args:
        model_output: Output from model forward pass (has .last_hidden_state).

    Returns:
        L2-normalized CLS embeddings of shape [B, 1280].
    """
    # CLS token is at index 0 of the sequence dimension
    cls_emb = model_output.last_hidden_state[:, 0]  # [B, 1280]
    # L2 normalize for cosine similarity / contrastive learning
    cls_emb = F.normalize(cls_emb, dim=1)
    return cls_emb


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dino(model, val_loader: DataLoader, device: str) -> dict:
    """Compute recall@1 and mean cosine similarity on validation set.

    Metric follows D-08: cosine similarity between same-product embeddings
    and recall@1 on held-out validation. Combined = 0.5 * recall@1 + 0.5 * mean_cosine.

    Args:
        model: DINOv3 model (base or with LoRA adapters).
        val_loader: DataLoader for validation set.
        device: CUDA device string.

    Returns:
        {"recall@1": float, "mean_cosine": float, "combined": float}
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device, dtype=torch.bfloat16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
        cls_emb = extract_cls_embedding(outputs)
        all_embeddings.append(cls_emb.float().cpu())
        all_labels.append(labels)

    emb = torch.cat(all_embeddings)   # [N, 1280]
    lab = torch.cat(all_labels)       # [N]

    # -- Recall@1 --
    # Pairwise cosine similarity (embeddings already L2-normalized)
    sim = emb @ emb.T                # [N, N]
    sim.fill_diagonal_(-float("inf"))  # exclude self-similarity
    _, nn_idx = sim.topk(1, dim=1)   # nearest neighbor index
    recall_1 = (lab[nn_idx.squeeze(1)] == lab).float().mean().item()

    # -- Mean intra-class cosine similarity --
    # For each class, compute average pairwise cosine among its members
    unique_labels = lab.unique()
    cosine_sums = []
    for lbl in unique_labels:
        mask = lab == lbl
        if mask.sum() < 2:
            continue
        class_emb = emb[mask]  # [K, D]
        class_sim = class_emb @ class_emb.T  # [K, K]
        # Take upper triangle (exclude diagonal)
        k = class_sim.size(0)
        triu_mask = torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
        cosine_sums.append(class_sim[triu_mask].mean().item())

    mean_cosine = float(np.mean(cosine_sums)) if cosine_sums else 0.0

    combined = 0.5 * recall_1 + 0.5 * mean_cosine

    logger.info(
        f"Eval: recall@1={recall_1:.4f}  mean_cosine={mean_cosine:.4f}  "
        f"combined={combined:.4f}"
    )

    return {"recall@1": recall_1, "mean_cosine": mean_cosine, "combined": combined}


# ---------------------------------------------------------------------------
# Adapter save / load
# ---------------------------------------------------------------------------

def save_adapter(model, output_dir: str = ADAPTER_OUTPUT_DIR) -> None:
    """Save LoRA adapter weights via PEFT's save_pretrained.

    Only saves the adapter weights (adapter_config.json + adapter_model.safetensors),
    NOT the full base model. Typically a few MB.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Adapter saved to {output_dir}")


def load_finetuned_model(adapter_path: str = ADAPTER_OUTPUT_DIR, device: str = "cuda"):
    """Load base DINOv3 model + LoRA adapter for inference.

    Used by the main system's DINOv3FTTeacher to produce fine-tuned embeddings.

    Args:
        adapter_path: Path to saved LoRA adapter directory.
        device: CUDA device string.

    Returns:
        PEFT model in eval mode, ready for inference.
    """
    from peft import PeftModel

    base_model = load_base_model(device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info(f"Fine-tuned DINOv3 loaded from {adapter_path}")
    return model


# ---------------------------------------------------------------------------
# VRAM smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Running VRAM smoke test for DINOv3 ViT-H+ ...")

    torch.cuda.reset_peak_memory_stats()

    model = load_base_model("cuda")
    processor = get_image_processor()

    # Create dummy batch of 4 images
    dummy_images = [Image.new("RGB", (224, 224), color=(128, 128, 128)) for _ in range(4)]
    inputs = processor(images=dummy_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)

    # Forward pass
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(pixel_values)
    cls_emb = extract_cls_embedding(outputs)
    logger.info(f"CLS embedding shape: {cls_emb.shape}")  # Expected: [4, 1280]

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak VRAM: {peak_vram_mb:.0f} MB")

    # Cleanup
    del model, outputs, cls_emb, pixel_values
    torch.cuda.empty_cache()

    logger.info("Smoke test complete.")
