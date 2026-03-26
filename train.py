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
    load_teacher_embeddings,
    TEACHER_REGISTRY, init_teachers, build_all_teacher_caches,
    # RADIO teacher
    load_radio_teacher_embeddings, RADIO_VERSION_MAP,
    build_radio_summary_cache, RADIOTeacher,
    # Evaluation
    run_retrieval_eval, compute_combined_metric,
    # Transforms
    build_val_transform,
    # Dataset builders
    build_distill_dataset, build_arcface_dataset, build_val_dataset,
    # Constants
    EPOCHS, EMBEDDING_DIM, IMAGE_SIZE,
    SKIP_CLASSES, VAL_DIR,
    # Utility
    set_seed,
)

import sys
import time
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
# Teacher selection (per D-03, D-05)
TEACHER = "trendyol_onnx"  # Single teacher mode (default, backward compatible)
# Multi-teacher mode: set TEACHERS to enable, overrides TEACHER
# Example: TEACHERS = {"trendyol_onnx": 0.5, "dinov2": 0.5}
TEACHERS: dict[str, float] | None = None
OUTPUT_DIR = "workspace/output/distill_trendyol_lcnet050_retail"
RETRIEVAL_MAX_SAMPLES = 5000
RETRIEVAL_TOPK = 5

# --- SSL Contrastive Loss ---
SSL_WEIGHT = 0.0              # 0 = disabled; 0.01-0.1 typical. NOTE: enabling doubles forward passes per batch (VRAM ~1.5x)
SSL_TEMPERATURE = 0.07        # InfoNCE temperature (learnable, this is init value)
SSL_PROJ_DIM = 128            # SSL projection head output dim

# --- Custom LCNet Backbone ---
LCNET_SCALE = 0.5             # Width multiplier (0.5 matches current lcnet_050)
SE_START_BLOCK = 10           # Block index where SE begins (0-indexed, 13 total blocks)
SE_REDUCTION = 0.25           # SE squeeze ratio
ACTIVATION = "h_swish"        # "h_swish" | "relu" | "gelu"
KERNEL_SIZES = [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5]  # Per-block kernel sizes (13 blocks)
USE_PRETRAINED = True          # Load timm pretrained weights when scale matches

# --- RADIO Teacher Configuration ---
RADIO_VARIANT = "so400m"              # "so400m" or "h" (C-RADIOv4 variant)
RADIO_ADAPTORS = ["backbone"]          # subset of ["backbone", "dino_v3", "siglip2-g"]
RADIO_CACHE_BASE = "workspace/output/teacher_cache"

# --- Spatial Distillation ---
SPATIAL_DISTILL_WEIGHT = 0.0   # 0.0 = disabled; agent sets positive value to enable spatial distillation from RADIO

# --- RADIO Training Techniques ---
# PHI-S: Hadamard isotropic standardization for multi-teacher gradient balancing
ENABLE_PHI_S = False
# Feature Normalizer: per-teacher whitening + rotation
ENABLE_FEATURE_NORMALIZER = False
NORMALIZER_WARMUP_BATCHES = 200  # ~1 full epoch for 50k images / 256 batch size


def _load_radio_metadata(
    variant: str,
    adaptors: list[str],
    cache_base: str,
) -> dict[str, int]:
    """Read adaptor summary dimensions from cached metadata.json files.

    Returns dict mapping adaptor name -> summary_dim.
    Raises FileNotFoundError if metadata not yet built.
    """
    import json

    dims: dict[str, int] = {}
    for adaptor in adaptors:
        meta_path = Path(cache_base) / f"radio_{variant}" / adaptor / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"RADIO metadata not found at {meta_path}. "
                f"Run build_radio_summary_cache first."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        dims[adaptor] = meta["summary_dim"]
    return dims


def _get_active_teachers() -> tuple[list[str], dict[str, float]]:
    """Resolve TEACHER/TEACHERS constants into teacher names and weights."""
    if TEACHERS is not None:
        return list(TEACHERS.keys()), dict(TEACHERS)
    return [TEACHER], {TEACHER: 1.0}


# ============================================================
# MODEL (agent edits architecture)
# ============================================================

def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round channel count to nearest divisor (matches timm implementation)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block (SE) with Hardsigmoid gating."""

    def __init__(self, channels: int, reduced_ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(channels, reduced_ch, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_ch, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.conv_reduce(scale))
        scale = self.gate(self.conv_expand(scale))
        return x * scale


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (DSConv) for LCNet."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = False,
        se_ratio: float = 0.25,
        act_layer: type = nn.Hardswish,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        # Depthwise conv
        self.conv_dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = act_layer()
        # Squeeze-and-Excitation (optional)
        if use_se:
            mid_ch = max(1, int(in_ch * se_ratio))
            self.se = SqueezeExcite(in_ch, mid_ch)
        else:
            self.se = nn.Identity()
        # Pointwise conv
        self.conv_pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv_dw(x)))
        x = self.se(x)
        x = self.act2(self.bn2(self.conv_pw(x)))
        return x


# Per-stage config: list of (kernel_size, stride, out_channels_base) per block
# Verified against timm lcnet_050 block-by-block output
LCNET_ARCH = [
    # Stage 0: 1 block
    [(3, 1, 32)],
    # Stage 1: 2 blocks
    [(3, 2, 64), (3, 1, 64)],
    # Stage 2: 2 blocks
    [(3, 2, 128), (3, 1, 128)],
    # Stage 3: 2 blocks
    [(3, 2, 256), (5, 1, 256)],
    # Stage 4: 4 blocks
    [(5, 1, 256), (5, 1, 256), (5, 1, 256), (5, 1, 256)],
    # Stage 5: 2 blocks (with SE)
    [(5, 2, 512), (5, 1, 512)],
]


class LCNet(nn.Module):
    """Custom LCNet backbone with agent-tunable architecture parameters.

    Reimplemented from scratch based on PP-LCNet paper + timm source.
    Uses timm-compatible naming for pretrained weight loading.
    """

    def __init__(
        self,
        scale: float = LCNET_SCALE,
        se_start_block: int = SE_START_BLOCK,
        se_reduction: float = SE_REDUCTION,
        activation: str = ACTIVATION,
        kernel_sizes: list[int] = KERNEL_SIZES,
        embedding_dim: int = 256,
        device: str = "cuda",
        teacher_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        act_lookup = {"h_swish": nn.Hardswish, "relu": nn.ReLU, "gelu": nn.GELU}
        act_layer = act_lookup.get(activation, nn.Hardswish)

        # Stem: Conv2d(3, stem_ch, 3x3, s=2) + BN + activation
        stem_ch = make_divisible(16 * scale)
        self.conv_stem = nn.Conv2d(3, stem_ch, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_ch)
        self.stem_act = act_layer()

        # Build blocks: stages of DepthwiseSeparableConv
        stages = []
        in_ch = stem_ch
        flat_block_idx = 0
        kernel_iter = iter(kernel_sizes)
        for stage_blocks in LCNET_ARCH:
            stage = []
            for _default_ks, stride, out_base in stage_blocks:
                out_ch = make_divisible(out_base * scale)
                # Use agent-tunable kernel size if available, else default
                try:
                    ks = next(kernel_iter)
                except StopIteration:
                    ks = _default_ks
                use_se = flat_block_idx >= se_start_block
                stage.append(DepthwiseSeparableConv(
                    in_ch, out_ch, kernel_size=ks, stride=stride,
                    use_se=use_se, se_ratio=se_reduction, act_layer=act_layer,
                ))
                in_ch = out_ch
                flat_block_idx += 1
            stages.append(nn.Sequential(*stage))
        self.blocks = nn.Sequential(*stages)

        # Conv head: 1x1 conv to 1280 (no BN per paper)
        self.conv_head = nn.Conv2d(in_ch, 1280, 1, bias=False)
        self.head_act = act_layer()
        self.num_features = 1280

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection head(s)
        # Multi-teacher mode (per D-06): per-teacher projection heads
        self.proj_heads: nn.ModuleDict | None = None
        if teacher_dims is not None and len(teacher_dims) > 1:
            self.proj_heads = nn.ModuleDict({
                t_name: nn.Sequential(
                    nn.Linear(1280, t_dim),
                    nn.BatchNorm1d(t_dim),
                )
                for t_name, t_dim in teacher_dims.items()
            })
            # self.proj points to first teacher's head for encode() compatibility
            first_name = next(iter(teacher_dims))
            self.proj = self.proj_heads[first_name]
        else:
            # Single-teacher mode (backward compatible)
            self.proj = nn.Sequential(
                nn.Linear(1280, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )

        # Weight initialization
        self._init_weights()

        # Freeze backbone (all conv_stem, bn1, blocks params) -- unfrozen at UNFREEZE_EPOCH
        for p in self.conv_stem.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False

    def _init_weights(self) -> None:
        """Initialize weights: kaiming for Conv2d, xavier for Linear, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward_stem_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through stem + all blocks (spatial features here)."""
        x = self.stem_act(self.bn1(self.conv_stem(x)))
        x = self.blocks(x)
        return x

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (spatial: [B, C, H, W], summary: [B, 1280]).

        spatial is the pre-conv_head feature map (output of last stage).
        summary is after conv_head + GAP.
        """
        spatial = self._forward_stem_blocks(x)
        x = self.head_act(self.conv_head(spatial))
        x = self.gap(x).flatten(1)
        return spatial, x

    def forward_backbone(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw backbone features [B, 1280] before any projection head."""
        _spatial, summary = self.forward_features(images)
        return summary

    def forward_embeddings_train(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradients for training. Returns L2-normalized [B, 256]."""
        _spatial, summary = self.forward_features(images)
        emb = self.proj(summary)
        return functional.normalize(emb, p=2, dim=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalized embeddings [B, 256]. No gradients."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _spatial, summary = self.forward_features(images)
            emb = self.proj(summary)
            result = functional.normalize(emb, p=2, dim=1)
        if was_training:
            self.train()
        return result

    def encode_with_spatial(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding: [B, 256], spatial: [B, C, H, W]). No gradients."""
        self.eval()
        with torch.no_grad():
            spatial, summary = self.forward_features(images)
            emb = self.proj(summary)
            emb = functional.normalize(emb, p=2, dim=1)
            return emb, spatial

    def unfreeze_last_stage(self) -> None:
        """Unfreeze last 2 stages and conv_head for fine-tuning."""
        # Unfreeze last 2 stages (blocks[-1] = stage 5, blocks[-2] = stage 4)
        for i in [-1, -2]:
            for p in self.blocks[i].parameters():
                p.requires_grad = True
        for p in self.conv_head.parameters():
            p.requires_grad = True


def load_pretrained_lcnet(model: LCNet, scale: float) -> None:
    """Load timm pretrained weights into custom LCNet."""
    import timm as _timm
    scale_to_variant = {0.5: "lcnet_050", 0.75: "lcnet_075", 1.0: "lcnet_100"}
    variant = scale_to_variant.get(scale)
    if variant is None:
        logger.warning(f"No pretrained weights for scale={scale}, training from scratch")
        return

    timm_model = _timm.create_model(f"hf-hub:timm/{variant}.ra2_in1k", pretrained=True, num_classes=0)
    timm_sd = timm_model.state_dict()

    model_sd = model.state_dict()
    loaded = 0
    for k, v in timm_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd, strict=False)
    logger.info(f"Loaded {loaded}/{len(timm_sd)} pretrained weights from {variant}")
    del timm_model


def _build_hadamard(n: int) -> torch.Tensor:
    """Build normalized Hadamard matrix of size n via Sylvester construction.

    Per PHI-S paper Eq. 18: recursive Kronecker product H_n = H_1 ⊗ H_{n-1},
    normalized by 1/sqrt(2) at each step so rows are orthonormal.

    Args:
        n: Matrix size, must be a positive power of 2.

    Returns:
        (n, n) orthonormal Hadamard matrix (H @ H^T = I).
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / (2 ** 0.5)
    return H


class PHISTransform(nn.Module):
    """PCA-Hadamard Isotropic Standardization (PHI-S).

    Rotates teacher features so all dimensions have equal variance,
    then scales uniformly. Prevents any single teacher from dominating
    gradient updates in multi-teacher distillation.

    IMPORTANT: PHI-S operates on RAW features BEFORE L2 normalization.
    For the current single-teacher mode where cached embeddings are already
    L2-normalized, PHI-S has limited effect but is correctly implemented
    for multi-teacher activation.

    Key property: transform is invertible and isotropic -- errors of equal
    magnitude in normalized space map to equal magnitude in original space.

    The transform is computed once via fit() from training data statistics,
    then frozen (no learnable parameters).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("R", torch.eye(feature_dim))       # rotation: H @ U^T
        self.register_buffer("alpha", torch.tensor(1.0))         # scale: 1/phi
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.ready = False

    def fit(self, features: torch.Tensor) -> None:
        """Compute PHI-S parameters from a batch of teacher features.

        Args:
            features: (N, D) tensor of raw teacher features.
        """
        D = self.feature_dim
        # Compute mean
        mu = features.mean(dim=0)

        # Compute covariance
        centered = features - mu
        cov = (centered.T @ centered) / (features.shape[0] - 1)

        # Eigendecomposition: cov = U @ diag(lambda) @ U^T
        eigenvalues, U = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.clamp(min=1e-8)  # numerical stability (Pitfall 3)

        # Build Hadamard matrix (power of 2 only via Sylvester)
        # For non-power-of-2 dims, pad to next power of 2 and truncate
        n_pad = 1
        while n_pad < D:
            n_pad *= 2
        H = _build_hadamard(n_pad)
        H = H[:D, :D]  # truncate if padded (approximate but functional)

        # PHI-S: R = H @ U^T, alpha = 1/phi
        # phi = sqrt(1/D * sum(lambda_i)) = sqrt(trace(cov) / D)
        phi = torch.sqrt(eigenvalues.sum() / D)
        self.R.copy_(H @ U.T)
        self.alpha.copy_(1.0 / phi)
        self.mean.copy_(mu)
        self.ready = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PHI-S transform: X' = alpha * (X - mu) @ R^T.

        Returns features unchanged (passthrough) if fit() has not been called.
        """
        if not self.ready:
            return features  # passthrough during warmup
        centered = features - self.mean
        return self.alpha * (centered @ self.R.T)


class FeatureNormalizer(nn.Module):
    """Per-teacher feature normalizer with online statistics warmup.

    During warmup (first `warmup_batches` forward calls), accumulates running
    mean and variance using Welford's online algorithm, passing features through
    unchanged. After warmup, standardizes features to zero mean and unit variance.

    Each teacher should have its own FeatureNormalizer instance (handled in
    Plan 03 integration).
    """

    def __init__(self, feature_dim: int, warmup_batches: int = 200) -> None:
        super().__init__()
        self.warmup_batches = warmup_batches
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.ready = False

    def _update_stats(self, features: torch.Tensor) -> None:
        """Update running mean/variance using Welford's online algorithm."""
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0, unbiased=False)
        n = self.count.item()
        self.count += 1
        # Welford's online update for running mean
        delta = batch_mean - self.running_mean
        self.running_mean += delta / (n + 1)
        # Online variance update (weighted combination)
        if n > 0:
            self.running_var = (n * self.running_var + batch_var) / (n + 1)
        else:
            self.running_var = batch_var
        if self.count >= self.warmup_batches:
            self.ready = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features after warmup, passthrough before.

        During warmup: updates statistics from detached features, returns input unchanged.
        After warmup: returns (features - mean) / (sqrt(var) + eps).
        """
        if not self.ready:
            self._update_stats(features.detach())
            return features  # passthrough during warmup
        # Numerical stability: epsilon prevents NaN from near-zero variance (Pitfall 3)
        return (features - self.running_mean) / (self.running_var.sqrt() + 1e-6)


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


def build_train_transform(image_size: int) -> transforms.Compose:
    """Build training transform (agent-editable augmentations)."""
    # ImageNet normalization (used by all lcnet variants)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
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


class SpatialAdapter(nn.Module):
    """Conv1x1 + BN to project student spatial features to RADIO spatial dim.

    Per D-08: lightweight adapter that aligns student's pre-GAP spatial
    feature channels to the RADIO teacher's spatial feature dimension.
    """

    def __init__(self, student_channels: int, radio_spatial_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(student_channels, radio_spatial_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(radio_spatial_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H_s, W_s] student spatial features
        return self.bn(self.proj(x))


def spatial_distillation_loss(
    student_spatial: torch.Tensor,   # [B, C, H_s, W_s] from LCNet pre-GAP
    teacher_spatial: torch.Tensor,   # [B, N, D] RADIO spatial (NLC format)
    adapter: SpatialAdapter,
    teacher_grid_h: int,
    teacher_grid_w: int,
) -> torch.Tensor:
    """Compute spatial distillation loss between student and teacher spatial features.

    Per D-06: Student spatial is bilinear-interpolated to match teacher spatial
    resolution, then projected via Conv1x1+BN to teacher dim. Both feature maps
    are L2-normalized before MSE to handle scale mismatch between student and
    RADIO teacher representations.
    """
    # Reshape teacher from NLC [B, N, D] to NCHW [B, D, H_t, W_t]
    B = teacher_spatial.shape[0]
    teacher_nchw = teacher_spatial.reshape(B, teacher_grid_h, teacher_grid_w, -1)
    teacher_nchw = teacher_nchw.permute(0, 3, 1, 2).contiguous()  # [B, D, H_t, W_t]

    # Interpolate student spatial to match teacher grid resolution
    student_interp = functional.interpolate(
        student_spatial,
        size=(teacher_grid_h, teacher_grid_w),
        mode="bilinear",
        align_corners=False,
    )  # [B, C, H_t, W_t]

    # Project student to teacher dim via Conv1x1 + BN
    student_proj = adapter(student_interp)  # [B, D, H_t, W_t]

    # L2-normalize along channel dim before loss (helps with scale mismatch)
    student_proj = functional.normalize(student_proj, p=2, dim=1)
    teacher_nchw = functional.normalize(teacher_nchw, p=2, dim=1)

    # MSE loss on aligned, normalized spatial feature maps
    loss = functional.mse_loss(student_proj, teacher_nchw)
    return loss


def vat_embedding_loss(
    model: LCNet,
    x: torch.Tensor,
    epsilon: float = 2.0,
    xi: float = 0.1,
    num_power_iter: int = 1,
) -> torch.Tensor:
    """Feature-level VAT loss (Miyato et al., 2018 -- arXiv 1704.03976).

    Perturbs summary features (post-GAP, pre-projection) to find adversarial
    direction, then penalises cosine distance between clean and perturbed
    embeddings.  Operating in 1280-dim feature space instead of 150K-dim
    pixel space is faster and finds better adversarial directions.
    """
    # Get clean summary features and embedding
    with torch.no_grad():
        _spatial, feat_clean = model.forward_features(x)
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


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with learnable temperature (CLIP-style).

    Pushes two augmented views of the same image together,
    different images apart. Uses in-batch negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        # CLIP-style learnable log temperature
        self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # z_a, z_b: [B, D] L2-normalized projections
        # Clamp log_scale to prevent temperature explosion (Pitfall 7)
        log_scale = self.log_scale.clamp(max=4.6052)  # temperature >= 0.01
        temperature = torch.exp(-log_scale)
        logits = z_a @ z_b.T / temperature  # [B, B]
        labels = torch.arange(len(z_a), device=z_a.device)
        loss = (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2
        return loss


class SSLProjectionHead(nn.Module):
    """Separate projection head for SSL contrastive learning.

    Maps 256d embeddings to 128d contrastive space.
    NOT part of the LCNet class -- does not affect .encode() output.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.net(x), p=2, dim=1)


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
    ssl_loss: float
    spatial_loss: float
    mean_cosine: float


def run_train_epoch(
    model: LCNet,
    distill_loader: DataLoader,
    arcface_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler,
    teachers: dict[str, object],
    teacher_weights: dict[str, float],
    device: torch.device,
    amp: bool,
    arc_margin: ArcMarginProduct | None,
    arc_loss_weight: float,
    drop_hard_ratio: float = 0.0,
    vat_weight: float = 0.0,
    vat_epsilon: float = 0.1,
    sep_weight: float = 0.0,
    blacklist_class_indices: set[int] | None = None,
    wl_centroid_ema: dict | None = None,
    backbone_unfrozen: bool = False,
    save_first_batch_path: Path | None = None,
    ssl_weight: float = 0.0,
    ssl_head: "SSLProjectionHead | None" = None,
    info_nce: "InfoNCELoss | None" = None,
    train_transform=None,
    radio_proj_heads: nn.ModuleDict | None = None,
    radio_adaptors: list[str] | None = None,
    radio_variant: str | None = None,
    radio_cache_base: str | None = None,
    spatial_distill_weight: float = 0.0,
    spatial_adapter: "SpatialAdapter | None" = None,
    spatial_radio_teacher: "RADIOTeacher | None" = None,
    spatial_grid_h: int = 0,
    spatial_grid_w: int = 0,
    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> EpochStats:
    """Run one training epoch with separate distillation and ArcFace data."""
    model.train()
    if ssl_head is not None:
        ssl_head.train()
    if radio_proj_heads is not None:
        radio_proj_heads.train()
    if spatial_adapter is not None:
        spatial_adapter.train()

    total_loss = 0.0
    total_distill = 0.0
    total_arc = 0.0
    total_vat = 0.0
    total_sep = 0.0
    total_ssl = 0.0
    total_spatial = 0.0
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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            student_emb = model.forward_embeddings_train(images)

            # --- Distillation Loss (multi-teacher weighted, per D-07) ---
            total_distill_loss = torch.tensor(0.0, device=device)
            first_cos_mean = 0.0
            _loss_idx = 0

            # Non-RADIO teachers: use existing load_teacher_embeddings
            for t_name, t_weight in teacher_weights.items():
                if t_name.startswith("radio_"):
                    continue  # RADIO handled below
                t_cache_dir = TEACHER_REGISTRY[t_name]["cache_dir"]
                t_emb = load_teacher_embeddings(paths, teachers[t_name], device, t_cache_dir, teacher_name=t_name)
                if hasattr(model, 'proj_heads') and model.proj_heads is not None:
                    backbone_feat = model.forward_backbone(images)
                    s_proj = model.proj_heads[t_name](backbone_feat)
                    s_proj = functional.normalize(s_proj, p=2, dim=1)
                else:
                    s_proj = student_emb  # single teacher, already projected + normalized
                t_emb = t_emb.to(device=device, dtype=s_proj.dtype)
                cos = functional.cosine_similarity(s_proj, t_emb, dim=1)
                total_distill_loss = total_distill_loss + t_weight * (1.0 - cos).mean()
                if _loss_idx == 0:
                    first_cos_mean = float(cos.mean().item())
                _loss_idx += 1

            # RADIO teachers: per-adaptor distillation via cached summaries
            if radio_proj_heads is not None and radio_adaptors and radio_variant and radio_cache_base:
                backbone_feat = model.forward_backbone(images)
                for r_name in [n for n in teacher_weights if n.startswith("radio_")]:
                    r_weight = teacher_weights[r_name]
                    # Weighted sum across all active adaptors (equal weight within a RADIO teacher)
                    adaptor_weight = r_weight / len(radio_adaptors)
                    for adaptor in radio_adaptors:
                        cache_dir = str(Path(radio_cache_base) / f"radio_{radio_variant}" / adaptor)
                        r_emb = load_radio_teacher_embeddings(paths, adaptor, cache_dir, device)
                        r_proj = radio_proj_heads[adaptor](backbone_feat)
                        r_proj = functional.normalize(r_proj, p=2, dim=1)
                        r_emb = r_emb.to(device=device, dtype=r_proj.dtype)
                        cos = functional.cosine_similarity(r_proj, r_emb, dim=1)
                        total_distill_loss = total_distill_loss + adaptor_weight * (1.0 - cos).mean()
                        if _loss_idx == 0:
                            first_cos_mean = float(cos.mean().item())
                        _loss_idx += 1

            distill_loss = total_distill_loss
            batch_align = first_cos_mean

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

            # --- SSL Contrastive Loss (dual-view, student-only) ---
            ssl_loss_val = torch.tensor(0.0, device=device)
            if ssl_weight > 0 and ssl_head is not None and info_nce is not None and train_transform is not None:
                # view_a = student_emb (from normal training forward pass, already computed above)
                # view_b = re-augment raw images from paths
                view_b_imgs = []
                for p in paths:
                    img = Image.open(p).convert("RGB")
                    view_b_imgs.append(train_transform(img))
                view_b_tensor = torch.stack(view_b_imgs).to(device, non_blocking=True)

                emb_b = model.forward_embeddings_train(view_b_tensor)

                # Project to contrastive space
                z_a = ssl_head(student_emb)  # gradients flow to encoder via student_emb
                z_b = ssl_head(emb_b)

                ssl_loss_val = info_nce(z_a, z_b)

            # --- Spatial Distillation Loss (on-the-fly RADIO spatial features) ---
            spatial_loss_val = torch.tensor(0.0, device=device)
            if spatial_distill_weight > 0 and spatial_adapter is not None and spatial_radio_teacher is not None:
                # Get student spatial features (pre-GAP, [B, C, H_s, W_s])
                student_spatial, _summary = model.forward_features(images)

                # Un-normalize images from ImageNet normalization back to [0, 1] for RADIO
                _mean = torch.tensor(imagenet_mean, device=device).view(1, 3, 1, 1)
                _std = torch.tensor(imagenet_std, device=device).view(1, 3, 1, 1)
                images_01 = images * _std + _mean  # reverse normalization to [0, 1]
                images_01 = images_01.clamp(0.0, 1.0)

                # Extract teacher spatial features on-the-fly (no disk caching)
                teacher_spatial = spatial_radio_teacher.extract_spatial_batch(
                    images_01, adaptor=radio_adaptors[0] if radio_adaptors else "backbone"
                )

                spatial_loss_val = spatial_distillation_loss(
                    student_spatial=student_spatial,
                    teacher_spatial=teacher_spatial,
                    adapter=spatial_adapter,
                    teacher_grid_h=spatial_grid_h,
                    teacher_grid_w=spatial_grid_w,
                )
                if n == 0:
                    logger.info(f"spatial_distill_loss={spatial_loss_val.item():.6f}")

            loss = distill_loss + arc_loss_weight * arc_loss + sep_weight * l_sep + ssl_weight * ssl_loss_val + spatial_distill_weight * spatial_loss_val

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
        total_ssl += float(ssl_loss_val.item())
        total_spatial += float(spatial_loss_val.item())
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
            ssl_str = f" - ssl: {total_ssl / n:.4f}" if ssl_weight > 0 else ""
            spatial_str = f" - spatial: {total_spatial / n:.4f}" if spatial_distill_weight > 0 else ""
            print(f"\r  {n}/{total_steps} [{bar}] - loss: {avg_loss:.4f} - cos: {avg_cos:.4f}{ssl_str}{spatial_str}", end="", flush=True)
    print()  # newline after epoch

    return EpochStats(
        loss=total_loss / max(n, 1),
        distill_loss=total_distill / max(n, 1),
        arc_loss=total_arc / max(n, 1),
        vat_loss=total_vat / max(n, 1),
        sep_loss=total_sep / max(n, 1),
        ssl_loss=total_ssl / max(n, 1),
        spatial_loss=total_spatial / max(n, 1),
        mean_cosine=total_align / max(n, 1),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Build training augmentations (agent controls these)
    train_transform = build_train_transform(IMAGE_SIZE)
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

    # Resolve active teachers (per D-03, D-05)
    teacher_names, teacher_weights = _get_active_teachers()

    # Collect all image paths for cache building
    all_image_paths = [s[0] for s in distill_dataset.samples]
    if distill_dataset.retail_samples:
        all_image_paths += [s[0] for s in distill_dataset.retail_samples]

    # Identify which teachers are RADIO-based
    radio_teacher_names = [n for n in teacher_names if n.startswith("radio_")]
    non_radio_teacher_names = [n for n in teacher_names if not n.startswith("radio_")]

    # Build RADIO caches first (per D-09, D-10) -- needs GPU, builds per-adaptor caches
    radio_adaptor_dims: dict[str, int] = {}  # adaptor_name -> summary_dim
    if radio_teacher_names:
        logger.info(f"Building RADIO summary caches: variant={RADIO_VARIANT}, adaptors={RADIO_ADAPTORS}")
        radio_adaptor_dims = build_radio_summary_cache(
            variant=RADIO_VARIANT,
            adaptor_names=RADIO_ADAPTORS,
            image_paths=all_image_paths,
            cache_base=RADIO_CACHE_BASE,
            batch_size=32,
            device=str(device),
        )
        logger.info(f"RADIO adaptor dims: {radio_adaptor_dims}")

    # Read RADIO metadata for projection head dims (from cached metadata.json)
    if radio_teacher_names and not radio_adaptor_dims:
        radio_adaptor_dims = _load_radio_metadata(RADIO_VARIANT, RADIO_ADAPTORS, RADIO_CACHE_BASE)

    # Validate non-RADIO teacher dimensions
    teacher_dims: dict[str, int] = {}
    for name in non_radio_teacher_names:
        dim = TEACHER_REGISTRY[name]["embedding_dim"]
        if dim is None:
            raise ValueError(f"Teacher {name} has no embedding_dim set -- is it a stub?")
        teacher_dims[name] = dim

    # For RADIO teachers, use backbone adaptor's dim as the "teacher dim" for LCNet proj_heads
    # Each RADIO adaptor gets its own projection head via radio_proj_heads (below)
    for name in radio_teacher_names:
        # Use first adaptor dim as teacher_dims entry for backward compat
        first_adaptor = RADIO_ADAPTORS[0]
        teacher_dims[name] = radio_adaptor_dims[first_adaptor]

    # Build non-RADIO caches sequentially (per D-11, VRAM safety)
    if non_radio_teacher_names:
        build_all_teacher_caches(non_radio_teacher_names, all_image_paths, device=str(device))

    # Initialize non-RADIO teachers for online inference during training
    # RADIO teachers use pre-cached embeddings only (no online inference needed)
    if non_radio_teacher_names:
        teachers_dict = init_teachers(non_radio_teacher_names, device=str(device))
    else:
        teachers_dict = {}
    logger.info(f"Teachers: {teacher_names} weights={teacher_weights}")
    if radio_teacher_names:
        logger.info(f"RADIO teachers use cached embeddings: variant={RADIO_VARIANT}, adaptors={RADIO_ADAPTORS}")

    # Model
    if len(teacher_names) > 1:
        model = LCNet(
            scale=LCNET_SCALE,
            se_start_block=SE_START_BLOCK,
            se_reduction=SE_REDUCTION,
            activation=ACTIVATION,
            kernel_sizes=KERNEL_SIZES,
            embedding_dim=EMBEDDING_DIM,
            device=str(device),
            teacher_dims=teacher_dims,
        ).to(device)
    else:
        model = LCNet(
            scale=LCNET_SCALE,
            se_start_block=SE_START_BLOCK,
            se_reduction=SE_REDUCTION,
            activation=ACTIVATION,
            kernel_sizes=KERNEL_SIZES,
            embedding_dim=EMBEDDING_DIM,
            device=str(device),
        ).to(device)

    if USE_PRETRAINED:
        load_pretrained_lcnet(model, LCNET_SCALE)

    # RADIO per-adaptor projection heads (per D-05)
    # Each adaptor gets a Linear projection from its summary_dim to EMBEDDING_DIM
    # Dims read from metadata.json (never hardcoded)
    radio_proj_heads: nn.ModuleDict | None = None
    if radio_teacher_names and radio_adaptor_dims:
        radio_proj_heads = nn.ModuleDict({
            adaptor: nn.Sequential(
                nn.Linear(summary_dim, EMBEDDING_DIM),
                nn.BatchNorm1d(EMBEDDING_DIM),
            )
            for adaptor, summary_dim in radio_adaptor_dims.items()
        }).to(device)
        logger.info(f"RADIO projection heads: {dict(radio_adaptor_dims)} -> {EMBEDDING_DIM}")

    # Spatial distillation components (per D-06, D-07, D-08)
    spatial_adapter: SpatialAdapter | None = None
    spatial_radio_teacher: RADIOTeacher | None = None
    spatial_grid_h = 0
    spatial_grid_w = 0
    if SPATIAL_DISTILL_WEIGHT > 0:
        # Load RADIO teacher for on-the-fly spatial extraction (lazy init, only when needed)
        spatial_radio_teacher = RADIOTeacher(
            variant=RADIO_VARIANT,
            adaptor_names=RADIO_ADAPTORS,
            device=str(device),
        )
        spatial_info = spatial_radio_teacher.get_spatial_info(RADIO_ADAPTORS[0])
        spatial_grid_h = spatial_info["grid_h"]
        spatial_grid_w = spatial_info["grid_w"]
        radio_spatial_dim = spatial_info["spatial_dim"]

        # Get student spatial channels (from LCNet last stage output)
        # The student spatial features are pre-conv_head, last stage output channels
        # For LCNet with scale=0.5, last stage has make_divisible(512*0.5) = 256 channels
        student_spatial_ch = make_divisible(512 * LCNET_SCALE)
        spatial_adapter = SpatialAdapter(student_spatial_ch, radio_spatial_dim).to(device)
        logger.info(
            f"Spatial distillation enabled: weight={SPATIAL_DISTILL_WEIGHT}, "
            f"student_ch={student_spatial_ch}, radio_dim={radio_spatial_dim}, "
            f"grid={spatial_grid_h}x{spatial_grid_w}"
        )

    # SSL components (per D-01, D-02, D-04)
    ssl_head: SSLProjectionHead | None = None
    info_nce_loss: InfoNCELoss | None = None
    if SSL_WEIGHT > 0:
        ssl_head = SSLProjectionHead(
            in_dim=EMBEDDING_DIM,
            hidden_dim=SSL_PROJ_DIM,
            out_dim=SSL_PROJ_DIM,
        ).to(device)
        info_nce_loss = InfoNCELoss(temperature=SSL_TEMPERATURE).to(device)
        logger.info(f"SSL enabled: weight={SSL_WEIGHT}, temperature={SSL_TEMPERATURE}, proj_dim={SSL_PROJ_DIM}")

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

    # Unfreeze backbone (per D-02, UNFREEZE_EPOCH=0 means unfreeze from start)
    backbone_unfrozen = False
    if UNFREEZE_EPOCH == 0:
        model.unfreeze_last_stage()
        backbone_unfrozen = True

    # Optimizer with differential LR
    # Backbone: conv_stem, bn1, blocks (frozen initially, unfrozen at UNFREEZE_EPOCH)
    # Head: proj (or proj_heads in multi-teacher mode), conv_head
    backbone_params = [p for p in list(model.conv_stem.parameters()) +
                       list(model.bn1.parameters()) +
                       list(model.blocks.parameters()) if p.requires_grad]
    if hasattr(model, 'proj_heads') and model.proj_heads is not None:
        proj_params: list = []
        for ph in model.proj_heads.values():
            proj_params += list(ph.parameters())
    else:
        proj_params = list(model.proj.parameters())
    head_params = proj_params + list(model.conv_head.parameters()) + list(model.head_act.parameters())
    if radio_proj_heads is not None:
        head_params += list(radio_proj_heads.parameters())
    if spatial_adapter is not None:
        head_params += list(spatial_adapter.parameters())
    if arc_margin is not None:
        head_params += list(arc_margin.parameters())
    if ssl_head is not None:
        head_params += list(ssl_head.parameters())
    if info_nce_loss is not None:
        head_params += list(info_nce_loss.parameters())  # learnable temperature

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

    # SWA: accumulate model weights from last SWA_EPOCHS epochs
    SWA_EPOCHS = 3
    swa_start = EPOCHS - SWA_EPOCHS
    swa_state: dict | None = None
    swa_count = 0

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
            backbone_params = [p for p in list(model.conv_stem.parameters()) +
                               list(model.bn1.parameters()) +
                               list(model.blocks.parameters()) if p.requires_grad]
            if hasattr(model, 'proj_heads') and model.proj_heads is not None:
                proj_params_rebuild: list = []
                for ph in model.proj_heads.values():
                    proj_params_rebuild += list(ph.parameters())
            else:
                proj_params_rebuild = list(model.proj.parameters())
            head_params = proj_params_rebuild + list(model.conv_head.parameters()) + list(model.head_act.parameters())
            if radio_proj_heads is not None:
                head_params += list(radio_proj_heads.parameters())
            if spatial_adapter is not None:
                head_params += list(spatial_adapter.parameters())
            if arc_margin is not None:
                head_params += list(arc_margin.parameters())
            if ssl_head is not None:
                head_params += list(ssl_head.parameters())
            if info_nce_loss is not None:
                head_params += list(info_nce_loss.parameters())
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
            teachers=teachers_dict,
            teacher_weights=teacher_weights,
            device=device,
            amp=(device.type == "cuda"),
            arc_margin=arc_margin,
            arc_loss_weight=effective_arc_weight,
            drop_hard_ratio=DROP_HARD_RATIO,
            vat_weight=VAT_WEIGHT,
            vat_epsilon=VAT_EPSILON,
            sep_weight=SEP_WEIGHT,
            blacklist_class_indices=distill_dataset.blacklist_class_indices,
            wl_centroid_ema=wl_centroid_ema,
            backbone_unfrozen=backbone_unfrozen,
            save_first_batch_path=first_batch_path,
            ssl_weight=SSL_WEIGHT,
            ssl_head=ssl_head,
            info_nce=info_nce_loss,
            train_transform=train_transform,
            radio_proj_heads=radio_proj_heads,
            radio_adaptors=RADIO_ADAPTORS if radio_teacher_names else None,
            radio_variant=RADIO_VARIANT if radio_teacher_names else None,
            radio_cache_base=RADIO_CACHE_BASE if radio_teacher_names else None,
            spatial_distill_weight=SPATIAL_DISTILL_WEIGHT,
            spatial_adapter=spatial_adapter,
            spatial_radio_teacher=spatial_radio_teacher,
            spatial_grid_h=spatial_grid_h,
            spatial_grid_w=spatial_grid_w,
        )
        elapsed_epoch = time.time() - t0

        arc_w_str = f" arc_w={effective_arc_weight:.4f}" if effective_arc_weight != ARCFACE_LOSS_WEIGHT else ""
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={stats.loss:.4f} distill={stats.distill_loss:.4f} "
            f"arc={stats.arc_loss:.4f} vat={stats.vat_loss:.4f} sep={stats.sep_loss:.4f} ssl={stats.ssl_loss:.4f} spatial={stats.spatial_loss:.4f} cosine={stats.mean_cosine:.4f}{arc_w_str} | "
            f"{elapsed_epoch:.1f}s"
        )

        # SWA: accumulate weights from last N epochs
        if epoch >= swa_start:
            sd = {k: v.clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = sd
            else:
                for k in swa_state:
                    swa_state[k] += sd[k]
            swa_count += 1
            logger.info(f"  SWA: accumulated epoch {epoch+1} ({swa_count}/{SWA_EPOCHS})")

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

    # Apply SWA averaged weights and re-evaluate
    if swa_state is not None and swa_count > 0:
        logger.info(f"Applying SWA weights (averaged over {swa_count} epochs)...")
        for k in swa_state:
            swa_state[k] /= swa_count
        model.load_state_dict(swa_state)

        # Re-evaluate with SWA weights
        if val_dataset is not None:
            # Get mean_cosine from a quick forward pass on distill data (use first/default teacher)
            default_t_name = teacher_names[0]
            default_t_cache = TEACHER_REGISTRY[default_t_name]["cache_dir"]
            model.eval()
            cos_sum, cos_n = 0.0, 0
            with torch.no_grad():
                for images, labels, paths in distill_loader:
                    images = images.to(device, non_blocking=True)
                    teacher_emb = load_teacher_embeddings(paths, teachers_dict[default_t_name], device, default_t_cache, teacher_name=default_t_name)
                    student_emb = model.encode(images)
                    teacher_emb = teacher_emb.to(device=device, dtype=student_emb.dtype)
                    cos = functional.cosine_similarity(student_emb, teacher_emb, dim=1)
                    cos_sum += cos.sum().item()
                    cos_n += len(cos)
            mean_cos = cos_sum / max(cos_n, 1)

            retrieval_metrics = run_retrieval_eval(
                model=model, dataset=val_dataset, device=device,
                amp=(device.type == "cuda"), max_samples=RETRIEVAL_MAX_SAMPLES,
                topk=RETRIEVAL_TOPK, seed=SEED, batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            recall_at_1 = retrieval_metrics["recall@1"]
            recall_at_5 = retrieval_metrics.get(f"recall@{RETRIEVAL_TOPK}", 0.0)
            combined = compute_combined_metric(recall_at_1, mean_cos)
            logger.info(f"SWA: recall@1={recall_at_1:.4f} mean_cos={mean_cos:.4f} combined={combined:.6f}")

    # Save model checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS,
        "combined_metric": combined,
        "recall_at_1": recall_at_1,
        "mean_cosine": mean_cos,
    }
    if arc_margin is not None:
        ckpt["arc_margin_state_dict"] = arc_margin.state_dict()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_dir / "checkpoint_last.pt")
    if not hasattr(main, "_best_combined") or combined > main._best_combined:
        main._best_combined = combined
        torch.save(ckpt, out_dir / "checkpoint_best.pt")
        logger.info(f"  -> New best combined metric: {combined:.6f}")

    # Compute final metrics
    elapsed = time.time() - t_start
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    final_distill_loss = stats.distill_loss
    final_arc_loss = stats.arc_loss
    final_vat_loss = stats.vat_loss
    final_sep_loss = stats.sep_loss
    final_ssl_loss = stats.ssl_loss
    final_spatial_loss = stats.spatial_loss

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
        "ssl_loss": final_ssl_loss,
        "spatial_distill_loss": final_spatial_loss,
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
    print(f"ssl_loss:         {final_ssl_loss:.6f}")
    print(f"spatial_distill_loss: {final_spatial_loss:.6f}")
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
