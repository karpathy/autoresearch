"""Immutable data/teacher/evaluation infrastructure for the autoresearch ReID pipeline.

This module owns all data loading, teacher inference/caching, evaluation, and metric
computation.  The agent cannot edit this file -- it is the trust boundary.
"""

from __future__ import annotations

import os
import site

# Ensure CUDA 12 libs from pip nvidia packages are on LD_LIBRARY_PATH (for onnxruntime-gpu)
_site_pkgs = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(os.path.dirname(os.__file__), "site-packages")
_nvidia_base = os.path.join(_site_pkgs, "nvidia")
if os.path.isdir(_nvidia_base):
    _lib_dirs = [
        os.path.join(_nvidia_base, sub, "lib")
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "cufft", "nvjitlink")
        if os.path.isdir(os.path.join(_nvidia_base, sub, "lib"))
    ]
    if _lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import hashlib
import numpy as np
import random
import torch
import timm
import torch.nn.functional as functional
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants (immutable)
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256
IMAGE_SIZE = 224
TRAIN_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/train"
VAL_DIR = "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/val"
ARCFACE_DIR = "/data/mnt/mnt_ml_shared/Vic/retail_product_checkout_crop"
REID_ROOT = Path("/data/mnt/mnt_ml_shared/joesu/reid/data/reid_train/train")
REID_PRODUCTS = str(REID_ROOT / "products")
REID_COMMODITY = str(REID_ROOT / "commodity")
REID_NEGATIVES = str(REID_ROOT / "negatives")
SKIP_CLASSES = {"0000000000"}
DEFAULT_TEACHER_CACHE_DIR = "workspace/output/trendyol_teacher_cache2"


# ---------------------------------------------------------------------------
# PadToSquare (correct version using TF.pad)
# ---------------------------------------------------------------------------

class PadToSquare:
    def __init__(self, color: int = 255) -> None:
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        if self.color != -1:
            padding = abs(width - height) // 2
            if width < height:
                return TF.pad(
                    img, (padding, 0, padding + (height - width) % 2, 0), fill=self.color, padding_mode="constant"
                )
            elif width > height:
                return TF.pad(
                    img, (0, padding, 0, padding + (width - height) % 2), fill=self.color, padding_mode="constant"
                )
        return img


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# TrendyolEmbedder (ONNX teacher model)
# ---------------------------------------------------------------------------

class TrendyolEmbedder:
    def __init__(
        self,
        onnx_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        if onnx_path is None:
            onnx_path = "/data/mnt/mnt_ml_shared/joesu/reid/distill_qwen_lcnet050_retail_2.onnx"
            # onnx_path = "/workspace/lcnet050_pfc_supcon_f256_224_20260309_onnx_fp32.onnx"

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        elif "CPUExecutionProvider" in available_providers:
            providers = ["CPUExecutionProvider"]
        else:
            providers = available_providers[:1]

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
        logger.info(f"TrendyolEmbedder: using provider {self.session.get_providers()[0]}")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        logger.info(f"TrendyolEmbedder: loaded ONNX model from {onnx_path}")

    def get_feature_dim(self) -> int:
        return 256

    def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
        if not images:
            return []

        try:
            pil_images = []
            for image in images:
                if isinstance(image, np.ndarray):
                    image_rgb = image[:, :, ::-1] if len(image.shape) == 3 and image.shape[2] == 3 else image
                    pil_image = Image.fromarray(image_rgb).convert("RGB")
                else:
                    pil_image = image.convert("RGB")
                pil_images.append(pil_image)

            input_tensors = np.stack([self.transform(img).numpy() for img in pil_images])

            input_name = self.session.get_inputs()[0].name
            embeddings = self.session.run(None, {input_name: input_tensors})[0]

            return [emb.flatten() for emb in embeddings]

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [None] * len(images)

    def extract_features(self, image_crops: list[np.ndarray]) -> np.ndarray:
        if not image_crops:
            return np.array([])

        embeddings = self.encode_batch(image_crops)
        valid_embeddings = [emb for emb in embeddings if emb is not None]

        if not valid_embeddings:
            return np.array([])

        return np.array(valid_embeddings, dtype=np.float32)

    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if features1.size == 0 or features2.size == 0:
            return 0.0

        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(features1, features2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# DINOv2Teacher kept for future use; v1 uses TrendyolEmbedder only (per D-03)
# ---------------------------------------------------------------------------

def _patch_transformers_compat() -> None:
    """Monkey-patch transformers 5.x compat for custom HF models."""
    from transformers import PreTrainedModel

    if getattr(PreTrainedModel, "_compat_patched", False):
        return
    _orig = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched(self: PreTrainedModel, loading_info: dict) -> None:  # type: ignore[override]
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, loading_info)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched  # type: ignore[assignment]
    PreTrainedModel._compat_patched = True  # type: ignore[attr-defined]


class DINOv2Teacher:
    """Trendyol DINO v2 as teacher. Same encode_batch interface as TrendyolEmbedder."""

    def __init__(self, model_name: str = "Trendyol/trendyol-dino-v2-ecommerce-256d", device: str = "cuda") -> None:
        import os
        from transformers import AutoModel

        os.environ["XFORMERS_DISABLED"] = "1"
        _patch_transformers_compat()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=False)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"DINOv2Teacher: loaded {model_name}, output_dim=256")

    @torch.no_grad()
    def encode_batch(self, images: list[np.ndarray | Image.Image]) -> list[np.ndarray | None]:
        if not images:
            return []
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        with torch.amp.autocast(self.device):
            out = self.model(batch)
        emb = out.last_hidden_state  # already L2-normalized 256d
        return [e.cpu().numpy() for e in emb]


# ---------------------------------------------------------------------------
# DistillImageFolder
# ---------------------------------------------------------------------------

class DistillImageFolder(datasets.ImageFolder):
    """ImageFolder that returns (image, label, path)."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        self.return_path = return_path

    def __getitem__(self, index: int) -> tuple[Image.Image, int] | tuple[Image.Image, int, str]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.return_path:
            return sample, target, path
        return sample, target


# ---------------------------------------------------------------------------
# SampledImageFolder
# ---------------------------------------------------------------------------

class SampledImageFolder(Dataset):
    """ImageFolder with max N samples per class."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        max_per_class: int = 100,
        return_path: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.max_per_class = max_per_class
        self.return_path = return_path

        # Find all classes
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Sample up to max_per_class from each class
        self.samples: list[tuple[str, int]] = []
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            image_files += list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPEG"))

            if len(image_files) > max_per_class:
                image_files = random.sample(image_files, max_per_class)

            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))

        logger.info(
            f"SampledImageFolder: {len(self.classes)} classes, {len(self.samples)} samples (max {max_per_class}/class)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, target, path
        return img, target


# ---------------------------------------------------------------------------
# CombinedDistillDataset
# ---------------------------------------------------------------------------

class CombinedDistillDataset(Dataset):
    """Mix product_code_dataset (train+val) with retail dataset for distillation.

    Uses random replacement: retail_ratio % of samples are replaced with retail.
    Returns (image, label, path) for teacher embedding lookup.
    """

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        transform: Callable | None = None,
        retail_ratio: float = 0.3,
        blacklist_root: str | None = None,
        blacklist_ratio: float = 0.0,
        skip_classes: set[str] | None = None,
        quality_degradation: Callable | None = None,
        skip_degradation_paths: list[str] | None = None,
    ) -> None:
        self.transform = transform
        self.retail_ratio = retail_ratio
        self.blacklist_ratio = blacklist_ratio
        self.quality_degradation = quality_degradation
        self.skip_degradation_paths = skip_degradation_paths or []
        self.samples: list[tuple[str, int]] = []
        self.retail_samples: list[tuple[str, int]] = []
        self.blacklist_samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        retail_path = Path(retail_root)

        # Collect all primary classes from all roots
        all_classes: set[str] = set()
        for root in primary_roots:
            root_path = Path(root)
            if root_path.exists():
                for d in root_path.iterdir():
                    if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                        all_classes.add(d.name)

        # Retail dataset classes (with prefix to avoid collision)
        retail_class_dirs: list[Path] = []
        if retail_path.exists():
            retail_class_dirs = sorted(
                [d for d in retail_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
            )
        retail_classes = [f"retail_{d.name}" for d in retail_class_dirs]

        # Blacklist classes (with prefix)
        bl_class_dirs: list[Path] = []
        if blacklist_root:
            bl_path = Path(blacklist_root)
            if bl_path.exists():
                bl_class_dirs = sorted(
                    [d for d in bl_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
                )
        bl_classes = [f"bl_{d.name}" for d in bl_class_dirs]

        self.classes = sorted(all_classes) + retail_classes + bl_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.blacklist_class_indices: set[int] = {self.class_to_idx[c] for c in bl_classes}

        # Load primary dataset samples from all roots (full)
        primary_count = 0
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            class_dirs = sorted(
                [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip]
            )
            for class_dir in class_dirs:
                class_idx = self.class_to_idx[class_dir.name]
                image_files = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.png"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.JPEG"))
                )
                for img_path in image_files:
                    self.samples.append((str(img_path), class_idx))
                    primary_count += 1

        # Load retail dataset samples (for random replacement)
        retail_count = 0
        for class_dir in retail_class_dirs:
            class_name = f"retail_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                self.retail_samples.append((str(img_path), class_idx))
                retail_count += 1

        # Load blacklist samples (capped to limit teacher cache misses)
        max_bl_samples = 50_000  # ~50K is enough for 10% ratio sampling
        all_bl: list[tuple[str, int]] = []
        for class_dir in bl_class_dirs:
            class_name = f"bl_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                all_bl.append((str(img_path), class_idx))
        if len(all_bl) > max_bl_samples:
            all_bl = random.sample(all_bl, max_bl_samples)
        self.blacklist_samples = all_bl
        bl_count = len(all_bl)

        logger.info(
            f"CombinedDistillDataset: {len(self.classes)} classes, "
            f"{len(self.samples)} primary, {len(self.retail_samples)} retail, "
            f"{len(self.blacklist_samples)} blacklist, "
            f"retail_ratio={retail_ratio}, blacklist_ratio={blacklist_ratio}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        # Randomly replace with blacklist or retail sample
        r = random.random()
        if self.blacklist_samples and r < self.blacklist_ratio:
            path, target = random.choice(self.blacklist_samples)
        elif self.retail_samples and r < self.blacklist_ratio + self.retail_ratio:
            path, target = random.choice(self.retail_samples)
        else:
            path, target = self.samples[index]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")

        # Apply quality degradation only if path is not in skip list
        if self.quality_degradation is not None:
            should_skip = any(skip_path in path for skip_path in self.skip_degradation_paths)
            if not should_skip:
                img = self.quality_degradation(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


# ---------------------------------------------------------------------------
# CombinedArcFaceDataset
# ---------------------------------------------------------------------------

class CombinedArcFaceDataset(Dataset):
    """Combine multiple primary datasets + retail dataset for ArcFace.

    Uses first primary_root's class IDs as the reference set.
    Other primary roots only include classes that exist in the reference set.
    """

    def __init__(
        self,
        primary_roots: list[str],
        retail_root: str,
        transform: Callable | None = None,
        retail_max_per_class: int = 100,
        skip_classes: set[str] | None = None,
        quality_degradation: Callable | None = None,
        skip_degradation_paths: list[str] | None = None,
    ) -> None:
        self.transform = transform
        self.quality_degradation = quality_degradation
        self.skip_degradation_paths = skip_degradation_paths or []
        self.samples: list[tuple[str, int]] = []
        _skip = skip_classes or set()

        if not primary_roots:
            raise ValueError("primary_roots must contain at least one path")

        # Collect reference class IDs from ALL primary roots (union)
        ref_class_ids = set()
        for root in primary_roots:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for d in root_path.iterdir():
                if d.is_dir() and not d.name.startswith((".", "@", "__")) and d.name not in _skip:
                    ref_class_ids.add(d.name)

        logger.info(f"Reference class IDs (union of {len(primary_roots)} roots): {len(ref_class_ids)} classes")

        # Collect all valid class directories from all primary roots
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

        # Deduplicate class names while keeping all directories
        seen_class_names = set()
        for d in primary_class_dirs:
            class_name = f"primary_{d.name}"
            if class_name not in seen_class_names:
                all_classes.append(class_name)
                seen_class_names.add(class_name)

        # Retail dataset classes
        retail_path = Path(retail_root)
        retail_class_dirs = sorted(
            [d for d in retail_path.iterdir() if d.is_dir() and not d.name.startswith((".", "@", "__"))]
        )
        retail_classes = [f"retail_{d.name}" for d in retail_class_dirs]
        all_classes.extend(retail_classes)

        self.classes = all_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

        # Load primary dataset samples from all roots
        primary_count = 0
        for class_dir in primary_class_dirs:
            class_name = f"primary_{class_dir.name}"
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                primary_count += 1

        # Load retail dataset samples (sampled)
        retail_count = 0
        for class_dir in retail_class_dirs:
            class_name = f"retail_{class_dir.name}"
            class_idx = self.class_to_idx[class_name]
            image_files = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.JPEG"))
            )
            if len(image_files) > retail_max_per_class:
                image_files = random.sample(image_files, retail_max_per_class)
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
                retail_count += 1

        logger.info(
            f"CombinedArcFaceDataset: {len(self.classes)} classes total "
            f"(primary={len(seen_class_names)}, retail={len(retail_classes)}), "
            f"{len(self.samples)} samples (primary={primary_count}, retail={retail_count})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            path, target = random.choice(self.samples)
            img = Image.open(path).convert("RGB")

        # Apply quality degradation only if path is not in skip list
        if self.quality_degradation is not None:
            should_skip = any(skip_path in path for skip_path in self.skip_degradation_paths)
            if not should_skip:
                img = self.quality_degradation(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


# ---------------------------------------------------------------------------
# Collation functions
# ---------------------------------------------------------------------------

def collate_distill(
    batch: Sequence[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate for distillation dataset (img, label, path)."""
    imgs, labels, paths = [], [], []
    for img, label, path in batch:
        imgs.append(img)
        labels.append(int(label))
        paths.append(path)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), paths


def collate_arcface(
    batch: Sequence[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate for ArcFace dataset (img, label, path)."""
    imgs, labels, paths = [], [], []
    for img, label, path in batch:
        imgs.append(img)
        labels.append(int(label))
        paths.append(path)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), paths


# ---------------------------------------------------------------------------
# Teacher embedding cache
# ---------------------------------------------------------------------------

_TEACHER_MEM_CACHE: dict[str, np.ndarray] = {}


def load_teacher_embeddings(
    image_paths: Sequence[str],
    teacher: TrendyolEmbedder,
    device: torch.device,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Load teacher embeddings with in-memory + disk caching and batch inference."""
    embeddings: list[np.ndarray] = [None] * len(image_paths)  # type: ignore[list-item]

    # Phase 1: in-memory cache -> disk cache
    uncached_indices: list[int] = []
    for i, path in enumerate(image_paths):
        if path in _TEACHER_MEM_CACHE:
            embeddings[i] = _TEACHER_MEM_CACHE[path]
            continue
        if cache_dir:
            cache_path = Path(cache_dir) / f"{hashlib.md5(path.encode()).hexdigest()}.npy"
            if cache_path.exists():
                emb = np.load(cache_path)
                _TEACHER_MEM_CACHE[path] = emb
                embeddings[i] = emb
                continue
        uncached_indices.append(i)

    # Phase 2: batch inference for uncached images
    if uncached_indices:
        pil_images = [Image.open(image_paths[i]).convert("RGB") for i in uncached_indices]
        emb_list = teacher.encode_batch(pil_images)

        for j, i in enumerate(uncached_indices):
            emb = emb_list[j]
            embeddings[i] = emb
            _TEACHER_MEM_CACHE[image_paths[i]] = emb
            if cache_dir and emb is not None:
                cache_path = Path(cache_dir) / f"{hashlib.md5(image_paths[i].encode()).hexdigest()}.npy"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, emb)

    return torch.tensor(np.stack(embeddings), device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_retrieval_eval(
    model,
    dataset: datasets.ImageFolder,
    device: torch.device,
    amp: bool,
    max_samples: int,
    topk: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Run retrieval evaluation on a validation dataset."""
    n_total = len(dataset)
    n_use = min(max_samples, n_total)
    if n_use < 2:
        raise ValueError(f"Not enough samples for retrieval eval: n={n_use}")

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)[:n_use].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.eval()
    embs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            e = model.encode(images)
        embs.append(e.detach().float().cpu())
        labels.append(y.detach().cpu())

    emb = torch.cat(embs, dim=0)
    lab = torch.cat(labels, dim=0)
    emb = functional.normalize(emb, dim=1)

    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))

    k = min(int(topk), sim.size(1) - 1)
    _, nn_idx = torch.topk(sim, k=k, dim=1)
    nn_lab = lab[nn_idx]

    correct = (nn_lab == lab.view(-1, 1)).any(dim=1).float()
    recall_at_k = float(correct.mean().item())

    _, nn1_idx = torch.topk(sim, k=1, dim=1)
    nn1_lab = lab[nn1_idx.squeeze(1)]
    recall_at_1 = float((nn1_lab == lab).float().mean().item())

    return {
        "retrieval_n": float(n_use),
        "recall@1": recall_at_1,
        f"recall@{k}": recall_at_k,
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_combined_metric(recall_at_1: float, mean_cosine: float) -> float:
    """Compute the single combined metric: 0.5 * recall@1 + 0.5 * mean_cosine."""
    return 0.5 * recall_at_1 + 0.5 * mean_cosine


# ---------------------------------------------------------------------------
# Convenience wrappers / builder functions
# ---------------------------------------------------------------------------

def build_val_transform(model_name: str, image_size: int) -> transforms.Compose:
    """Build validation/evaluation transform (immutable)."""
    tmp_model = timm.create_model(model_name, pretrained=True)
    data_config = resolve_data_config(tmp_model.pretrained_cfg)
    mean = data_config["mean"]
    std = data_config["std"]
    del tmp_model
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def init_teacher(device: str = "cuda") -> TrendyolEmbedder:
    """Initialize the ONNX teacher model. Per D-03/D-04, always TrendyolEmbedder."""
    return TrendyolEmbedder(device=device)


def build_distill_dataset(
    transform: Callable,
    quality_degradation: Callable,
    blacklist_ratio: float = 0.10,
) -> CombinedDistillDataset:
    """Build the distillation dataset. Data paths are immutable."""
    return CombinedDistillDataset(
        primary_roots=[REID_PRODUCTS, REID_COMMODITY],
        retail_root=ARCFACE_DIR,
        blacklist_root=REID_NEGATIVES,
        blacklist_ratio=blacklist_ratio,
        skip_classes=SKIP_CLASSES,
        transform=transform,
        quality_degradation=quality_degradation,
        skip_degradation_paths=[],
    )


def build_arcface_dataset(
    transform: Callable,
    quality_degradation: Callable,
    skip_extra_classes: set[str] | None = None,
    max_per_class: int = 100,
) -> tuple[CombinedArcFaceDataset, int]:
    """Build the ArcFace dataset. Returns (dataset, num_classes).

    Automatically excludes val barcodes from ArcFace training to prevent data leakage.
    """
    val_dir = Path(VAL_DIR)
    val_barcodes = {
        d.name for d in val_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    } if val_dir.exists() else set()
    arcface_skip = SKIP_CLASSES | val_barcodes
    if skip_extra_classes:
        arcface_skip |= skip_extra_classes
    logger.info(f"ArcFace: skipping {len(arcface_skip)} classes (1 empty + {len(val_barcodes)} val barcodes)")
    dataset = CombinedArcFaceDataset(
        primary_roots=[REID_PRODUCTS],
        retail_root=ARCFACE_DIR,
        transform=transform,
        retail_max_per_class=max_per_class,
        skip_classes=arcface_skip,
        quality_degradation=quality_degradation,
        skip_degradation_paths=[],
    )
    return dataset, len(dataset.classes)


def build_val_dataset(model_name: str, image_size: int) -> datasets.ImageFolder | None:
    """Build the validation dataset for retrieval evaluation."""
    val_dir = Path(VAL_DIR)
    if not val_dir.exists():
        logger.warning(f"Validation directory not found: {val_dir}")
        return None
    val_transform = build_val_transform(model_name, image_size)
    dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    logger.info(f"Validation dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
    return dataset
