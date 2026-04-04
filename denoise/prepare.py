"""
Image Denoising Benchmark — Data & Evaluation Harness
DO NOT MODIFY (agent reads only)

24 Kodak test images (or synthetic fallback):
  - 20 images for training (random patch extraction)
  - 4 images for validation (fixed noise, deterministic)

Primary metric: val_psnr (dB) — higher is better.
Visual output: samples/latest.png — noisy | denoised | clean grid.

Usage:
    uv run prepare.py          # download / prepare data (one-time)
"""

import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Constants (DO NOT CHANGE)
# ──────────────────────────────────────────────────────────────
NOISE_SIGMA   = 25.0 / 255.0     # Gaussian noise σ in [0,1] scale
PATCH_SIZE    = 64                # training patch size
TIME_BUDGET   = 300               # 5-minute wall-clock training budget
SEED          = 42                # reproducible evaluation noise
CACHE_DIR     = Path.home() / ".cache" / "autoresearch-denoise"
NUM_TRAIN     = 20                # kodim01–20
NUM_VAL       = 4                 # kodim21–24
VAL_CROP      = 256               # center-crop val images to this
PATCHES_PER_IMAGE = 200           # random patches per training image

# ──────────────────────────────────────────────────────────────
# Data Download / Generation
# ──────────────────────────────────────────────────────────────

def _download_kodak():
    """Download 24 Kodak PNG images → list of (H,W,3) float32 [0,1]."""
    import requests

    img_dir = CACHE_DIR / "kodak"
    img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for i in range(1, 25):
        path = img_dir / f"kodim{i:02d}.png"
        if not path.exists():
            url = f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
            print(f"  Downloading kodim{i:02d}.png ...")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                return None
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        images.append(img)
    return images


def _generate_synthetic(n=24, h=512, w=768):
    """Fallback: smooth colour images built from upsampled random fields."""
    print("  Kodak download failed — generating synthetic images …")
    rng = np.random.RandomState(SEED)
    images = []
    for _ in range(n):
        # low-freq base
        lo = rng.rand(h // 32, w // 32, 3).astype(np.float32)
        lo_up = np.stack([
            np.array(Image.fromarray((lo[:, :, c] * 255).astype(np.uint8), "L")
                     .resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
            for c in range(3)
        ], axis=-1)
        # mid-freq detail
        mi = rng.rand(h // 4, w // 4, 3).astype(np.float32)
        mi_up = np.stack([
            np.array(Image.fromarray((mi[:, :, c] * 255).astype(np.uint8), "L")
                     .resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
            for c in range(3)
        ], axis=-1)
        images.append(np.clip(0.7 * lo_up + 0.3 * mi_up, 0, 1))
    return images


def prepare_data():
    """Download (or generate) images, split train/val, cache as .npy."""
    if (CACHE_DIR / "ready").exists():
        print("Data already prepared.")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    images = _download_kodak()
    if images is None or len(images) < 24:
        images = _generate_synthetic()

    with open(CACHE_DIR / "train_images.pkl", "wb") as f:
        pickle.dump(images[:NUM_TRAIN], f)
    with open(CACHE_DIR / "val_images.pkl", "wb") as f:
        pickle.dump(images[NUM_TRAIN:NUM_TRAIN + NUM_VAL], f)
    (CACHE_DIR / "ready").touch()
    print(f"Data ready: {NUM_TRAIN} train, {NUM_VAL} val images  →  {CACHE_DIR}")


# ──────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────

def load_train_images():
    with open(CACHE_DIR / "train_images.pkl", "rb") as f:
        return pickle.load(f)

def load_val_images():
    with open(CACHE_DIR / "val_images.pkl", "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────
# Patch Dataset + DataLoader
# ──────────────────────────────────────────────────────────────

def extract_patches(images, patch_size=PATCH_SIZE, per_image=PATCHES_PER_IMAGE):
    """list of (H,W,3) arrays → (N_patches, 3, pH, pW) float32"""
    rng = np.random.RandomState(SEED)
    patches = []
    for img in images:
        h, w = img.shape[:2]
        for _ in range(per_image):
            y = rng.randint(0, h - patch_size)
            x = rng.randint(0, w - patch_size)
            patches.append(img[y:y + patch_size, x:x + patch_size].transpose(2, 0, 1))
    return np.array(patches, dtype=np.float32)


class DenoiseDataset(torch.utils.data.Dataset):
    """Returns (noisy, clean) pairs with on-the-fly noise + augmentation."""

    def __init__(self, clean_patches, sigma=NOISE_SIGMA, augment=True):
        self.clean = torch.from_numpy(clean_patches)
        self.sigma = sigma
        self.augment = augment

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = self.clean[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                clean = clean.flip(-1)
            if torch.rand(1).item() > 0.5:
                clean = clean.flip(-2)
            k = torch.randint(0, 4, (1,)).item()
            if k:
                clean = torch.rot90(clean, k, [-2, -1])
        noisy = torch.clamp(clean + torch.randn_like(clean) * self.sigma, 0, 1)
        return noisy, clean


def get_train_loader(batch_size=32):
    patches = extract_patches(load_train_images())
    ds = DenoiseDataset(patches)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_psnr(clean: np.ndarray, other: np.ndarray) -> float:
    mse = np.mean((clean - other) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


# ──────────────────────────────────────────────────────────────
# Model Evaluation  (called from train.py)
# ──────────────────────────────────────────────────────────────

def _center_crop(img, size):
    h, w = img.shape[:2]
    s = min(size, h, w)
    s = (s // 16) * 16  # multiple of 16
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


@torch.no_grad()
def evaluate_model(model, device) -> float:
    """
    Run model on fixed validation set.
    Returns average PSNR (dB).
    Saves visual grid → samples/latest.png
    """
    model.eval()
    val_imgs = load_val_images()
    rng = np.random.RandomState(SEED + 1000)

    rows = []          # (noisy_np, denoised_np, clean_np)
    psnrs_noisy = []
    psnrs_denoised = []

    for img in val_imgs:
        img = _center_crop(img, VAL_CROP)
        noise = rng.randn(*img.shape).astype(np.float32) * NOISE_SIGMA
        noisy = np.clip(img + noise, 0, 1)

        inp = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)
        out = model(inp).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out = np.clip(out, 0, 1)

        psnrs_noisy.append(compute_psnr(img, noisy))
        psnrs_denoised.append(compute_psnr(img, out))
        rows.append((noisy, out, img))

    avg_noisy  = float(np.mean(psnrs_noisy))
    avg_psnr   = float(np.mean(psnrs_denoised))

    _save_grid(rows, avg_psnr, avg_noisy)
    model.train()
    return avg_psnr


def _save_grid(rows, avg_psnr, avg_noisy):
    os.makedirs("samples", exist_ok=True)
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (noisy, denoised, clean) in enumerate(rows):
        pn = compute_psnr(clean, noisy)
        pd = compute_psnr(clean, denoised)
        for j, (im, label) in enumerate(zip(
            [noisy, denoised, clean],
            [f"Noisy ({pn:.2f} dB)", f"Denoised ({pd:.2f} dB)", "Clean"],
        )):
            axes[i, j].imshow(np.clip(im, 0, 1))
            axes[i, j].set_title(label, fontsize=12)
            axes[i, j].axis("off")

    fig.suptitle(
        f"val_psnr: {avg_psnr:.2f} dB    (noisy baseline: {avg_noisy:.2f} dB)",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("samples/latest.png", dpi=100, bbox_inches="tight")

    # commit-stamped copy
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        plt.savefig(f"samples/{h}.png", dpi=100, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("  Visual comparison → samples/latest.png")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prepare_data()
