"""Pre-build all teacher embedding caches before training.

Usage: python build_caches.py [teacher_names...]
  No args = build all teachers
  With args = build only specified teachers (e.g., python build_caches.py dinov3_ft radio_so400m)
"""

import sys
import torch
from pathlib import Path
from loguru import logger

from prepare import (
    build_all_teacher_caches, build_radio_summary_cache,
    build_distill_dataset, build_val_dataset,
    TEACHER_REGISTRY, IMAGE_SIZE,
    set_seed,
)

# All available teachers (non-RADIO + RADIO variants)
NON_RADIO_TEACHERS = ["trendyol_onnx", "dinov2", "dinov3_ft"]
RADIO_TEACHERS = {
    "radio_so400m": {"variant": "so400m", "adaptors": ["backbone", "dino_v3", "siglip2-g"]},
    "radio_h": {"variant": "h", "adaptors": ["backbone", "dino_v3", "siglip2-g"]},
}

MODEL_NAME = "lcnet_100"  # For val dataset transform


def collect_all_image_paths() -> list[str]:
    """Collect all image paths from distill + val datasets."""
    from torchvision import transforms

    # Minimal transform just to build the dataset objects
    dummy_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Quality degradation placeholder (identity)
    identity_degrade = lambda img: img

    distill_dataset = build_distill_dataset(dummy_transform, identity_degrade)
    all_paths = [s[0] if isinstance(s, (tuple, list)) else s for s in distill_dataset.samples]

    # Add val paths
    val_dataset = build_val_dataset(MODEL_NAME, IMAGE_SIZE)
    if val_dataset is not None:
        val_paths = [s[0] for s in val_dataset.samples]
        all_paths.extend(val_paths)

    # Deduplicate
    all_paths = list(dict.fromkeys(all_paths))
    logger.info(f"Total unique image paths: {len(all_paths)}")
    return all_paths


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse which teachers to build
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    all_teacher_names = NON_RADIO_TEACHERS + list(RADIO_TEACHERS.keys())

    if requested:
        for t in requested:
            if t not in all_teacher_names:
                logger.error(f"Unknown teacher: {t}. Available: {all_teacher_names}")
                sys.exit(1)
        teachers_to_build = requested
    else:
        teachers_to_build = all_teacher_names

    logger.info("=" * 60)
    logger.info("Building teacher embedding caches")
    logger.info("=" * 60)
    logger.info(f"Teachers: {teachers_to_build}")
    logger.info(f"Device: {device}")

    # Collect all image paths
    image_paths = collect_all_image_paths()

    # Build RADIO caches first (if requested)
    for radio_name, radio_cfg in RADIO_TEACHERS.items():
        if radio_name not in teachers_to_build:
            continue
        logger.info(f"\n{'='*40}")
        logger.info(f"Building RADIO cache: {radio_name}")
        logger.info(f"Variant: {radio_cfg['variant']}, Adaptors: {radio_cfg['adaptors']}")
        logger.info(f"{'='*40}")
        cache_base = f"workspace/output/teacher_cache/{radio_name}"
        build_radio_summary_cache(
            variant=radio_cfg["variant"],
            adaptor_names=radio_cfg["adaptors"],
            image_paths=image_paths,
            cache_base=cache_base,
            batch_size=32,
            device=device,
        )
        logger.info(f"RADIO {radio_name} cache complete!")

    # Build non-RADIO caches
    non_radio_to_build = [t for t in teachers_to_build if t in NON_RADIO_TEACHERS]
    if non_radio_to_build:
        logger.info(f"\n{'='*40}")
        logger.info(f"Building non-RADIO caches: {non_radio_to_build}")
        logger.info(f"{'='*40}")
        build_all_teacher_caches(non_radio_to_build, image_paths, device=device)

    logger.info("\n" + "=" * 60)
    logger.info("ALL CACHES BUILT SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
