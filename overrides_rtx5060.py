# Autoresearch — RTX 5060 (8 GB VRAM) Tuning Guide
#
# The stock autoresearch defaults target H100 (80 GB). These overrides
# keep runs within the 5060's memory envelope while remaining useful for
# autonomous experimentation.
#
# Apply by editing the HYPERPARAMETER block at the bottom of train.py.

# ── Model architecture ────────────────────────────────────────────────
DEPTH              = 4            # 8 → 4 (halve layers)
ASPECT_RATIO       = 48           # 64 → 48 (smaller dim = 192)
HEAD_DIM           = 64           # 128 → 64 (smaller heads)
WINDOW_PATTERN     = "L"          # SSSL → L (banded attention is slow without FA3 Hopper)

# ── Batch sizes ───────────────────────────────────────────────────────
TOTAL_BATCH_SIZE   = 2**16        # 524K → 64K tokens per step
DEVICE_BATCH_SIZE  = 16           # 128 → 16 (fits 8 GB)

# ── Learning rates (unchanged — let autoresearch optimise these) ──────
EMBEDDING_LR       = 0.6
UNEMBEDDING_LR     = 0.004
MATRIX_LR          = 0.04
SCALAR_LR          = 0.5

# ── Schedule ──────────────────────────────────────────────────────────
WARMUP_RATIO       = 0.05
WARMDOWN_RATIO     = 0.5
