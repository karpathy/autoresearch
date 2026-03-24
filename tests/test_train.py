"""Tests for train.py -- agent-editable training script.

Covers: REFAC-01 (split exists), REFAC-05 (module-level constants),
        REFAC-06 (encode contract).
"""
import inspect
import torch
import train


def test_train_importable():
    """REFAC-01: train.py is importable."""
    assert train is not None


def test_prepare_importable():
    """REFAC-01: prepare.py is importable."""
    import prepare
    assert prepare is not None


def test_train_imports_from_prepare():
    """REFAC-01: train.py imports from prepare.py."""
    src = inspect.getsource(train)
    assert "from prepare import" in src


# --- REFAC-02: No evaluation code in train.py ---

def test_no_eval_code_in_train():
    """REFAC-02: Zero evaluation code in train.py."""
    src = inspect.getsource(train)
    # No evaluation function definitions
    assert "def run_retrieval_eval" not in src
    assert "def compute_combined_metric" not in src
    # (train.py may CALL compute_combined_metric but must not DEFINE it)


# --- REFAC-05: Module-level constants ---

def test_no_argparse():
    """REFAC-05: train.py has no argparse."""
    src = inspect.getsource(train)
    assert "argparse" not in src
    assert "parse_args" not in src
    assert "add_argument" not in src


def test_module_level_constants_exist():
    """REFAC-05: All tunable parameters are module-level constants."""
    required_constants = [
        "MODEL_NAME", "BATCH_SIZE", "ARCFACE_BATCH_SIZE",
        "LR", "WEIGHT_DECAY", "EPOCHS", "NUM_WORKERS",
        "SEED", "DEVICE", "QUALITY_DEGRADATION_PROB",
        "DROP_HARD_RATIO", "USE_ARCFACE", "ARCFACE_S",
        "ARCFACE_M", "ARCFACE_LOSS_WEIGHT", "VAT_WEIGHT",
        "VAT_EPSILON", "SEP_WEIGHT", "UNFREEZE_EPOCH",
        "BACKBONE_LR_MULT", "TEACHER_CACHE_DIR", "OUTPUT_DIR",
        "RETRIEVAL_MAX_SAMPLES", "RETRIEVAL_TOPK",
    ]
    for const in required_constants:
        assert hasattr(train, const), f"Missing constant: {const}"


def test_constant_values():
    """REFAC-05: Key constants have correct default values."""
    assert train.MODEL_NAME == "hf-hub:timm/lcnet_050.ra2_in1k"
    assert train.EPOCHS == 10, f"EPOCHS should be 10, got {train.EPOCHS}"
    assert train.EMBEDDING_DIM == 256
    assert train.BATCH_SIZE == 256
    assert train.LR == 1e-1
    assert train.SEED == 42
    assert train.UNFREEZE_EPOCH == 0, f"UNFREEZE_EPOCH should be 0, got {train.UNFREEZE_EPOCH}"
    assert train.BACKBONE_LR_MULT == 0.1


# --- REFAC-06: .encode() contract ---

def test_encode_contract_shape():
    """REFAC-06: model.encode() returns Tensor[B, 256]."""
    model = train.FrozenBackboneWithHead("hf-hub:timm/lcnet_050.ra2_in1k", 256, "cpu")
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"


def test_encode_contract_l2_normalized():
    """REFAC-06: model.encode() output is L2-normalized."""
    model = train.FrozenBackboneWithHead("hf-hub:timm/lcnet_050.ra2_in1k", 256, "cpu")
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Not L2-normalized: {norms}"


def test_encode_contract_no_grad():
    """REFAC-06: model.encode() runs without gradients."""
    model = train.FrozenBackboneWithHead("hf-hub:timm/lcnet_050.ra2_in1k", 256, "cpu")
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    assert not out.requires_grad, "encode() output should not require grad"


# --- No early stopping, no ONNX export ---

def test_no_early_stopping():
    """No early stopping in train.py (Pitfall 5)."""
    src = inspect.getsource(train)
    assert "patience" not in src.lower() or "patience" not in src
    assert "no_improve" not in src
    assert "early_stop" not in src.lower()


def test_no_onnx_export():
    """No ONNX export in train.py (Pitfall 6)."""
    src = inspect.getsource(train)
    assert "torch.onnx.export" not in src
    assert "_ExportWrapper" not in src


# --- OOM handling ---

def test_oom_handler_exists():
    """train.py catches OOM and prints status."""
    src = inspect.getsource(train)
    assert "OutOfMemoryError" in src
    assert "status: OOM" in src


# --- Greppable output ---

def test_greppable_output():
    """train.py prints greppable summary (per D-08)."""
    src = inspect.getsource(train)
    assert "combined_metric:" in src
    assert "recall@1:" in src
    assert "mean_cosine:" in src
    assert "peak_vram_mb:" in src
    assert "elapsed_seconds:" in src
    assert "epochs:" in src


# --- Training classes ---

def test_training_classes_in_train():
    """Agent-editable components are in train.py."""
    assert hasattr(train, "ProjectionHead")
    assert hasattr(train, "ArcMarginProduct")
    assert hasattr(train, "FrozenBackboneWithHead")
    assert hasattr(train, "RandomQualityDegradation")
    assert hasattr(train, "vat_embedding_loss")
    assert hasattr(train, "EpochStats")
    assert hasattr(train, "run_train_epoch")
    assert hasattr(train, "main")
