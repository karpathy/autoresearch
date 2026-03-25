"""Tests for train.py -- agent-editable training script.

Covers: REFAC-01 (split exists), REFAC-05 (module-level constants),
        REFAC-06 (encode contract), LCNET-01..04 (custom LCNet),
        INFRA-09 (einops dependency).
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
        # New LCNet constants
        "LCNET_SCALE", "SE_START_BLOCK", "SE_REDUCTION",
        "ACTIVATION", "KERNEL_SIZES", "USE_PRETRAINED",
    ]
    for const in required_constants:
        assert hasattr(train, const), f"Missing constant: {const}"


def test_constant_values():
    """REFAC-05: Key constants have correct default values."""
    assert train.MODEL_NAME == "hf-hub:timm/lcnet_050.ra2_in1k"
    assert train.EPOCHS == 10, f"EPOCHS should be 10, got {train.EPOCHS}"
    assert train.EMBEDDING_DIM == 256
    assert train.BATCH_SIZE == 256
    assert train.LR == 2e-3
    assert train.SEED == 42
    assert train.UNFREEZE_EPOCH == 0, f"UNFREEZE_EPOCH should be 0, got {train.UNFREEZE_EPOCH}"
    assert train.BACKBONE_LR_MULT == 0.1


# --- LCNET-01: .encode() contract with custom LCNet ---

def _make_lcnet():
    """Helper: create LCNet and warm up BN running stats."""
    model = train.LCNet(scale=0.5, embedding_dim=256, device="cpu")
    # Warm up BN running stats with a train-mode forward pass
    model.train()
    dummy = torch.randn(2, 3, 224, 224)
    _ = model.forward_embeddings_train(dummy)
    return model


def test_encode_contract_shape():
    """LCNET-01: model.encode() returns Tensor[B, 256]."""
    model = _make_lcnet()
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"


def test_encode_contract_l2_normalized():
    """LCNET-01: model.encode() output is L2-normalized."""
    model = _make_lcnet()
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Not L2-normalized: {norms}"


def test_encode_contract_no_grad():
    """LCNET-01: model.encode() runs without gradients."""
    model = _make_lcnet()
    dummy = torch.randn(4, 3, 224, 224)
    out = model.encode(dummy)
    assert not out.requires_grad, "encode() output should not require grad"


# --- LCNET-02: Tunable constants ---

def test_lcnet_tunable_constants():
    """LCNET-02: All 6 LCNet tunable constants exist with correct defaults."""
    assert train.LCNET_SCALE == 0.5
    assert train.SE_START_BLOCK == 10
    assert train.SE_REDUCTION == 0.25
    assert train.ACTIVATION == "h_swish"
    assert train.KERNEL_SIZES == [3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5]
    assert len(train.KERNEL_SIZES) == 13
    assert train.USE_PRETRAINED is True


def test_lcnet_different_scales():
    """LCNET-02: Different scales produce different model sizes, same encode output."""
    model_05 = train.LCNet(scale=0.5, embedding_dim=256, device="cpu")
    model_10 = train.LCNet(scale=1.0, embedding_dim=256, device="cpu")
    params_05 = sum(p.numel() for p in model_05.parameters())
    params_10 = sum(p.numel() for p in model_10.parameters())
    assert params_10 > params_05, f"scale=1.0 ({params_10}) should have more params than 0.5 ({params_05})"

    # Both produce [B, 256] embeddings
    dummy = torch.randn(2, 3, 224, 224)
    model_05.train()
    _ = model_05.forward_embeddings_train(dummy)
    model_10.train()
    _ = model_10.forward_embeddings_train(dummy)
    assert model_05.encode(dummy).shape == (2, 256)
    assert model_10.encode(dummy).shape == (2, 256)


# --- LCNET-03: Pretrained weight loading ---

def test_pretrained_loading():
    """LCNET-03: Pretrained weight loading changes model weights."""
    model = train.LCNet(scale=0.5, embedding_dim=256, device="cpu")
    # Save initial conv_stem weight
    initial_stem = model.conv_stem.weight.clone()

    # Load pretrained weights
    train.load_pretrained_lcnet(model, 0.5)

    # At least conv_stem should change (it's a standard layer with pretrained weights)
    assert not torch.equal(initial_stem, model.conv_stem.weight), \
        "conv_stem.weight should differ after loading pretrained weights"


# --- LCNET-04: Spatial feature APIs ---

def test_forward_features_spatial():
    """LCNET-04: forward_features returns (spatial [B,C,7,7], summary [B,1280])."""
    model = _make_lcnet()
    dummy = torch.randn(2, 3, 224, 224)
    model.train()
    spatial, summary = model.forward_features(dummy)

    assert isinstance(spatial, torch.Tensor)
    assert isinstance(summary, torch.Tensor)
    assert len(spatial.shape) == 4, f"spatial should be 4D: {spatial.shape}"
    assert spatial.shape[0] == 2
    assert spatial.shape[2] == 7 and spatial.shape[3] == 7, f"spatial resolution: {spatial.shape}"
    # For scale=0.5, last stage output is make_divisible(512*0.5) = 256
    assert spatial.shape[1] == train.make_divisible(512 * 0.5), f"spatial channels: {spatial.shape[1]}"
    assert summary.shape == (2, 1280), f"summary shape: {summary.shape}"


def test_encode_with_spatial():
    """LCNET-04: encode_with_spatial returns (embedding [B,256], spatial [B,C,7,7])."""
    model = _make_lcnet()
    dummy = torch.randn(2, 3, 224, 224)
    emb, spatial = model.encode_with_spatial(dummy)

    assert emb.shape == (2, 256), f"embedding shape: {emb.shape}"
    norms = torch.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"Not L2-normalized: {norms}"
    assert len(spatial.shape) == 4
    assert spatial.shape[0] == 2
    assert spatial.shape[2] == 7 and spatial.shape[3] == 7


# --- INFRA-09: einops in pyproject.toml ---

def test_einops_in_pyproject():
    """INFRA-09: einops is listed in pyproject.toml dependencies."""
    from pathlib import Path
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject.read_text()
    assert "einops" in content, "einops not found in pyproject.toml"


# --- No timm runtime dependency ---

def test_lcnet_no_timm_runtime_dep():
    """LCNet.__init__ does not use timm.create_model (timm only in load_pretrained_lcnet)."""
    src = inspect.getsource(train.LCNet.__init__)
    assert "timm.create_model" not in src, "LCNet.__init__ should not use timm.create_model"
    assert "import timm" not in src, "LCNet.__init__ should not import timm"


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
    # New custom LCNet classes
    assert hasattr(train, "LCNet")
    assert hasattr(train, "DepthwiseSeparableConv")
    assert hasattr(train, "SqueezeExcite")
    assert hasattr(train, "make_divisible")
    assert hasattr(train, "load_pretrained_lcnet")
    # Preserved classes
    assert hasattr(train, "ArcMarginProduct")
    assert hasattr(train, "RandomQualityDegradation")
    assert hasattr(train, "vat_embedding_loss")
    assert hasattr(train, "EpochStats")
    assert hasattr(train, "run_train_epoch")
    assert hasattr(train, "main")
