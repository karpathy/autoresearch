"""Tests for prepare.py -- immutable infrastructure.

Covers: REFAC-02 (eval in prepare), REFAC-03 (teacher cache),
        REFAC-04 (dataset loading), REFAC-07 (combined metric).
"""
import inspect
import prepare


def test_prepare_importable():
    """REFAC-01: prepare.py is importable."""
    assert prepare is not None


def test_no_training_code_in_prepare():
    """Verify prepare.py does not contain training-specific code."""
    src = inspect.getsource(prepare)
    assert "class ProjectionHead" not in src
    assert "class FrozenBackboneWithHead" not in src
    assert "class ArcMarginProduct" not in src
    assert "class RandomQualityDegradation" not in src
    assert "def vat_embedding_loss" not in src
    assert "def run_train_epoch" not in src
    assert "argparse" not in src


def test_no_train_import_in_prepare():
    """Verify prepare.py never imports from train.py (Pitfall 2)."""
    src = inspect.getsource(prepare)
    assert "from train import" not in src
    assert "import train" not in src


# --- REFAC-02: Evaluation in prepare.py ---

def test_run_retrieval_eval_exists():
    """REFAC-02: run_retrieval_eval exists in prepare.py."""
    assert hasattr(prepare, "run_retrieval_eval")
    assert callable(prepare.run_retrieval_eval)


def test_run_retrieval_eval_signature():
    """REFAC-02: run_retrieval_eval accepts model as first param (duck-typed)."""
    sig = inspect.signature(prepare.run_retrieval_eval)
    params = list(sig.parameters.keys())
    assert params[0] == "model", f"First param is {params[0]}, expected 'model'"


# --- REFAC-03: Teacher cache ---

def test_trendyol_embedder_exists():
    """REFAC-03: TrendyolEmbedder class exists in prepare.py."""
    assert hasattr(prepare, "TrendyolEmbedder")


def test_load_teacher_embeddings_exists():
    """REFAC-03: load_teacher_embeddings function exists."""
    assert hasattr(prepare, "load_teacher_embeddings")
    assert callable(prepare.load_teacher_embeddings)


def test_teacher_mem_cache_exists():
    """REFAC-03: In-memory cache dict exists."""
    assert hasattr(prepare, "_TEACHER_MEM_CACHE")
    assert isinstance(prepare._TEACHER_MEM_CACHE, dict)


def test_init_teacher_exists():
    """REFAC-03: init_teacher convenience function exists."""
    assert hasattr(prepare, "init_teacher")
    assert callable(prepare.init_teacher)


# --- REFAC-04: Dataset loading ---

def test_dataset_classes_exist():
    """REFAC-04: All dataset classes exist in prepare.py."""
    assert hasattr(prepare, "CombinedDistillDataset")
    assert hasattr(prepare, "CombinedArcFaceDataset")
    assert hasattr(prepare, "DistillImageFolder")
    assert hasattr(prepare, "SampledImageFolder")


def test_dataset_builder_functions_exist():
    """REFAC-04: Builder functions exist."""
    assert hasattr(prepare, "build_distill_dataset")
    assert callable(prepare.build_distill_dataset)
    assert hasattr(prepare, "build_arcface_dataset")
    assert callable(prepare.build_arcface_dataset)
    assert hasattr(prepare, "build_val_dataset")
    assert callable(prepare.build_val_dataset)


def test_collate_functions_exist():
    """REFAC-04: Collate functions exist."""
    assert hasattr(prepare, "collate_distill")
    assert callable(prepare.collate_distill)
    assert hasattr(prepare, "collate_arcface")
    assert callable(prepare.collate_arcface)


def test_dataset_path_constants():
    """REFAC-04: Dataset path constants exist with correct values."""
    assert prepare.TRAIN_DIR == "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/train"
    assert prepare.VAL_DIR == "/data/mnt/mnt_ml_shared/Vic/product_code_dataset/val"
    assert prepare.ARCFACE_DIR == "/data/mnt/mnt_ml_shared/Vic/retail_product_checkout_crop"
    assert prepare.REID_PRODUCTS.endswith("products")
    assert prepare.REID_COMMODITY.endswith("commodity")
    assert prepare.REID_NEGATIVES.endswith("negatives")
    assert prepare.SKIP_CLASSES == {"0000000000"}


# --- REFAC-07: Combined metric ---

def test_compute_combined_metric_exists():
    """REFAC-07: compute_combined_metric exists."""
    assert hasattr(prepare, "compute_combined_metric")
    assert callable(prepare.compute_combined_metric)


def test_compute_combined_metric_formula():
    """REFAC-07: Formula is 0.5 * recall@1 + 0.5 * mean_cosine."""
    assert abs(prepare.compute_combined_metric(0.8, 0.6) - 0.7) < 1e-9
    assert abs(prepare.compute_combined_metric(1.0, 1.0) - 1.0) < 1e-9
    assert abs(prepare.compute_combined_metric(0.0, 0.0) - 0.0) < 1e-9
    assert abs(prepare.compute_combined_metric(0.5, 0.5) - 0.5) < 1e-9
    # Asymmetric test
    assert abs(prepare.compute_combined_metric(0.9, 0.3) - 0.6) < 1e-9


# --- Constants ---

def test_immutable_constants():
    """Constants that define the contract."""
    assert prepare.EMBEDDING_DIM == 256
    assert prepare.IMAGE_SIZE == 224
    assert prepare.DEFAULT_TEACHER_CACHE_DIR == "workspace/output/trendyol_teacher_cache2"


# --- PadToSquare uses correct TF.pad ---

def test_pad_to_square_exists():
    """PadToSquare uses torchvision.transforms.functional (not broken tf.pad)."""
    assert hasattr(prepare, "PadToSquare")
    src = inspect.getsource(prepare.PadToSquare)
    assert "TF.pad" in src, "PadToSquare should use TF.pad (torchvision.transforms.functional)"
    assert "tf.pad" not in src.replace("TF.pad", ""), "PadToSquare should NOT use tf.pad (broken version)"
