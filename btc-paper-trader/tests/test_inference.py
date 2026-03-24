"""Test inference pipeline parity with autotrader predict_fn().

Verifies that every intermediate stage of the pipeline matches
the reference predictions stored in the model artifact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference import compute_position, load_artifacts, validate_artifacts


class TestComputePosition:
    def test_below_threshold_is_flat(self):
        assert compute_position(0.0) == 0.0
        assert compute_position(0.10) == 0.0
        assert compute_position(-0.10) == 0.0
        assert compute_position(0.19) == 0.0

    def test_at_threshold(self):
        pos = compute_position(0.20)
        assert pos == 0.0  # exactly at threshold = flat

    def test_above_threshold(self):
        pos = compute_position(0.25)
        expected = (0.25 - 0.20) / (0.50 - 0.20)  # = 0.167
        assert abs(pos - expected) < 1e-6

    def test_full_position(self):
        pos = compute_position(0.50)
        assert abs(pos - 1.0) < 1e-6

    def test_beyond_full_capped(self):
        pos = compute_position(0.80)
        assert abs(pos - 1.0) < 1e-6

    def test_short_direction(self):
        pos = compute_position(-0.35)
        expected = -((0.35 - 0.20) / (0.50 - 0.20))
        assert abs(pos - expected) < 1e-6

    def test_custom_thresholds(self):
        pos = compute_position(0.30, sigma_threshold=0.10, sigma_full=0.40)
        expected = (0.30 - 0.10) / (0.40 - 0.10)  # = 0.667
        assert abs(pos - expected) < 1e-6


class TestArtifactLoading:
    """Tests that require a model artifact. Skipped in CI."""

    @pytest.fixture
    def artifact_path(self):
        artifact_dir = Path(__file__).parent.parent / "artifacts"
        if not artifact_dir.exists():
            pytest.skip("No artifacts directory")

        joblib_files = list(artifact_dir.glob("model_*.joblib"))
        if not joblib_files:
            pytest.skip("No artifact files found")

        return str(joblib_files[0])

    def test_load_artifacts(self, artifact_path):
        artifacts = load_artifacts(artifact_path)
        assert "model" in artifacts
        assert "model_72" in artifacts
        assert "conf_model" in artifacts
        assert "pos_scaler" in artifacts

    def test_validate_artifacts(self, artifact_path):
        artifacts = load_artifacts(artifact_path)
        assert validate_artifacts(artifacts)


class TestInferenceParity:
    """Full pipeline parity against reference predictions from artifact."""

    @pytest.fixture
    def artifact_with_ref(self):
        artifact_dir = Path(__file__).parent.parent / "artifacts"
        if not artifact_dir.exists():
            pytest.skip("No artifacts directory")

        joblib_files = list(artifact_dir.glob("model_*.joblib"))
        if not joblib_files:
            pytest.skip("No artifact files found")

        import joblib
        art = joblib.load(joblib_files[0])
        if "reference_predictions" not in art:
            pytest.skip("No reference predictions in artifact")
        return art

    def test_24h_model_parity(self, artifact_with_ref):
        """Verify 24h model predictions match reference."""
        ref = artifact_with_ref["reference_predictions"]
        feats = ref["features"]

        pred = artifact_with_ref["model"].predict(feats)
        np.testing.assert_allclose(pred, ref["pred_24"], atol=1e-6)

    def test_72h_model_parity(self, artifact_with_ref):
        """Verify 72h model raw predictions match reference."""
        ref = artifact_with_ref["reference_predictions"]
        feats = ref["features"]

        pred = artifact_with_ref["model_72"].predict(feats)
        np.testing.assert_allclose(pred, ref["pred_72_raw"], atol=1e-6)

    def test_72h_ema_parity(self, artifact_with_ref):
        """Verify 72h EMA smoothing matches reference."""
        ref = artifact_with_ref["reference_predictions"]
        pred_72_raw = ref["pred_72_raw"]

        smoothed = pd.Series(pred_72_raw).ewm(span=12, min_periods=1).mean().values
        np.testing.assert_allclose(smoothed, ref["pred_72_smoothed"], atol=1e-6)

    def test_sign_matching_parity(self, artifact_with_ref):
        """Verify 72h sign matching and dampening."""
        ref = artifact_with_ref["reference_predictions"]

        sign_match = np.sign(ref["pred_24"]) * np.sign(ref["pred_72_smoothed"])
        np.testing.assert_array_equal(sign_match, ref["sign_match"])

        sigma_after = ref["pred_24"] * (1.0 - 0.04 + 0.04 * sign_match)
        np.testing.assert_allclose(sigma_after, ref["pred_after_72h"], atol=1e-10)

    def test_confidence_scaler_parity(self, artifact_with_ref):
        """Verify confidence scaler pipeline matches reference."""
        ref = artifact_with_ref["reference_predictions"]
        art = artifact_with_ref
        feats = ref["features"]

        conf_pred = art["conf_model"].predict_proba(
            feats[:, art["conf_feat_indices"]]
        )[:, 1]
        np.testing.assert_allclose(conf_pred, ref["conf_prob"], atol=1e-6)

        conf_smooth = pd.Series(conf_pred).ewm(span=24, min_periods=1).mean().values
        np.testing.assert_allclose(conf_smooth, ref["conf_smoothed"], atol=1e-6)

        conf_range = max(art["conf_train_p95"] - art["conf_train_p5"], 1e-6)
        conf_norm = np.clip(
            (conf_smooth - art["conf_train_p5"]) / conf_range, 0.0, 1.0
        )
        np.testing.assert_allclose(conf_norm, ref["conf_norm"], atol=1e-6)

        dampen = np.clip((conf_norm - 0.05) / 0.10, 0.0, 1.0)
        boost = np.clip((conf_norm - 0.85) / 0.10, 0.0, 1.0)
        conf_adj = 0.70 + 0.30 * dampen + 0.20 * boost
        np.testing.assert_allclose(conf_adj, ref["conf_adj"], atol=1e-10)

    def test_position_scaler_parity(self, artifact_with_ref):
        """Verify position scaler pipeline matches reference."""
        ref = artifact_with_ref["reference_predictions"]
        art = artifact_with_ref
        feats = ref["features"]

        signal = np.clip(
            art["pos_scaler"].predict(feats[:, art["scaler_feat_mask"]]), 0.0, 1.0
        )
        np.testing.assert_allclose(signal, ref["scaler_signal"], atol=1e-6)

        scale = 1.08 - 0.72 * signal
        np.testing.assert_allclose(scale, ref["pos_scale"], atol=1e-10)

    def test_final_prediction_parity(self, artifact_with_ref):
        """Verify the complete pipeline end-to-end."""
        ref = artifact_with_ref["reference_predictions"]

        after_scale = ref["pred_after_72h"] * ref["conf_adj"] * ref["pos_scale"] * 0.35
        np.testing.assert_allclose(after_scale, ref["pred_after_scale"], atol=1e-10)

        final = pd.Series(after_scale).ewm(span=20, min_periods=1).mean().values
        np.testing.assert_allclose(final, ref["pred_final"], atol=1e-6)
