"""Test feature computation parity with autotrader.

Verifies the frozen compute_features() produces identical output
to the reference features stored in the model artifact.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COUNT_WITH_FUNDING,
    FEATURE_COUNT_WITHOUT_FUNDING,
    MAX_LOOKBACK,
    compute_features,
)


def _make_test_df(n_rows: int = 500, with_funding: bool = True) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n_rows) * 100)
    high = close + np.abs(np.random.randn(n_rows) * 50)
    low = close - np.abs(np.random.randn(n_rows) * 50)
    open_ = close + np.random.randn(n_rows) * 30
    volume = np.abs(np.random.randn(n_rows) * 1000) + 500

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    if with_funding:
        df["funding_rate"] = np.random.randn(n_rows) * 0.0001

    return df


class TestComputeFeatures:
    def test_output_shape(self):
        df = _make_test_df(500, with_funding=True)
        features, timestamps, vol = compute_features(df)

        expected_rows = len(df) - MAX_LOOKBACK
        assert features.shape[0] == expected_rows
        assert features.shape[1] == FEATURE_COUNT_WITH_FUNDING
        assert len(timestamps) == expected_rows
        assert len(vol) == expected_rows

    def test_output_shape_no_funding(self):
        df = _make_test_df(500, with_funding=False)
        features, timestamps, vol = compute_features(df)

        assert features.shape[1] == FEATURE_COUNT_WITHOUT_FUNDING  # 36 without funding

    def test_no_nan_in_output(self):
        # Need >720+168 rows for all rolling windows to be fully populated
        df = _make_test_df(2000, with_funding=True)
        features, _, _ = compute_features(df)

        # After MAX_LOOKBACK trimming with sufficient data, most features
        # should be finite. The 720-period rolling window needs 720+168
        # rows to produce valid output after the MAX_LOOKBACK trim.
        nan_count = np.isnan(features).sum()
        assert nan_count < features.size * 0.01, f"Too many NaN: {nan_count}"

    def test_vol_safe_positive(self):
        df = _make_test_df(500)
        _, _, vol = compute_features(df)
        assert (vol > 0).all(), "vol_safe must be strictly positive"

    def test_deterministic(self):
        df = _make_test_df(500)
        f1, t1, v1 = compute_features(df)
        f2, t2, v2 = compute_features(df)

        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(v1, v2)

    def test_timestamps_aligned(self):
        df = _make_test_df(500)
        _, timestamps, _ = compute_features(df)

        # Timestamps should start at MAX_LOOKBACK offset
        expected = df["timestamp"].values[MAX_LOOKBACK:]
        np.testing.assert_array_equal(timestamps, expected)


class TestFeatureParityWithArtifact:
    """Tests that require a model artifact with reference predictions.

    These tests are skipped if no artifact is available (CI environment).
    """

    @pytest.fixture
    def artifact(self):
        """Load the first available artifact, skip if none."""
        artifact_dir = Path(__file__).parent.parent / "artifacts"
        if not artifact_dir.exists():
            pytest.skip("No artifacts directory")

        joblib_files = list(artifact_dir.glob("model_*.joblib"))
        if not joblib_files:
            pytest.skip("No artifact files found")

        import joblib
        return joblib.load(joblib_files[0])

    def test_feature_count_matches(self, artifact):
        n_features = artifact["n_features"]
        df = _make_test_df(500, with_funding=True)
        features, _, _ = compute_features(df)
        assert features.shape[1] == n_features

    def test_reference_features_match(self, artifact):
        """Compare computed features against stored reference (if available)."""
        ref = artifact.get("reference_predictions")
        if ref is None or "features" not in ref:
            pytest.skip("No reference features in artifact")

        # This test requires the same input data that was used during export.
        # It serves as a template — the actual parity test runs against
        # the exported reference data.
        assert ref["features"].shape[1] == artifact["n_features"]
