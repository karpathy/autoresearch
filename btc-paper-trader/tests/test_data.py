"""Test data fetching and parquet management."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data import append_candle, load_parquet, save_parquet


def _make_ohlcv_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame for testing."""
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.random.uniform(40000, 41000, n_rows),
        "high": np.random.uniform(41000, 42000, n_rows),
        "low": np.random.uniform(39000, 40000, n_rows),
        "close": np.random.uniform(40000, 41000, n_rows),
        "volume": np.random.uniform(100, 1000, n_rows),
    })


class TestParquetIO:
    def test_save_and_load(self):
        df = _make_ohlcv_df()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            save_parquet(df, path)
            loaded = load_parquet(path)
            assert len(loaded) == len(df)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(path)

    def test_atomic_write_no_tmp_left(self):
        df = _make_ohlcv_df()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            save_parquet(df, path)
            assert os.path.exists(path)
            assert not os.path.exists(path + ".tmp")
        finally:
            os.unlink(path)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_parquet("/nonexistent/file.parquet")

    def test_load_invalid_columns_raises(self):
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            df.to_parquet(path)
            with pytest.raises(ValueError, match="missing columns"):
                load_parquet(path)
        finally:
            os.unlink(path)


class TestAppendCandle:
    def test_append_new_candle(self):
        df = _make_ohlcv_df(10)
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {
            "timestamp": new_ts,
            "open": 40500.0,
            "high": 40600.0,
            "low": 40400.0,
            "close": 40550.0,
            "volume": 500.0,
        }

        result = append_candle(df, candle)
        assert len(result) == 11
        assert result["timestamp"].iloc[-1] == new_ts

    def test_dedup_existing_timestamp(self):
        df = _make_ohlcv_df(10)
        existing_ts = df["timestamp"].iloc[5]
        candle = {
            "timestamp": existing_ts,
            "open": 99999.0,
            "high": 99999.0,
            "low": 99999.0,
            "close": 99999.0,
            "volume": 99999.0,
        }

        result = append_candle(df, candle)
        assert len(result) == 10  # No new row added

    def test_funding_rate_forward_fill(self):
        df = _make_ohlcv_df(10)
        df["funding_rate"] = 0.0001

        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {
            "timestamp": new_ts,
            "open": 40500.0,
            "high": 40600.0,
            "low": 40400.0,
            "close": 40550.0,
            "volume": 500.0,
        }

        # No funding_rate provided — should forward-fill
        result = append_candle(df, candle, funding_rate=None)
        assert result["funding_rate"].iloc[-1] == 0.0001

    def test_funding_rate_explicit(self):
        df = _make_ohlcv_df(10)
        df["funding_rate"] = 0.0001

        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {
            "timestamp": new_ts,
            "open": 40500.0,
            "high": 40600.0,
            "low": 40400.0,
            "close": 40550.0,
            "volume": 500.0,
        }

        result = append_candle(df, candle, funding_rate=0.0005)
        assert result["funding_rate"].iloc[-1] == 0.0005

    def test_original_not_modified(self):
        df = _make_ohlcv_df(10)
        original_len = len(df)

        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {
            "timestamp": new_ts,
            "open": 40500.0,
            "high": 40600.0,
            "low": 40400.0,
            "close": 40550.0,
            "volume": 500.0,
        }

        result = append_candle(df, candle)
        assert len(df) == original_len  # original unchanged
        assert len(result) == original_len + 1
