import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open

# Create a mock RequestException that inherits from Exception
class MockRequestException(Exception):
    pass

# Mock dependencies before importing prepare
mock_modules = [
    "requests",
    "pyarrow",
    "pyarrow.parquet",
    "rustbpe",
    "tiktoken",
    "torch",
]
for module_name in mock_modules:
    mock = MagicMock()
    if module_name == "requests":
        mock.RequestException = MockRequestException
    sys.modules[module_name] = mock

import prepare

def test_download_single_shard_success():
    with patch("os.path.exists", return_value=False), \
         patch("requests.get") as mock_get, \
         patch("builtins.open", mock_open()) as mock_f, \
         patch("os.rename") as mock_rename:

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        result = prepare.download_single_shard(1)

        assert result is True
        mock_get.assert_called_once()
        mock_rename.assert_called_once()

def test_download_single_shard_already_exists():
    with patch("os.path.exists", return_value=True):
        result = prepare.download_single_shard(1)
        assert result is True

def test_download_single_shard_retry_logic():
    with patch("os.path.exists", return_value=False), \
         patch("requests.get") as mock_get, \
         patch("time.sleep") as mock_sleep, \
         patch("os.remove") as mock_remove:

        mock_get.side_effect = MockRequestException("Failed")

        result = prepare.download_single_shard(1)

        assert result is False
        assert mock_get.call_count == 5
        assert mock_sleep.call_count == 4

def test_list_parquet_files():
    with patch("os.listdir", return_value=["shard_00001.parquet", "shard_00002.parquet", "temp.tmp", "other.txt"]):
        files = prepare.list_parquet_files()
        assert len(files) == 2
        assert all(f.endswith(".parquet") for f in files)
        assert any("shard_00001.parquet" in f for f in files)

def test_tokenizer_wrapper():
    mock_enc = MagicMock()
    mock_enc.n_vocab = 100
    mock_enc.encode_single_token.return_value = 0
    mock_enc.encode_ordinary.return_value = [1, 2, 3]
    mock_enc.decode.return_value = "hello"

    tokenizer = prepare.Tokenizer(mock_enc)

    assert tokenizer.get_vocab_size() == 100
    # Tokenizer.encode("test") calls enc.encode_ordinary("test") -> [1, 2, 3]
    # prepend is None by default.
    assert tokenizer.encode("test") == [1, 2, 3]
    assert tokenizer.decode([1, 2, 3]) == "hello"

def test_get_token_bytes():
    with patch("builtins.open", mock_open()), \
         patch("prepare.torch.load") as mock_load:
        mock_load.return_value = "mock_tensor"
        result = prepare.get_token_bytes()
        assert result == "mock_tensor"

def test_document_batches():
    mock_pf = MagicMock()
    mock_pf.num_row_groups = 1
    mock_rg = MagicMock()
    mock_rg.column.return_value.to_pylist.return_value = ["doc1", "doc2"]
    mock_pf.read_row_group.return_value = mock_rg

    with patch("prepare.list_parquet_files", return_value=["file1.parquet", "file2.parquet"]), \
         patch("prepare.pq.ParquetFile", return_value=mock_pf), \
         patch("os.path.exists", return_value=True):

        gen = prepare._document_batches("train")
        batch, epoch = next(gen)
        assert batch == ["doc1", "doc2"]
        assert epoch == 1
