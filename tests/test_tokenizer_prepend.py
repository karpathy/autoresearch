import sys
import types
import unittest

pyarrow_module = sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
pyarrow_parquet = sys.modules.setdefault("pyarrow.parquet", types.ModuleType("pyarrow.parquet"))
pyarrow_module.parquet = pyarrow_parquet
sys.modules.setdefault("rustbpe", types.ModuleType("rustbpe"))
torch_module = sys.modules.setdefault("torch", types.ModuleType("torch"))
torch_module.no_grad = lambda: (lambda fn: fn)
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))

from prepare import Tokenizer


class FakeEncoding:
    def __init__(self):
        self.n_vocab = 999

    def encode_single_token(self, text):
        if text in {"<bos>", "<|reserved_0|>"}:
            return 42
        raise ValueError(f"not a single token: {text}")

    def encode_ordinary(self, text):
        mapping = {
            "hello": [1, 2],
            "world": [3],
            "ab": [7, 8],
            "": [],
        }
        if text not in mapping:
            raise AssertionError(f"unexpected encode_ordinary input: {text!r}")
        return list(mapping[text])

    def encode_ordinary_batch(self, texts, num_threads=8):
        return [self.encode_ordinary(text) for text in texts]

    def decode(self, ids):
        return str(ids)


class TokenizerPrependTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(FakeEncoding())

    def test_single_token_string_prepend_uses_single_token_id(self):
        ids = self.tokenizer.encode("hello", prepend="<bos>")
        self.assertEqual(ids, [42, 1, 2])

    def test_multi_token_string_prepend_falls_back_to_ordinary_encoding(self):
        ids = self.tokenizer.encode("hello", prepend="ab")
        self.assertEqual(ids, [7, 8, 1, 2])

    def test_multi_token_string_prepend_applies_to_every_batch_row(self):
        rows = self.tokenizer.encode(["hello", "world"], prepend="ab")
        self.assertEqual(rows, [[7, 8, 1, 2], [7, 8, 3]])


if __name__ == "__main__":
    unittest.main()
