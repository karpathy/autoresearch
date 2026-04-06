import sys
import types
import unittest

pyarrow_module = types.ModuleType("pyarrow")
pyarrow_parquet_module = types.ModuleType("pyarrow.parquet")
pyarrow_module.parquet = pyarrow_parquet_module
sys.modules.setdefault("pyarrow", pyarrow_module)
sys.modules.setdefault("pyarrow.parquet", pyarrow_parquet_module)
sys.modules.setdefault("rustbpe", types.ModuleType("rustbpe"))
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
sys.modules.setdefault("requests", types.ModuleType("requests"))

torch_module = types.ModuleType("torch")


class DummyTensor:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


def tensor(values, dtype=None):
    return DummyTensor(values)


def no_grad():
    def decorator(fn):
        return fn
    return decorator


torch_module.tensor = tensor
torch_module.int32 = "int32"
torch_module.no_grad = no_grad
sys.modules.setdefault("torch", torch_module)

from prepare import build_token_bytes_lookup


class DummyEncoding:
    def __init__(self, n_vocab):
        self.n_vocab = n_vocab


class BuildTokenBytesLookupTests(unittest.TestCase):
    def test_uses_raw_mergeable_bytes_instead_of_utf8_reencoding(self):
        enc = DummyEncoding(4)
        mergeable_ranks = {
            b"a": 0,
            b"\x80": 1,  # invalid standalone UTF-8 continuation byte
            b"\xff\xfe": 2,  # invalid two-byte sequence
        }
        special_tokens = {"<|reserved_0|>": 3}

        token_bytes = build_token_bytes_lookup(enc, mergeable_ranks, special_tokens)

        self.assertEqual(token_bytes.tolist(), [1, 1, 2, 0])

    def test_raises_when_vocab_contains_unknown_non_special_token(self):
        enc = DummyEncoding(2)
        mergeable_ranks = {b"a": 0}
        special_tokens = {}

        with self.assertRaisesRegex(ValueError, "Token ID 1 missing"):
            build_token_bytes_lookup(enc, mergeable_ranks, special_tokens)


if __name__ == "__main__":
    unittest.main()
