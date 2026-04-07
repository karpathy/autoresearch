import sys
import types
import unittest

sys.modules.setdefault('pyarrow', types.ModuleType('pyarrow'))
sys.modules.setdefault('pyarrow.parquet', types.ModuleType('pyarrow.parquet'))
sys.modules.setdefault('requests', types.ModuleType('requests'))
sys.modules.setdefault('rustbpe', types.ModuleType('rustbpe'))
sys.modules.setdefault('tiktoken', types.ModuleType('tiktoken'))

torch = types.ModuleType('torch')
torch.no_grad = lambda: (lambda fn: fn)
sys.modules.setdefault('torch', torch)

from prepare import Tokenizer


class FakeEncoder:
    def encode_single_token(self, text):
        if text == '<|reserved_0|>':
            return 999
        raise ValueError('not a single token')

    def encode(self, text, allowed_special='all'):
        mapping = {
            'hello ': [10, 11],
            '<|reserved_0|>': [999],
        }
        return list(mapping[text])

    def encode_ordinary(self, text):
        mapping = {
            'world': [20, 21],
            'alpha': [30],
            'beta': [40],
        }
        return list(mapping[text])

    def encode_ordinary_batch(self, texts, num_threads=8):
        return [self.encode_ordinary(text) for text in texts]


class TokenizerPrependTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = object.__new__(Tokenizer)
        self.tokenizer.enc = FakeEncoder()

    def test_single_special_token_prepend_still_uses_single_id(self):
        self.assertEqual(self.tokenizer.encode('world', prepend='<|reserved_0|>'), [999, 20, 21])

    def test_multi_token_string_prepend_is_encoded_and_prefixed(self):
        self.assertEqual(self.tokenizer.encode('world', prepend='hello '), [10, 11, 20, 21])

    def test_multi_token_string_prepend_is_applied_to_each_batch_row(self):
        self.assertEqual(
            self.tokenizer.encode(['alpha', 'beta'], prepend='hello '),
            [[10, 11, 30], [10, 11, 40]],
        )

    def test_explicit_token_id_sequences_are_supported(self):
        self.assertEqual(self.tokenizer.encode('world', prepend=[7, 8]), [7, 8, 20, 21])


if __name__ == '__main__':
    unittest.main()
