import pytest

from autoresearch.tokenizer import CharTokenizer


def test_roundtrip_shakespeare_snippet():
    text = "To be, or not to be, that is the question.\n"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_vocab_is_sorted_and_stable():
    tok1 = CharTokenizer.from_text("bac")
    tok2 = CharTokenizer.from_text("cab")
    assert tok1.itos == tok2.itos == ["a", "b", "c"]
    assert tok1.encode("abc") == [0, 1, 2]


def test_empty_rejected():
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_unknown_char_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("z")
