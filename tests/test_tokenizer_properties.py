import pytest

# Skips when checking the starter project in an environment without tiktoken.
tiktoken = pytest.importorskip("tiktoken")


def test_unicode_round_trip():
    encoding = tiktoken.get_encoding("o200k_base")
    for sample in ("Nigeria", "Ẹ jọ̀ọ́", "₦234,500", "💸✅🇳🇬", "\n  "):
        assert encoding.decode(encoding.encode(sample)) == sample


def test_count_is_not_the_same_as_whitespace_words():
    encoding = tiktoken.get_encoding("o200k_base")
    text = "supercalifragilisticexpialidocious"
    assert len(text.split()) == 1
    assert len(encoding.encode(text)) > 1
