"""Byte-level BPE: round-trip losslessness, merge correctness, and persistence.

The single most important property is that ``decode(encode(s)) == s`` for ANY
string. Because the base vocabulary is the 256 byte values there is no ``<unk>``,
so a failure here means a real bug rather than an out-of-vocabulary character.
A lossy tokenizer would silently cap the loss a model could ever reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm.data import get_batch
from csllm.tokenizer import BPETokenizer

CORPUS = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
) * 12


# 320 (= 64 merges) is comfortably within what this small corpus supports. Its
# 29 unique words only admit ~92 merges in total — see
# test_merge_count_is_bounded_by_diversity_not_length.
TEST_VOCAB_SIZE = 320


@pytest.fixture(scope="module")
def trained():
    tok = BPETokenizer()
    tok.train(CORPUS, TEST_VOCAB_SIZE)
    return tok


# ── round-trip ───────────────────────────────────────────────────────────────

ADVERSARIAL = [
    "",
    " ",
    "\n\n\t  \r\n",
    "hello world",
    "To be, or not to be",
    "ünïcödé àccënts",
    "emoji: 🦀🔥👨‍👩‍👧‍👦 and a ZWJ family",
    "中文字符与日本語のテキスト",
    "\x00\x01\x02 control bytes \x7f",
    "math: ∑∫√≈∞ ± 3.14159",
    "mixed 123 numbers 456 and 'contractions' don't won't",
    "a" * 500,
    "🦀" * 40,
]


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_round_trip_is_lossless(trained, text):
    assert trained.decode(trained.encode(text)) == text


@pytest.mark.parametrize("text", ADVERSARIAL)
def test_round_trip_lossless_before_training(text):
    """An untrained tokenizer is pure bytes — it must still round-trip exactly."""
    assert BPETokenizer().decode(BPETokenizer().encode(text)) == text


def test_round_trip_on_the_training_corpus(trained):
    assert trained.decode(trained.encode(CORPUS)) == CORPUS


def test_round_trip_over_random_unicode(trained):
    """Fuzz across the BMP, skipping surrogates (not encodable in UTF-8)."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        points = rng.integers(1, 0xFFFF, size=64)
        text = "".join(chr(c) for c in points if not 0xD800 <= c <= 0xDFFF)
        assert trained.decode(trained.encode(text)) == text


def test_decode_bytes_supports_streaming_reassembly(trained):
    """Concatenating per-token bytes must equal decoding them all at once.

    This is what lets the gateway buffer a partial UTF-8 sequence instead of
    emitting mojibake mid-stream.
    """
    text = "héllo 🌍 wörld"
    ids = trained.encode(text)
    piecewise = b"".join(trained.decode_bytes([i]) for i in ids)
    assert piecewise == trained.decode_bytes(ids) == text.encode("utf-8")


def test_a_token_can_split_a_multibyte_character(trained):
    """Demonstrates why the gateway must buffer: some single token is not valid UTF-8."""
    ids = trained.encode("🌍🦀🔥")
    assert any(
        not _is_valid_utf8(trained.decode_bytes([i])) for i in ids
    ), "expected at least one token to be a partial UTF-8 sequence"


def _is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ── training behaviour ───────────────────────────────────────────────────────


def test_vocab_size_is_reached(trained):
    assert trained.vocab_size == TEST_VOCAB_SIZE
    assert len(trained.merges) == TEST_VOCAB_SIZE - 256


def test_training_compresses(trained):
    """Merges must actually shorten the sequence versus raw bytes."""
    raw = len(CORPUS.encode("utf-8"))
    encoded = len(trained.encode(CORPUS))
    assert encoded < raw * 0.7, f"weak compression: {raw} bytes -> {encoded} tokens"


def test_merge_count_is_bounded_by_diversity_not_length():
    """Repeating the same text buys no extra merges.

    Merging stops once no adjacent pair occurs twice, which depends on how many
    DISTINCT words exist, not on corpus size. This is why train_tokenizer.py
    hard-fails when the learned vocab is smaller than the config asks for: a
    silent shortfall would mismatch the model's embedding table.
    """
    short, long = BPETokenizer(), BPETokenizer()
    short.train(CORPUS, 4096)  # CORPUS is the 4 lines x 12
    long.train(CORPUS * 5, 4096)
    assert len(short.merges) == len(long.merges)
    assert len(short.merges) < 4096 - 256, "expected this corpus to exhaust its pairs"


def test_merges_are_learned_in_frequency_order(trained):
    """The first merge must be the most frequent adjacent byte pair."""
    first_pair = min(trained.merges.items(), key=lambda kv: kv[1])[0]
    merged = trained.vocab[trained.merges[first_pair]]
    # In this corpus the most common pair is a common English bigram.
    assert merged in (b"th", b"e ", b"t ", b"in", b"o ", b" t", b"re", b"ou", b"s ")


def test_all_byte_values_are_present_in_the_vocab(trained):
    for i in range(256):
        assert trained.vocab[i] == bytes([i])


def test_pretokenization_prevents_merges_across_whitespace(trained):
    """The GPT-2 split keeps BPE from learning tokens that straddle words."""
    for token in trained.vocab.values():
        stripped = token.strip()
        if stripped and b" " in stripped:
            pytest.fail(f"token {token!r} merges across a word boundary")


def test_rejects_vocab_smaller_than_the_byte_alphabet():
    with pytest.raises(ValueError, match="must be >= 256"):
        BPETokenizer().train("hello", 100)


def test_training_stops_early_when_no_pairs_repeat():
    """A corpus with nothing to merge must not invent junk merges."""
    tok = BPETokenizer()
    tok.train("abcdefg", 4096)
    assert tok.vocab_size < 4096
    assert tok.decode(tok.encode("abcdefg")) == "abcdefg"


def test_decode_rejects_unknown_ids(trained):
    with pytest.raises(ValueError, match="outside the vocabulary"):
        trained.decode([999_999])


# ── persistence ──────────────────────────────────────────────────────────────


def test_save_load_round_trip(trained, tmp_path):
    trained.save(tmp_path)
    restored = BPETokenizer.load(tmp_path)

    assert restored.vocab_size == trained.vocab_size
    assert restored.merges == trained.merges
    assert restored.vocab == trained.vocab

    text = "Whether 'tis nobler 🦀 in the mind"
    assert restored.encode(text) == trained.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_saved_files_are_the_documented_artifacts(trained, tmp_path):
    trained.save(tmp_path)
    assert (tmp_path / "merges.txt").exists()
    assert (tmp_path / "vocab.json").exists()
    lines = (tmp_path / "merges.txt").read_text().strip().splitlines()
    assert len(lines) == len(trained.merges)
    assert all(len(line.split()) == 2 for line in lines)


# ── batching ─────────────────────────────────────────────────────────────────


def test_get_batch_shapes_and_shift():
    data = np.arange(1000, dtype=np.uint16)
    rng = np.random.default_rng(0)
    x, y = get_batch(data, batch_size=4, block_size=16, rng=rng)

    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == np.int32
    # Targets are inputs shifted by one — the autoregressive objective.
    np.testing.assert_array_equal(y[:, :-1], x[:, 1:])


def test_get_batch_never_reads_past_the_end():
    data = np.arange(64, dtype=np.uint16)
    rng = np.random.default_rng(1)
    for _ in range(100):
        x, y = get_batch(data, batch_size=8, block_size=16, rng=rng)
        assert x.max() < 64 and y.max() < 64


def test_get_batch_rejects_a_too_short_split():
    with pytest.raises(ValueError, match="block_size"):
        get_batch(np.arange(8, dtype=np.uint16), 2, 16, np.random.default_rng(0))
