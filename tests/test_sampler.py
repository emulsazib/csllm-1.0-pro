"""Sampling strategy: temperature, top-k, top-p, and the multinomial draw.

The order of operations is fixed and load-bearing:
    temperature -> top-k (on logits) -> softmax -> top-p (on probabilities) -> draw
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core


def softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@pytest.fixture
def logits():
    rng = np.random.default_rng(0)
    return rng.standard_normal(50).astype(np.float32) * 2.0


def draw(logits, *, draws=1, seed=0, **params):
    return core.sample_logits(logits, core.SamplingParams(**params), seed, draws)


def test_temperature_zero_is_deterministic_argmax(logits):
    expected = int(np.argmax(logits))
    picks = draw(logits, draws=32, temperature=0.0)
    assert set(picks.tolist()) == {expected}


def test_top_k_one_is_argmax(logits):
    expected = int(np.argmax(logits))
    picks = draw(logits, draws=32, temperature=1.0, top_k=1)
    assert set(picks.tolist()) == {expected}


def test_top_k_restricts_to_the_k_largest(logits):
    k = 5
    allowed = set(np.argsort(logits)[-k:].tolist())
    picks = draw(logits, draws=400, temperature=1.0, top_k=k, seed=3)
    assert set(picks.tolist()) <= allowed


def test_top_p_restricts_to_the_nucleus(logits):
    p = 0.5
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    # The nucleus keeps everything up to and including the crossing token.
    nucleus = set(order[: int(np.searchsorted(cum, p)) + 1].tolist())
    picks = draw(logits, draws=400, temperature=1.0, top_p=p, seed=4)
    assert set(picks.tolist()) <= nucleus


def test_untruncated_draws_follow_the_softmax_distribution():
    """With no truncation the empirical frequencies must match softmax."""
    values = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    expected = softmax(values)
    picks = draw(values, draws=20000, temperature=1.0, seed=11)
    observed = np.bincount(picks, minlength=4) / len(picks)
    np.testing.assert_allclose(observed, expected, atol=0.02)


def test_temperature_scales_the_distribution():
    """Low temperature sharpens toward the argmax; high temperature flattens."""
    values = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)

    def entropy(temp):
        picks = draw(values, draws=8000, temperature=temp, seed=5)
        freq = np.bincount(picks, minlength=4) / len(picks)
        freq = freq[freq > 0]
        return float(-(freq * np.log(freq)).sum())

    assert entropy(0.3) < entropy(1.0) < entropy(3.0)


def test_low_temperature_concentrates_on_the_argmax():
    values = np.array([3.0, 1.0, 0.0], dtype=np.float32)
    picks = draw(values, draws=2000, temperature=0.1, seed=6)
    assert (picks == 0).mean() > 0.98


def test_same_seed_reproduces_the_sequence(logits):
    a = draw(logits, draws=64, temperature=0.9, seed=99)
    b = draw(logits, draws=64, temperature=0.9, seed=99)
    c = draw(logits, draws=64, temperature=0.9, seed=100)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_all_draws_are_valid_token_ids(logits):
    picks = draw(logits, draws=200, temperature=1.2, top_k=20, top_p=0.9, seed=7)
    assert picks.min() >= 0
    assert picks.max() < len(logits)


def test_top_k_larger_than_vocab_is_harmless():
    values = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    picks = draw(values, draws=100, temperature=1.0, top_k=1000, seed=8)
    assert set(picks.tolist()) <= {0, 1, 2}


def test_top_p_one_keeps_everything():
    """top_p=1 must not truncate — every token stays reachable."""
    values = np.zeros(8, dtype=np.float32)  # uniform
    picks = draw(values, draws=4000, temperature=1.0, top_p=1.0, seed=9)
    assert len(set(picks.tolist())) == 8
