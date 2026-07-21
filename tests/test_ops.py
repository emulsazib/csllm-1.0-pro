"""Forward-pass checks: every C++ op against its NumPy oracle.

Run in float64 so a mismatch means a real disagreement in the mathematics, not
accumulated fp32 rounding. float32 is spot-checked separately at looser tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core

from . import reference as ref

RTOL = 1e-11
ATOL = 1e-12


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_matmul(rng):
    x = rng.standard_normal((7, 5))
    w = rng.standard_normal((5, 3))
    np.testing.assert_allclose(core.matmul_f64(x, w), ref.matmul(x, w), rtol=RTOL, atol=ATOL)


def test_matmul_bt(rng):
    x = rng.standard_normal((7, 5))
    w = rng.standard_normal((3, 5))
    np.testing.assert_allclose(core.matmul_bt_f64(x, w), ref.matmul_bt(x, w), rtol=RTOL, atol=ATOL)


def test_add(rng):
    x = rng.standard_normal((4, 6))
    y = rng.standard_normal((4, 6))
    np.testing.assert_allclose(core.add_f64(x, y), ref.add(x, y), rtol=RTOL, atol=ATOL)


def test_rmsnorm(rng):
    x = rng.standard_normal((9, 16))
    gain = rng.standard_normal(16)
    np.testing.assert_allclose(
        core.rmsnorm_f64(x, gain, 1e-5), ref.rmsnorm(x, gain, 1e-5), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("pos_offset", [0, 5])
def test_rope(rng, pos_offset):
    x = rng.standard_normal((2, 3, 7, 8))
    np.testing.assert_allclose(
        core.rope_f64(x, pos_offset, 10000.0),
        ref.rope(x, pos_offset, 10000.0),
        rtol=RTOL,
        atol=ATOL,
    )


def test_rope_is_norm_preserving(rng):
    """A rotation cannot change vector length — a cheap structural invariant."""
    x = rng.standard_normal((2, 2, 6, 8))
    y = core.rope_f64(x, 3, 10000.0)
    np.testing.assert_allclose(
        (y**2).sum(axis=-1), (x**2).sum(axis=-1), rtol=1e-12, atol=1e-12
    )


def test_rope_inverse_round_trips(rng):
    """Rotating forward then back must be the identity (orthogonality)."""
    x = rng.standard_normal((1, 2, 5, 4))
    y = core.rope_f64(x, 2, 10000.0)
    # The VJP path applies the negated rotation; feeding it y recovers x.
    (back,) = core.rope_f64(np.zeros_like(x), 2, 10000.0, grad_output=y)
    np.testing.assert_allclose(back, x, rtol=1e-11, atol=1e-12)


def test_softmax_causal(rng):
    s = rng.standard_normal((3, 6, 6))
    out = core.softmax_causal_f64(s)
    np.testing.assert_allclose(out, ref.softmax_causal(s), rtol=RTOL, atol=ATOL)


def test_softmax_causal_masks_future_and_sums_to_one(rng):
    s = rng.standard_normal((2, 5, 5))
    out = core.softmax_causal_f64(s)
    for i in range(5):
        assert np.all(out[:, i, i + 1 :] == 0.0), "future positions must be exactly zero"
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-12)


def test_silu(rng):
    x = rng.standard_normal((5, 9))
    np.testing.assert_allclose(core.silu_f64(x), ref.silu(x), rtol=RTOL, atol=ATOL)


def test_swiglu(rng):
    x = rng.standard_normal((6, 8))
    wg = rng.standard_normal((8, 12))
    wu = rng.standard_normal((8, 12))
    wd = rng.standard_normal((12, 8))
    np.testing.assert_allclose(
        core.swiglu_f64(x, wg, wu, wd), ref.swiglu(x, wg, wu, wd), rtol=RTOL, atol=ATOL
    )


def test_embedding(rng):
    table = rng.standard_normal((20, 6))
    ids = rng.integers(0, 20, size=11).astype(np.int32)
    np.testing.assert_allclose(
        core.embedding_f64(table, ids), ref.embedding(table, ids), rtol=RTOL, atol=ATOL
    )


def test_cross_entropy(rng):
    logits = rng.standard_normal((10, 17))
    targets = rng.integers(0, 17, size=10).astype(np.int32)
    assert core.cross_entropy_f64(logits, targets) == pytest.approx(
        ref.cross_entropy(logits, targets), rel=1e-12
    )


def test_cross_entropy_uniform_logits_equals_log_vocab():
    """Uniform logits must give exactly log(V) — an absolute, analytic anchor."""
    v = 32
    logits = np.zeros((4, v))
    targets = np.zeros(4, dtype=np.int32)
    assert core.cross_entropy_f64(logits, targets) == pytest.approx(np.log(v), rel=1e-12)


def test_attention(rng):
    b, t, h, dh = 2, 5, 3, 4
    c = h * dh
    x = rng.standard_normal((b, t, c))
    w = [rng.standard_normal((c, c)) * 0.1 for _ in range(4)]
    np.testing.assert_allclose(
        core.attention_f64(x, *w, h, 10000.0),
        ref.attention(x, *w, h, 10000.0),
        rtol=1e-10,
        atol=1e-11,
    )


def test_attention_first_position_ignores_weights_of_later_tokens(rng):
    """Causality: position 0 attends only to itself, so later tokens cannot change it."""
    b, t, h, dh = 1, 6, 2, 4
    c = h * dh
    x = rng.standard_normal((b, t, c))
    w = [rng.standard_normal((c, c)) * 0.1 for _ in range(4)]
    out_a = core.attention_f64(x, *w, h, 10000.0)

    x2 = x.copy()
    x2[:, 3:, :] += rng.standard_normal((b, t - 3, c))  # perturb the future
    out_b = core.attention_f64(x2, *w, h, 10000.0)

    np.testing.assert_allclose(out_a[:, :3], out_b[:, :3], rtol=1e-11, atol=1e-12)
    assert not np.allclose(out_a[:, 3:], out_b[:, 3:]), "the future should have changed"


def test_float32_ops_agree_at_reduced_precision(rng):
    """Spot-check that the f32 instantiation computes the same thing, just noisier."""
    x = rng.standard_normal((6, 8))
    gain = rng.standard_normal(8)
    np.testing.assert_allclose(
        core.rmsnorm_f32(x.astype(np.float32), gain.astype(np.float32), 1e-5),
        ref.rmsnorm(x, gain, 1e-5),
        rtol=1e-5,
        atol=1e-6,
    )
