"""Double-precision gradient checks — the load-bearing verification of this project.

Every backward pass in CSLLM is derived by hand. Each is checked here against
central finite differences of the corresponding C++ forward:

    dL/dxᵢ ≈ [ L(x + εeᵢ) − L(x − εeᵢ) ] / 2ε

with L = Σ(output ⊙ g) for a fixed random g. All of it runs in float64: fp32
central differences carry ~1e-3 relative noise, indistinguishable from a subtly
wrong gradient, which is exactly why the C++ ops are templated on scalar type.

Tensors are kept small — finite differences cost two forward passes per element.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core

from . import reference as ref

EPS = 1e-6
RTOL = 1e-6
ATOL = 1e-9


def check(fn, arrays, *, eps=EPS, rtol=RTOL, atol=ATOL, seed=0, names=None):
    """Compare fn's analytic gradients to finite differences of fn's own forward.

    ``fn(*arrays)`` returns the forward value; ``fn(*arrays, grad_output=g)``
    returns a tuple of input gradients in the same order as ``arrays``.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(fn(*arrays), dtype=np.float64)
    is_scalar = out.ndim == 0

    # A random cotangent makes the check sensitive to every output element;
    # ones() could mask errors that cancel across a row.
    g = np.ones(1) if is_scalar else rng.standard_normal(out.shape)
    analytic = fn(*arrays, grad_output=g)

    def loss(_unused=None) -> float:
        o = np.asarray(fn(*arrays), dtype=np.float64)
        return float(o) if is_scalar else float((o * g).sum())

    for i, arr in enumerate(arrays):
        numeric = ref.finite_difference_grad(loss, arr, eps)
        label = names[i] if names else f"input[{i}]"
        np.testing.assert_allclose(
            analytic[i], numeric, rtol=rtol, atol=atol, err_msg=f"gradient mismatch for {label}"
        )


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_matmul_grad(rng):
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((3, 5))
    check(core.matmul_f64, [x, w], names=["x", "w"])


def test_matmul_bt_grad(rng):
    x = rng.standard_normal((4, 3))
    w = rng.standard_normal((5, 3))
    check(core.matmul_bt_f64, [x, w], names=["x", "w"])


def test_add_grad(rng):
    check(core.add_f64, [rng.standard_normal((3, 4)), rng.standard_normal((3, 4))])


def test_rmsnorm_grad(rng):
    x = rng.standard_normal((5, 8))
    gain = rng.standard_normal(8)
    check(lambda a, b, grad_output=None: core.rmsnorm_f64(a, b, 1e-5, grad_output),
          [x, gain], names=["x", "gain"])


def test_rmsnorm_grad_with_tiny_inputs(rng):
    """Near-zero rows exercise the 1/r³ term where eps dominates the denominator."""
    x = rng.standard_normal((4, 6)) * 1e-3
    gain = rng.standard_normal(6)
    check(lambda a, b, grad_output=None: core.rmsnorm_f64(a, b, 1e-5, grad_output),
          [x, gain], names=["x", "gain"], rtol=1e-5)


@pytest.mark.parametrize("pos_offset", [0, 4])
def test_rope_grad(rng, pos_offset):
    x = rng.standard_normal((2, 2, 5, 6))
    check(lambda a, grad_output=None: core.rope_f64(a, pos_offset, 10000.0, grad_output), [x])


def test_softmax_causal_grad(rng):
    check(core.softmax_causal_f64, [rng.standard_normal((2, 5, 5))])


def test_silu_grad(rng):
    check(core.silu_f64, [rng.standard_normal((4, 7))])


def test_swiglu_grad(rng):
    x = rng.standard_normal((3, 5))
    wg = rng.standard_normal((5, 6))
    wu = rng.standard_normal((5, 6))
    wd = rng.standard_normal((6, 5))
    check(core.swiglu_f64, [x, wg, wu, wd], names=["x", "wg", "wu", "wd"])


def test_embedding_grad(rng):
    table = rng.standard_normal((9, 4))
    ids = rng.integers(0, 9, size=7).astype(np.int32)
    check(lambda t, grad_output=None: core.embedding_f64(t, ids, grad_output), [table],
          names=["table"])


def test_embedding_grad_with_repeated_ids(rng):
    """Repeated ids must ACCUMULATE into the same row, not overwrite it."""
    table = rng.standard_normal((5, 3))
    ids = np.array([2, 2, 2, 0, 2], dtype=np.int32)
    check(lambda t, grad_output=None: core.embedding_f64(t, ids, grad_output), [table],
          names=["table"])


def test_cross_entropy_grad(rng):
    logits = rng.standard_normal((6, 11))
    targets = rng.integers(0, 11, size=6).astype(np.int32)
    check(lambda x, grad_output=None: core.cross_entropy_f64(x, targets, grad_output), [logits],
          names=["logits"])


def test_attention_grad(rng):
    """The composite one: RoPE + causal mask + softmax + three projections."""
    b, t, h, dh = 1, 4, 2, 4
    c = h * dh
    x = rng.standard_normal((b, t, c)) * 0.5
    w = [rng.standard_normal((c, c)) * 0.3 for _ in range(4)]
    check(
        lambda *a, grad_output=None: core.attention_f64(*a, h, 10000.0, grad_output),
        [x, *w],
        names=["x", "wq", "wk", "wv", "wo"],
        rtol=1e-5,
        atol=1e-8,
    )


def test_attention_grad_multi_batch(rng):
    b, t, h, dh = 2, 3, 2, 4
    c = h * dh
    x = rng.standard_normal((b, t, c)) * 0.5
    w = [rng.standard_normal((c, c)) * 0.3 for _ in range(4)]
    check(
        lambda *a, grad_output=None: core.attention_f64(*a, h, 10000.0, grad_output),
        [x, *w],
        rtol=1e-5,
        atol=1e-8,
    )


# ── end-to-end: the whole network, including weight tying ────────────────────


def _tiny_model():
    cfg = core.ModelConfig()
    cfg.vocab_size = 7
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 8
    cfg.block_size = 6
    cfg.ffn_hidden = 12
    return core.ModelF64(cfg, 3)


@pytest.mark.parametrize(
    "param",
    [
        "tok_emb",
        "blocks.0.attn_norm.gain",
        "blocks.0.attn.wq",
        "blocks.0.attn.wo",
        "blocks.0.ffn.wg",
        "blocks.0.ffn.wd",
        "blocks.1.attn.wv",
        "norm_f.gain",
    ],
)
def test_model_parameter_gradients(param):
    """Full-network gradcheck in float64.

    ``tok_emb`` is the important case: it is weight-tied to lm_head, so its
    gradient must accumulate from BOTH the embedding scatter-add and the output
    projection. Dropping either path still trains — just wrongly — and this is
    the only check that catches it.
    """
    model = _tiny_model()
    rng = np.random.default_rng(11)
    ids = rng.integers(0, 7, size=(2, 5)).astype(np.int32)
    targets = rng.integers(0, 7, size=(2, 5)).astype(np.int32)

    model.zero_grad()
    model.forward_loss(ids, targets)
    model.backward()
    analytic = np.array(model.get_grad(param), copy=True)

    weights = model.get_param(param)  # zero-copy: mutating this edits the model
    numeric = np.zeros_like(analytic)
    flat_w, flat_n = weights.reshape(-1), numeric.reshape(-1)
    eps = 1e-6
    for i in range(flat_w.size):
        original = flat_w[i]
        flat_w[i] = original + eps
        plus = model.forward_loss(ids, targets)
        flat_w[i] = original - eps
        minus = model.forward_loss(ids, targets)
        flat_w[i] = original
        flat_n[i] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-9)


def test_weight_tying_uses_both_gradient_paths():
    """Rows never used as INPUT tokens still get gradient — via lm_head only.

    If tying were implemented as two independent tensors, or if the lm_head
    contribution were dropped, these rows would have exactly zero gradient.
    """
    model = _tiny_model()
    # Only ids 0 and 1 appear as inputs; every row still receives lm_head gradient.
    ids = np.array([[0, 1, 0, 1, 0]], dtype=np.int32)
    targets = np.array([[1, 0, 1, 0, 1]], dtype=np.int32)

    model.zero_grad()
    model.forward_loss(ids, targets)
    model.backward()
    grad = np.array(model.get_grad("tok_emb"))

    unused_rows = grad[2:]  # never embedded, never a target
    assert np.all(np.abs(unused_rows).sum(axis=1) > 1e-12), (
        "rows absent from the input must still receive lm_head gradient under weight tying"
    )
