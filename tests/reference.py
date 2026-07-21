"""NumPy reference implementations — the oracle every C++ forward op is checked against.

Verification is split in two, and the split is what makes it trustworthy:

  * ``test_ops.py``       C++ forward  ==  the NumPy forward here
  * ``test_gradcheck.py`` C++ backward ==  finite differences of the C++ forward

Together they pin the backward to the true mathematics. Neither alone would:
matching NumPy says nothing about gradients, and a self-consistent fwd/bwd pair
could both be wrong in the same way.

Everything here is float64. fp32 central differences carry roughly 1e-3 relative
noise — the same order as a subtly wrong gradient — so the C++ ops are templated
on scalar type specifically to make this double-precision check possible.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "add",
    "attention",
    "cross_entropy",
    "embedding",
    "finite_difference_grad",
    "matmul",
    "matmul_bt",
    "rmsnorm",
    "rope",
    "silu",
    "softmax_causal",
    "swiglu",
]


def matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w


def matmul_bt(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w.T


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def rmsnorm(x: np.ndarray, gain: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    r = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return gain * x / r


def rope(x: np.ndarray, pos_offset: int = 0, theta: float = 10000.0) -> np.ndarray:
    """Rotary embedding over [B, H, T, Dh], INTERLEAVED pair convention.

    Channels (2p, 2p+1) form the 2-D vector that gets rotated. The other common
    convention pairs d with d+Dh/2; the two are not interchangeable, and this one
    must match core/src/ops_rope.cpp.
    """
    _, _, t_len, dh = x.shape
    half = dh // 2
    p = np.arange(half, dtype=np.float64)
    inv_freq = theta ** (-2.0 * p / dh)
    pos = np.arange(t_len, dtype=np.float64) + pos_offset
    angle = pos[:, None] * inv_freq[None, :]  # [T, half]
    cos, sin = np.cos(angle), np.sin(angle)

    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def softmax_causal(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax over [n, T, T] where key j is visible to query i iff j <= i."""
    _, tq, tk = scores.shape
    mask = np.tril(np.ones((tq, tk), dtype=bool))
    s = np.where(mask, scores, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.where(mask, np.exp(s), 0.0)
    return e / e.sum(axis=-1, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
    return (silu(x @ wg) * (x @ wu)) @ wd


def embedding(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return table[ids]


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    n = logits.shape[0]
    m = logits.max(axis=-1, keepdims=True)
    logsumexp = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    return float((logsumexp - logits[np.arange(n), targets]).mean())


def attention(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    n_head: int,
    theta: float = 10000.0,
    pos_offset: int = 0,
) -> np.ndarray:
    """Causal multi-head self-attention with RoPE. x is [B, T, C]."""
    b, t_len, c = x.shape
    dh = c // n_head
    xf = x.reshape(b * t_len, c)

    def to_heads(proj: np.ndarray) -> np.ndarray:
        return (xf @ proj).reshape(b, t_len, n_head, dh).transpose(0, 2, 1, 3)

    q = rope(to_heads(wq), pos_offset, theta)
    k = rope(to_heads(wk), pos_offset, theta)
    v = to_heads(wv)  # V carries no positional information

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    probs = softmax_causal(scores.reshape(b * n_head, t_len, t_len))
    ctx = probs.reshape(b, n_head, t_len, t_len) @ v

    ctx_flat = ctx.transpose(0, 2, 1, 3).reshape(b * t_len, c)
    return (ctx_flat @ wo).reshape(b, t_len, c)


def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar-valued ``f`` at ``x``, in place-safe form.

    Intended for float64 only — see the module docstring.
    """
    grad = np.zeros_like(x)
    flat = x.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = f(x)
        flat[i] = original - eps
        minus = f(x)
        flat[i] = original
        gflat[i] = (plus - minus) / (2.0 * eps)
    return grad
