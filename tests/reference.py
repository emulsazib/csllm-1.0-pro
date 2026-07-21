"""NumPy reference implementations — Phase 2.

The oracle every C++ op is checked against. Each op needs a forward reference
here plus a double-precision finite-difference gradcheck before it counts as
done (memory-bank/rules.md).

Planned references: matmul, rmsnorm, rope, softmax_causal, silu, swiglu,
embedding, cross_entropy, attention.

Why double precision matters: fp32 central differences carry roughly 1e-3
relative noise, which is the same order as a subtly wrong gradient. Running the
check in float64 separates a real bug from rounding. This is why the C++ ops are
templated on scalar type.
"""

from __future__ import annotations

import numpy as np

__all__ = ["finite_difference_grad"]


def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar-valued ``f`` at ``x``.

    Intended for float64 inputs only — see the module docstring.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad
