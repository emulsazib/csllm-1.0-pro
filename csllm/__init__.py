"""CSLLM — an autoregressive Transformer built from scratch.

The C++ engine (``_csllm_core``) owns all tensor math, memory, and gradients.
This package is the Python surface: tokenizer, config, and data handling.

Deliberately absent: torch / tensorflow / jax. NumPy is used for tests and data
preparation only, never on the model's forward or backward path.
"""

from __future__ import annotations

__version__ = "1.0.0"

try:
    from ._csllm_core import (
        AdamWConfig,
        Arena,
        BuildInfo,
        ModelConfig,
        SamplingParams,
        build_info,
        cosine_lr,
        matmul_f32,
        matmul_f64,
        thread_pool_size,
    )
except ImportError as exc:  # pragma: no cover - build guidance only
    raise ImportError(
        "The CSLLM C++ extension is not built.\n"
        "  Build it with:  make build      (or: pip install -e .)\n"
        f"Original error: {exc}"
    ) from exc

from .config import load_config  # noqa: E402

__all__ = [
    "AdamWConfig",
    "Arena",
    "BuildInfo",
    "ModelConfig",
    "SamplingParams",
    "__version__",
    "build_info",
    "cosine_lr",
    "load_config",
    "matmul_f32",
    "matmul_f64",
    "thread_pool_size",
]
