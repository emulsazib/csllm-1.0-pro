"""Analytic parameter and memory accounting for a model configuration.

The configurator UI needs to answer "how big is this model?" while a slider is
moving — before any model exists. Building one to find out costs a full
allocation and weight init per keystroke, so the count is derived from the
config instead.

The decomposition here MUST agree with ``ModelConfig::num_params()`` in
``core/src/model.cpp``; that is the engine's own count and the only one that can
be wrong in a way users notice (a bundle that reports 12.19 M and loads 13 M).
``tests/test_params.py`` asserts equality across a sweep of configs, so a change
to the C++ layout that is not mirrored here fails the suite rather than silently
misreporting.

Two things the naive formula gets wrong:

* **``lm_head`` is weight-tied to ``tok_emb``** (README architecture table), so
  the embedding matrix is counted exactly ONCE. Counting it twice overstates the
  12 M config by 1.57 M.
* **SwiGLU's FFN has three matrices, not two** — gate, up, and down — so the
  block is ``3 * n_embd * ffn_hidden``, not ``2 *``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from . import _csllm_core as core
from .config import config_from_dict

__all__ = ["MemoryEstimate", "ParamBreakdown", "calculate_model_params"]

#: Weights, gradients, and both AdamW moments are all fp32 in this engine.
BYTES_PER_ELEM = 4

#: AdamW keeps exp_avg and exp_avg_sq per parameter.
OPTIMIZER_STATES = 2


@dataclass(frozen=True)
class ParamBreakdown:
    """Trainable parameters, split by where they live."""

    embedding: int
    attention: int
    ffn: int
    norms: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryEstimate:
    """Bytes needed to *train* at a given batch/sequence shape.

    ``activations`` dominates at realistic batch sizes and is the number that
    decides whether a run fits: at B=8/T=256 the 12 M config needs ~1.5 GB of
    activation arena against 49 MB of weights.
    """

    params: int
    gradients: int
    optimizer: int
    activations: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def calculate_model_params(
    config: dict[str, Any] | Any,
    batch_size: int = 8,
    seq_len: int | None = None,
) -> tuple[ParamBreakdown, MemoryEstimate]:
    """Count trainable parameters and estimate training memory for ``config``.

    ``config`` may be a plain dict or an already-validated ``ModelConfig``. A
    dict goes through ``config_from_dict``, so the C++ invariants (n_embd
    divisible by n_head, even head_dim for RoPE) are enforced here too and an
    impossible architecture raises rather than returning a meaningless number.

    ``seq_len`` defaults to the config's ``block_size`` — the worst case, and
    what the training loop actually uses.
    """
    cfg = config_from_dict(config) if isinstance(config, dict) else config

    n_embd, n_layer = cfg.n_embd, cfg.n_layer

    # Mirrors ModelConfig::num_params() in core/src/model.cpp.
    embedding = cfg.vocab_size * n_embd  # tied with lm_head — counted once
    attention = n_layer * 4 * n_embd * n_embd  # wq, wk, wv, wo
    ffn = n_layer * 3 * n_embd * cfg.ffn_hidden  # SwiGLU: gate, up, down
    norms = n_layer * 2 * n_embd + n_embd  # two RMSNorm gains per block + final

    breakdown = ParamBreakdown(
        embedding=embedding,
        attention=attention,
        ffn=ffn,
        norms=norms,
        total=embedding + attention + ffn + norms,
    )

    params_bytes = breakdown.total * BYTES_PER_ELEM
    # The arena estimate is the engine's own, so the number the UI shows is the
    # number that will actually be allocated.
    activations = core.estimate_activation_bytes(
        cfg, batch_size, seq_len if seq_len is not None else cfg.block_size
    )
    memory = MemoryEstimate(
        params=params_bytes,
        gradients=params_bytes,
        optimizer=params_bytes * OPTIMIZER_STATES,
        activations=activations,
        total=params_bytes * (2 + OPTIMIZER_STATES) + activations,
    )
    return breakdown, memory
