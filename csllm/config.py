"""Model configuration: JSON on disk, ``ModelConfig`` in C++.

Python owns parsing (``json`` is right there); the C++ side owns validation, so
there is exactly one implementation of the invariants — notably that ``head_dim``
must be even for RoPE's pairwise rotation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._csllm_core import ModelConfig

_FIELDS = (
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "block_size",
    "ffn_hidden",
    "rope_theta",
    "norm_eps",
)


def config_from_dict(d: dict[str, Any]) -> ModelConfig:
    """Build and validate a ModelConfig from a plain dict."""
    unknown = set(d) - set(_FIELDS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    cfg = ModelConfig()
    for key in _FIELDS:
        if key in d:
            setattr(cfg, key, d[key])
    cfg.validate()  # C++ owns the invariants
    return cfg


def load_config(path: str | Path) -> ModelConfig:
    """Load ``configs/*.json`` into a validated ModelConfig."""
    data = json.loads(Path(path).read_text())
    data.pop("_comment", None)
    return config_from_dict(data)


def config_to_dict(cfg: ModelConfig) -> dict[str, Any]:
    return {key: getattr(cfg, key) for key in _FIELDS}
