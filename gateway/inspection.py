"""Shared next-token analysis for the REST endpoint and the WebSocket stream.

Both surfaces must report the same thing, so the computation lives here once.
The filtered probabilities come from the C++ ``filtered_distribution`` that
``Sampler::sample`` is itself built on — the same no-drift argument, one level up.
"""

from __future__ import annotations

import numpy as np

from csllm import _csllm_core as core

from .schemas import CandidateToken

__all__ = ["distribution_stats", "top_candidates"]


def _raw_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax at temperature 1 — the model's own belief, independent of the knobs."""
    shifted = logits.astype(np.float64) - logits.max()
    raw = np.exp(shifted)
    return raw / raw.sum()


def top_candidates(
    logits: np.ndarray,
    tokenizer,
    params: core.SamplingParams,
    top_n: int,
) -> tuple[list[CandidateToken], np.ndarray, np.ndarray]:
    """Return the top-N candidates plus the raw and filtered distributions."""
    filtered = np.asarray(core.filtered_distribution(logits, params))
    raw = _raw_probabilities(logits)

    # Rank by whichever is larger so a token the filters just excluded stays
    # visible — those are exactly the ones a user is judging the settings by.
    order = np.argsort(np.maximum(raw, filtered))[::-1][:top_n]

    candidates = [
        CandidateToken(
            id=int(i),
            text=tokenizer.decode_bytes([int(i)]).decode("utf-8", errors="replace"),
            logit=float(logits[i]),
            raw_prob=float(raw[i]),
            prob=float(filtered[i]),
            kept=bool(filtered[i] > 0),
        )
        for i in order
    ]
    return candidates, raw, filtered


def distribution_stats(raw: np.ndarray, filtered: np.ndarray) -> tuple[float, float, int]:
    """(raw entropy, filtered entropy, kept count), in nats.

    max(0, …) because a one-hot (greedy) distribution yields -0.0, which renders
    as "-0.000" and reads like a bug.
    """
    raw_nz = raw[raw > 0]
    filt_nz = filtered[filtered > 0]
    return (
        max(0.0, float(-(raw_nz * np.log(raw_nz)).sum())),
        max(0.0, float(-(filt_nz * np.log(filt_nz)).sum())),
        int((filtered > 0).sum()),
    )
