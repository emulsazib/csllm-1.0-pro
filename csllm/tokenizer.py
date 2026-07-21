"""Byte-level BPE tokenizer — Phase 3.

Lives in the shared package rather than ``train/`` because the FastAPI gateway
needs ``encode``/``decode`` at request time; one implementation serves both
training and serving so they cannot drift.

Planned surface:
    class BPETokenizer:
        def train(self, text: str, vocab_size: int) -> None
        def encode(self, s: str) -> list[int]
        def decode(self, ids: list[int]) -> str
        def save(self, directory: str) -> None
        @classmethod
        def load(cls, directory: str) -> "BPETokenizer"

Design notes carried into implementation:
  * GPT-2 style regex pre-tokenization, then iterative highest-frequency pair merges.
  * Operates on raw bytes, so round-trips are lossless for arbitrary UTF-8.
  * Artifacts: ``vocab.json`` + ``merges.txt``.
  * A single token may decode to a PARTIAL UTF-8 sequence; the gateway must buffer
    incomplete code points rather than emit invalid UTF-8 in an SSE frame.
"""

from __future__ import annotations

__all__ = ["BPETokenizer"]


class BPETokenizer:
    def __init__(self) -> None:
        raise NotImplementedError("BPETokenizer arrives in Phase 3")
