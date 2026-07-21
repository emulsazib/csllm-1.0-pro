"""Corpus download, binarization, and batch sampling — Phase 3.

Planned surface:
    def download_tinyshakespeare(dest: Path) -> Path
    def binarize(text: str, tok: BPETokenizer, out: Path) -> None   # -> uint16 memmap
    def split(ids: np.ndarray, frac: float = 0.9) -> tuple[np.ndarray, np.ndarray]
    def get_batch(data, batch_size, block_size, rng) -> tuple[np.ndarray, np.ndarray]

NumPy is used here for memmap I/O and batch slicing only. It must never appear
on the model's forward or backward path — that is the C++ engine's job.
"""

from __future__ import annotations

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

__all__ = ["TINYSHAKESPEARE_URL"]
