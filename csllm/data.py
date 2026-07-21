"""Corpus download, binarization, and batch sampling.

NumPy appears here for memmap I/O and batch slicing only — never on the model's
forward or backward path, which is the C++ engine's job.

Token ids are stored as ``uint16``: the vocabulary is 4096, so 2 bytes per token
halves the file and the page-cache footprint versus int32. The batch sampler
widens to int32 at the last moment because that is what the C++ binding takes.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .tokenizer import BPETokenizer

__all__ = [
    "TINYSHAKESPEARE_URL",
    "binarize",
    "download_tinyshakespeare",
    "encode_documents",
    "get_batch",
    "load_split",
    "prepare_dataset",
]

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

DATA_DIR = Path("data")
TOKENIZER_DIR = DATA_DIR / "tokenizer"


def download_tinyshakespeare(dest: str | Path = DATA_DIR / "tinyshakespeare.txt") -> Path:
    """Fetch the corpus once; subsequent calls reuse the local copy."""
    path = Path(dest)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {TINYSHAKESPEARE_URL} -> {path}")
    with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def binarize(
    text: str,
    tokenizer: BPETokenizer,
    out_dir: str | Path = DATA_DIR,
    val_fraction: float = 0.1,
) -> tuple[Path, Path]:
    """Encode the corpus once and write ``train.bin`` / ``val.bin`` as uint16.

    The split is by position, not shuffled: validation is the tail of the text,
    so it measures generalisation to unseen passages rather than to unseen
    samples of memorised ones.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if tokenizer.vocab_size > 65536:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} exceeds what uint16 can store")

    ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    split = int(len(ids) * (1.0 - val_fraction))
    train_path, val_path = out / "train.bin", out / "val.bin"
    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)

    ratio = len(text) / max(1, len(ids))
    print(
        f"binarized {len(text):,} chars -> {len(ids):,} tokens "
        f"({ratio:.2f} chars/token); train={split:,} val={len(ids) - split:,}"
    )
    return train_path, val_path


def encode_documents(documents: Iterable[str], tokenizer: BPETokenizer) -> np.ndarray:
    """Encode a document stream into one uint16 id array.

    Documents are joined with the tokenizer's end-of-text id when it has one.
    Without a separator, a training window sampled across a document boundary
    would ask the model to predict the start of an unrelated document from the
    end of the previous one — teaching it a transition that does not exist.

    A separator is only inserted BETWEEN documents, never trailing, so a
    single-document corpus is byte-identical to ``binarize``.
    """
    eot = tokenizer.eot_id
    chunks: list[np.ndarray] = []
    for i, doc in enumerate(documents):
        if i and eot is not None:
            chunks.append(np.array([eot], dtype=np.uint16))
        chunks.append(np.asarray(tokenizer.encode(doc), dtype=np.uint16))
    if not chunks:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(chunks)


def prepare_dataset(
    paths: Sequence[str | Path],
    tokenizer: BPETokenizer,
    out_dir: str | Path = DATA_DIR,
    val_fraction: float = 0.1,
    **plugin_options,
) -> tuple[Path, Path]:
    """Read files through the dataset registry and write train/val splits.

    The registry picks a reader per extension, so this works for any format a
    plugin covers. See ``datasets/``.
    """
    from datasets import iter_documents  # local: keeps `datasets` optional for the gateway

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if tokenizer.vocab_size > 65536:
        raise ValueError(f"vocab_size {tokenizer.vocab_size} exceeds what uint16 can store")

    ids = encode_documents(iter_documents(paths, **plugin_options), tokenizer)
    if len(ids) < 2:
        raise ValueError("dataset produced fewer than 2 tokens; check the files and plugin options")

    split = int(len(ids) * (1.0 - val_fraction))
    train_path, val_path = out / "train.bin", out / "val.bin"
    ids[:split].tofile(train_path)
    ids[split:].tofile(val_path)
    print(
        f"prepared {len(paths)} file(s) -> {len(ids):,} tokens; "
        f"train={split:,} val={len(ids) - split:,}"
    )
    return train_path, val_path


def load_split(path: str | Path) -> np.ndarray:
    """Memory-map a ``.bin`` split — the OS page cache does the buffering."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``batch_size`` random windows.

    Targets are inputs shifted by one: predicting token t+1 from tokens <= t is
    the autoregressive objective.
    """
    if len(data) <= block_size + 1:
        raise ValueError(f"split has {len(data)} tokens, need more than block_size+1")
    offsets = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[o : o + block_size] for o in offsets]).astype(np.int32)
    y = np.stack([data[o + 1 : o + 1 + block_size] for o in offsets]).astype(np.int32)
    return x, y
