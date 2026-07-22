"""Standalone loader for a CSLLM export bundle.

    from load import CSLLMBundle
    bundle = CSLLMBundle(".")
    ids = bundle.tokenizer.encode("KING RICHARD:")
    weights = bundle.weights            # {name: np.ndarray}

No torch, no TensorFlow, and no dependency on the repository that produced this
bundle. Requires only numpy, safetensors and regex (see requirements.txt).

Run it directly for a self-check:

    python load.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import regex
from safetensors.numpy import load_file


def _merge_sequence(symbols, pair, new_id):
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
    out = []
    i = 0
    n = len(symbols)
    first, second = pair
    while i < n:
        if i < n - 1 and symbols[i] == first and symbols[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BundleTokenizer:
    """Byte-level BPE rebuilt from ``tokenizer.json`` alone."""

    def __init__(self, payload: dict):
        self.pattern = payload["pattern"]
        self.vocab_size = payload["vocab_size"]
        self.special_tokens = payload.get("special_tokens", {}) or {}
        self._re = regex.compile(self.pattern)

        # Index in the merges list IS the rank, and rank + 256 is the token id.
        # Encoding reproduces the training-time segmentation only if merges are
        # applied lowest-rank-first, so this ordering is load bearing.
        self.merges = {
            (int(a), int(b)): 256 + i for i, (a, b) in enumerate(payload["merges"])
        }
        # Byte lists, not strings: a token need not be valid UTF-8 on its own.
        self.vocab = {int(i): bytes(b) for i, b in payload["vocab"].items()}
        self._cache: dict[str, list[int]] = {}

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        symbols = list(chunk.encode("utf-8"))
        while len(symbols) >= 2:
            best_pair, best_rank = None, None
            # strict=False: the two views differ in length by one, deliberately.
            for pair in zip(symbols, symbols[1:], strict=False):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            symbols = _merge_sequence(symbols, best_pair, best_rank)
        self._cache[chunk] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """Encode to token ids. Special tokens are NOT recognised in the input,
        so untrusted text cannot inject a control token by containing its
        literal spelling."""
        ids = []
        for chunk in self._re.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode_bytes(self, ids) -> bytes:
        return b"".join(self.vocab[int(i)] for i in ids)

    def decode(self, ids, errors: str = "replace") -> str:
        # A single token can end mid-UTF-8-sequence, so decode the joined bytes
        # rather than each token independently.
        return self.decode_bytes(ids).decode("utf-8", errors=errors)


class CSLLMBundle:
    """A loaded export: config, tokenizer and weights."""

    def __init__(self, directory: str | Path = "."):
        root = Path(directory)
        self.config = json.loads((root / "config.json").read_text())
        self.architecture = self.config["architecture"]
        self.hparams = self.config["config"]
        self.tokenizer = BundleTokenizer(json.loads((root / "tokenizer.json").read_text()))
        self.weights = load_file(str(root / "model.safetensors"))

        vocab = self.hparams["vocab_size"]
        if self.tokenizer.vocab_size != vocab:
            raise ValueError(
                f"tokenizer vocab_size={self.tokenizer.vocab_size} != model {vocab}; "
                "this bundle is inconsistent"
            )

    def __repr__(self) -> str:
        h = self.hparams
        return (
            f"<CSLLMBundle {self.config['num_params']:,} params "
            f"{h['n_layer']}L x {h['n_head']}H x {h['n_embd']}d ctx {h['block_size']}>"
        )

    def embed(self, ids) -> np.ndarray:
        """Token ids -> embedding rows. ``lm_head`` is TIED to this matrix."""
        return self.weights["tok_emb"][np.asarray(ids, dtype=np.int64)]


def main() -> None:
    bundle = CSLLMBundle(Path(__file__).resolve().parent.parent)
    print(bundle)
    print(f"architecture : {bundle.architecture}")
    print(f"tensors      : {len(bundle.weights)}")

    probe = "KING RICHARD:"
    ids = bundle.tokenizer.encode(probe)
    print(f"encode({probe!r}) -> {ids}")
    assert bundle.tokenizer.decode(ids) == probe, "tokenizer round-trip failed"
    print(f"round-trip   : OK ({len(ids)} tokens)")
    print(f"embed shape  : {bundle.embed(ids).shape}")


if __name__ == "__main__":
    main()
