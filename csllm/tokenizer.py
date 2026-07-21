"""Byte-level Byte-Pair Encoding.

Lives in the shared package rather than ``train/`` because the FastAPI gateway
needs ``encode``/``decode`` at request time; one implementation serves both
training and serving so they cannot drift.

Design notes:

* **Byte-level.** Base vocabulary is the 256 byte values, so any input round-trips
  losslessly — including emoji, control characters, and invalid-looking sequences.
  There is no ``<unk>`` because there cannot be one.
* **GPT-2 style pre-tokenization.** A regex splits text into word-ish chunks before
  merging, which stops BPE from learning merges that straddle word boundaries
  (e.g. ``"e th"``).
* **Incremental merge training.** Naively recounting all pairs each merge is
  O(corpus x merges) and takes minutes. Instead we keep unique pre-tokens with
  frequencies, an inverted index from pair to the words containing it, and update
  only the affected words after each merge.
* **Special tokens are opt-in.** A single continuous corpus (TinyShakespeare) has
  no document boundaries, so an EOS token would never be trained and could never
  be emitted. Multi-document corpora (`.jsonl`, `.csv`) DO have boundaries, and
  without a separator a training window silently straddles unrelated documents.
  So `train(..., special_tokens=("<|endoftext|>",))` reserves ids above the
  merges. With none requested the tokenizer behaves byte-identically to before.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import regex

__all__ = ["END_OF_TEXT", "BPETokenizer", "GPT2_SPLIT_PATTERN"]

#: Conventional document separator, matching GPT-2's name.
END_OF_TEXT = "<|endoftext|>"

# GPT-2's pre-tokenization pattern: contractions, letter runs, digit runs,
# punctuation runs, and whitespace (trailing whitespace kept with the next word).
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge_sequence(symbols: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``."""
    out: list[int] = []
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


class BPETokenizer:
    """Trainable byte-level BPE tokenizer."""

    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self._re = regex.compile(pattern)
        # (a, b) -> merged id. Ids increase with merge order, so the id doubles
        # as the merge rank, which is what encoding needs.
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Special token text -> id. Ids sit above the merges and are never
        # produced by ordinary encoding, only by encode(..., allow_special=True).
        self.special_tokens: dict[str, int] = {}
        self._special_re: regex.Pattern[str] | None = None
        self._cache: dict[str, list[int]] = {}

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int | None:
        """Id of the end-of-text separator, or None if this tokenizer has none."""
        return self.special_tokens.get(END_OF_TEXT)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        extra = f" special={len(self.special_tokens)}" if self.special_tokens else ""
        return f"<BPETokenizer vocab_size={self.vocab_size} merges={len(self.merges)}{extra}>"

    # ── special tokens ───────────────────────────────────────────────────────

    def add_special_tokens(self, tokens: Sequence[str]) -> None:
        """Append special tokens, each taking the next free id."""
        for token in tokens:
            if not token:
                raise ValueError("special tokens must be non-empty")
            if token in self.special_tokens:
                continue
            new_id = len(self.vocab)
            self.special_tokens[token] = new_id
            # Store the literal text so decode() renders it readably.
            self.vocab[new_id] = token.encode("utf-8")
        self._rebuild_special_pattern()

    def _rebuild_special_pattern(self) -> None:
        if not self.special_tokens:
            self._special_re = None
            return
        # Longest-first so "<|a|><|ab|>" cannot mis-split on a shorter prefix.
        alternatives = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("(" + "|".join(map(regex.escape, alternatives)) + ")")

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: Sequence[str] = (),
        verbose: bool = False,
    ) -> None:
        """Learn merges until the vocabulary reaches ``vocab_size``.

        ``vocab_size`` is the TOTAL including any special tokens, so it matches
        the model config's ``vocab_size`` exactly and the embedding table is
        sized correctly.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (the byte alphabet), got {vocab_size}")
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} leaves no room for {len(special_tokens)} "
                f"special tokens on top of the 256-byte alphabet"
            )

        # Unique pre-tokens with frequencies: the corpus collapses from ~1 MB of
        # characters to a few tens of thousands of distinct words.
        word_freq = Counter(self._re.findall(text))
        symbols: list[list[int]] = [list(w.encode("utf-8")) for w in word_freq]
        counts: list[int] = list(word_freq.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, syms in enumerate(symbols):
            c = counts[wi]
            for pair in zip(syms, syms[1:], strict=False):
                pair_counts[pair] += c
                pair_words[pair].add(wi)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens = {}
        self._special_re = None
        self._cache.clear()

        for i in range(num_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < 2:
                break  # nothing left worth merging

            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Only words containing `best` can change.
            for wi in list(pair_words[best]):
                syms = symbols[wi]
                c = counts[wi]
                for pair in zip(syms, syms[1:], strict=False):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_words[pair].discard(wi)

                merged = _merge_sequence(syms, best, new_id)
                symbols[wi] = merged
                for pair in zip(merged, merged[1:], strict=False):
                    pair_counts[pair] += c
                    pair_words[pair].add(wi)

            pair_words.pop(best, None)
            pair_counts.pop(best, None)

            if verbose and (i + 1) % 500 == 0:
                token = self.vocab[new_id]
                print(f"  merge {i + 1}/{num_merges}: {best} -> {new_id} {token!r}")

        # Specials go last so their ids sit above every merge.
        self.add_special_tokens(special_tokens)

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        symbols = list(chunk.encode("utf-8"))
        while len(symbols) >= 2:
            # Apply the lowest-ranked (earliest-learned) applicable merge, which
            # is what makes encoding reproduce the training-time segmentation.
            best_pair = None
            best_rank = None
            for pair in zip(symbols, symbols[1:], strict=False):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            symbols = _merge_sequence(symbols, best_pair, best_rank)

        self._cache[chunk] = symbols
        return symbols

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._re.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode to token ids.

        ``allow_special`` is off by default so untrusted input can never inject a
        control token by simply containing its literal text — the same guarantee
        tiktoken makes. Turn it on only for text you construct yourself.
        """
        if not allow_special or self._special_re is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            special = self.special_tokens.get(part)
            if special is not None:
                ids.append(special)
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    # ── decoding ─────────────────────────────────────────────────────────────

    def decode_bytes(self, ids: list[int] | tuple[int, ...]) -> bytes:
        """Raw bytes for these ids.

        The gateway streams with this: a single token can end mid-UTF-8-sequence,
        so bytes must be buffered and decoded only at code-point boundaries.
        """
        try:
            return b"".join(self.vocab[int(i)] for i in ids)
        except KeyError as exc:
            raise ValueError(f"token id {exc.args[0]} is outside the vocabulary") from None

    def decode(self, ids: list[int] | tuple[int, ...], errors: str = "replace") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Write ``merges.txt`` (canonical) and ``vocab.json`` (for inspection)."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        lines = [f"{a} {b}" for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])]
        (path / "merges.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        # The vocabulary is fully determined by the merges; this file exists so a
        # human can see what was learned. Byte lists keep it unambiguous for
        # tokens that are not valid UTF-8 on their own.
        meta = {
            "vocab_size": self.vocab_size,
            "pattern": self.pattern,
            # Canonical: ids are NOT derivable from merges alone once specials exist.
            "special_tokens": self.special_tokens,
            "tokens": {
                str(i): {"bytes": list(b), "text": b.decode("utf-8", errors="replace")}
                for i, b in sorted(self.vocab.items())
            },
        }
        (path / "vocab.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, directory: str | Path) -> BPETokenizer:
        path = Path(directory)
        meta_file = path / "vocab.json"
        pattern = GPT2_SPLIT_PATTERN
        meta: dict = {}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            pattern = meta.get("pattern", GPT2_SPLIT_PATTERN)

        tok = cls(pattern)
        merges_text = (path / "merges.txt").read_text().strip()
        if merges_text:
            for i, line in enumerate(merges_text.splitlines()):
                a, b = line.split()
                new_id = 256 + i
                tok.merges[(int(a), int(b))] = new_id
                tok.vocab[new_id] = tok.vocab[int(a)] + tok.vocab[int(b)]

        # Restored by explicit id rather than re-appended: a tokenizer trained
        # with a truncated merge list would otherwise get different ids on load,
        # silently invalidating every checkpoint trained against it.
        for token, token_id in (meta.get("special_tokens") or {}).items():
            tok.special_tokens[token] = int(token_id)
            tok.vocab[int(token_id)] = token.encode("utf-8")
        tok._rebuild_special_pattern()
        return tok
