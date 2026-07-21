"""BPE tokenizer training entrypoint — Phase 3.

Downloads TinyShakespeare, trains byte-level BPE merges to the config's
``vocab_size``, and writes ``vocab.json`` + ``merges.txt`` under ``data/tokenizer/``.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Tokenizer training arrives in Phase 3")


if __name__ == "__main__":
    main()
