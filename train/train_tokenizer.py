"""Train the byte-level BPE tokenizer and binarize the corpus.

    python -m train.train_tokenizer --config configs/shakespeare.json

Downloads TinyShakespeare if needed, learns merges up to the config's
``vocab_size``, writes ``data/tokenizer/{merges.txt,vocab.json}``, then encodes
the corpus once into ``data/train.bin`` and ``data/val.bin``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import csllm
from csllm.data import DATA_DIR, TOKENIZER_DIR, binarize, download_tinyshakespeare
from csllm.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shakespeare.json")
    parser.add_argument("--corpus", default=None, help="text file (defaults to TinyShakespeare)")
    parser.add_argument("--out", default=str(TOKENIZER_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = csllm.load_config(args.config)
    corpus = Path(args.corpus) if args.corpus else download_tinyshakespeare()
    text = corpus.read_text(encoding="utf-8")
    print(f"corpus: {corpus} ({len(text):,} chars)")

    tokenizer = BPETokenizer()
    print(f"training BPE to vocab_size={cfg.vocab_size} ...")
    start = time.time()
    tokenizer.train(text, cfg.vocab_size, verbose=args.verbose)
    print(f"  learned {len(tokenizer.merges):,} merges in {time.time() - start:.1f}s")

    if tokenizer.vocab_size != cfg.vocab_size:
        # Fewer merges than requested means the corpus ran out of repeated pairs.
        # The model's embedding table is sized from the config, so this must be
        # loud rather than silently producing an unusable vocab mismatch.
        raise SystemExit(
            f"tokenizer produced vocab_size={tokenizer.vocab_size} but the config "
            f"expects {cfg.vocab_size}; lower vocab_size or use a larger corpus"
        )

    tokenizer.save(args.out)
    print(f"saved tokenizer -> {args.out}")

    # Round-trip on the real corpus before committing to a training run: a lossy
    # tokenizer would silently cap achievable loss.
    if tokenizer.decode(tokenizer.encode(text)) != text:
        raise SystemExit("tokenizer round-trip FAILED on the corpus")
    print("round-trip on full corpus: OK")

    binarize(text, tokenizer, args.data_dir, args.val_fraction)


if __name__ == "__main__":
    main()
