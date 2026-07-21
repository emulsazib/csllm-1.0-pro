"""Train the byte-level BPE tokenizer and binarize a corpus.

    python -m train.train_tokenizer --config configs/shakespeare.json
    python -m train.train_tokenizer --config configs/debug.json --dataset notes.jsonl

Three ways to name the corpus, in precedence order:

``--dataset NAME``   a file in ``datasets/raw/``, read through the plugin
                     registry — so ``.jsonl``/``.csv`` work, not just ``.txt``
``--corpus PATH``    any single text file, read directly
*(neither)*          downloads TinyShakespeare

Writes ``{out}/{merges.txt,vocab.json}`` and ``{data_dir}/{train,val}.bin``.

``--emit-jsonl`` makes this drivable from the dashboard: the gateway supervises
it exactly like a training run (``gateway/telemetry.py``, job kind ``prepare``),
because a dataset must be tokenized and binarized before it can be trained on
and that takes minutes, not milliseconds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import csllm
from csllm.data import DATA_DIR, TOKENIZER_DIR, binarize, download_tinyshakespeare, prepare_dataset
from csllm.tokenizer import BPETokenizer

from .emit import make_emitter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shakespeare.json")
    parser.add_argument("--corpus", default=None, help="text file (defaults to TinyShakespeare)")
    parser.add_argument("--dataset", default=None, help="file in datasets/raw/, read via plugins")
    parser.add_argument("--out", default=str(TOKENIZER_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--emit-jsonl",
        action="store_true",
        help="emit machine-readable progress rows for the gateway",
    )
    parser.add_argument("--run-id", default="local", help="identifier stamped on emitted rows")
    args = parser.parse_args()

    emit = make_emitter(args.emit_jsonl, args.run_id)
    cfg = csllm.load_config(args.config)

    # ── source the text ──────────────────────────────────────────────────────
    dataset_path: Path | None = None
    if args.dataset:
        import datasets as ds

        dataset_path = ds.RAW_DIR / args.dataset
        # `--dataset` is a bare name, not a path: reject traversal the same way
        # the gateway's dataset routes do.
        if dataset_path.parent.resolve() != ds.RAW_DIR.resolve() or not dataset_path.is_file():
            raise SystemExit(f"no such dataset: {args.dataset}")
        print(f"dataset: {dataset_path} (via plugin registry)")
        # BPE needs the whole corpus as one string; iter_documents is lazy for
        # the binarize pass, but merge counting is global by nature.
        text = "\n".join(ds.iter_documents([dataset_path]))
    else:
        corpus = Path(args.corpus) if args.corpus else download_tinyshakespeare()
        print(f"corpus: {corpus} ({corpus.stat().st_size:,} bytes)")
        text = corpus.read_text(encoding="utf-8")

    print(f"corpus: {len(text):,} chars")
    emit(
        "start",
        stage="tokenizer",
        source=str(dataset_path or args.corpus or "tinyshakespeare"),
        num_chars=len(text),
        vocab_size=cfg.vocab_size,
    )

    # ── train BPE ────────────────────────────────────────────────────────────
    tokenizer = BPETokenizer()
    print(f"training BPE to vocab_size={cfg.vocab_size} ...")
    emit("stage", stage="bpe", message=f"learning merges to vocab_size={cfg.vocab_size}")
    start = time.time()
    # Verbose under --emit-jsonl: those progress lines become `log` events, which
    # is the only signal the UI gets during an otherwise opaque multi-minute call.
    tokenizer.train(text, cfg.vocab_size, verbose=args.verbose or args.emit_jsonl)
    elapsed = time.time() - start
    print(f"  learned {len(tokenizer.merges):,} merges in {elapsed:.1f}s")
    emit("stage", stage="bpe_done", merges=len(tokenizer.merges), elapsed_s=elapsed)

    if tokenizer.vocab_size != cfg.vocab_size:
        # Fewer merges than requested means the corpus ran out of repeated pairs.
        # The model's embedding table is sized from the config, so this must be
        # loud rather than silently producing an unusable vocab mismatch.
        emit(
            "error",
            message=f"tokenizer vocab_size={tokenizer.vocab_size} != config {cfg.vocab_size}",
        )
        raise SystemExit(
            f"tokenizer produced vocab_size={tokenizer.vocab_size} but the config "
            f"expects {cfg.vocab_size}; lower vocab_size or use a larger corpus"
        )

    tokenizer.save(args.out)
    print(f"saved tokenizer -> {args.out}")

    # Round-trip on the real corpus before committing to a training run: a lossy
    # tokenizer would silently cap achievable loss.
    emit("stage", stage="roundtrip", message="verifying round-trip on the full corpus")
    if tokenizer.decode(tokenizer.encode(text)) != text:
        emit("error", message="tokenizer round-trip FAILED on the corpus")
        raise SystemExit("tokenizer round-trip FAILED on the corpus")
    print("round-trip on full corpus: OK")

    # ── binarize ─────────────────────────────────────────────────────────────
    emit("stage", stage="binarize", message="encoding corpus to train/val splits")
    if dataset_path is not None:
        train_path, val_path = prepare_dataset(
            [dataset_path], tokenizer, args.data_dir, args.val_fraction
        )
    else:
        train_path, val_path = binarize(text, tokenizer, args.data_dir, args.val_fraction)

    train_tokens = train_path.stat().st_size // 2  # uint16
    val_tokens = val_path.stat().st_size // 2
    emit(
        "done",
        stage="complete",
        tokenizer_dir=str(args.out),
        data_dir=str(args.data_dir),
        vocab_size=tokenizer.vocab_size,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        compression=len(text) / max(1, train_tokens + val_tokens),
    )


if __name__ == "__main__":
    main()
