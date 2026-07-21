"""Generate text from a trained checkpoint.

    python -m train.sample --prompt "KING RICHARD:" --max-tokens 300

Uses the same C++ ``GenerationSession`` (KV-cache decode path) the FastAPI
gateway will use, so what you see here is what the service will produce.

Bytes are accumulated and flushed only at UTF-8 boundaries: a single BPE token
can end mid-code-point, and printing it immediately would emit mojibake. The
gateway needs exactly this buffering for SSE frames.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from csllm import _csllm_core as core
from csllm.data import TOKENIZER_DIR
from csllm.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/model.csllm")
    parser.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR))
    parser.add_argument("--prompt", default="KING RICHARD:\n")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer_dir)
    model = core.Model.load(args.checkpoint)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise SystemExit(
            f"tokenizer vocab_size={tokenizer.vocab_size} != model {model.config.vocab_size}"
        )

    block = model.config.block_size
    prompt_ids = tokenizer.encode(args.prompt) or [0]
    if len(prompt_ids) >= block:
        prompt_ids = prompt_ids[-(block - 1) :]

    session = core.GenerationSession(model, args.seed)
    session.prefill(np.asarray(prompt_ids, dtype=np.int32), model.config.vocab_size)
    params = core.SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )

    budget = min(args.max_tokens, block - len(prompt_ids) - 1)
    if budget <= 0:
        raise SystemExit(f"prompt fills the {block}-token context; nothing left to generate")

    if args.stream:
        sys.stdout.write(args.prompt)
        sys.stdout.flush()

    token = int(prompt_ids[-1])
    generated: list[int] = []
    pending = b""  # bytes not yet at a code-point boundary
    start = time.time()

    for _ in range(budget):
        token = session.step(token, params)
        generated.append(token)
        if args.stream:
            pending += tokenizer.decode_bytes([token])
            try:
                text = pending.decode("utf-8")
            except UnicodeDecodeError:
                continue  # incomplete sequence — wait for the next token
            sys.stdout.write(text)
            sys.stdout.flush()
            pending = b""

    elapsed = time.time() - start
    if args.stream:
        if pending:
            sys.stdout.write(pending.decode("utf-8", errors="replace"))
        print()
    else:
        print(args.prompt + tokenizer.decode(generated))

    print(
        f"\n[{len(generated)} tokens in {elapsed:.2f}s = {len(generated) / elapsed:.1f} tok/s]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
