"""Training loop — Python orchestrates, the C++ engine does every tensor operation.

    python -m train.train --config configs/shakespeare.json --steps 4000

Per step:
    x, y  = get_batch(...)                  # NumPy: slicing a memmap
    loss  = model.forward_loss(x, y)        # C++: builds the tape
    model.backward()                        # C++: walks it in reverse
    norm  = opt.clip_grad_norm(1.0)         # C++
    opt.step(cosine_lr(...))                # C++

Checkpoints store weights plus a small sidecar with the step counter. Adam's
moment estimates are NOT persisted — on resume they rebuild within a few dozen
steps, which is not worth extending the checkpoint format for.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import csllm
from csllm import _csllm_core as core
from csllm.data import DATA_DIR, TOKENIZER_DIR, get_batch, load_split
from csllm.tokenizer import BPETokenizer


def evaluate(model, data, batch_size, block_size, iters, rng) -> float:
    """Mean loss over a few fixed-size batches (no grad graph is walked)."""
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, block_size, rng)
        total += model.forward_loss(x, y)
    return total / iters


def generate(model, tokenizer, prompt, max_tokens, temperature, top_k, top_p, seed) -> str:
    """Sample a continuation using the KV-cache decode path."""
    ids = tokenizer.encode(prompt) or [0]
    block = model.config.block_size
    ids = ids[-(block - 1) :]

    session = core.GenerationSession(model, seed)
    session.prefill(np.asarray(ids, dtype=np.int32), model.config.vocab_size)
    params = core.SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

    # The first token comes from the prefill logits; step(ids[-1]) would feed
    # the prompt's last token twice.
    out: list[int] = []
    token = session.sample_last(params)
    for i in range(min(max_tokens, block - len(ids) - 1)):
        if i > 0:
            token = session.step(token, params)
        out.append(token)
    return prompt + tokenizer.decode(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/shakespeare.json")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--tokenizer-dir", default=str(TOKENIZER_DIR))
    parser.add_argument("--out", default="data/model.csllm")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-prompt", default="KING RICHARD:\n")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = csllm.load_config(args.config)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    train_data = load_split(data_dir / "train.bin")
    val_data = load_split(data_dir / "val.bin")
    tokenizer = BPETokenizer.load(args.tokenizer_dir)

    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab_size={tokenizer.vocab_size} != config {cfg.vocab_size}; "
            "re-run train.train_tokenizer"
        )

    start_step = 0
    sidecar = Path(args.out).with_suffix(".state.json")
    if args.resume and Path(args.out).exists():
        model = core.Model.load(args.out)
        if sidecar.exists():
            start_step = json.loads(sidecar.read_text()).get("step", 0)
        print(f"resumed from {args.out} at step {start_step} (Adam moments restart)")
    else:
        model = core.Model(cfg, args.seed)

    adam = core.AdamWConfig()
    adam.lr = args.lr
    adam.weight_decay = args.weight_decay
    opt = core.AdamW(model, adam)

    tokens_per_step = args.batch_size * cfg.block_size
    print(f"model    : {model.num_params():,} params, {cfg}")
    print(f"data     : train={len(train_data):,} val={len(val_data):,} tokens")
    print(f"batch    : {args.batch_size} x {cfg.block_size} = {tokens_per_step:,} tokens/step")
    arena_mb = core.estimate_activation_bytes(cfg, args.batch_size, cfg.block_size) / 1e6
    print(f"arena    : ~{arena_mb:.0f} MB")
    print(f"schedule : lr {args.lr} -> {args.min_lr}, warmup {args.warmup}, {args.steps} steps")
    print(f"epochs   : {args.steps * tokens_per_step / max(1, len(train_data)):.1f}", flush=True)

    best_val = math.inf
    t_start = time.time()
    t_window = time.time()

    for step in range(start_step, args.steps):
        lr = core.cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, rng)
        model.zero_grad()
        loss = model.forward_loss(x, y)
        model.backward()
        grad_norm = opt.clip_grad_norm(args.grad_clip)
        opt.step(lr)

        if not math.isfinite(loss):
            raise SystemExit(f"loss went non-finite at step {step}; aborting")

        if step % 50 == 0:
            dt = (time.time() - t_window) / max(1, 50 if step else 1)
            print(
                f"step {step:5d} | loss {loss:6.4f} | lr {lr:.2e} | "
                f"|g| {grad_norm:6.3f} | {dt * 1000:6.1f} ms/step | "
                f"{tokens_per_step / dt:,.0f} tok/s",
                flush=True,
            )
            t_window = time.time()

        if step > 0 and step % args.eval_every == 0:
            val = evaluate(model, val_data, args.batch_size, cfg.block_size, args.eval_iters, rng)
            marker = ""
            if val < best_val:
                best_val = val
                model.save(args.out)
                sidecar.write_text(json.dumps({"step": step, "val_loss": val}, indent=1))
                marker = "  <- saved"
            print(f"  eval @ {step}: val_loss {val:.4f} (best {best_val:.4f}){marker}", flush=True)
            t_window = time.time()

        if args.sample_every and step > 0 and step % args.sample_every == 0:
            text = generate(model, tokenizer, args.sample_prompt, 200, 0.8, 40, 0.95, args.seed)
            print(f"  --- sample @ {step} ---\n{text}\n  ---", flush=True)
            t_window = time.time()

    val = evaluate(model, val_data, args.batch_size, cfg.block_size, args.eval_iters, rng)
    if val < best_val:
        best_val = val
        model.save(args.out)
        sidecar.write_text(json.dumps({"step": args.steps, "val_loss": val}, indent=1))

    elapsed = (time.time() - t_start) / 60
    print(f"\ndone in {elapsed:.1f} min | final val {val:.4f} | best {best_val:.4f}")
    print(f"checkpoint: {args.out}")
    sample = generate(model, tokenizer, args.sample_prompt, 300, 0.8, 40, 0.95, args.seed)
    print(f"\n--- final sample ---\n{sample}")


if __name__ == "__main__":
    main()
