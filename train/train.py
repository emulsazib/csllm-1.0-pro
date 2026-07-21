"""Training loop — Phase 3.

Python orchestrates; the C++ engine does every tensor operation.

    for step in range(max_steps):
        x, y  = get_batch(train_data, B, T, rng)
        loss  = model.forward_loss(x, y)        # C++ builds the tape
        model.backward()                        # C++ walks it in reverse
        norm  = opt.clip_grad_norm(1.0)         # C++
        opt.step(cosine_lr(step, ...))          # C++
        model.zero_grad()

Plan: AdamW (lr 3e-4, cosine schedule + warmup, wd 0.1, grad-clip 1.0), periodic
validation, sample generation every N steps, resumable ``.csllm`` checkpoints.

Validate on ``configs/debug.json`` first — it must overfit a single batch to ~0
loss, which is the strongest end-to-end proof the hand-written autograd is right.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("The training loop arrives in Phase 3")


if __name__ == "__main__":
    main()
