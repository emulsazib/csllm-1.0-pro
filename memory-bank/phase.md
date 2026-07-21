# Phases & Roadmap

## Current Phase

**Phase 3 — Tokenizer, Data & Training Loop: COMPLETE.**
A trained model exists at `data/model.csllm` (12,194,688 params, val **4.39/token = 1.41 nats/char**)
producing coherent Shakespearean dialogue. 134 tests pass, ruff clean.
**Awaiting explicit user approval to start Phase 4.**

## Completed

**Phase 1 — Scaffolding & build system**
- [x] knbase governance, environment audit, `cmake`, `git init`.
- [x] `CMakeLists.txt` (C++20, Accelerate + fallback, three-tier pybind11 discovery),
      `pyproject.toml` via scikit-build-core, `Makefile`, configs, `README.md`.

**Phase 2 — C++ core + bindings**
- [x] Arena, `Tensor<T>`, tape autograd; ten ops with hand-derived backwards; fused causal MHA with
      RoPE; pre-norm block stack with weight tying; AdamW + clipping; sampler; JSON; `.csllm`;
      KV-cache decode; pybind11 surface with GIL released.
- [x] Verified: forwards match NumPy to 1e-11; **all backwards match float64 finite differences**;
      overfit-one-batch 2.803 → 0.000043.

**Phase 3 — Tokenizer, data & training loop**
- [x] `csllm/tokenizer.py` — byte-level BPE, incremental merge training (inverted pair→word index):
      3,840 merges over 1.1 MB in **1.9s**; lossless round-trip on the full corpus, emoji, CJK,
      control bytes, and BMP fuzzing.
- [x] `csllm/data.py` — TinyShakespeare download, uint16 binarization (344,120 tokens @ 3.24
      chars/token), 90/10 split, batch sampler.
- [x] `train/train_tokenizer.py`, `train/train.py` (AdamW, cosine+warmup, grad clipping, periodic
      eval, best-val checkpointing, sampling, `--resume`), `train/sample.py` (streaming with
      UTF-8-boundary buffering).
- [x] `tests/test_tokenizer.py` (44) + `tests/test_training.py` (7) → **134 total**.
- [x] Makefile targets: `tokenizer`, `train`, `sample`, `debug`.
- [x] **Trained the 12M model:** 1500 steps, B=8×T=256, lr 1e-3→5e-5, **11.9 min**, ~4,400 tok/s.
      Best val **4.2996** at step 1000 → 1.41 nats/char, 2.03 bits/char, perplexity 80.5.
      *A 3000-step schedule scored only 4.4714; matching the schedule to the actual run length was
      worth 3.8% in half the wall-clock.*

## In Progress

Nothing — Phase 3 is closed and the project is paused at the approval gate.

## Next Up

1. **Phase 4 — FastAPI gateway.**
   - Lifespan loads `data/model.csllm` (mmap) + `data/tokenizer/` once at startup.
   - `POST /generate` with Pydantic v2 bounds on prompt, max_tokens, temperature, top_k, top_p, seed.
   - **SSE streaming** via `EventSourceResponse`, terminating with `[DONE]`.
   - Per-request `GenerationSession` (private 4.7 MB KV cache); per-token compute dispatched through
     `asyncio.to_thread` (the bindings already release the GIL); `Semaphore` caps concurrency.
   - **Buffer partial UTF-8 sequences** — reuse the byte-buffering logic proven in `train/sample.py`.
   - Abort generation on `await request.is_disconnected()`.
   - `GET /health` + structured errors; `tests/test_gateway.py` via httpx ASGI.

*Each phase stops for explicit user approval before the next begins.*

## Backlog

- Regularization (dropout) — the model overfits 310k tokens badly (train/val gap +1.26).
- Persist Adam moments so `--resume` is exact.
- Gradient checkpointing / flash-style tiled attention to cut the ~1.5 GB arena at B=8/T=256.
- Benchmark thread-pool granularity against Accelerate's internal threading.
- Metal / hand-written NEON kernels; bf16 checkpoints; KV-cache quantization.
- Continuous/batched decoding across concurrent gateway requests.
- Larger corpora (TinyStories, WikiText-2) and a bigger vocab.
- Benchmark suite: tokens/sec for training and inference versus a NumPy baseline.
