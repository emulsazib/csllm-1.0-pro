# Phases & Roadmap

## Current Phase

**COMPLETE — all four phases delivered.**
A ~12M-parameter Transformer built from scratch in C++, trained on TinyShakespeare, and served over
SSE. **164 tests pass, 0 compiler warnings, ruff clean.** Remaining items are backlog, not scope.

## Completed

**Phase 1 — Scaffolding & build system**
- [x] `CMakeLists.txt` (C++20, Accelerate + naive fallback, three-tier pybind11 discovery),
      `pyproject.toml` via scikit-build-core, `Makefile`, configs, `README.md`, knbase governance.

**Phase 2 — C++ core + bindings**
- [x] Arena, `Tensor<T>`, tape autograd; ten ops with hand-derived backwards; fused causal MHA with
      RoPE; pre-norm block stack with weight tying; AdamW + clipping; sampler; JSON; `.csllm`;
      KV-cache decode; pybind11 surface with the GIL released on all compute.
- [x] Verified: forwards match NumPy oracles to 1e-11; **all backwards match float64 finite
      differences**; overfit-one-batch 2.803 → 0.000043.

**Phase 3 — Tokenizer, data & training loop**
- [x] Byte-level BPE with incremental merge training (3,840 merges over 1.1 MB in **1.9s**), lossless
      round-trip on emoji/CJK/control bytes/BMP fuzz; TinyShakespeare → uint16 memmap; AdamW loop with
      cosine+warmup, clipping, best-val checkpointing, `--resume`; `train/sample.py`.
- [x] **Trained the 12M model:** 1500 steps in **11.9 min**, best val **4.2996/token**
      = 1.41 nats/char, 2.03 bits/char, perplexity 80.5.

**Phase 4 — FastAPI gateway**
- [x] Lifespan loads model (mmap) + tokenizer once; `POST /generate` with Pydantic v2 bounds and
      `extra="forbid"`; SSE streaming terminated with `[DONE]`; non-streaming JSON mode;
      `GET /health` with live session count.
- [x] Per-request `GenerationSession` (private 4.7 MB KV cache), `asyncio.to_thread` dispatch,
      concurrency semaphore with timeout → 503, UTF-8 boundary buffering, disconnect abort.
- [x] **Fixed a latent generation bug**: `prefill()` + `step(prompt[-1])` fed the prompt's last token
      twice. Added `sample_last()`, fixed both callers, added regression tests.
- [x] **Measured live:** TTFT **19 ms**, ~565 tok/s single stream, 4 concurrent streams in 1.95x one
      stream's wall-clock, disconnect frees the slot within 0.5 s, all invalid requests → 422.

## In Progress

Nothing. The project meets every goal in `prd.md`.

## Next Up

No committed work. If the project is picked up again, the highest-value items are:

1. **Regularization (dropout)** — the model overfits 310k tokens badly (train/val gap +1.26); this
   would buy the largest quality improvement per unit effort.
2. **Continuous/batched decoding** across concurrent sessions — would turn today's 1.95x concurrency
   scaling into something closer to linear.
3. **Gradient checkpointing** — the ~1.5 GB activation arena at B=8/T=256 is what caps batch size.

## Backlog

- Persist Adam moments so `--resume` is exact.
- Benchmark thread-pool granularity against Accelerate's internal threading (oversubscription risk).
- Metal / hand-written NEON kernels for the hot loops.
- bf16 checkpoint storage; KV-cache quantization for longer contexts.
- Larger corpora (TinyStories, WikiText-2) and a bigger vocab.
- Flash-attention-style tiled attention to cut O(T²) activation memory.
- Benchmark suite: tokens/sec for training and inference versus a NumPy baseline.
- Gateway auth / rate limiting (explicit non-goal in `prd.md`, but needed for real exposure).
