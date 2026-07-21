# Phases & Roadmap

## Current Phase

**Phase 2 — C++ Core + Bindings: COMPLETE.**
The engine trains and generates. 83 tests pass, 0 compiler warnings, ruff clean; every hand-derived
backward is confirmed by double-precision finite differences, and the model overfits one batch to
~0 loss. **Awaiting explicit user approval to start Phase 3.**

## Completed

**Phase 1 — Scaffolding & build system**
- [x] knbase governance; environment audit; `brew install cmake`; `git init`.
- [x] `CMakeLists.txt` (C++20, Accelerate + naive fallback, three-tier pybind11 discovery),
      `pyproject.toml` via scikit-build-core, `Makefile`, configs, `README.md`.

**Phase 2 — C++ core + bindings**
- [x] Arena allocator, `Tensor<T>`, tape autograd (`NoGradGuard`, reachability walk — no sort needed).
- [x] Ops with hand-derived backwards: `matmul`, `matmul_bt`, `add`, `rmsnorm`, `rope`,
      `softmax_causal`, `silu`, `swiglu`, `embedding`, fused `cross_entropy`.
- [x] Fused causal multi-head attention with RoPE; backward re-masks `dS`; buffer aliasing saves ~25%.
- [x] Pre-norm block stack, weight-tied `lm_head`, GPT-2 style init with 1/√(2L) residual scaling.
- [x] AdamW with decoupled decay (skipping 1-D gains) + global-norm clipping; cosine LR schedule.
- [x] Sampler: temperature → top-k → softmax → top-p → seeded multinomial.
- [x] Minimal JSON reader/writer; mmap-able `.csllm` checkpoints.
- [x] Incremental KV-cache decode path, cross-checked against the full-sequence forward.
- [x] pybind11 surface with GIL released on all compute; per-op gradcheck harness; **f64 `Model`**.
- [x] `tests/reference.py` NumPy oracles; `test_ops.py` (18), `test_gradcheck.py` (24),
      `test_model.py` (17), `test_sampler.py` (11), `test_build.py` (13).
- [x] **Verified:** forwards match NumPy to 1e-11; backwards match f64 finite differences (~2e-10
      against gradients of magnitude ~1; a 1% perturbation is detected). Overfit: 2.803 → 0.000043
      in 300 steps, all 16 target tokens reproduced. 12M config: 0.28 s/step (~3,600 tok/s) at
      B=4/T=256, 510 MB arena, 4.7 MB KV cache, init loss 8.337 > ln(4096)=8.318.

## In Progress

Nothing — Phase 2 is closed and the project is paused at the approval gate.

## Next Up

1. **Phase 3 — Tokenizer, data & training loop.**
   - `csllm/tokenizer.py`: byte-level BPE (GPT-2 style regex pre-split, iterative pair merges to
     vocab 4096), `save`/`load` as `vocab.json` + `merges.txt`, lossless round-trip on arbitrary UTF-8.
   - `csllm/data.py`: fetch TinyShakespeare, encode once to a `uint16` memmap, 90/10 split, batch sampler.
   - `train/train.py`: AdamW (lr 3e-4, cosine + warmup, wd 0.1, clip 1.0), periodic validation,
     sample generation, resumable `.csllm` checkpoints.
   - Validate on `configs/debug.json` first, then the real run to val loss ≈ 1.5.
2. **Phase 4 — FastAPI gateway.** Lifespan model load, Pydantic-validated `/generate`, SSE streaming,
   per-request `GenerationSession`, `asyncio.to_thread` dispatch, disconnect handling.

*Each phase stops for explicit user approval before the next begins.*

## Backlog

- Gradient checkpointing / flash-style tiled attention to cut the 510 MB activation footprint.
- Benchmark thread-pool granularity against Accelerate's internal threading (oversubscription risk).
- Metal / hand-written NEON kernels for the hot loops.
- bf16 checkpoint storage; KV-cache quantization for longer contexts.
- Continuous/batched decoding across concurrent gateway requests.
- Larger corpora (TinyStories, WikiText-2) and a bigger vocab.
- Streaming-detokenization policy for partial UTF-8 sequences (`design.md` Open Questions #2).
- Benchmark suite: tokens/sec for training and inference versus a NumPy baseline.
