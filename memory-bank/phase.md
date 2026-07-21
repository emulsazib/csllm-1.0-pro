# Phases & Roadmap

## Current Phase

**Phase 1 — Project Scaffolding & Build System: COMPLETE.**
The C++/Python bridge compiles and imports, 13 tests pass, zero compiler warnings, ruff clean.
**Awaiting explicit user approval to start Phase 2.**

## Completed

- [x] knbase governance bootstrapped (all six `memory-bank/` docs).
- [x] Environment audit: arm64 + Apple clang 21, Accelerate present, Homebrew Python 3.14.6.
- [x] Architecture decisions locked: pure C++ engine, Python BPE, TinyShakespeare @ ~12M params,
      RoPE + RMSNorm + SwiGLU.
- [x] `brew install cmake` (4.4.0).
- [x] `git init` + `.gitignore`.
- [x] Directory tree: `core/{include,src,bindings}`, `csllm/`, `train/`, `gateway/`, `tests/`, `configs/`.
- [x] `CMakeLists.txt` — C++20, Accelerate discovery + naive fallback, three-tier pybind11 discovery
      (config → pip cmakedir → FetchContent), `-O3` without `-ffast-math`.
- [x] `pyproject.toml` (scikit-build-core), `Makefile`, `README.md`.
- [x] 11 C++ headers defining the Phase 2 contracts; build-critical pieces implemented for real
      (GEMM, arena, thread pool, autograd tape, build info, config validation, cosine LR, KV cache).
- [x] `configs/debug.json` (~0.14M params) and `configs/shakespeare.json` (12,194,688 params).
- [x] **Exit check passed:** `make setup && make smoke && make test` →
      Accelerate active, fast-math off, `__cplusplus`=202002, GEMM matches NumPy.

## In Progress

Nothing — Phase 1 is closed and the project is paused at the approval gate.

## Next Up

1. **Phase 2 — C++ core + bindings.** Bottom-up, each op gated on a NumPy oracle **and** a
   double-precision gradcheck before the next is written:
   `Tensor<T>` storage → tape wiring → matmul → RMSNorm → RoPE → causal MHA → SwiGLU →
   embedding → fused cross-entropy → AdamW + clipping → sampler → `.csllm` serialization →
   expanded pybind11 surface (GIL released throughout).
2. **Phase 3 — Tokenizer, data & training loop.** Byte-level BPE; TinyShakespeare → uint16 memmap;
   AdamW loop with cosine schedule, warmup, clipping, resumable checkpoints. Must overfit one batch
   to ≈0 loss on `debug.json` before the real run.
3. **Phase 4 — FastAPI gateway.** Lifespan model load, Pydantic-validated `/generate`, SSE streaming,
   per-request `GenerationSession`, `asyncio.to_thread` dispatch, disconnect handling.

*Each phase stops for explicit user approval before the next begins.*

## Backlog

- Metal / hand-written NEON kernels for the hot loops (currently CPU + Accelerate only).
- bf16 checkpoint storage to halve file size.
- Flash-attention-style tiled attention to cut O(T²) activation memory.
- Continuous/batched decoding across concurrent gateway requests.
- Larger corpora (TinyStories, WikiText-2) and a bigger vocab.
- KV-cache quantization for longer contexts.
- Streaming-detokenization policy for partial UTF-8 sequences (`design.md` Open Questions #2).
- Benchmark suite: tokens/sec for training and inference versus a NumPy baseline.
- C++ JSON reader (`ModelConfig::from_json`), currently stubbed — arrives with the checkpoint loader.
