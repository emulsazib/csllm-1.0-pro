# Project Memory

## Summary

**CSLLM 1.0 Pro** builds a ~12M-parameter autoregressive Transformer entirely from scratch:
a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch, no autograd library),
a **Python** byte-level BPE tokenizer and training loop, and a **FastAPI** gateway streaming tokens
over SSE. Architecture is Llama-style: **RoPE + RMSNorm + SwiGLU**, pre-norm residuals, weight-tied
`lm_head`. Target corpus is TinyShakespeare (~1.1 MB, vocab 4096, 6L/6H/384d/256ctx).

**Phase 1 is complete and verified.** The C++/Python bridge builds and imports; 13 tests pass;
zero compiler warnings; ruff clean. The model math itself is stubbed (`CSLLM_NOT_IMPLEMENTED`)
and lands in Phase 2.

## Recent Changes

*(reverse-chronological)*

- **2026-07-21 — Phase 1: scaffolding & build system.** Created the full tree (`core/`, `csllm/`,
  `train/`, `gateway/`, `tests/`, `configs/`), `CMakeLists.txt`, `pyproject.toml` (scikit-build-core),
  `Makefile`, `README.md`, `.gitignore`; `git init`. Installed cmake. Authored 11 C++ headers as the
  Phase 2 interface contracts, with these pieces **fully implemented** because they de-risk the
  build: `gemm.cpp` (Accelerate `cblas_sgemm`/`dgemm` + naive fallback), `arena.cpp`, `threadpool.cpp`,
  `autograd.cpp` (tape push/backward/NoGradGuard), `build_info.cpp`, `ModelConfig::validate/num_params`,
  `cosine_lr`, `KVCache`. Everything else throws `CSLLM_NOT_IMPLEMENTED`.
  **Verified:** BLAS backend = Accelerate, fast-math = False, `__cplusplus` = 202002, 8 threads,
  `shakespeare.json` = **12,194,688 params**, GEMM matches NumPy at rtol 1e-4 (f32) / 1e-12 (f64).
- **2026-07-21 — Governance bootstrap.** Authored all six `memory-bank/` docs from `doc/prompt.md`.
  Recorded the four locked decisions and the full hand-derived gradient table in `design.md`.

## Learnings & Gotchas

**Environment (verified 2026-07-21 on this machine)**

- **`cmake` was absent; now installed at `/opt/homebrew/bin/cmake` — version 4.4.0.** Note CMake 4
  **removed compatibility with `cmake_minimum_required(VERSION < 3.5)`**, so any future FetchContent
  dependency with an ancient minimum will fail to configure.
- **pybind11 must be >= 3.0 (we resolve 3.0.4).** The 2.13.x series predates Python 3.14 and does not
  support its C-API. This is pinned in both `pyproject.toml` build-requires and the CMake
  FetchContent tag — **do not lower either.**
- Python: use Homebrew **3.14.6** (`/opt/homebrew/bin/python3.14`) for `.venv`; system 3.9.6 is EOL.
  The 3.14 wheel risk did **not** materialize: numpy 2.5.1, fastapi 0.139.2, pydantic 2.13.4,
  uvicorn 0.51.0, ruff, pytest 9.1.1 all had cp314 wheels. Choosing a zero-PyTorch stack is what
  made 3.14 safe.
- **arm64 + Apple clang 21.0.0**; `Accelerate.framework` present, giving tuned multithreaded BLAS
  free of third-party dependencies. 8 hardware threads.

**Build system**

- `pyproject.toml`'s `readme = "README.md"` makes metadata generation **hard-fail** if the file is
  missing — that was the only build error hit during Phase 1.
- Editable installs: `pip install -e . --no-build-isolation` (what `make build` runs) is much faster
  for iterating than a full isolated rebuild.
- `-Wunused-private-field` fires on classes whose methods are still stubbed. Fixed honestly by
  implementing `MappedCheckpoint`'s destructor (`munmap`) and `tensor_data()` for real rather than
  suppressing the warning. Expect the same pattern for other stubbed classes in Phase 2.

**Design traps to remember**

- **Accelerate on modern SDKs needs `ACCELERATE_NEW_LAPACK`** defined, or deprecated prototypes bite.
- **Weight tying makes the embedding gradient accumulate from two paths** (input lookup *and* the
  output projection). Forgetting the second yields a model that trains but converges wrong — only a
  gradcheck on the embedding catches it.
- **RoPE's backward is free**: the rotation is orthogonal, so the VJP is the same kernel with negated
  angles. Don't derive a Jacobian.
- **Attention backward must re-apply the causal mask** to `dS`, or gradient leaks into masked positions.
- **fp32 finite differences are too noisy for gradchecking** — hence `template<typename T>` ops with a
  double instantiation. Most important testability decision in the project.
- **`-ffast-math` is banned**; `test_fast_math_is_disabled` asserts `__FAST_MATH__` is undefined so it
  cannot creep back in.
- **Bindings must release the GIL** (`py::gil_scoped_release`), or the gateway cannot serve concurrent
  streams even with `asyncio.to_thread`.
- **Byte-level BPE can emit a partial UTF-8 sequence** mid-stream; the gateway must buffer incomplete
  code points instead of pushing invalid UTF-8 into an SSE frame.
- `head_dim` must be **even** for RoPE pairing (64 at the 12M config ✓); `ModelConfig::validate()`
  enforces this and is covered by a test.

## Known Issues

- All model math is stubbed and throws `CSLLM_NOT_IMPLEMENTED` — by design, pending Phase 2.
- `ModelConfig::from_json` is stubbed; Python parses `configs/*.json` and sets fields across the
  binding. A C++ JSON reader arrives with the checkpoint loader in Phase 2.
- `MappedCheckpoint`'s constructor (header parsing) is stubbed; its destructor and accessor are real.
- Open design questions tracked in `design.md` → *Open Questions*: thread-pool granularity vs.
  Accelerate's internal threading, streaming detokenization policy, checkpoint dtype, batched
  sessions, Metal kernels.
