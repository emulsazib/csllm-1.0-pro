# Project Memory

## Summary

**CSLLM 1.0 Pro** builds a ~12M-parameter autoregressive Transformer entirely from scratch:
a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch, no autograd library),
a **Python** byte-level BPE tokenizer and training loop, and a **FastAPI** gateway streaming tokens
over SSE. Llama-style: **RoPE + RMSNorm + SwiGLU**, pre-norm residuals, weight-tied `lm_head`.

**Phases 1 and 2 are complete and verified.** The C++ engine trains and generates: 83 tests pass,
every backward pass is confirmed by double-precision finite differences, and the model overfits a
single batch to ~0 loss. Phase 3 (tokenizer, data, training loop) has not started — `csllm/tokenizer.py`
and `csllm/data.py` are still placeholders.

## Recent Changes

*(reverse-chronological)*

- **2026-07-21 — Phase 2: the C++ core.** Implemented the full engine: arena-backed `Tensor<T>`,
  tape autograd, `matmul`/`matmul_bt`/`add`/`rmsnorm`/`rope`/`softmax_causal`/`silu`/`swiglu`/
  `embedding`/`cross_entropy`, fused causal multi-head attention with RoPE, the pre-norm block stack
  with weight tying, AdamW + global-norm clipping, the sampler (temperature/top-k/top-p), a minimal
  JSON reader (`core/src/json.cpp`), mmap-able `.csllm` checkpoints, and an incremental KV-cache
  decode path. Expanded the pybind11 surface with a per-op gradcheck harness plus an **f64 `Model`**
  so the whole network can be gradchecked in double precision.
  **Verified:** 83 tests, 0 compiler warnings, ruff clean. Per-op forwards match NumPy oracles to
  1e-11; all backwards match double-precision finite differences (analytic vs numeric agree to
  ~2e-10 against gradients of magnitude ~1, and a deliberate 1% perturbation IS detected).
  Overfit-one-batch: loss 2.803 → 0.000043 in 300 steps, reproducing all 16 target tokens exactly.
  At the 12M config: 0.28 s/step (≈3,600 tok/s) at B=4/T=256, 510 MB activation arena, 4.7 MB KV cache.
- **2026-07-21 — Phase 1: scaffolding & build system.** Tree, `CMakeLists.txt`, `pyproject.toml`
  (scikit-build-core), `Makefile`, configs, `git init`. Build green, Accelerate active.
- **2026-07-21 — Governance bootstrap.** Authored all six `memory-bank/` docs from `doc/prompt.md`.

## Learnings & Gotchas

**Environment (verified on this machine)**

- `cmake` **4.4.0** at `/opt/homebrew/bin/cmake` (installed during Phase 1). CMake 4 **removed
  compatibility with `cmake_minimum_required(VERSION < 3.5)`** — a future FetchContent dependency
  with an ancient minimum will fail to configure.
- **pybind11 must be >= 3.0** (we resolve 3.0.4). The 2.13.x series predates Python 3.14 and does not
  support its C-API. Pinned in both `pyproject.toml` and the CMake FetchContent tag — **do not lower**.
- Use Homebrew **Python 3.14.6** for `.venv`; system 3.9.6 is EOL. No PyTorch means no wheel risk;
  numpy 2.5.1 / fastapi 0.139.2 / pydantic 2.13.4 all have cp314 wheels.
- arm64 + Apple clang 21; `Accelerate.framework` gives multithreaded BLAS free. 8 hardware threads.
- `pyproject.toml`'s `readme = "README.md"` hard-fails metadata generation if the file is missing.
- Fast iteration: `pip install -e . --no-build-isolation` (what `make build` runs).

**Implementation lessons from Phase 2**

- **The tape needs no topological sort.** Nodes are appended in forward order, so a node's index
  always exceeds its parents'. Walking indices downward from the root is already a valid reverse
  order — only a reachability mark is required.
- **Weight tying works via `matmul_bt`** sharing storage with `tok_emb`; the gradient accumulates
  from the embedding scatter-add *and* `matmul_bt`'s dW into one buffer. `test_weight_tying_uses_both_
  gradient_paths` catches a dropped path by asserting that rows never used as input tokens still get
  gradient.
- **A fingerprint of correct tying:** at init the argmax equals the *input* token ~85% of the time
  (chance 0.02%), because the residual stream still carries the input embedding.
- **ln(V) is a hard floor** for initial loss when targets are independent of the logits
  (`logsumexp(z) − mean(z) ≥ log V` by Jensen). A loss below it means the loss or target indexing is
  wrong. Measured 8.337 vs ln(4096)=8.318 ✓. *Careful when reading training logs:* a loss printed
  after a few optimizer steps is legitimately below the floor — that is learning, not a bug.
- **RoPE convention matters and is not interchangeable.** We use INTERLEAVED pairs (2p, 2p+1); the
  half-split convention (d with d+Dh/2) gives different results. `tests/reference.py` must match.
- **RoPE's backward is free** — the rotation is orthogonal, so the VJP is the same kernel with
  negated angles.
- **Attention backward must re-apply the causal mask** to `dS`, and the `1/√dₕ` scale folds in there.
- **Cached keys must be stored already-rotated.** The decode path applies RoPE at the current
  position before writing K to the cache, exactly as training does. `test_kv_cache_decode_matches_
  full_forward` cross-checks the two independent implementations (agree to 2e-4 in fp32).
- **`-Wunused-private-field` fires on stubbed classes.** Fixed honestly by implementing the real
  destructor/accessor rather than suppressing.
- **fp32 finite differences are useless for gradchecking** — hence `template<typename T>` with an f64
  instantiation, including an f64 `Model`. Verify the harness is *sensitive* (perturb an analytic
  gradient by 1% and confirm the test fails) before trusting a clean pass.
- **Attention dominates activation memory.** `qf/kf/vf/ctx` are dead after packing, so the backward
  aliases them — ~25% saving. Still 510 MB at B=4/T=256 with no recomputation.

## Known Issues

- **Phase 3 not started:** `csllm/tokenizer.py`, `csllm/data.py`, `train/train.py`,
  `train/train_tokenizer.py` are placeholders that raise `NotImplementedError`.
- **Phase 4 not started:** everything in `gateway/` is a placeholder.
- No gradient checkpointing; activation memory grows linearly with batch × context.
- Thread-pool granularity vs Accelerate's internal threading is unbenchmarked (possible
  oversubscription). See `design.md` → Open Questions #1.
- Checkpoints are fp32 only; `GenerationSession` handles one sequence at a time.
