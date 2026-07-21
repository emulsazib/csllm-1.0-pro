# Project Memory

## Summary

**CSLLM 1.0 Pro is complete.** A ~12M-parameter autoregressive Transformer built entirely from
scratch: a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch, no autograd
library), a **Python** byte-level BPE tokenizer and training loop, and a **FastAPI** gateway
streaming tokens over SSE. Llama-style: **RoPE + RMSNorm + SwiGLU**, pre-norm residuals,
weight-tied `lm_head`.

All four phases are done. `data/model.csllm` holds a trained model (12,194,688 params,
val 4.39/token = **1.41 nats/char**) that generates coherent Shakespearean dialogue, served at
**19 ms time-to-first-token**. **164 tests pass, 0 compiler warnings, ruff clean.**

## Recent Changes

*(reverse-chronological)*

- **2026-07-21 — Phase 4: FastAPI gateway.** `gateway/settings.py` (env-driven), `schemas.py`
  (Pydantic v2 with `extra="forbid"` and bounds on every sampling parameter), `engine.py`
  (per-request `GenerationSession`, `asyncio.to_thread` dispatch, concurrency semaphore with
  timeout → 503, UTF-8 boundary buffering), `main.py` (lifespan load, SSE `/generate`, `/health`).
  Added `tests/test_gateway.py` (27) → **164 total**.
  **Also fixed a real generation bug** (see Learnings): `prefill()` + `step(prompt[-1])` fed the
  prompt's last token twice. Added `GenerationSession::sample_last()`, fixed `train/train.py` and
  `train/sample.py`, added three regression tests.
  **Measured live:** TTFT 19 ms, ~565 tok/s single stream, 4 concurrent streams in 1.95× one
  stream's wall-clock, disconnect frees the slot within 0.5 s.
- **2026-07-21 — Phase 3: tokenizer, data, training loop.** Byte-level BPE with incremental merge
  training (3,840 merges over 1.1 MB in **1.9s**), TinyShakespeare binarization, AdamW training loop,
  `train/sample.py`. Trained the 12M model in 11.9 min → best val 4.2996 at step 1000.
- **2026-07-21 — Phase 2: the C++ core.** Arena-backed `Tensor<T>`, tape autograd, ten ops with
  hand-derived backwards, fused causal MHA with RoPE, weight-tied block stack, AdamW, sampler, JSON,
  mmap-able `.csllm`, KV-cache decode. All gradients verified in float64; overfit-one-batch → 4e-5.
- **2026-07-21 — Phase 1: scaffolding & build system.**
- **2026-07-21 — Governance bootstrap.**

## Learnings & Gotchas

**Environment (verified on this machine)**

- `cmake` **4.4.0**; CMake 4 removed compatibility with `cmake_minimum_required(VERSION < 3.5)`.
- **pybind11 must be >= 3.0** (we resolve 3.0.4) — 2.13.x predates Python 3.14's C-API.
  Pinned in `pyproject.toml` and the CMake FetchContent tag; **do not lower**.
- Use Homebrew **Python 3.14.6** for `.venv`; system 3.9.6 is EOL. No PyTorch ⇒ no wheel risk.
- arm64 + Apple clang 21; Accelerate gives multithreaded BLAS free. 8 hardware threads.
- Fast iteration: `pip install -e . --no-build-isolation` (`make build`).

**Engine (Phase 2)**

- **The tape needs no topological sort** — nodes are appended in forward order, so descending from
  the root is already valid reverse order; only a reachability mark is needed.
- **Weight tying**: `matmul_bt` shares storage with `tok_emb`; the gradient accumulates from the
  embedding scatter-add *and* `matmul_bt`'s dW. Fingerprint: at init the argmax equals the *input*
  token ~85% of the time (chance 0.02%).
- **ln(V) is a hard floor** for initial loss when targets are independent of the logits — but a loss
  printed *after* a few optimizer steps is legitimately below it.
- **RoPE uses INTERLEAVED pairs** (2p, 2p+1). Backward is free (orthogonal ⇒ negated angles).
  Attention backward must **re-apply the causal mask** to `dS`. Cached keys are stored
  already-rotated.
- **fp32 finite differences are useless for gradchecking** — hence the f64 instantiation including an
  f64 `Model`. Always confirm the harness is *sensitive* (perturb an analytic gradient by 1% and
  check the test fails) before trusting a clean pass.

**Training (Phase 3)**

- **BPE merge count is bounded by corpus DIVERSITY, not length.** 12 and 60 repetitions of the same
  29 unique words both yield exactly 92 merges. `train_tokenizer.py` hard-fails on a shortfall.
- **Naive BPE training is too slow**; the inverted pair→word index makes it 1.9s instead of minutes.
- **Match the cosine schedule to the steps you will actually run.** A 3000-step schedule plateaued at
  val 4.4714 with LR still near maximum; rerunning at 1500 steps reached **4.2996** — 3.8% better in
  *half* the wall-clock. A checkpoint stranded mid-schedule is not a converged model.
- **A 20-batch eval is optimistically biased** (4.2996 vs 4.3877 over 200 batches).
- **Per-token loss is not comparable across vocabularies** — divide by chars/token (3.11) for nats/char.
- 12M params on 310k tokens overfits hard (train/val gap +1.26); best-val checkpointing is essential.

**Serving (Phase 4)**

- **`prefill()` consumes the ENTIRE prompt.** The first generated token must come from
  `sample_last()`; calling `step(prompt[-1])` feeds the prompt's last token a second time and
  generates from a corrupted context. This bug produced *plausible* text and was invisible in every
  loss metric — with prompt `"KING RICHARD:\n"` it duplicated the newline so the model started a new
  speaker instead of continuing the line. Only comparing against
  `forward_logits(prompt)[-1].argmax()` catches it (`test_first_generated_token_continues_the_prompt`).
- **`Sampler::sample()` scales logits in place**, so `sample_last()` must work on a copy — otherwise
  repeated calls apply temperature T^k and the distribution collapses onto the argmax.
- **Concurrent `GenerationSession`s are safe**: each owns its KV cache and scratch arena, the tape is
  thread-local, and `forward_token` only reads shared weights. `Model::forward_logits` is *not*
  safe to share — it uses the single shared activation arena.
- **Testing a stochastic sampler needs care.** "Call it twice, expect the same answer" is wrong at
  temperature > 0. Reseed between draws and assert a *distributional* property instead.
- `httpx.ASGITransport` runs the whole gateway in-process — no ports, no flakiness.
- SSE disconnect is handled twice over: `sse-starlette` cancels the generator, and the explicit
  `await request.is_disconnected()` poll stops the C++ work promptly and sets `finish_reason`.

## Known Issues

- Adam moment estimates are **not** persisted; `--resume` reloads weights and the step counter only.
- No dropout — the only regularizers are weight decay and best-val early stopping.
- No gradient checkpointing; ~1.5 GB activation arena at B=8/T=256.
- Thread-pool granularity vs Accelerate's internal threading is unbenchmarked.
- Checkpoints are fp32 only; `GenerationSession` handles one sequence at a time (no continuous
  batching), so 4 concurrent streams give ~1.95x, not 4x.
- No auth or rate limiting on the gateway (explicit non-goal in `prd.md`).
