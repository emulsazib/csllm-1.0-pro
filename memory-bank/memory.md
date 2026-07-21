# Project Memory

## Summary

**CSLLM** is a ~12M-parameter autoregressive Transformer built entirely from scratch: a pure
**C++20 engine** with hand-derived reverse-mode autograd (no PyTorch), a **Python** byte-level BPE
tokenizer and training loop, and a **FastAPI** gateway streaming tokens over SSE. Llama-style:
RoPE + RMSNorm + SwiGLU, pre-norm residuals, weight-tied `lm_head`.

**1.0 is complete** — `data/model.csllm` holds a trained model (val 4.39/token = 1.41 nats/char)
served at 19 ms TTFT. **2.0 Phase 1 is complete** — a dataset plugin system, optional tokenizer
special tokens, and a portable safetensors export. **224 tests pass, 0 compiler warnings, ruff clean.**

2.0 Phases 2-4 (WebSocket telemetry + `/configure_model`; React tokenizer/probability UI;
React Three Fiber attention animation + training dashboard) are **not started**.

## Recent Changes

*(reverse-chronological)*

- **2026-07-21 — 2.0 Phase 1: datasets & export.**
  `datasets/` plugin system: `DatasetPlugin` ABC with `__init_subclass__` auto-registration,
  extension-keyed registry, built-in `.txt`/`.md` (whole-file or blank-line), `.jsonl`/`.ndjson`
  (configurable field), `.csv`/`.tsv` (stdlib `csv`, so quoted commas/newlines survive).
  `csllm/tokenizer.py` gained **optional** special tokens (`train(..., special_tokens=…)`,
  `encode(..., allow_special=…)`, `eot_id`), persisted by id in `vocab.json`.
  `csllm/data.py` gained `encode_documents()` (separator BETWEEN documents only) and
  `prepare_dataset()`. `csllm/export.py` writes `model.safetensors` + `tokenizer.json` +
  `config.json`. Added `make export` / `make datasets`; +60 tests (224 total).
  **Verified:** exported the real 12M model — 56 tensors, 12,194,688 params, **bitwise identical**
  to the live engine (max diff 0.0), readable with numpy+safetensors alone, `torch` never imported.
- **2026-07-21 — 1.0 Phase 4: FastAPI gateway.** SSE `/generate`, `/health`, per-request
  `GenerationSession`, `asyncio.to_thread` dispatch, semaphore, UTF-8 buffering, disconnect abort.
  Fixed a latent generation bug (see Learnings). TTFT 19 ms; 4 concurrent streams at 1.95x.
- **2026-07-21 — 1.0 Phase 3: tokenizer, data, training loop.** Trained the 12M model in 11.9 min.
- **2026-07-21 — 1.0 Phase 2: the C++ core.** All gradients verified in float64.
- **2026-07-21 — 1.0 Phase 1 + governance bootstrap.**

## Learnings & Gotchas

**Environment**

- `cmake` **4.4.0** (dropped `cmake_minimum_required(VERSION < 3.5)` support).
- **pybind11 must be >= 3.0** — 2.13.x predates Python 3.14's C-API. Pinned in `pyproject.toml`
  and the CMake FetchContent tag; **do not lower**.
- Homebrew **Python 3.14.6** for `.venv`; system 3.9.6 is EOL. **Node v26.3.0 / npm 11.16.0** present.
- arm64 + Apple clang 21; Accelerate gives multithreaded BLAS free. 8 hardware threads.
- Fast iteration: `pip install -e . --no-build-isolation` (`make build`).

**Engine**

- **The tape needs no topological sort** — nodes are appended in forward order, so descending from
  the root is already valid reverse order; only a reachability mark is needed.
- **Weight tying**: `matmul_bt` shares storage with `tok_emb`; the gradient accumulates from the
  embedding scatter-add *and* `matmul_bt`'s dW. At init the argmax equals the *input* token ~85%
  of the time (chance 0.02%) — a useful fingerprint that tying is wired correctly.
- **ln(V) is a hard floor** for initial loss when targets are independent of the logits.
- **RoPE uses INTERLEAVED pairs**; its backward is free (orthogonal ⇒ negated angles). Attention
  backward must **re-apply the causal mask** to `dS`. Cached keys are stored already-rotated.
- **fp32 finite differences are useless for gradchecking** — hence the f64 instantiation including
  an f64 `Model`. Verify the harness is *sensitive* (perturb by 1%, confirm failure) before
  trusting a clean pass.
- **Attention probabilities are NOT exposed to Python.** `attention.cpp` keeps `prob` in the arena;
  the decode path keeps per-head scores in local `sc[]` ([model.cpp:429-437](core/src/model.cpp)).
  2.0 Phase 2 must add a capture API before any attention visualization can show real data.

**Training / data**

- **BPE merge count is bounded by corpus DIVERSITY, not length.** Repeating the same 29 unique words
  12x or 60x both yield exactly 92 merges. Bit me twice while writing tests — use a lexically
  diverse corpus whenever a test asserts an exact `vocab_size`.
- **Match the cosine schedule to the steps you will actually run.** A 3000-step schedule plateaued
  at val 4.4714 with LR still near maximum; 1500 steps reached **4.2996** — 3.8% better in half the
  wall-clock. A checkpoint stranded mid-schedule is not a converged model.
- **A 20-batch eval is optimistically biased** (4.2996 vs 4.3877 over 200 batches).
- **Per-token loss is not comparable across vocabularies** — divide by chars/token for nats/char.
- **Multi-document corpora need a separator.** Without one, a training window sampled across a
  boundary teaches a transition that does not exist. `encode_documents()` inserts the EOT id
  BETWEEN documents only, so single-document corpora stay byte-identical.
- **Special tokens must be inert by default.** `encode()` ignores their literal text unless
  `allow_special=True`, so untrusted input cannot inject a control token (tiktoken's rule).
- **Special token ids are persisted explicitly**, not recomputed on load: a truncated merge list
  would otherwise shift them and silently invalidate every checkpoint trained against them.

**Serving**

- **`prefill()` consumes the ENTIRE prompt.** The first generated token must come from
  `sample_last()`; `step(prompt[-1])` feeds the last prompt token twice and generates from a
  corrupted context. Invisible in loss metrics and it produced *plausible* text — with
  `"KING RICHARD:\n"` it duplicated the newline so the model started a new speaker. Only comparing
  against `forward_logits(prompt)[-1].argmax()` catches it.
- **`Sampler::sample()` scales logits in place**, so `sample_last()` works on a copy.
- **Concurrent `GenerationSession`s are safe** (own KV cache + scratch, thread-local tape).
  `Model::forward_logits` is **not** — it uses the single shared activation arena.
- **Testing a stochastic sampler needs care**: "call twice, expect equality" is wrong at
  temperature > 0. Reseed between draws and assert a *distributional* property.
- `httpx.ASGITransport` runs the whole gateway in-process — no ports, no flakiness.

**Tooling**

- **`make datasets` / `make export` must be `.PHONY`** — a `datasets/` directory exists, so make
  otherwise reports "up to date" and does nothing.
- The local top-level `datasets/` package **shadows HuggingFace `datasets`** if that is ever
  installed (cwd precedes site-packages). Intentional here; rename if HF datasets is needed.
- `safetensors` is a **numpy-backend** dependency — it does **not** pull in torch.

## Known Issues

- 2.0 Phases 2-4 not started; `gateway/` has no WebSocket, telemetry, or config routes yet, and
  there is no `web/` frontend.
- Adam moments are not persisted; `--resume` reloads weights and the step counter only.
- No dropout — only weight decay and best-val early stopping regularize.
- No gradient checkpointing; ~1.5 GB activation arena at B=8/T=256.
- Checkpoints are fp32 only; one sequence per `GenerationSession` (no continuous batching), so
  4 concurrent streams give ~1.95x rather than 4x.
- No auth or rate limiting on the gateway (explicit non-goal in `prd.md`).
