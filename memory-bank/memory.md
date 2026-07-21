# Project Memory

## Summary

**CSLLM 1.0 Pro** builds a ~12M-parameter autoregressive Transformer entirely from scratch:
a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch, no autograd library),
a **Python** byte-level BPE tokenizer and training loop, and a **FastAPI** gateway streaming tokens
over SSE. Llama-style: **RoPE + RMSNorm + SwiGLU**, pre-norm residuals, weight-tied `lm_head`.

**Phases 1-3 are complete.** A trained model exists at `data/model.csllm` (12,194,688 params,
val 4.39/token = **1.41 nats/char**) that generates coherent Shakespearean dialogue with correct
speaker formatting and real character names. Phase 4 (FastAPI gateway) has not started — everything
in `gateway/` is still a placeholder.

## Recent Changes

*(reverse-chronological)*

- **2026-07-21 — Phase 3: tokenizer, data, training loop.** Implemented `csllm/tokenizer.py`
  (byte-level BPE with incremental merge training — inverted pair→word index, so 3,840 merges over
  1.1 MB take **1.9s**), `csllm/data.py` (download, uint16 binarization, batch sampler),
  `train/train_tokenizer.py`, `train/train.py` (AdamW, cosine+warmup, clipping, eval,
  best-val checkpointing, periodic sampling, `--resume`), and `train/sample.py` (streaming
  generation with UTF-8-boundary buffering). Added `tests/test_tokenizer.py` (44) and
  `tests/test_training.py` (7). **134 tests pass, ruff clean.**
  **Trained the 12M model:** 1500 steps, B=8×T=256, lr 1e-3→5e-5, **11.9 min**, ~4,400 tok/s.
  Best val **4.2996/token** at step 1000 (20-batch estimate; 200-batch re-eval gives 4.3877).
  = **1.41 nats/char = 2.03 bits/char**, perplexity 80.5, 47% better than uniform.
  Train/val gap +1.26 → heavily overfit, as expected for 12M params on 310k tokens.
- **2026-07-21 — Phase 2: the C++ core.** Full engine: arena-backed `Tensor<T>`, tape autograd, ten
  ops with hand-derived backwards, fused causal MHA with RoPE, weight-tied block stack, AdamW,
  sampler, JSON reader, mmap-able `.csllm`, incremental KV-cache decode. 83 tests; all gradients
  verified in float64; overfit-one-batch 2.803 → 0.000043.
- **2026-07-21 — Phase 1: scaffolding & build system.** Tree, CMake/scikit-build-core bridge,
  `Makefile`, configs. Build green, Accelerate active.
- **2026-07-21 — Governance bootstrap.**

## Learnings & Gotchas

**Environment (verified on this machine)**

- `cmake` **4.4.0**; CMake 4 removed compatibility with `cmake_minimum_required(VERSION < 3.5)`.
- **pybind11 must be >= 3.0** (we resolve 3.0.4) — 2.13.x predates Python 3.14's C-API.
  Pinned in `pyproject.toml` and the CMake FetchContent tag; **do not lower**.
- Use Homebrew **Python 3.14.6** for `.venv`; system 3.9.6 is EOL. No PyTorch ⇒ no wheel risk.
- arm64 + Apple clang 21; Accelerate gives multithreaded BLAS free. 8 hardware threads.
- Fast iteration: `pip install -e . --no-build-isolation` (`make build`).

**Engine lessons (Phase 2)**

- **The tape needs no topological sort** — nodes are appended in forward order, so descending from
  the root is already valid reverse order; only a reachability mark is needed.
- **Weight tying**: `matmul_bt` shares storage with `tok_emb`; the gradient accumulates from the
  embedding scatter-add *and* `matmul_bt`'s dW. Fingerprint: at init the argmax equals the *input*
  token ~85% of the time (chance 0.02%).
- **ln(V) is a hard floor** for initial loss when targets are independent of the logits. But a loss
  printed *after* a few optimizer steps is legitimately below it — that is learning, not a bug.
- **RoPE uses INTERLEAVED pairs** (2p, 2p+1), not the half-split convention. Its backward is free
  (orthogonal ⇒ negated angles). Attention backward must **re-apply the causal mask** to `dS`.
- **Cached keys are stored already-rotated**, matching training.
- **fp32 finite differences are useless for gradchecking** — hence the f64 instantiation, including
  an f64 `Model`. Always verify the harness is *sensitive* (perturb an analytic gradient by 1% and
  confirm failure) before trusting a clean pass.

**Training lessons (Phase 3)**

- **BPE merge count is bounded by corpus DIVERSITY, not length.** 12 and 60 repetitions of the same
  29 unique words both yield exactly 92 merges. `train_tokenizer.py` therefore hard-fails when the
  learned vocab is smaller than the config asks — a silent shortfall would mismatch the embedding
  table. TinyShakespeare supports the full 3,840 merges.
- **Naive BPE training is too slow.** Recounting all pairs per merge is O(corpus × merges). Keeping
  unique pre-tokens with frequencies plus an inverted pair→word index makes it 1.9s instead of minutes.
- **A single BPE token can end mid-UTF-8-sequence** (verified on emoji). This is why `decode_bytes()`
  is separate from `decode()`; `train/sample.py` buffers bytes and flushes only at code-point
  boundaries. **Phase 4's SSE gateway needs exactly this.**
- **Match the cosine schedule to the steps you will actually run.** The first run used a 3000-step
  schedule, plateaued at val 4.4714 by step 750 with the LR still near maximum, then overfit. Rerunning
  with a 1500-step schedule reached **4.2996** — 3.8% better in *half* the wall-clock, purely from
  letting the decay complete. A checkpoint stranded mid-schedule is not a converged model.
- **A 20-batch eval is noticeably optimistic**: 4.2996 vs 4.3877 over 200 batches. Use a wider eval
  before quoting a number, and expect ±0.1 jitter between adjacent evals.
- **Per-token loss is not comparable across vocabularies.** Divide by chars/token (3.11 on the val
  split) to get nats/char, which is what char-level baselines report.
- 12M params on 310k tokens overfits hard (train/val gap +1.26). Best-val checkpointing is what makes
  the run usable; the last-step weights are worse than the saved ones.

## Known Issues

- **Phase 4 not started:** everything in `gateway/` is a placeholder.
- Adam moment estimates are **not** persisted; `--resume` reloads weights and the step counter only,
  so the moments rebuild over a few dozen steps.
- No dropout or data augmentation, so the only regularizers are weight decay and early stopping via
  best-val checkpointing.
- No gradient checkpointing; ~1.5 GB activation arena at B=8/T=256.
- Thread-pool granularity vs Accelerate's internal threading is unbenchmarked.
- Checkpoints are fp32 only; `GenerationSession` handles one sequence at a time.
