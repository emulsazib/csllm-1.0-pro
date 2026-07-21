# CSLLM 1.0 Pro

An autoregressive Transformer built **from scratch** — no PyTorch, no TensorFlow, no autograd
library. A C++20 engine owns tensors, memory, and hand-derived gradients; Python owns the BPE
tokenizer, data, and the training loop; FastAPI streams generated tokens over SSE.

```
text → BPE → ids → [C++] embed → 6 × (RMSNorm → RoPE attention → RMSNorm → SwiGLU)
     → RMSNorm → tied lm_head → logits → sampler → id → BPE decode → SSE chunk
```

## Architecture

Llama-style: **RoPE + RMSNorm + SwiGLU**, pre-norm residuals, weight-tied `lm_head`.
The same compiled engine serves both training and inference, so the served model cannot
drift from the trained one.

| Config | Layers | Heads | d_model | Context | Vocab | Params |
| --- | --- | --- | --- | --- | --- | --- |
| `configs/debug.json` | 2 | 2 | 64 | 32 | 512 | ~0.14 M |
| `configs/shakespeare.json` | 6 | 6 | 384 | 256 | 4096 | **12.19 M** |

## Layout

| Path | Role |
| --- | --- |
| `core/` | C++ backend — `include/csllm/`, `src/`, `bindings/` |
| `csllm/` | Shared Python package (tokenizer, config, data) + the compiled `_csllm_core` |
| `train/` | Training entrypoints |
| `gateway/` | FastAPI app |
| `tests/` | NumPy oracles, gradient checks, gateway tests |
| `memory-bank/` | Governance docs (knbase) — start with `prd.md` and `design.md` |

## Requirements

- macOS on Apple Silicon (uses **Accelerate** for BLAS; falls back to a naive GEMM elsewhere)
- `cmake` ≥ 3.20 — `brew install cmake`
- Python ≥ 3.10 (developed on Homebrew Python 3.14)
- A C++20 compiler (Apple clang)

## Quick start

```bash
make setup     # create .venv and compile the C++ extension
make smoke     # print BLAS backend, thread count, parameter count
make test      # run the suite
```

## Verification

Because every gradient is derived by hand, correctness is enforced structurally:

- Forward ops are checked against **NumPy oracles** in `tests/reference.py`.
- Backward passes are checked by **double-precision** finite differences — fp32 central
  differences are too noisy to distinguish a real bug from rounding, which is why the C++ ops
  are templated on scalar type.
- The debug config must **overfit a single batch to ≈0 loss** — the strongest end-to-end proof
  that autograd is wired correctly.

## Status

Built in four approval-gated phases:

- [x] **Phase 1** — scaffolding, CMake/scikit-build-core bridge, configs
- [x] **Phase 2** — C++ core: tensors, tape autograd, RoPE/RMSNorm/SwiGLU/attention, AdamW, sampler,
      KV-cache decoding, `.csllm` checkpoints — *83 tests, all gradients verified in float64*
- [ ] **Phase 3** — BPE tokenizer, data pipeline, training loop
- [ ] **Phase 4** — FastAPI gateway with SSE streaming

Measured at the 12M config (8 CPU threads): **~0.28 s/step** (≈3,600 tok/s) for
forward+backward+AdamW at B=4/T=256, 510 MB activation arena, 4.7 MB KV cache per session.

## Ground rules

Documented in `memory-bank/rules.md`. The load-bearing ones:

1. **Never add PyTorch/TensorFlow/JAX** — the constraint is the point.
2. **No op ships without a NumPy oracle and a double-precision gradcheck.**
3. **No `-ffast-math`** — it breaks NaN/Inf guards and reproducibility.
4. **Never block the FastAPI event loop** — C++ releases the GIL; routes use `asyncio.to_thread`.
