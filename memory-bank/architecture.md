# Architecture

## Overview

CSLLM is a three-layer system built around a single compiled artifact. A **C++ engine**
(`core/`) owns all numerics, memory, and gradients, and is compiled by CMake into a pybind11
extension module `_csllm_core`, which CMake places inside the Python package `csllm/`.
Python never performs tensor math — it orchestrates.

The decisive property: **the same compiled engine serves both training and inference.** There
is no separate inference reimplementation, so the served model cannot drift from the trained
one. Weights are shared read-only across requests; only the KV cache is per-request.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Arena allocator | `core/src/arena.cpp` | Bump allocation for activations; reset each step. Parameters live in a separate persistent arena. |
| Tensor | `core/src/tensor.cpp` | `Tensor<T>`: shape, strides, data, grad, `requires_grad`. Templated on scalar for float (train) / double (gradcheck). |
| Autograd | `core/src/autograd.cpp` | `Node{backward_fn, parents}` appended to a `Tape`; `backward()` topo-sorts and walks in reverse accumulating grads. |
| GEMM | `core/src/gemm.cpp` | `cblas_sgemm`/`cblas_dgemm` via Accelerate; portable naive fallback off-Apple. |
| Ops | `core/src/ops_*.cpp` | RMSNorm, RoPE, softmax, SiLU/SwiGLU, fused cross-entropy — each with a hand-derived backward. |
| Attention | `core/src/attention.cpp` | Causal multi-head self-attention fwd/bwd; KV cache for incremental decode. |
| Model | `core/src/model.cpp` | `Block`, `CSLLM`, `GenerationSession`. Weight-tied `lm_head`. |
| Optimizer | `core/src/optim.cpp` | AdamW + global-norm gradient clipping, in C++ over parameter arenas. |
| Sampler | `core/src/sampler.cpp` | Temperature, top-k, top-p, seeded `std::mt19937` multinomial. |
| Serialization | `core/src/serialize.cpp` | `.csllm` format: magic, version, JSON manifest, aligned fp32 payload; mmap-able. |
| Thread pool | `core/src/threadpool.cpp` | Parallelism over `B×H` (Accelerate already threads GEMM internally). |
| Bindings | `core/bindings/py_module.cpp` | pybind11 surface; releases the GIL around all compute. |
| Python package | `csllm/` | `tokenizer.py` (byte-level BPE), `config.py` (`ModelConfig` ⇄ JSON), `data.py` (corpus → uint16 memmap, batching). Re-exports `_csllm_core`. |
| Training | `train/` | `train_tokenizer.py`, `train.py` — the orchestration loop. |
| Gateway | `gateway/` | FastAPI app: `main.py`, `engine.py`, `schemas.py`, `settings.py`. |
| Tests | `tests/` | `reference.py` (NumPy oracle) plus op, gradcheck, tokenizer, serialize, sampler, gateway suites. |

**Why the tokenizer sits in `csllm/` rather than `train/`:** the gateway needs `encode`/`decode`
at request time. A shared package means one implementation for both training and serving.

## Data Flow

**Training**

```
corpus.txt → BPE train (Python) → vocab.json + merges.txt
           → encode once → uint16 memmap → 90/10 split
           → batch sampler → (B,T) int32 ids
           → [C++] embed → ×N blocks → RMSNorm → tied lm_head → logits
           → [C++] fused cross-entropy → loss
           → [C++] loss.backward() walks the tape → grads
           → [C++] clip + AdamW.step() → updated params
           → periodic: val loss, sample, checkpoint → model.csllm
```

**Inference**

```
HTTP POST /generate → Pydantic validation
  → BPE encode → prompt ids
  → [C++] GenerationSession (own KV cache) prefill
  → loop: forward_incremental(tok,pos) → logits
        → sampler(temperature, top_k, top_p) → next id
        → BPE decode → SSE frame → client
  → until max_tokens, EOS, or client disconnect
```

A single Transformer block, in order:

```
x = x + attention(RMSNorm(x))     # RoPE applied to Q/K inside attention, causal mask
x = x + swiglu(RMSNorm(x))        # silu(x·Wg) ⊙ (x·Wu) then ·Wd
```

## Tech Stack

- **C++20**, Apple clang, `-O3 -funroll-loops -fno-math-errno`. Deliberately **not** `-ffast-math`
  (it would defeat NaN/Inf guards and reproducibility).
- **CMake ≥ 3.20**, driven by **scikit-build-core** so `pip install -e .` compiles the C++.
- **pybind11** for bindings (found via `find_package`, with a `FetchContent` fallback).
- **Apple Accelerate** for BLAS — no third-party math library.
- **Python 3.14** (Homebrew) in `.venv`; NumPy for tests/data only, never in the model path.
- **FastAPI + Pydantic v2 + uvicorn + sse-starlette** for the gateway.
- **pytest** for verification.

## External Dependencies

| Dependency | Role | Notes |
| --- | --- | --- |
| Accelerate.framework | BLAS (`cblas_sgemm`/`dgemm`) | Ships with macOS; needs `ACCELERATE_NEW_LAPACK` on modern SDKs |
| pybind11 | C++ ↔ Python bindings | Build-time; FetchContent fallback if not pip-installed |
| scikit-build-core | PEP 517 backend bridging pip → CMake | Build-time only |
| NumPy | Test oracle, data binarization | **Never** on the model's forward/backward path |
| FastAPI / Pydantic v2 / uvicorn / sse-starlette | HTTP gateway, validation, SSE | Runtime, gateway only |
| regex | GPT-2-style pre-tokenization pattern | Tokenizer training/encoding |
| TinyShakespeare (~1.1 MB) | Training corpus | Downloaded on first run |

**Explicitly absent: PyTorch / TensorFlow / JAX / Eigen.** Their absence is the point.
