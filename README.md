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

# train the tokenizer + binarize the corpus, then train the model
python -m train.train_tokenizer --config configs/shakespeare.json
python -m train.train --config configs/shakespeare.json --steps 3000 --batch-size 8 --lr 1e-3
```

Iterate on `configs/debug.json` first — the whole pipeline runs in seconds there:

```bash
python -m train.train_tokenizer --config configs/debug.json \
    --out data/tokenizer-debug --data-dir data/debug
python -m train.train --config configs/debug.json --steps 300 --batch-size 16 --lr 3e-3 \
    --tokenizer-dir data/tokenizer-debug --data-dir data/debug --out data/debug/model.csllm
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
      KV-cache decoding, `.csllm` checkpoints — *all gradients verified in float64*
- [x] **Phase 3** — BPE tokenizer, data pipeline, training loop — *134 tests, model trained*
- [ ] **Phase 4** — FastAPI gateway with SSE streaming

### Trained model

The 12M config trains on TinyShakespeare in **11.9 minutes** on 8 CPU threads
(1500 steps, B=8 x T=256, ~4,400 tok/s):

| Metric | Value |
| --- | --- |
| Validation loss | 4.39 / token |
| **Nats per character** | **1.41** (2.03 bits/char) |
| Perplexity | 80.5 |
| vs. uniform ln(4096) | 47% better |

```
KING RICHARD:
DUCHESS OF YORK:
O, thy lord, it was my mother.
O sir, I hear, I'll not have a king.

DUKE OF AUMERLE:
My husband's good, poor Clarence, to my heart.
```

Two notes on reading these numbers. Per-token loss is **not** comparable across vocabularies —
divide by chars/token (3.11 here) to get nats/char, which is what char-level baselines report.
And with 12M parameters against 310k training tokens the model overfits hard (train/val gap +1.26),
so best-validation checkpointing is what makes the run usable.

## Ground rules

Documented in `memory-bank/rules.md`. The load-bearing ones:

1. **Never add PyTorch/TensorFlow/JAX** — the constraint is the point.
2. **No op ships without a NumPy oracle and a double-precision gradcheck.**
3. **No `-ffast-math`** — it breaks NaN/Inf guards and reproducibility.
4. **Never block the FastAPI event loop** — C++ releases the GIL; routes use `asyncio.to_thread`.
