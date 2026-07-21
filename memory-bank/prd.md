# Product Requirements (PRD)

## Problem

Modern LLM work is almost always done through frameworks (PyTorch/JAX) that hide the
mathematics behind autograd and fused kernels. That leaves a gap: an engineer can train a
model without ever deriving a backward pass, managing an activation arena, or understanding
what happens between a token ID and a streamed HTTP chunk.

**CSLLM** closes that gap by building a complete, working autoregressive Transformer from
first principles — no PyTorch, no TensorFlow, no autograd library. Every gradient in the
system is derived by hand and verified numerically. The result is both a teaching artifact
and a genuinely deployable inference service.

## Goals

1. A **pure C++ numerics engine** owning tensors, memory (arena allocation), and reverse-mode
   autograd, with no third-party tensor library. GEMM is delegated to Apple Accelerate (BLAS).
2. **Hand-derived backward passes** for every operation: matmul, RMSNorm, RoPE, softmax,
   causal multi-head attention, SiLU/SwiGLU, fused cross-entropy, embedding scatter-add.
3. **Numerical proof of correctness** — double-precision finite-difference gradient checks
   against the analytic gradients, plus NumPy reference implementations of every forward op.
4. A **byte-level BPE tokenizer** in Python, trained on the corpus, lossless over arbitrary UTF-8.
5. A **Python training loop** that drives the C++ backend and converges on TinyShakespeare
   to readable output (target val loss ≈ 1.5).
6. A **FastAPI gateway** streaming tokens over SSE, staying responsive under concurrent load.

## Users & Personas

- **The engineer/author (primary).** Wants to understand and own every layer of the stack,
  from `cblas_sgemm` to the SSE frame. Values correctness proofs over convenience.
- **API consumers.** Any HTTP client (curl, browser, another service) hitting `/generate`
  and consuming a token stream. They need validated inputs, predictable errors, and
  low time-to-first-token.
- **Future agents/contributors.** Extend the project without re-deriving context — served by
  the `memory-bank/` governance docs.

## Functional Requirements

1. **Tokenization** — byte-level BPE: train merges to a target vocab, `encode`/`decode`,
   persist as `vocab.json` + `merges.txt`. Round-trip must be lossless for any UTF-8 input.
2. **Embeddings & positions** — token embedding lookup; **RoPE** applied to Q/K inside attention
   (not an additive position table).
3. **Transformer blocks** — pre-norm residual stack: `x += attn(RMSNorm(x))`,
   `x += swiglu(RMSNorm(x))`. Causal masked multi-head self-attention.
4. **Logits & probabilities** — final RMSNorm, then `lm_head` **weight-tied** to the token
   embedding; softmax to a distribution over the vocabulary.
5. **Sampling** — temperature scaling, top-k, top-p (nucleus), seeded multinomial draw.
6. **Training** — AdamW with weight decay, cosine LR schedule with warmup, global-norm gradient
   clipping, periodic validation, resumable checkpointing.
7. **Persistence** — a self-describing, mmap-able `.csllm` checkpoint format.
8. **Serving** — `POST /generate` (prompt, max_tokens, temperature, top_k, top_p, seed, stream)
   streaming SSE; `GET /health`. Per-request KV cache; generation aborts on client disconnect.

## Non-Goals

- **Any dependency on PyTorch, TensorFlow, JAX, or an existing autograd library.** This is the
  defining constraint; violating it defeats the project's purpose.
- GPU/Metal/CUDA kernels. CPU-only (Accelerate + a thread pool). Metal is backlog.
- Distributed or multi-node training.
- Competing with production LLMs on quality. ~12M params on ~1 MB of text is a correctness and
  architecture demonstration, not a frontier model.
- Training a tokenizer beyond byte-level BPE (no WordPiece/Unigram/SentencePiece).
- Authentication, rate limiting, or multi-tenancy in the gateway.

## Success Metrics

| Metric | Target |
| --- | --- |
| Gradient check (double precision, central differences) | all ops within `rtol=1e-6` |
| Forward ops vs NumPy reference | within `rtol=1e-5` |
| Overfit a single batch (debug config) | loss → ≈0, proving autograd end-to-end |
| TinyShakespeare validation loss | ≈1.5, samples readable as Shakespearean English |
| Tokenizer round-trip | byte-identical on the full corpus + emoji/adversarial UTF-8 |
| Gateway time-to-first-token | < 500 ms on the ~12M config |
| Gateway concurrency | 4 simultaneous streams without event-loop starvation |
| Client disconnect | generation aborts promptly; no orphaned CPU work |
