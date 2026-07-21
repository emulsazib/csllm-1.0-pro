# Design

## Modules & Interfaces

### C++ core (`core/include/csllm/`)

```cpp
template <typename T> struct Tensor {          // tensor.hpp
  T* data; T* grad; Shape shape; Strides strides;
  bool requires_grad; int node_id;
};

struct Node { std::function<void()> backward_fn; std::vector<int> parents; };  // autograd.hpp
struct Tape { std::vector<Node> nodes; void backward(int root); void clear(); };

// ops.hpp — every op appends a Node to the active Tape
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);
Tensor<T> rmsnorm(const Tensor<T>& x, const Tensor<T>& gain, T eps);
void      rope_(Tensor<T>& q, Tensor<T>& k, int pos_offset);   // in-place
Tensor<T> softmax_causal(const Tensor<T>& scores);
Tensor<T> swiglu(const Tensor<T>& x, const Tensor<T>& wg, const Tensor<T>& wu, const Tensor<T>& wd);
Tensor<T> cross_entropy(const Tensor<T>& logits, const int32_t* targets);  // fused log-softmax + NLL

// model.hpp
class CSLLM {
  static CSLLM load(const std::string& path);          // mmap the .csllm
  Tensor<float> forward(const int32_t* ids, int B, int T);
  float forward_loss(const int32_t* ids, const int32_t* targets, int B, int T);
  void  zero_grad();
  size_t num_params() const;
};

class GenerationSession {                     // one per HTTP request; owns its KV cache
  GenerationSession(const CSLLM& model, uint64_t seed);
  void prefill(const int32_t* ids, int n);
  int32_t step(float temperature, int top_k, float top_p);   // returns next token id
  void reset();
};

class AdamW {                                  // optim.hpp
  AdamW(ParamGroup& params, float lr, float beta1, float beta2, float eps, float wd);
  float clip_grad_norm(float max_norm);        // returns pre-clip global norm
  void step(float lr);
};
```

### Python (`csllm/`)

```python
@dataclass
class ModelConfig:                    # config.py — mirrors the C++ struct, (de)serialized to JSON
    vocab_size: int; n_layer: int; n_head: int; n_embd: int
    block_size: int; ffn_hidden: int; rope_theta: float = 10000.0; norm_eps: float = 1e-5

class BPETokenizer:                   # tokenizer.py — byte-level, lossless over any UTF-8
    def train(self, text: str, vocab_size: int) -> None: ...
    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, dir: str) -> None: ...       # vocab.json + merges.txt
    @classmethod
    def load(cls, dir: str) -> "BPETokenizer": ...
```

## Key Decisions

| # | Decision | Rationale / trade-off |
| --- | --- | --- |
| D1 | **Pure C++, zero PyTorch** | The project's purpose. One engine serves training *and* inference, so the served model cannot drift from the trained one. Cost: every backward derived by hand. |
| D2 | **Apple Accelerate for GEMM** | Tuned, multithreaded `sgemm` with zero third-party dependencies on arm64 macOS. A naive fallback keeps the code portable. |
| D3 | **RoPE + RMSNorm + SwiGLU, pre-norm** | Matches contemporary production models (Llama-style). Pre-norm trains stably without warmup tricks — important when the optimizer is hand-written. RoPE extrapolates past the training context. |
| D4 | **Tape-based reverse-mode autograd** | Closure-per-node is simple and debuggable; the graph is rebuilt each step, so control flow is free. |
| D5 | **Arena allocation** | Activations bump-allocate and reset per step: no per-op malloc traffic, no fragmentation, trivially freed. Parameters sit in a separate persistent arena. |
| D6 | **Ops templated on scalar type** | Enables **double-precision** gradient checking. fp32 central differences are too noisy to distinguish a real gradient bug from rounding — this is the single most important testability decision. |
| D7 | **Weight-tied `lm_head`** | Saves 1.57M params (~13% of the model) and regularizes. Consequence: the embedding gradient **accumulates from two paths** — a classic and easy-to-miss bug. |
| D8 | **Tokenizer in Python, in the shared package** | Encoding cost is negligible beside the forward pass; Python keeps BPE debuggable. Shared so training and gateway use one implementation. |
| D9 | **Bindings release the GIL** | `py::gil_scoped_release` around compute is what lets FastAPI serve concurrent streams via `asyncio.to_thread`. |
| D10 | **Per-request `GenerationSession`** | Weights shared read-only; only the KV cache (~4.5 MiB) is per-request, so concurrency is cheap and requests cannot corrupt each other's state. |
| D11 | **No `-ffast-math`** | It would license reassociation and break NaN/Inf guards and reproducibility. `-O3 -funroll-loops -fno-math-errno` instead. |
| D12 | **mmap-able `.csllm` checkpoints** | Near-instant gateway startup and page-cache sharing across worker processes. |

### Hand-derived gradients (the critical-path work)

| Op | Forward | Backward |
| --- | --- | --- |
| matmul | `C = A·B` | `dA = dC·Bᵀ`, `dB = Aᵀ·dC` (both via BLAS) |
| RMSNorm | `y = g⊙x/r`, `r = √(mean(x²)+ε)` | `s = Σ(dyᵢgᵢxᵢ)`; `dxᵢ = gᵢdyᵢ/r − xᵢs/(d·r³)`; `dgᵢ = dyᵢxᵢ/r` |
| RoPE | per-pair rotation `R(θ)` | rotation is **orthogonal** ⇒ VJP is the same kernel with **negated angles** |
| softmax | `y = softmax(x)` | `dx = y ⊙ (dy − Σ(dy⊙y))` |
| SiLU | `y = x·σ(x)` | `dx = dy·σ(1 + x(1−σ))` |
| SwiGLU | `h = silu(a)⊙b`, `a=x·Wg`, `b=x·Wu` | `da = dh⊙b⊙silu′(a)`, `db = dh⊙silu(a)`, then matmul rules |
| Attention | `S=QKᵀ/√dₕ` (masked), `P=softmax(S)`, `O=PV` | `dV=Pᵀ·dO`; `dP=dO·Vᵀ`; `dS=softmax_bwd(P,dP)` **re-masked**; `dQ=dS·K/√dₕ`; `dK=dSᵀ·Q/√dₕ` |
| Cross-entropy | fused log-softmax + NLL | `dlogits = (p − onehot)/(B·T)` |
| Embedding | gather rows | scatter-add; **accumulates twice** under weight tying (D7) |

## Data Models

### `.csllm` checkpoint (safetensors-like, mmap-able)

```
offset 0  : magic   "CSLLM\0\0\0"        (8 bytes)
offset 8  : version uint32               (little-endian)
offset 12 : header_len uint32
offset 16 : JSON header {
              "config": { ...ModelConfig... },
              "tensors": [{"name","dtype","shape","offset"}, ...]
            }
then      : 64-byte-aligned fp32 payload
```

### Model configs

| Config | n_layer | n_head | n_embd | block | vocab | ffn_hidden | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `debug.json` | 2 | 2 | 64 | 32 | 512 | 192 | ~0.1 M |
| `shakespeare.json` | 6 | 6 | 384 | 256 | 4096 | 1024 | **~12.2 M** |

`head_dim = n_embd / n_head = 64` (must be **even** for RoPE pairing).
`ffn_hidden ≈ ⅔·4·n_embd`, rounded to a multiple of 64 — the standard SwiGLU sizing that
keeps parameter count comparable to a 4× GELU MLP despite SwiGLU's third matrix.
KV cache per session = `2 · n_layer · block · n_embd · 4 B` ≈ **4.5 MiB** at the 12M config.

### Tokenizer artifacts
`vocab.json` (id → token bytes, base64) and `merges.txt` (ordered merge rules) in `data/tokenizer/`.

## Conventions

- **C++**: C++20; headers `core/include/csllm/*.hpp`, one `.cpp` per header; `snake_case` functions,
  `PascalCase` types, trailing `_` for in-place ops (`rope_`). No exceptions on the hot path.
- **Python**: PEP 8, `ruff`, type hints throughout; dataclasses for config; no NumPy in the model path.
- **Tests**: every C++ op needs (a) a NumPy forward oracle in `tests/reference.py` and
  (b) a double-precision gradcheck entry — **both before the op is considered done**.
- **Errors**: C++ throws `std::runtime_error` translated to Python exceptions by pybind11;
  the gateway maps them to structured JSON error responses.
- **Naming**: parameter tensors follow `blocks.{i}.attn.wq`, `blocks.{i}.ffn.wg`, `tok_emb`,
  `norm_f.gain` — flat dotted keys in the checkpoint manifest.

## Open Questions

1. **Thread-pool granularity** — parallelise attention over `B×H` only, or also tile the FFN?
   Accelerate already threads GEMM internally; nesting our pool inside it may oversubscribe cores.
   Measure before adding.
2. **Streaming detokenization** — byte-level BPE can emit a token that is a partial UTF-8
   sequence. The gateway must buffer incomplete code points rather than emit invalid UTF-8 in an
   SSE frame. Decide the buffering policy in Phase 4.
3. **Checkpoint dtype** — fp32 only for now; bf16 storage would halve file size but needs
   conversion on load.
4. Should `GenerationSession` support batched/continuous batching across requests, or stay
   one-sequence-per-session? Currently the latter, for simplicity.
5. Metal/NEON hand-written kernels — deferred to backlog; revisit only if CPU training proves
   too slow at the 12M config.
