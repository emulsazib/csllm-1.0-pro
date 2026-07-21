# Design

## Modules & Interfaces

### C++ core (`core/include/csllm/`) — implemented in Phase 2

```cpp
template <typename T> struct Tensor {          // tensor.hpp
  T* data; T* grad; Shape shape; bool requires_grad; int node_id;
};
template <typename T> Tensor<T> make_tensor(Arena&, const Shape&, bool requires_grad);

struct Node { std::function<void()> backward_fn; std::vector<int> parents; std::string name; };
template <typename T> class Tape {             // autograd.hpp
  int push(Node); void backward(int root); void clear(); static Tape& active();
};
class NoGradGuard { ... };                     // inference / no_grad regions

// ops.hpp — every op appends a Node to the active Tape
template <typename T> Tensor<T> matmul(Arena&, const Tensor<T>& x, const Tensor<T>& w);
template <typename T> Tensor<T> matmul_bt(Arena&, const Tensor<T>& x, const Tensor<T>& w);
template <typename T> Tensor<T> add(Arena&, const Tensor<T>&, const Tensor<T>&);
template <typename T> Tensor<T> rmsnorm(Arena&, const Tensor<T>& x, const Tensor<T>& gain, T eps);
template <typename T> void      rope_apply(T*, i64 B, i64 H, i64 T, i64 dh, i64 pos, T theta, bool inverse);
template <typename T> Tensor<T> rope(Arena&, const Tensor<T>&, i64 pos_offset, T theta);
template <typename T> Tensor<T> softmax_causal(Arena&, const Tensor<T>& scores);
template <typename T> Tensor<T> silu(Arena&, const Tensor<T>&);
template <typename T> Tensor<T> swiglu(Arena&, const Tensor<T>& x, wg, wu, wd);
template <typename T> Tensor<T> embedding(Arena&, const Tensor<T>& table, const i32* ids, i64 n);
template <typename T> Tensor<T> cross_entropy(Arena&, const Tensor<T>& logits, const i32*, i64 n);

// attention.hpp — one fused node, not a composition of primitives
template <typename T> Tensor<T> attention(Arena&, const Tensor<T>& x /*[B,T,C]*/,
                                          wq, wk, wv, wo, const AttentionParams&);
template <typename T> class KVCache { T* key(i64 layer); T* value(i64 layer); ... };

// model.hpp
template <typename T> class Model {
  Model(const ModelConfig&, u64 seed);
  static Model load(const std::string&);  void save(const std::string&) const;
  T forward_loss(const i32* ids, const i32* targets, i64 B, i64 T);
  void backward();  void zero_grad();
  const T* forward_logits(const i32* ids, i64 B, i64 T);   // no-grad
};
class GenerationSession {                      // one per HTTP request
  const f32* prefill(const i32* ids, i64 n);
  const f32* decode(i32 token);
  i32 step(i32 token, const SamplingParams&);
  void reset(); void reseed(u64);
};

template <typename T> class AdamW {            // optim.hpp
  T clip_grad_norm(T max_norm);   // returns the PRE-clip global norm
  void step(T lr);
};
```

`json.hpp` is a minimal reader/writer added so `.csllm` headers stay human-inspectable
without a third-party dependency.

### Python (`csllm/`)

```python
@dataclass-like ModelConfig                    # C++ type, exposed via pybind11
class BPETokenizer:                            # tokenizer.py — Phase 3
    train / encode / decode / save / load
def load_config(path) -> ModelConfig           # config.py
```

## Key Decisions

| # | Decision | Rationale / trade-off |
| --- | --- | --- |
| D1 | **Pure C++, zero PyTorch** | The project's purpose. One engine serves training *and* inference, so the served model cannot drift from the trained one. |
| D2 | **Apple Accelerate for GEMM** | Tuned, multithreaded `sgemm`/`dgemm`, no third-party dependency. Naive fallback keeps it portable. |
| D3 | **RoPE + RMSNorm + SwiGLU, pre-norm** | Matches contemporary production models. Pre-norm trains stably without warmup tricks. |
| D4 | **Tape-based reverse-mode autograd** | Nodes are appended in forward order, so a node's index always exceeds its parents'. Walking indices downward from the root is therefore already a valid reverse-topological order — **no sort is needed**, only a reachability mark. |
| D5 | **Arena allocation** | Activations bump-allocate and reset per step. Parameters and their gradients live in separate persistent arenas so they survive the reset. |
| D6 | **Ops templated on scalar type** | Enables **double-precision** gradient checking. The single most important testability decision — see Verification. |
| D7 | **Weight-tied `lm_head` via `matmul_bt`** | `logits = h @ tok_embᵀ`. Saves 1.57M params (~13%). The embedding gradient accumulates from **two** paths (scatter-add + `matmul_bt`'s dW), both `+=` into one buffer. |
| D8 | **Attention is one fused node** | The `[B,H,T,T]` probability matrix is materialised once and shared by every backward branch. |
| D9 | **Bindings release the GIL** | `py::gil_scoped_release` around all compute — what lets FastAPI serve concurrent streams. |
| D10 | **Per-request `GenerationSession`** | Weights shared read-only; only the KV cache (4.7 MB measured) is per-request. |
| D11 | **No `-ffast-math`** | Would break NaN/Inf guards and reproducibility. Asserted by a test on `__FAST_MATH__`. |
| D12 | **mmap-able `.csllm` checkpoints** | Near-instant gateway startup and page-cache sharing. |
| D13 | **RoPE uses INTERLEAVED pairs** | Channels (2p, 2p+1) rotate together. The alternative (d paired with d+Dh/2) is **not** interchangeable; `tests/reference.py` must match this choice. |
| D14 | **AdamW skips decay on 1-D tensors** | Decaying RMSNorm gains toward zero would suppress the signal they scale. Standard exclusion. |
| D15 | **Attention backward reuses dead forward buffers** | `qf/kf/vf/ctx` are dead after packing, so the backward aliases them instead of allocating four more `[B*T,C]` buffers (~25% of attention activation memory). |

### Hand-derived gradients (all verified in double precision)

| Op | Forward | Backward |
| --- | --- | --- |
| matmul | `C = A·B` | `dA = dC·Bᵀ`, `dB = Aᵀ·dC` |
| matmul_bt | `C = A·Bᵀ` | `dA = dC·B`, `dB = dCᵀ·A` |
| RMSNorm | `y = g⊙x/r`, `r = √(mean(x²)+ε)` | `s = Σ(dyᵢgᵢxᵢ)`; `dxᵢ = gᵢdyᵢ/r − xᵢs/(d·r³)`; `dgᵢ = dyᵢxᵢ/r` |
| RoPE | per-pair rotation `R(θ)` | orthogonal ⇒ VJP is the same kernel with **negated angles** |
| softmax | `y = softmax(x)` | `dx = y ⊙ (dy − Σ(dy⊙y))` |
| SiLU | `y = x·σ(x)` | `dx = dy·σ(1 + x(1−σ))` |
| SwiGLU | `h = silu(a)⊙b`, `a=x·Wg`, `b=x·Wu` | `da = dh⊙b⊙silu′(a)`, `db = dh⊙silu(a)`, then matmul rules |
| Attention | `S=QKᵀ/√dₕ` (masked), `P=softmax(S)`, `O=PV` | `dV=Pᵀ·dO`; `dP=dO·Vᵀ`; `dS=softmax_bwd(P,dP)` **re-masked**, scale folded in; `dQ=dS·K`; `dK=dSᵀ·Q` |
| Cross-entropy | fused log-softmax + NLL | `dlogits = (p − onehot)/n` |
| Embedding | gather rows | scatter-add; **accumulates twice** under weight tying (D7) |

## Data Models

### `.csllm` checkpoint (mmap-able)

```
offset 0  : magic "CSLLM\0\0\0"   (8 bytes)
offset 8  : version    uint32     (little-endian)
offset 12 : header_len uint32
offset 16 : JSON header { "config": {...}, "tensors": [{"name","dtype","shape","offset"}, ...] }
then      : payload, 64-byte aligned
```

### Model configs

| Config | n_layer | n_head | n_embd | block | vocab | ffn_hidden | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `debug.json` | 2 | 2 | 64 | 32 | 512 | 192 | ~0.14 M |
| `shakespeare.json` | 6 | 6 | 384 | 256 | 4096 | 1024 | **12,194,688** |

`head_dim = 64` (must be **even** for RoPE). `ffn_hidden ≈ ⅔·4·n_embd` rounded to a multiple of 64.
Measured at the 12M config: activation arena **510 MB** at B=4/T=256, KV cache **4.7 MB** per session,
**~0.28 s/step (≈3,600 tok/s)** for forward+backward+AdamW on 8 CPU threads.

### Parameter names
`tok_emb`, `blocks.{i}.attn_norm.gain`, `blocks.{i}.attn.{wq,wk,wv,wo}`,
`blocks.{i}.ffn_norm.gain`, `blocks.{i}.ffn.{wg,wu,wd}`, `norm_f.gain`. Insertion-ordered,
so checkpoints are byte-stable.

## Conventions

- **C++**: C++20; one `.cpp` per header; `snake_case` functions, `PascalCase` types, trailing `_`
  for in-place ops. Ops are `template<typename T>` with explicit `f32`/`f64` instantiation.
- **Python**: PEP 8, ruff-clean, type hints; no NumPy on the model path.
- **Tests**: every op needs (a) a NumPy forward oracle in `tests/reference.py` and
  (b) a double-precision gradcheck — both before the op counts as done.
- **Errors**: C++ throws `csllm::Error`, translated to Python `RuntimeError` by pybind11.

## Open Questions

1. **Thread-pool granularity** — attention parallelises over `B×H` while Accelerate threads GEMM
   internally. Not yet benchmarked for oversubscription; measure before tuning.
2. **Streaming detokenization** — byte-level BPE can emit a partial UTF-8 sequence; the gateway must
   buffer incomplete code points. Policy to be decided in Phase 4.
3. **Activation memory** — 510 MB at B=4/T=256 with no recomputation. Gradient checkpointing or
   tiled (flash-style) attention would cut it; deferred to backlog.
4. **Checkpoint dtype** — fp32 only; bf16 storage would halve file size.
5. Batched/continuous decoding across concurrent sessions — currently one sequence per session.
