#pragma once
//
// Differentiable ops. Every entry appends a Node to the active Tape.
//
// The backward of each op is derived by hand; the formulas live in
// memory-bank/design.md and MUST be kept in sync with these implementations.
// No op is done until it has (a) a NumPy forward oracle in tests/reference.py
// and (b) a double-precision gradcheck.
//
#include "csllm/arena.hpp"
#include "csllm/autograd.hpp"
#include "csllm/common.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

// C[M,N] = X[M,K] @ W[K,N].    bwd: dX = dC·Wᵀ,  dW = Xᵀ·dC
template <typename T>
Tensor<T> matmul(Arena& a, const Tensor<T>& x, const Tensor<T>& w);

// C[M,N] = X[M,K] @ W[N,K]ᵀ.   bwd: dX = dC·W,   dW = dCᵀ·X
// Used for the weight-tied lm_head, which shares storage with the embedding
// table — so this op's dW accumulates into the SAME buffer the embedding
// scatter-add writes to (design.md D7).
template <typename T>
Tensor<T> matmul_bt(Arena& a, const Tensor<T>& x, const Tensor<T>& w);

// Elementwise residual add. bwd: gradient flows unchanged to both inputs.
template <typename T>
Tensor<T> add(Arena& a, const Tensor<T>& x, const Tensor<T>& y);

// x[rows,C], gain[C]:  y = gain ⊙ x / r,   r = sqrt(mean(x²) + eps)
// bwd: s = Σⱼ(dyⱼ·gainⱼ·xⱼ);  dxⱼ = gainⱼ·dyⱼ/r − xⱼ·s/(C·r³);  dgainⱼ = Σᵢ dyᵢⱼ·xᵢⱼ/rᵢ
template <typename T>
Tensor<T> rmsnorm(Arena& a, const Tensor<T>& x, const Tensor<T>& gain, T eps);

// Rotary position embedding over a [B,H,T,Dh] buffer. Dh must be EVEN.
// `inverse` rotates by the negated angle: because the rotation is orthogonal,
// that is exactly the vector-Jacobian product — no Jacobian is needed.
template <typename T>
void rope_apply(T* x, i64 B, i64 H, i64 T_len, i64 head_dim, i64 pos_offset, T theta,
                bool inverse);

// Tape-aware RoPE over x[B,H,T,Dh]; exists so the op can be gradchecked alone.
template <typename T>
Tensor<T> rope(Arena& a, const Tensor<T>& x, i64 pos_offset, T theta);

// Row-wise softmax over scores[n,T,T] with a causal mask (key j ≤ query i).
// bwd: dx = y ⊙ (dy − Σ(dy⊙y)); masked entries stay exactly zero.
template <typename T>
Tensor<T> softmax_causal(Arena& a, const Tensor<T>& scores);

// y = x·σ(x).   bwd: dx = dy·σ·(1 + x·(1−σ))
template <typename T>
Tensor<T> silu(Arena& a, const Tensor<T>& x);

// Fused SwiGLU FFN: out = ( silu(x·wg) ⊙ (x·wu) ) · wd
//   x[rows,C], wg[C,H], wu[C,H], wd[H,C] -> out[rows,C]
// bwd: dh = dout·wdᵀ;  da = dh⊙b⊙silu′(a);  db = dh⊙silu(a);  then matmul rules.
template <typename T>
Tensor<T> swiglu(Arena& a, const Tensor<T>& x, const Tensor<T>& wg, const Tensor<T>& wu,
                 const Tensor<T>& wd);

// Row gather from table[V,C] by ids[n] -> [n,C].
// bwd: scatter-add. Under weight tying this table's gradient also receives the
// lm_head contribution from matmul_bt — missing either path trains but
// converges wrong, and only a gradcheck catches it.
template <typename T>
Tensor<T> embedding(Arena& a, const Tensor<T>& table, const i32* ids, i64 n);

// Fused log-softmax + NLL over logits[n,V], mean-reduced. Returns a scalar
// tensor so the loss participates in the tape like any other node.
// bwd: dlogits = (p − onehot)/n
template <typename T>
Tensor<T> cross_entropy(Arena& a, const Tensor<T>& logits, const i32* targets, i64 n);

}  // namespace csllm
