#pragma once
//
// Causal multi-head self-attention with RoPE, plus the KV cache used for
// incremental decoding.
//
// Forward : S = QKᵀ/√dₕ (causally masked), P = softmax(S), O = PV
// Backward: dV = Pᵀ·dO;  dP = dO·Vᵀ;  dS = softmax_bwd(P, dP) RE-MASKED;
//           dQ = dS·K/√dₕ;  dK = dSᵀ·Q/√dₕ
//
// The re-mask in the backward pass is essential: without it, gradient leaks
// into positions the forward pass masked out.
//
#include <vector>

#include "csllm/arena.hpp"
#include "csllm/common.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

struct AttentionParams {
  i64 n_head = 0;
  i64 head_dim = 0;      // must be even, for RoPE pair rotation
  double rope_theta = 10000.0;
  i64 pos_offset = 0;    // >0 when decoding with a warm KV cache
};

// Fused causal multi-head self-attention with RoPE.
//   x: [B,T,C] -> out: [B,T,C];  weights are [C,C] row-major, C = n_head*head_dim.
//
// A single tape node rather than a composition of primitives: the T×T
// probability matrix is materialised once and shared by every branch of the
// backward.
template <typename T>
Tensor<T> attention(Arena& a, const Tensor<T>& x, const Tensor<T>& wq, const Tensor<T>& wk,
                    const Tensor<T>& wv, const Tensor<T>& wo, const AttentionParams& p);

// Per-request key/value cache. One per GenerationSession — never shared
// between concurrent requests (see rules.md).
template <typename T>
class KVCache {
 public:
  KVCache(i64 n_layer, i64 n_head, i64 head_dim, i64 max_seq);

  T* key(i64 layer);
  T* value(i64 layer);

  i64 position() const noexcept { return pos_; }
  void advance(i64 n) noexcept { pos_ += n; }
  void reset() noexcept { pos_ = 0; }
  std::size_t bytes() const noexcept;

 private:
  i64 n_layer_, n_head_, head_dim_, max_seq_, pos_ = 0;
  std::vector<T> k_, v_;
};

}  // namespace csllm
