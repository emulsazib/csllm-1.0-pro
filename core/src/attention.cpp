#include "csllm/attention.hpp"

namespace csllm {

// ── KVCache ─────────────────────────────────────────────────────────────────
// Implemented now: sizing is needed to reason about gateway concurrency cost.
template <typename T>
KVCache<T>::KVCache(i64 n_layer, i64 n_head, i64 head_dim, i64 max_seq)
    : n_layer_(n_layer), n_head_(n_head), head_dim_(head_dim), max_seq_(max_seq) {
  const std::size_t per_layer = static_cast<std::size_t>(n_head * max_seq * head_dim);
  k_.assign(per_layer * static_cast<std::size_t>(n_layer), T(0));
  v_.assign(per_layer * static_cast<std::size_t>(n_layer), T(0));
}

template <typename T>
T* KVCache<T>::key(i64 layer) {
  CSLLM_CHECK(layer >= 0 && layer < n_layer_, "KVCache: layer out of range");
  return k_.data() + static_cast<std::size_t>(layer * n_head_ * max_seq_ * head_dim_);
}

template <typename T>
T* KVCache<T>::value(i64 layer) {
  CSLLM_CHECK(layer >= 0 && layer < n_layer_, "KVCache: layer out of range");
  return v_.data() + static_cast<std::size_t>(layer * n_head_ * max_seq_ * head_dim_);
}

template <typename T>
std::size_t KVCache<T>::bytes() const noexcept {
  return (k_.size() + v_.size()) * sizeof(T);
}

// ── Attention ───────────────────────────────────────────────────────────────
// Phase 2.  fwd: S = QKᵀ/√dₕ (causally masked), P = softmax(S), O = PV
//           bwd: dV = Pᵀ·dO; dP = dO·Vᵀ; dS = softmax_bwd(P,dP) then RE-MASK;
//                dQ = dS·K/√dₕ; dK = dSᵀ·Q/√dₕ
// The re-mask is not optional — skipping it leaks gradient into positions the
// forward pass masked out.
template <typename T>
Tensor<T> attention(Arena&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&,
                    const Tensor<T>&, const AttentionParams&) {
  CSLLM_NOT_IMPLEMENTED();
}

template class KVCache<f32>;
template class KVCache<f64>;
template Tensor<f32> attention<f32>(Arena&, const Tensor<f32>&, const Tensor<f32>&,
                                    const Tensor<f32>&, const Tensor<f32>&, const Tensor<f32>&,
                                    const AttentionParams&);
template Tensor<f64> attention<f64>(Arena&, const Tensor<f64>&, const Tensor<f64>&,
                                    const Tensor<f64>&, const Tensor<f64>&, const Tensor<f64>&,
                                    const AttentionParams&);

}  // namespace csllm
