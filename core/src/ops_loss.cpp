#include "csllm/ops.hpp"

// Phase 2.
//   embedding     : gather rows; bwd is a scatter-add. Under weight tying the
//                   table gradient accumulates from TWO paths (input lookup and
//                   the output projection) — design.md D7. Missing the second
//                   path yields a model that trains but converges wrong, and
//                   only a gradcheck on the embedding catches it.
//   cross_entropy : fused log-softmax + NLL; dlogits = (p − onehot)/(B·T).
//                   Fused for numerical stability (max-subtraction) and to avoid
//                   materialising a second [B,T,V] tensor.
namespace csllm {

template <typename T>
Tensor<T> embedding(Arena&, const Tensor<T>&, const i32*, i64) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
T cross_entropy(Arena&, const Tensor<T>&, const i32*, Tensor<T>*) {
  CSLLM_NOT_IMPLEMENTED();
}

template Tensor<f32> embedding<f32>(Arena&, const Tensor<f32>&, const i32*, i64);
template Tensor<f64> embedding<f64>(Arena&, const Tensor<f64>&, const i32*, i64);
template f32 cross_entropy<f32>(Arena&, const Tensor<f32>&, const i32*, Tensor<f32>*);
template f64 cross_entropy<f64>(Arena&, const Tensor<f64>&, const i32*, Tensor<f64>*);

}  // namespace csllm
