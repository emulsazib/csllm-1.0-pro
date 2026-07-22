#pragma once
//
// AdamW with decoupled weight decay, plus global-norm gradient clipping.
//
//   m ← β₁m + (1−β₁)g
//   v ← β₂v + (1−β₂)g²
//   m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ)
//   θ ← θ − lr·(m̂/(√v̂+ε) + λθ)      ← decay decoupled from the gradient
//
#include <string>
#include <vector>

#include "csllm/common.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

struct AdamWConfig {
  f32 lr = 3e-4f;
  f32 beta1 = 0.9f;
  f32 beta2 = 0.95f;
  f32 eps = 1e-8f;
  f32 weight_decay = 0.1f;
};

template <typename T>
class AdamW {
 public:
  AdamW(std::vector<Tensor<T>*> params, const AdamWConfig& cfg);

  // Returns the pre-clip global gradient norm (useful for monitoring spikes).
  T clip_grad_norm(T max_norm);

  void step(T lr);
  void zero_grad();

  i64 step_count() const noexcept { return t_; }

 private:
  std::vector<Tensor<T>*> params_;
  std::vector<std::vector<T>> m_, v_;
  AdamWConfig cfg_;
  i64 t_ = 0;
};

// Cosine decay with linear warmup — the standard LLM schedule.
f32 cosine_lr(i64 step, i64 warmup_steps, i64 max_steps, f32 lr_max, f32 lr_min);

}  // namespace csllm
