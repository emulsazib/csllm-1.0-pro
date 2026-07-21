#include "csllm/optim.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace csllm {

// Implemented now — pure arithmetic, no tensors needed, and the training script
// wants to plot the schedule before the model exists.
f32 cosine_lr(i64 step, i64 warmup_steps, i64 max_steps, f32 lr_max, f32 lr_min) {
  if (step < warmup_steps && warmup_steps > 0) {
    return lr_max * static_cast<f32>(step + 1) / static_cast<f32>(warmup_steps);
  }
  if (step >= max_steps) return lr_min;
  const f32 progress = static_cast<f32>(step - warmup_steps) /
                       static_cast<f32>(std::max<i64>(1, max_steps - warmup_steps));
  const f32 coeff = 0.5f * (1.0f + std::cos(3.14159265358979323846f * progress));
  return lr_min + coeff * (lr_max - lr_min);
}

// Phase 2 — AdamW with decoupled weight decay + global-norm clipping.
template <typename T>
AdamW<T>::AdamW(std::vector<Tensor<T>*> params, const AdamWConfig& cfg)
    : params_(std::move(params)), cfg_(cfg) {}

template <typename T>
T AdamW<T>::clip_grad_norm(T) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void AdamW<T>::step(T) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void AdamW<T>::zero_grad() {
  CSLLM_NOT_IMPLEMENTED();
}

template class AdamW<f32>;
template class AdamW<f64>;

}  // namespace csllm
