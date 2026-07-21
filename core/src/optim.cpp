#include "csllm/optim.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace csllm {

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

template <typename T>
AdamW<T>::AdamW(std::vector<Tensor<T>*> params, const AdamWConfig& cfg)
    : params_(std::move(params)), cfg_(cfg) {
  m_.resize(params_.size());
  v_.resize(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const auto n = static_cast<std::size_t>(params_[i]->numel());
    m_[i].assign(n, T(0));
    v_[i].assign(n, T(0));
  }
}

// Global-norm clipping: rescales every gradient by the same factor so the
// update direction is preserved. Returns the PRE-clip norm, which is the useful
// diagnostic — a spiking norm is the first sign of a diverging run.
template <typename T>
T AdamW<T>::clip_grad_norm(T max_norm) {
  T total = T(0);
  for (auto* p : params_) {
    const i64 n = p->numel();
    for (i64 i = 0; i < n; ++i) total += p->grad[i] * p->grad[i];
  }
  const T norm = std::sqrt(total);
  if (norm > max_norm && norm > T(0)) {
    const T scale = max_norm / (norm + T(1e-6));
    for (auto* p : params_) {
      const i64 n = p->numel();
      for (i64 i = 0; i < n; ++i) p->grad[i] *= scale;
    }
  }
  return norm;
}

template <typename T>
void AdamW<T>::step(T lr) {
  ++t_;
  const T bc1 = T(1) - std::pow(cfg_.beta1, static_cast<T>(t_));
  const T bc2 = T(1) - std::pow(cfg_.beta2, static_cast<T>(t_));

  for (std::size_t pi = 0; pi < params_.size(); ++pi) {
    Tensor<T>* p = params_[pi];
    const i64 n = p->numel();
    T* m = m_[pi].data();
    T* v = v_[pi].data();

    // Weight decay applies to matrices only. Decaying the 1-D RMSNorm gains
    // would pull them toward zero and suppress the signal they scale — the
    // standard exclusion in every modern LLM recipe.
    const T wd = (p->shape.ndim >= 2) ? static_cast<T>(cfg_.weight_decay) : T(0);

    for (i64 i = 0; i < n; ++i) {
      const T g = p->grad[i];
      m[i] = cfg_.beta1 * m[i] + (T(1) - cfg_.beta1) * g;
      v[i] = cfg_.beta2 * v[i] + (T(1) - cfg_.beta2) * g * g;
      const T mhat = m[i] / bc1;
      const T vhat = v[i] / bc2;
      // Decoupled decay: applied to the weight, not folded into the gradient.
      p->data[i] -= lr * (mhat / (std::sqrt(vhat) + cfg_.eps) + wd * p->data[i]);
    }
  }
}

template <typename T>
void AdamW<T>::zero_grad() {
  for (auto* p : params_) zero_<T>(p->grad, p->numel());
}

template class AdamW<f32>;
template class AdamW<f64>;

}  // namespace csllm
