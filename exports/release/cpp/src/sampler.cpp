#include "csllm/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace csllm {

Sampler::Sampler(u64 seed) : rng_(seed) {}

void Sampler::reseed(u64 seed) { rng_.seed(seed); }

// Fixed order (see sampler.hpp):
//   temperature scale -> top-k -> softmax -> top-p -> renormalise.
// top-k truncates on LOGITS and top-p on PROBABILITIES, which is why the
// softmax sits between them.
void Sampler::distribution(const f32* logits, i64 vocab_size, const SamplingParams& p, f32* out) {
  CSLLM_CHECK(vocab_size > 0, "sampler: empty vocabulary");
  std::memset(out, 0, static_cast<std::size_t>(vocab_size) * sizeof(f32));

  // Greedy: temperature 0 means "no randomness", so short-circuit rather than
  // dividing by zero. The distribution is a one-hot on the argmax.
  if (p.temperature <= 0.0f) {
    const auto best = static_cast<i64>(std::max_element(logits, logits + vocab_size) - logits);
    out[best] = 1.0f;
    return;
  }

  const f32 inv_temp = 1.0f / p.temperature;

  index_buf_.resize(static_cast<std::size_t>(vocab_size));
  std::iota(index_buf_.begin(), index_buf_.end(), 0);

  // ── top-k (on temperature-scaled logits; scaling is monotonic so the ranking
  // is the same, but we keep the scale for the softmax below) ──
  if (p.top_k > 0 && p.top_k < vocab_size) {
    const auto kept = static_cast<std::size_t>(p.top_k);
    std::nth_element(index_buf_.begin(), index_buf_.begin() + kept, index_buf_.end(),
                     [&](i32 a, i32 b) { return logits[a] > logits[b]; });
    index_buf_.resize(kept);
  }

  // ── softmax over the surviving candidates ──
  f32 maxv = -std::numeric_limits<f32>::infinity();
  for (i32 idx : index_buf_) maxv = std::max(maxv, logits[idx] * inv_temp);

  prob_buf_.resize(index_buf_.size());
  f32 sum = 0.0f;
  for (std::size_t i = 0; i < index_buf_.size(); ++i) {
    prob_buf_[i] = std::exp(logits[index_buf_[i]] * inv_temp - maxv);
    sum += prob_buf_[i];
  }
  for (auto& v : prob_buf_) v /= sum;

  // ── top-p (nucleus): smallest prefix whose mass reaches p, renormalised ──
  std::vector<std::size_t> order(prob_buf_.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return prob_buf_[a] > prob_buf_[b]; });

  std::size_t cutoff = order.size();
  if (p.top_p < 1.0f) {
    f32 cum = 0.0f;
    for (std::size_t i = 0; i < order.size(); ++i) {
      cum += prob_buf_[order[i]];
      if (cum >= p.top_p) {
        cutoff = i + 1;  // always keep the token that crosses the threshold
        break;
      }
    }
  }

  f32 mass = 0.0f;
  for (std::size_t i = 0; i < cutoff; ++i) mass += prob_buf_[order[i]];
  const f32 renorm = (mass > 0.0f) ? 1.0f / mass : 0.0f;
  for (std::size_t i = 0; i < cutoff; ++i) {
    out[index_buf_[order[i]]] = prob_buf_[order[i]] * renorm;
  }
}

i32 Sampler::sample(const f32* logits, i64 vocab_size, const SamplingParams& p) {
  // Greedy needs no distribution or RNG draw.
  if (p.temperature <= 0.0f) {
    return static_cast<i32>(std::max_element(logits, logits + vocab_size) - logits);
  }

  dist_buf_.resize(static_cast<std::size_t>(vocab_size));
  distribution(logits, vocab_size, p, dist_buf_.data());

  std::uniform_real_distribution<f32> uniform(0.0f, 1.0f);
  const f32 target = uniform(rng_);
  f32 cum = 0.0f;
  i64 last_nonzero = 0;
  for (i64 i = 0; i < vocab_size; ++i) {
    if (dist_buf_[static_cast<std::size_t>(i)] <= 0.0f) continue;
    last_nonzero = i;
    cum += dist_buf_[static_cast<std::size_t>(i)];
    if (cum >= target) return static_cast<i32>(i);
  }
  // Only reachable through floating-point round-off.
  return static_cast<i32>(last_nonzero);
}

}  // namespace csllm
