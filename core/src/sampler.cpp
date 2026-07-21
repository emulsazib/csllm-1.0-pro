#include "csllm/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace csllm {

Sampler::Sampler(u64 seed) : rng_(seed) {}

void Sampler::reseed(u64 seed) { rng_.seed(seed); }

// Fixed order (see sampler.hpp):
//   temperature scale → top-k → softmax → top-p → multinomial.
// top-k truncates on LOGITS and top-p on PROBABILITIES, which is why the
// softmax sits between them.
i32 Sampler::sample(f32* logits, i64 vocab_size, const SamplingParams& p) {
  CSLLM_CHECK(vocab_size > 0, "sampler: empty vocabulary");

  // Greedy: temperature 0 means "no randomness", so short-circuit rather than
  // dividing by zero.
  if (p.temperature <= 0.0f) {
    return static_cast<i32>(std::max_element(logits, logits + vocab_size) - logits);
  }

  const f32 inv_temp = 1.0f / p.temperature;
  for (i64 i = 0; i < vocab_size; ++i) logits[i] *= inv_temp;

  index_buf_.resize(static_cast<std::size_t>(vocab_size));
  std::iota(index_buf_.begin(), index_buf_.end(), 0);

  // ── top-k ──
  i64 kept = vocab_size;
  if (p.top_k > 0 && p.top_k < vocab_size) {
    kept = p.top_k;
    std::nth_element(index_buf_.begin(), index_buf_.begin() + kept, index_buf_.end(),
                     [&](i32 a, i32 b) { return logits[a] > logits[b]; });
    index_buf_.resize(static_cast<std::size_t>(kept));
  }

  // ── softmax over the surviving candidates ──
  f32 maxv = -std::numeric_limits<f32>::infinity();
  for (i32 idx : index_buf_) maxv = std::max(maxv, logits[idx]);
  f32 sum = 0.0f;
  std::vector<f32> probs(index_buf_.size());
  for (std::size_t i = 0; i < index_buf_.size(); ++i) {
    probs[i] = std::exp(logits[index_buf_[i]] - maxv);
    sum += probs[i];
  }
  for (auto& v : probs) v /= sum;

  // ── top-p (nucleus) ──
  // Sort descending, keep the smallest prefix whose mass reaches p, renormalise.
  std::vector<std::size_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });

  std::size_t cutoff = order.size();
  if (p.top_p < 1.0f) {
    f32 cum = 0.0f;
    for (std::size_t i = 0; i < order.size(); ++i) {
      cum += probs[order[i]];
      if (cum >= p.top_p) {
        cutoff = i + 1;  // always keep the token that crosses the threshold
        break;
      }
    }
  }

  f32 mass = 0.0f;
  for (std::size_t i = 0; i < cutoff; ++i) mass += probs[order[i]];

  // ── multinomial draw ──
  std::uniform_real_distribution<f32> uniform(0.0f, 1.0f);
  const f32 target = uniform(rng_) * mass;
  f32 cum = 0.0f;
  for (std::size_t i = 0; i < cutoff; ++i) {
    cum += probs[order[i]];
    if (cum >= target) return index_buf_[order[i]];
  }
  // Only reachable through floating-point round-off.
  return index_buf_[order[cutoff - 1]];
}

}  // namespace csllm
