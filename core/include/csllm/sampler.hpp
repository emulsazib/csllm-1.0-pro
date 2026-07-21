#pragma once
//
// Decoding strategy: temperature scaling, top-k, top-p (nucleus), and a
// seeded multinomial draw.
//
// Order of operations matters and is fixed:
//   1. scale logits by 1/temperature   (temperature == 0 → deterministic argmax)
//   2. top-k   truncation  (keep the k largest logits)
//   3. softmax to probabilities
//   4. top-p   truncation  (smallest set whose cumulative mass ≥ p), renormalise
//   5. multinomial draw from the surviving distribution
//
#include <random>
#include <vector>

#include "csllm/common.hpp"

namespace csllm {

struct SamplingParams {
  f32 temperature = 1.0f;   // 0 => greedy argmax
  i32 top_k = 0;            // 0 => disabled
  f32 top_p = 1.0f;         // 1 => disabled
  f32 repetition_penalty = 1.0f;
};

class Sampler {
 public:
  explicit Sampler(u64 seed);

  // logits is [vocab_size]; mutated in place. Returns the chosen token id.
  i32 sample(f32* logits, i64 vocab_size, const SamplingParams& p);

  void reseed(u64 seed);

 private:
  std::mt19937_64 rng_;
  std::vector<i32> index_buf_;
};

}  // namespace csllm
