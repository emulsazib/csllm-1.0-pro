#include "csllm/sampler.hpp"

namespace csllm {

Sampler::Sampler(u64 seed) : rng_(seed) {}

void Sampler::reseed(u64 seed) { rng_.seed(seed); }

// Phase 2. Fixed order (see sampler.hpp):
//   temperature scale → top-k → softmax → top-p → multinomial.
// temperature == 0 short-circuits to a deterministic argmax.
i32 Sampler::sample(f32*, i64, const SamplingParams&) { CSLLM_NOT_IMPLEMENTED(); }

}  // namespace csllm
