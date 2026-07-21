#include "csllm/model.hpp"

#include <string>

namespace csllm {

// ── ModelConfig ─────────────────────────────────────────────────────────────
// Implemented now so configs can be validated and sized before any math exists.

void ModelConfig::validate() const {
  CSLLM_CHECK(vocab_size > 0, "vocab_size must be positive");
  CSLLM_CHECK(n_layer > 0, "n_layer must be positive");
  CSLLM_CHECK(n_head > 0, "n_head must be positive");
  CSLLM_CHECK(n_embd > 0, "n_embd must be positive");
  CSLLM_CHECK(block_size > 0, "block_size must be positive");
  CSLLM_CHECK(ffn_hidden > 0, "ffn_hidden must be positive");
  CSLLM_CHECK(n_embd % n_head == 0,
              "n_embd (" + std::to_string(n_embd) + ") must be divisible by n_head (" +
                  std::to_string(n_head) + ")");
  // RoPE rotates adjacent channel pairs, so head_dim must be even.
  CSLLM_CHECK(head_dim() % 2 == 0,
              "head_dim (" + std::to_string(head_dim()) + ") must be even for RoPE pairing");
  CSLLM_CHECK(norm_eps > 0.0f, "norm_eps must be positive");
  CSLLM_CHECK(rope_theta > 0.0f, "rope_theta must be positive");
}

i64 ModelConfig::num_params() const {
  // tok_emb is weight-tied with lm_head, so it is counted exactly once.
  const i64 embed = vocab_size * n_embd;
  const i64 attn = 4 * n_embd * n_embd;              // wq, wk, wv, wo
  const i64 ffn = 3 * n_embd * ffn_hidden;           // wg, wu (up) + wd (down)
  const i64 norms = 2 * n_embd;                      // two RMSNorm gains per block
  return embed + n_layer * (attn + ffn + norms) + n_embd /* final norm */;
}

std::string ModelConfig::to_json() const {
  std::string s = "{";
  s += "\"vocab_size\":" + std::to_string(vocab_size);
  s += ",\"n_layer\":" + std::to_string(n_layer);
  s += ",\"n_head\":" + std::to_string(n_head);
  s += ",\"n_embd\":" + std::to_string(n_embd);
  s += ",\"block_size\":" + std::to_string(block_size);
  s += ",\"ffn_hidden\":" + std::to_string(ffn_hidden);
  s += ",\"rope_theta\":" + std::to_string(rope_theta);
  s += ",\"norm_eps\":" + std::to_string(norm_eps);
  s += "}";
  return s;
}

ModelConfig ModelConfig::from_json(const std::string&) {
  // Phase 2 — arrives with the checkpoint reader. Until then Python parses
  // configs/*.json and sets the fields directly across the binding.
  CSLLM_NOT_IMPLEMENTED();
}

// ── ParamStore / Model / GenerationSession ──────────────────────────────────
// Phase 2.

template <typename T>
ParamStore<T>::ParamStore(const ModelConfig& cfg) : cfg_(cfg) {
  cfg_.validate();
}

template <typename T>
Tensor<T>& ParamStore<T>::get(const std::string&) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
const Tensor<T>& ParamStore<T>::get(const std::string&) const {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
std::vector<std::string> ParamStore<T>::names() const {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void ParamStore<T>::zero_grad() {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
i64 ParamStore<T>::num_params() const {
  return cfg_.num_params();
}

template <typename T>
Model<T>::Model(const ModelConfig& cfg) : cfg_(cfg), params_(cfg) {
  cfg_.validate();
}

template <typename T>
Model<T> Model<T>::load(const std::string&) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void Model<T>::save(const std::string&) const {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
T Model<T>::forward_loss(const i32*, const i32*, i64, i64) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
const T* Model<T>::forward_logits(const i32*, i64, i64) {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void Model<T>::backward() {
  CSLLM_NOT_IMPLEMENTED();
}

template <typename T>
void Model<T>::zero_grad() {
  CSLLM_NOT_IMPLEMENTED();
}

struct GenerationSession::Impl {
  u64 seed = 0;
};

GenerationSession::GenerationSession(const Model<f32>&, u64 seed) : impl_(new Impl{seed}) {}
GenerationSession::~GenerationSession() = default;
GenerationSession::GenerationSession(GenerationSession&&) noexcept = default;
GenerationSession& GenerationSession::operator=(GenerationSession&&) noexcept = default;

void GenerationSession::prefill(const i32*, i64) { CSLLM_NOT_IMPLEMENTED(); }
i32 GenerationSession::step(const SamplingParams&) { CSLLM_NOT_IMPLEMENTED(); }
void GenerationSession::reset() { CSLLM_NOT_IMPLEMENTED(); }
i64 GenerationSession::position() const noexcept { return 0; }
std::size_t GenerationSession::cache_bytes() const noexcept { return 0; }

template class ParamStore<f32>;
template class ParamStore<f64>;
template class Model<f32>;
template class Model<f64>;

}  // namespace csllm
