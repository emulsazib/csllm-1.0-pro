#pragma once
//
// The model: config, parameter storage, transformer blocks, and the
// per-request generation session.
//
// Block (pre-norm):
//     x = x + attention(rmsnorm(x))      // RoPE applied to Q/K inside attention
//     x = x + swiglu(rmsnorm(x))
//
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "csllm/arena.hpp"
#include "csllm/attention.hpp"
#include "csllm/common.hpp"
#include "csllm/sampler.hpp"
#include "csllm/tensor.hpp"

namespace csllm {

struct ModelConfig {
  i64 vocab_size = 4096;
  i64 n_layer = 6;
  i64 n_head = 6;
  i64 n_embd = 384;
  i64 block_size = 256;
  i64 ffn_hidden = 1024;   // ≈ ⅔·4·n_embd, rounded to a multiple of 64
  f32 rope_theta = 10000.0f;
  f32 norm_eps = 1e-5f;

  i64 head_dim() const { return n_embd / n_head; }
  i64 num_params() const;
  void validate() const;   // throws if head_dim is odd, dims mismatch, etc.

  std::string to_json() const;
  static ModelConfig from_json(const std::string& s);
};

// Flat dotted parameter names: "tok_emb", "blocks.{i}.attn.wq",
// "blocks.{i}.ffn.wg", "norm_f.gain", ... (see design.md → Conventions).
template <typename T>
class ParamStore {
 public:
  explicit ParamStore(const ModelConfig& cfg);

  Tensor<T>& get(const std::string& name);
  const Tensor<T>& get(const std::string& name) const;
  std::vector<std::string> names() const;
  void zero_grad();
  i64 num_params() const;

 private:
  ModelConfig cfg_;
  std::unique_ptr<Arena> params_;
  std::unique_ptr<Arena> grads_;
  std::unordered_map<std::string, Tensor<T>> tensors_;
};

template <typename T>
class Model {
 public:
  explicit Model(const ModelConfig& cfg);

  static Model load(const std::string& path);
  void save(const std::string& path) const;

  // ids/targets are [B,T]. Returns mean cross-entropy loss and builds the tape.
  T forward_loss(const i32* ids, const i32* targets, i64 B, i64 T_);

  // Inference-only forward for the last position; used by GenerationSession.
  const T* forward_logits(const i32* ids, i64 B, i64 T_);

  void backward();
  void zero_grad();

  const ModelConfig& config() const noexcept { return cfg_; }
  ParamStore<T>& params() noexcept { return params_; }

 private:
  ModelConfig cfg_;
  ParamStore<T> params_;
  std::unique_ptr<Arena> activations_;
};

// One per HTTP request. Weights are shared read-only; only the KV cache is
// per-session, so concurrency costs ~4.5 MiB at the 12M config.
class GenerationSession {
 public:
  GenerationSession(const Model<f32>& model, u64 seed);
  // Declared (not defaulted) because Impl is incomplete here — the definition
  // lives in model.cpp where Impl is complete.
  ~GenerationSession();
  GenerationSession(GenerationSession&&) noexcept;
  GenerationSession& operator=(GenerationSession&&) noexcept;

  void prefill(const i32* ids, i64 n);
  i32 step(const SamplingParams& p);   // returns the next token id
  void reset();

  i64 position() const noexcept;
  std::size_t cache_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace csllm
