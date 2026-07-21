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

// Upper bound on activation-arena bytes for a [B,T] batch. Deliberately
// generous: the arena throws on exhaustion rather than silently reallocating.
std::size_t estimate_activation_bytes(const ModelConfig& cfg, i64 B, i64 T_len,
                                      std::size_t elem_size);

// Flat dotted parameter names: "tok_emb", "blocks.{i}.attn.wq",
// "blocks.{i}.ffn.wg", "norm_f.gain", ... (see design.md → Conventions).
template <typename T>
class ParamStore {
 public:
  ParamStore(const ModelConfig& cfg, u64 seed);

  Tensor<T>& get(const std::string& name);
  const Tensor<T>& get(const std::string& name) const;

  // Insertion-ordered, so checkpoints are byte-stable across runs.
  const std::vector<std::string>& names() const noexcept { return order_; }
  std::vector<Tensor<T>*> parameter_list();

  void zero_grad();
  i64 num_params() const;

 private:
  Tensor<T>& add(const std::string& name, const Shape& shape);

  ModelConfig cfg_;
  std::unique_ptr<Arena> params_;
  std::unique_ptr<Arena> grads_;
  std::unordered_map<std::string, Tensor<T>> tensors_;
  std::vector<std::string> order_;
};

template <typename T>
class Model {
 public:
  explicit Model(const ModelConfig& cfg, u64 seed = 1337);

  static Model load(const std::string& path);
  void save(const std::string& path) const;

  // ids/targets are [B,T] token ids. Resets the activation arena, clears the
  // tape, builds the graph, and returns the mean cross-entropy loss.
  T forward_loss(const i32* ids, const i32* targets, i64 B, i64 T_len);

  // Seeds the loss gradient with 1 and walks the tape in reverse.
  void backward();

  // No-grad full-sequence forward. Returns logits [B*T, vocab_size], valid
  // until the next forward call.
  const T* forward_logits(const i32* ids, i64 B, i64 T_len);

  void zero_grad();

  const ModelConfig& config() const noexcept { return cfg_; }
  ParamStore<T>& params() noexcept { return params_; }
  const ParamStore<T>& params() const noexcept { return params_; }
  std::size_t activation_high_water() const noexcept;

 private:
  void ensure_arena(i64 B, i64 T_len);
  // Shared graph builder for forward_loss / forward_logits.
  Tensor<T> forward_hidden(const i32* ids, i64 B, i64 T_len);

  ModelConfig cfg_;
  ParamStore<T> params_;
  std::unique_ptr<Arena> activations_;
  Tensor<T> loss_;
  int loss_node_ = -1;
};

// One per HTTP request. Weights are shared read-only; only the KV cache is
// per-session, so concurrency costs ~4.5 MiB at the 12M config.
class GenerationSession {
 public:
  GenerationSession(const Model<f32>& model, u64 seed);
  ~GenerationSession();
  GenerationSession(GenerationSession&&) noexcept;
  GenerationSession& operator=(GenerationSession&&) noexcept;

  // Consumes the prompt, filling the KV cache. Returns logits for the last
  // position so the caller can sample the first generated token.
  const f32* prefill(const i32* ids, i64 n);

  // Feeds one token at the current position and returns the next-token logits.
  const f32* decode(i32 token);

  // decode() + sample in one call; returns the chosen token id.
  i32 step(i32 token, const SamplingParams& p);

  void reset();
  void reseed(u64 seed);

  i64 position() const noexcept;
  std::size_t cache_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace csllm
