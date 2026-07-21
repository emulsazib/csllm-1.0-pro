#include "csllm/model.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

#include "csllm/autograd.hpp"
#include "csllm/gemm.hpp"
#include "csllm/json.hpp"
#include "csllm/ops.hpp"
#include "csllm/serialize.hpp"

namespace csllm {
namespace {

std::string block_key(i64 layer, const char* suffix) {
  return "blocks." + std::to_string(layer) + "." + suffix;
}

}  // namespace

// ── ModelConfig ─────────────────────────────────────────────────────────────

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
  s += ",\"rope_theta\":" + json::number_to_string(rope_theta);
  s += ",\"norm_eps\":" + json::number_to_string(norm_eps);
  s += "}";
  return s;
}

ModelConfig ModelConfig::from_json(const std::string& s) {
  const json::Value v = json::parse(s);
  ModelConfig c;
  c.vocab_size = v.integer("vocab_size");
  c.n_layer = v.integer("n_layer");
  c.n_head = v.integer("n_head");
  c.n_embd = v.integer("n_embd");
  c.block_size = v.integer("block_size");
  c.ffn_hidden = v.integer("ffn_hidden");
  if (v.has("rope_theta")) c.rope_theta = static_cast<f32>(v.real("rope_theta"));
  if (v.has("norm_eps")) c.norm_eps = static_cast<f32>(v.real("norm_eps"));
  c.validate();
  return c;
}

std::size_t estimate_activation_bytes(const ModelConfig& cfg, i64 B, i64 T_len,
                                      std::size_t elem_size) {
  const i64 rows = B * T_len;
  const i64 C = cfg.n_embd, Hf = cfg.ffn_hidden, V = cfg.vocab_size, H = cfg.n_head;
  // Per block: ~26 [rows,C] buffers across norm/attention/residuals, two
  // [B,H,T,T] probability matrices (P and dP), and ~8 [rows,Hf] in the FFN.
  const i64 per_layer = 26 * rows * C + 2 * B * H * T_len * T_len + 8 * rows * Hf;
  // Embedding, final norm, logits (+grad), and the saved softmax in the loss.
  const i64 head = 8 * rows * C + 4 * rows * V;
  const i64 elems = cfg.n_layer * per_layer + head + 4096;
  // 25% margin: the arena throws on exhaustion rather than silently growing.
  return static_cast<std::size_t>(elems) * elem_size * 5 / 4;
}

// ── ParamStore ──────────────────────────────────────────────────────────────

template <typename T>
ParamStore<T>::ParamStore(const ModelConfig& cfg, u64 seed) : cfg_(cfg) {
  cfg_.validate();

  const auto bytes =
      static_cast<std::size_t>(cfg_.num_params()) * sizeof(T) + 64 * 512 /* alignment slack */;
  params_ = std::make_unique<Arena>(bytes);
  grads_ = std::make_unique<Arena>(bytes);

  const i64 C = cfg_.n_embd, Hf = cfg_.ffn_hidden, V = cfg_.vocab_size;

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 0.02);
  // GPT-2 style: scale the residual-path output projections by 1/sqrt(2L) so the
  // residual stream variance does not grow with depth.
  const double resid_scale = 1.0 / std::sqrt(2.0 * static_cast<double>(cfg_.n_layer));

  auto init_normal = [&](Tensor<T>& t, double scale) {
    const i64 n = t.numel();
    for (i64 i = 0; i < n; ++i) t.data[i] = static_cast<T>(normal(rng) * scale);
  };
  auto init_ones = [](Tensor<T>& t) {
    const i64 n = t.numel();
    for (i64 i = 0; i < n; ++i) t.data[i] = T(1);
  };

  init_normal(add("tok_emb", Shape{V, C}), 1.0);
  for (i64 l = 0; l < cfg_.n_layer; ++l) {
    init_ones(add(block_key(l, "attn_norm.gain"), Shape{C}));
    init_normal(add(block_key(l, "attn.wq"), Shape{C, C}), 1.0);
    init_normal(add(block_key(l, "attn.wk"), Shape{C, C}), 1.0);
    init_normal(add(block_key(l, "attn.wv"), Shape{C, C}), 1.0);
    init_normal(add(block_key(l, "attn.wo"), Shape{C, C}), resid_scale);
    init_ones(add(block_key(l, "ffn_norm.gain"), Shape{C}));
    init_normal(add(block_key(l, "ffn.wg"), Shape{C, Hf}), 1.0);
    init_normal(add(block_key(l, "ffn.wu"), Shape{C, Hf}), 1.0);
    init_normal(add(block_key(l, "ffn.wd"), Shape{Hf, C}), resid_scale);
  }
  init_ones(add("norm_f.gain", Shape{C}));
}

template <typename T>
Tensor<T>& ParamStore<T>::add(const std::string& name, const Shape& shape) {
  const auto n = static_cast<std::size_t>(shape.numel());
  Tensor<T> t;
  t.shape = shape;
  t.requires_grad = true;
  t.node_id = -1;  // parameters are graph leaves
  t.data = params_->alloc_n<T>(n);
  t.grad = grads_->alloc_n<T>(n);
  zero_<T>(t.grad, shape.numel());
  order_.push_back(name);
  return tensors_.emplace(name, t).first->second;
}

template <typename T>
Tensor<T>& ParamStore<T>::get(const std::string& name) {
  auto it = tensors_.find(name);
  CSLLM_CHECK(it != tensors_.end(), "unknown parameter '" + name + "'");
  return it->second;
}

template <typename T>
const Tensor<T>& ParamStore<T>::get(const std::string& name) const {
  auto it = tensors_.find(name);
  CSLLM_CHECK(it != tensors_.end(), "unknown parameter '" + name + "'");
  return it->second;
}

template <typename T>
std::vector<Tensor<T>*> ParamStore<T>::parameter_list() {
  std::vector<Tensor<T>*> out;
  out.reserve(order_.size());
  for (const auto& name : order_) out.push_back(&tensors_.at(name));
  return out;
}

template <typename T>
void ParamStore<T>::zero_grad() {
  for (const auto& name : order_) {
    auto& t = tensors_.at(name);
    zero_<T>(t.grad, t.numel());
  }
}

template <typename T>
i64 ParamStore<T>::num_params() const {
  return cfg_.num_params();
}

// ── Model ───────────────────────────────────────────────────────────────────

template <typename T>
Model<T>::Model(const ModelConfig& cfg, u64 seed) : cfg_(cfg), params_(cfg, seed) {
  cfg_.validate();
}

template <typename T>
void Model<T>::ensure_arena(i64 B, i64 T_len) {
  const std::size_t need = estimate_activation_bytes(cfg_, B, T_len, sizeof(T));
  if (!activations_ || activations_->capacity() < need) {
    activations_ = std::make_unique<Arena>(need);
  }
  activations_->reset();
}

template <typename T>
std::size_t Model<T>::activation_high_water() const noexcept {
  return activations_ ? activations_->high_water() : 0;
}

// Builds the block stack and returns the final normalised hidden state [B*T, C].
template <typename T>
Tensor<T> Model<T>::forward_hidden(const i32* ids, i64 B, i64 T_len) {
  CSLLM_CHECK(T_len <= cfg_.block_size,
              "sequence length " + std::to_string(T_len) + " exceeds block_size " +
                  std::to_string(cfg_.block_size));
  Arena& a = *activations_;
  const i64 rows = B * T_len, C = cfg_.n_embd;
  const T eps = static_cast<T>(cfg_.norm_eps);

  AttentionParams ap;
  ap.n_head = cfg_.n_head;
  ap.head_dim = cfg_.head_dim();
  ap.rope_theta = cfg_.rope_theta;

  Tensor<T> h = embedding<T>(a, params_.get("tok_emb"), ids, rows);

  for (i64 l = 0; l < cfg_.n_layer; ++l) {
    // x = x + attention(rmsnorm(x))
    Tensor<T> hn = rmsnorm<T>(a, h, params_.get(block_key(l, "attn_norm.gain")), eps);
    hn.shape = Shape{B, T_len, C};  // attention wants [B,T,C]; same storage
    Tensor<T> att = attention<T>(a, hn, params_.get(block_key(l, "attn.wq")),
                                 params_.get(block_key(l, "attn.wk")),
                                 params_.get(block_key(l, "attn.wv")),
                                 params_.get(block_key(l, "attn.wo")), ap);
    att.shape = Shape{rows, C};
    h = add<T>(a, h, att);

    // x = x + swiglu(rmsnorm(x))
    Tensor<T> hn2 = rmsnorm<T>(a, h, params_.get(block_key(l, "ffn_norm.gain")), eps);
    Tensor<T> ff = swiglu<T>(a, hn2, params_.get(block_key(l, "ffn.wg")),
                             params_.get(block_key(l, "ffn.wu")),
                             params_.get(block_key(l, "ffn.wd")));
    h = add<T>(a, h, ff);
  }

  return rmsnorm<T>(a, h, params_.get("norm_f.gain"), eps);
}

template <typename T>
T Model<T>::forward_loss(const i32* ids, const i32* targets, i64 B, i64 T_len) {
  ensure_arena(B, T_len);
  Tape<T>::active().clear();

  Tensor<T> h = forward_hidden(ids, B, T_len);
  // Weight-tied lm_head: logits = h @ tok_embᵀ. The SAME tensor also feeds the
  // embedding lookup, so its gradient accumulates from both paths (design.md D7).
  Tensor<T> logits = matmul_bt<T>(*activations_, h, params_.get("tok_emb"));
  loss_ = cross_entropy<T>(*activations_, logits, targets, B * T_len);
  loss_node_ = loss_.node_id;
  return loss_.data[0];
}

template <typename T>
void Model<T>::backward() {
  CSLLM_CHECK(loss_node_ >= 0, "backward() called before a forward pass that built a graph");
  loss_.grad[0] = T(1);  // seed dL/dL
  Tape<T>::active().backward(loss_node_);
  loss_node_ = -1;  // a tape may only be walked once
}

template <typename T>
const T* Model<T>::forward_logits(const i32* ids, i64 B, i64 T_len) {
  NoGradGuard no_grad;
  ensure_arena(B, T_len);
  Tape<T>::active().clear();
  Tensor<T> h = forward_hidden(ids, B, T_len);
  Tensor<T> logits = matmul_bt<T>(*activations_, h, params_.get("tok_emb"));
  return logits.data;
}

template <typename T>
void Model<T>::zero_grad() {
  params_.zero_grad();
}

template <typename T>
void Model<T>::save(const std::string& path) const {
  std::vector<TensorEntry> entries;
  std::vector<const T*> sources;
  u64 offset = 0;
  for (const auto& name : params_.names()) {
    const Tensor<T>& t = params_.get(name);
    TensorEntry e;
    e.name = name;
    e.dtype = sizeof(T) == 4 ? "f32" : "f64";
    for (int i = 0; i < t.shape.ndim; ++i) e.shape.push_back(t.shape[i]);
    e.offset = offset;
    entries.push_back(std::move(e));
    sources.push_back(t.data);
    offset += static_cast<u64>(t.numel()) * sizeof(T);
  }

  std::vector<std::byte> payload(offset);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const Tensor<T>& t = params_.get(entries[i].name);
    std::memcpy(payload.data() + entries[i].offset, sources[i],
                static_cast<std::size_t>(t.numel()) * sizeof(T));
  }
  save_checkpoint(path, cfg_, entries, payload.data(), payload.size());
}

template <typename T>
Model<T> Model<T>::load(const std::string& path) {
  MappedCheckpoint ckpt(path);
  Model<T> model(ckpt.header().config);
  for (const auto& entry : ckpt.header().tensors) {
    Tensor<T>& t = model.params_.get(entry.name);
    i64 expect = 1;
    for (i64 d : entry.shape) expect *= d;
    CSLLM_CHECK(expect == t.numel(),
                "checkpoint tensor '" + entry.name + "' has " + std::to_string(expect) +
                    " elements but the model expects " + std::to_string(t.numel()));
    CSLLM_CHECK(entry.dtype == (sizeof(T) == 4 ? "f32" : "f64"),
                "checkpoint tensor '" + entry.name + "' has dtype " + entry.dtype);
    std::memcpy(t.data, ckpt.tensor_data(entry.name),
                static_cast<std::size_t>(t.numel()) * sizeof(T));
  }
  return model;
}

// ── GenerationSession ───────────────────────────────────────────────────────
//
// Incremental decoding. Each session owns a private KV cache; weights are
// shared read-only, so sessions never interfere and concurrency is cheap.

struct GenerationSession::Impl {
  const Model<f32>& model;
  KVCache<f32> cache;
  Sampler sampler;
  Arena scratch;
  std::vector<f32> logits;
  i64 pos = 0;

  Impl(const Model<f32>& m, u64 seed)
      : model(m),
        cache(m.config().n_layer, m.config().n_head, m.config().head_dim(),
              m.config().block_size),
        sampler(seed),
        // One token's worth of activations, generously sized.
        scratch(static_cast<std::size_t>(64 * (m.config().n_embd + m.config().ffn_hidden) +
                                         16 * m.config().block_size) *
                    sizeof(f32) +
                (1u << 16)),
        logits(static_cast<std::size_t>(m.config().vocab_size)) {}

  // Runs one token through the stack at position `pos`, updating the cache.
  const f32* forward_token(i32 token);
};

const f32* GenerationSession::Impl::forward_token(i32 token) {
  const ModelConfig& cfg = model.config();
  const i64 C = cfg.n_embd, H = cfg.n_head, dh = cfg.head_dim(), Hf = cfg.ffn_hidden;
  const i64 V = cfg.vocab_size, maxT = cfg.block_size;
  const f32 eps = cfg.norm_eps;
  const f32 scale = 1.0f / std::sqrt(static_cast<f32>(dh));
  auto& P = const_cast<ParamStore<f32>&>(model.params());

  CSLLM_CHECK(pos < maxT, "generation exceeded block_size (" + std::to_string(maxT) + ")");
  CSLLM_CHECK(token >= 0 && token < V, "token id out of range");

  scratch.reset();
  f32* h = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* hn = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* q = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* kv = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* ctx = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* att = scratch.alloc_n<f32>(static_cast<std::size_t>(C));
  f32* sc = scratch.alloc_n<f32>(static_cast<std::size_t>(maxT));
  f32* gate = scratch.alloc_n<f32>(static_cast<std::size_t>(Hf));
  f32* up = scratch.alloc_n<f32>(static_cast<std::size_t>(Hf));

  auto rmsnorm_row = [&](const f32* x, const f32* gain, f32* out) {
    f32 ss = 0.0f;
    for (i64 j = 0; j < C; ++j) ss += x[j] * x[j];
    const f32 inv = 1.0f / std::sqrt(ss / static_cast<f32>(C) + eps);
    for (i64 j = 0; j < C; ++j) out[j] = gain[j] * x[j] * inv;
  };

  const Tensor<f32>& emb = P.get("tok_emb");
  std::memcpy(h, emb.data + static_cast<i64>(token) * C, static_cast<std::size_t>(C) * sizeof(f32));

  for (i64 l = 0; l < cfg.n_layer; ++l) {
    rmsnorm_row(h, P.get(block_key(l, "attn_norm.gain")).data, hn);

    f32* kcache = cache.key(l);
    f32* vcache = cache.value(l);
    // Cache layout mirrors training: [H, maxT, dh] within each layer.
    auto cache_at = [&](f32* base, i64 head, i64 t) { return base + (head * maxT + t) * dh; };

    gemm<f32>(false, false, 1, C, C, 1.0f, hn, C, P.get(block_key(l, "attn.wq")).data, C, 0.0f, q,
              C);
    gemm<f32>(false, false, 1, C, C, 1.0f, hn, C, P.get(block_key(l, "attn.wk")).data, C, 0.0f, kv,
              C);
    // RoPE this position's Q and K before the K goes into the cache — cached
    // keys are stored already-rotated, exactly as training produces them.
    rope_apply<f32>(q, 1, H, 1, dh, pos, cfg.rope_theta, false);
    rope_apply<f32>(kv, 1, H, 1, dh, pos, cfg.rope_theta, false);
    for (i64 hd = 0; hd < H; ++hd) {
      std::memcpy(cache_at(kcache, hd, pos), kv + hd * dh,
                  static_cast<std::size_t>(dh) * sizeof(f32));
    }

    gemm<f32>(false, false, 1, C, C, 1.0f, hn, C, P.get(block_key(l, "attn.wv")).data, C, 0.0f, kv,
              C);
    for (i64 hd = 0; hd < H; ++hd) {
      std::memcpy(cache_at(vcache, hd, pos), kv + hd * dh,
                  static_cast<std::size_t>(dh) * sizeof(f32));
    }

    for (i64 hd = 0; hd < H; ++hd) {
      const f32* qh = q + hd * dh;
      f32 maxv = -std::numeric_limits<f32>::infinity();
      for (i64 t = 0; t <= pos; ++t) {
        const f32* kt = cache_at(kcache, hd, t);
        f32 dot = 0.0f;
        for (i64 d = 0; d < dh; ++d) dot += qh[d] * kt[d];
        sc[t] = dot * scale;
        maxv = std::max(maxv, sc[t]);
      }
      f32 sum = 0.0f;
      for (i64 t = 0; t <= pos; ++t) {
        sc[t] = std::exp(sc[t] - maxv);
        sum += sc[t];
      }
      const f32 inv = 1.0f / sum;
      f32* out = ctx + hd * dh;
      for (i64 d = 0; d < dh; ++d) out[d] = 0.0f;
      for (i64 t = 0; t <= pos; ++t) {
        const f32 w = sc[t] * inv;
        const f32* vt = cache_at(vcache, hd, t);
        for (i64 d = 0; d < dh; ++d) out[d] += w * vt[d];
      }
    }

    gemm<f32>(false, false, 1, C, C, 1.0f, ctx, C, P.get(block_key(l, "attn.wo")).data, C, 0.0f,
              att, C);
    for (i64 j = 0; j < C; ++j) h[j] += att[j];

    rmsnorm_row(h, P.get(block_key(l, "ffn_norm.gain")).data, hn);
    gemm<f32>(false, false, 1, Hf, C, 1.0f, hn, C, P.get(block_key(l, "ffn.wg")).data, Hf, 0.0f,
              gate, Hf);
    gemm<f32>(false, false, 1, Hf, C, 1.0f, hn, C, P.get(block_key(l, "ffn.wu")).data, Hf, 0.0f, up,
              Hf);
    for (i64 j = 0; j < Hf; ++j) {
      const f32 s = 1.0f / (1.0f + std::exp(-gate[j]));
      gate[j] = gate[j] * s * up[j];
    }
    gemm<f32>(false, false, 1, C, Hf, 1.0f, gate, Hf, P.get(block_key(l, "ffn.wd")).data, C, 0.0f,
              att, C);
    for (i64 j = 0; j < C; ++j) h[j] += att[j];
  }

  rmsnorm_row(h, P.get("norm_f.gain").data, hn);
  // Tied lm_head: logits = hn @ tok_embᵀ
  gemm<f32>(false, true, 1, V, C, 1.0f, hn, C, emb.data, C, 0.0f, logits.data(), V);

  ++pos;
  cache.advance(1);
  return logits.data();
}

GenerationSession::GenerationSession(const Model<f32>& model, u64 seed)
    : impl_(new Impl(model, seed)) {}
GenerationSession::~GenerationSession() = default;
GenerationSession::GenerationSession(GenerationSession&&) noexcept = default;
GenerationSession& GenerationSession::operator=(GenerationSession&&) noexcept = default;

const f32* GenerationSession::prefill(const i32* ids, i64 n) {
  CSLLM_CHECK(n > 0, "prefill needs at least one token");
  const f32* out = nullptr;
  for (i64 i = 0; i < n; ++i) out = impl_->forward_token(ids[i]);
  return out;
}

const f32* GenerationSession::decode(i32 token) { return impl_->forward_token(token); }

i32 GenerationSession::step(i32 token, const SamplingParams& p) {
  const f32* logits = impl_->forward_token(token);
  return impl_->sampler.sample(const_cast<f32*>(logits), impl_->model.config().vocab_size, p);
}

void GenerationSession::reset() {
  impl_->pos = 0;
  impl_->cache.reset();
}

void GenerationSession::reseed(u64 seed) { impl_->sampler.reseed(seed); }
i64 GenerationSession::position() const noexcept { return impl_->pos; }
std::size_t GenerationSession::cache_bytes() const noexcept { return impl_->cache.bytes(); }

template class ParamStore<f32>;
template class ParamStore<f64>;
template class Model<f32>;
template class Model<f64>;

}  // namespace csllm
