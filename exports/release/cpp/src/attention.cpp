#include "csllm/attention.hpp"

#include <algorithm>
#include <cmath>

#include "csllm/gemm.hpp"
#include "csllm/ops.hpp"
#include "csllm/threadpool.hpp"

namespace csllm {
namespace {

// [B,T,H,dh] (the natural layout of a [B*T, C] projection) -> [B,H,T,dh]
template <typename T>
void pack_heads(const T* src, T* dst, i64 B, i64 T_len, i64 H, i64 dh) {
  for (i64 b = 0; b < B; ++b) {
    for (i64 t = 0; t < T_len; ++t) {
      for (i64 h = 0; h < H; ++h) {
        const T* s = src + ((b * T_len + t) * H + h) * dh;
        T* d = dst + (((b * H + h) * T_len) + t) * dh;
        for (i64 i = 0; i < dh; ++i) d[i] = s[i];
      }
    }
  }
}

// [B,H,T,dh] -> [B,T,H,dh];  accumulate=true does += instead of =.
template <typename T>
void unpack_heads(const T* src, T* dst, i64 B, i64 T_len, i64 H, i64 dh, bool accumulate) {
  for (i64 b = 0; b < B; ++b) {
    for (i64 h = 0; h < H; ++h) {
      for (i64 t = 0; t < T_len; ++t) {
        const T* s = src + (((b * H + h) * T_len) + t) * dh;
        T* d = dst + ((b * T_len + t) * H + h) * dh;
        if (accumulate) {
          for (i64 i = 0; i < dh; ++i) d[i] += s[i];
        } else {
          for (i64 i = 0; i < dh; ++i) d[i] = s[i];
        }
      }
    }
  }
}

}  // namespace

// ── KVCache ─────────────────────────────────────────────────────────────────
template <typename T>
KVCache<T>::KVCache(i64 n_layer, i64 n_head, i64 head_dim, i64 max_seq)
    : n_layer_(n_layer), n_head_(n_head), head_dim_(head_dim), max_seq_(max_seq) {
  const std::size_t per_layer = static_cast<std::size_t>(n_head * max_seq * head_dim);
  k_.assign(per_layer * static_cast<std::size_t>(n_layer), T(0));
  v_.assign(per_layer * static_cast<std::size_t>(n_layer), T(0));
}

template <typename T>
T* KVCache<T>::key(i64 layer) {
  CSLLM_CHECK(layer >= 0 && layer < n_layer_, "KVCache: layer out of range");
  return k_.data() + static_cast<std::size_t>(layer * n_head_ * max_seq_ * head_dim_);
}

template <typename T>
T* KVCache<T>::value(i64 layer) {
  CSLLM_CHECK(layer >= 0 && layer < n_layer_, "KVCache: layer out of range");
  return v_.data() + static_cast<std::size_t>(layer * n_head_ * max_seq_ * head_dim_);
}

template <typename T>
std::size_t KVCache<T>::bytes() const noexcept {
  return (k_.size() + v_.size()) * sizeof(T);
}

// ── attention ───────────────────────────────────────────────────────────────
template <typename T>
Tensor<T> attention(Arena& arena, const Tensor<T>& x, const Tensor<T>& wq, const Tensor<T>& wk,
                    const Tensor<T>& wv, const Tensor<T>& wo, const AttentionParams& p) {
  CSLLM_CHECK(x.shape.ndim == 3, "attention expects x[B,T,C]");
  const i64 B = x.shape[0], T_len = x.shape[1], C = x.shape[2];
  const i64 H = p.n_head, dh = p.head_dim;
  CSLLM_CHECK(H * dh == C, "attention: n_head*head_dim must equal C");
  CSLLM_CHECK(dh % 2 == 0, "attention: head_dim must be even for RoPE");

  const i64 rows = B * T_len;
  const auto nbt = static_cast<std::size_t>(rows * C);
  const auto nprob = static_cast<std::size_t>(B * H * T_len * T_len);
  const T scale = T(1) / std::sqrt(static_cast<T>(dh));
  const T theta = static_cast<T>(p.rope_theta);

  // Projections, then head-major layout.
  T* qf = arena.alloc_n<T>(nbt);
  T* kf = arena.alloc_n<T>(nbt);
  T* vf = arena.alloc_n<T>(nbt);
  gemm<T>(false, false, rows, C, C, T(1), x.data, C, wq.data, C, T(0), qf, C);
  gemm<T>(false, false, rows, C, C, T(1), x.data, C, wk.data, C, T(0), kf, C);
  gemm<T>(false, false, rows, C, C, T(1), x.data, C, wv.data, C, T(0), vf, C);

  T* q = arena.alloc_n<T>(nbt);
  T* k = arena.alloc_n<T>(nbt);
  T* v = arena.alloc_n<T>(nbt);
  pack_heads<T>(qf, q, B, T_len, H, dh);
  pack_heads<T>(kf, k, B, T_len, H, dh);
  pack_heads<T>(vf, v, B, T_len, H, dh);

  // RoPE rotates Q and K only — V carries no positional information.
  rope_apply<T>(q, B, H, T_len, dh, p.pos_offset, theta, /*inverse=*/false);
  rope_apply<T>(k, B, H, T_len, dh, p.pos_offset, theta, /*inverse=*/false);

  T* prob = arena.alloc_n<T>(nprob);   // saved for backward
  T* ctx = arena.alloc_n<T>(nbt);

  ThreadPool::global().parallel_for(static_cast<std::size_t>(B * H), [&](std::size_t idx) {
    const i64 bh = static_cast<i64>(idx);
    const T* qh = q + bh * T_len * dh;
    const T* kh = k + bh * T_len * dh;
    const T* vh = v + bh * T_len * dh;
    T* ph = prob + bh * T_len * T_len;
    T* oh = ctx + bh * T_len * dh;

    // S = Q·Kᵀ·scale
    gemm<T>(false, true, T_len, T_len, dh, scale, qh, dh, kh, dh, T(0), ph, T_len);

    // Causal softmax, in place.
    for (i64 i = 0; i < T_len; ++i) {
      T* row = ph + i * T_len;
      T maxv = row[0];
      for (i64 j = 1; j <= i; ++j) maxv = std::max(maxv, row[j]);
      T sum = T(0);
      for (i64 j = 0; j <= i; ++j) {
        row[j] = std::exp(row[j] - maxv);
        sum += row[j];
      }
      const T inv = T(1) / sum;
      for (i64 j = 0; j <= i; ++j) row[j] *= inv;
      for (i64 j = i + 1; j < T_len; ++j) row[j] = T(0);
    }

    // O = P·V
    gemm<T>(false, false, T_len, dh, T_len, T(1), ph, T_len, vh, dh, T(0), oh, dh);
  });

  T* ctxf = arena.alloc_n<T>(nbt);
  unpack_heads<T>(ctx, ctxf, B, T_len, H, dh, /*accumulate=*/false);

  const bool rg = any_requires_grad(x, wq, wk, wv, wo);
  Tensor<T> out = make_tensor<T>(arena, Shape{B, T_len, C}, rg);
  gemm<T>(false, false, rows, C, C, T(1), ctxf, C, wo.data, C, T(0), out.data, C);

  if (rg) {
    // Backward scratch, pre-allocated so the closure never touches the arena.
    //
    // qf/kf/vf and ctx are dead once the forward pass has packed them, so the
    // backward reuses their storage instead of allocating four more [B*T,C]
    // buffers. At the 12M config that is ~25% of attention's activation memory.
    T* dqf = qf;
    T* dkf = kf;
    T* dvf = vf;
    T* dctx = ctx;
    T* dctxf = arena.alloc_n<T>(nbt);
    T* dq = arena.alloc_n<T>(nbt);
    T* dk = arena.alloc_n<T>(nbt);
    T* dv = arena.alloc_n<T>(nbt);
    T* dprob = arena.alloc_n<T>(nprob);

    out.node_id = record<T>(
        "attention", {x.node_id, wq.node_id, wk.node_id, wv.node_id, wo.node_id},
        [x, wq, wk, wv, wo, out, q, k, v, prob, ctxf, dctxf, dctx, dq, dk, dv, dqf, dkf, dvf,
         dprob, B, T_len, C, H, dh, rows, scale, theta, pos_offset = p.pos_offset, nbt] {
          // ── output projection ──
          // dCtx = dOut·woᵀ ;  dwo = Ctxᵀ·dOut
          gemm<T>(false, true, rows, C, C, T(1), out.grad, C, wo.data, C, T(0), dctxf, C);
          if (wo.grad) {
            gemm<T>(true, false, C, C, rows, T(1), ctxf, C, out.grad, C, T(1), wo.grad, C);
          }
          pack_heads<T>(dctxf, dctx, B, T_len, H, dh);

          zero_<T>(dq, static_cast<i64>(nbt));
          zero_<T>(dk, static_cast<i64>(nbt));
          zero_<T>(dv, static_cast<i64>(nbt));

          // ── per-head attention backward ──
          ThreadPool::global().parallel_for(
              static_cast<std::size_t>(B * H), [&](std::size_t idx) {
                const i64 bh = static_cast<i64>(idx);
                const T* qh = q + bh * T_len * dh;
                const T* kh = k + bh * T_len * dh;
                const T* vh = v + bh * T_len * dh;
                const T* ph = prob + bh * T_len * T_len;
                const T* doh = dctx + bh * T_len * dh;
                T* dph = dprob + bh * T_len * T_len;
                T* dqh = dq + bh * T_len * dh;
                T* dkh = dk + bh * T_len * dh;
                T* dvh = dv + bh * T_len * dh;

                // dV = Pᵀ·dO
                gemm<T>(true, false, T_len, dh, T_len, T(1), ph, T_len, doh, dh, T(1), dvh, dh);
                // dP = dO·Vᵀ
                gemm<T>(false, true, T_len, T_len, dh, T(1), doh, dh, vh, dh, T(0), dph, T_len);

                // dS = softmax_bwd(P, dP), then RE-MASK. Without the re-mask,
                // gradient leaks into positions the forward pass masked out.
                // The scale from S = Q·Kᵀ·scale folds in here.
                for (i64 i = 0; i < T_len; ++i) {
                  const T* prow = ph + i * T_len;
                  T* drow = dph + i * T_len;
                  T dot = T(0);
                  for (i64 j = 0; j <= i; ++j) dot += drow[j] * prow[j];
                  for (i64 j = 0; j <= i; ++j) {
                    drow[j] = prow[j] * (drow[j] - dot) * scale;
                  }
                  for (i64 j = i + 1; j < T_len; ++j) drow[j] = T(0);
                }

                // dQ = dS·K ;  dK = dSᵀ·Q
                gemm<T>(false, false, T_len, dh, T_len, T(1), dph, T_len, kh, dh, T(1), dqh, dh);
                gemm<T>(true, false, T_len, dh, T_len, T(1), dph, T_len, qh, dh, T(1), dkh, dh);
              });

          // RoPE is orthogonal ⇒ its VJP is the same kernel with negated angles.
          rope_apply<T>(dq, B, H, T_len, dh, pos_offset, theta, /*inverse=*/true);
          rope_apply<T>(dk, B, H, T_len, dh, pos_offset, theta, /*inverse=*/true);

          unpack_heads<T>(dq, dqf, B, T_len, H, dh, /*accumulate=*/false);
          unpack_heads<T>(dk, dkf, B, T_len, H, dh, /*accumulate=*/false);
          unpack_heads<T>(dv, dvf, B, T_len, H, dh, /*accumulate=*/false);

          // ── input projections ──
          if (x.grad) {
            gemm<T>(false, true, rows, C, C, T(1), dqf, C, wq.data, C, T(1), x.grad, C);
            gemm<T>(false, true, rows, C, C, T(1), dkf, C, wk.data, C, T(1), x.grad, C);
            gemm<T>(false, true, rows, C, C, T(1), dvf, C, wv.data, C, T(1), x.grad, C);
          }
          if (wq.grad) {
            gemm<T>(true, false, C, C, rows, T(1), x.data, C, dqf, C, T(1), wq.grad, C);
          }
          if (wk.grad) {
            gemm<T>(true, false, C, C, rows, T(1), x.data, C, dkf, C, T(1), wk.grad, C);
          }
          if (wv.grad) {
            gemm<T>(true, false, C, C, rows, T(1), x.data, C, dvf, C, T(1), wv.grad, C);
          }
        });
  }
  return out;
}

template class KVCache<f32>;
template class KVCache<f64>;
template Tensor<f32> attention<f32>(Arena&, const Tensor<f32>&, const Tensor<f32>&,
                                    const Tensor<f32>&, const Tensor<f32>&, const Tensor<f32>&,
                                    const AttentionParams&);
template Tensor<f64> attention<f64>(Arena&, const Tensor<f64>&, const Tensor<f64>&,
                                    const Tensor<f64>&, const Tensor<f64>&, const Tensor<f64>&,
                                    const AttentionParams&);

}  // namespace csllm
